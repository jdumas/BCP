// Diagnostic-only implementation for the GIF_controlled validation target.
#include "ControlledValidation.h"

#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace bcp::cv {
namespace {

struct Artifact {
    std::string name, file, dtype, stage, indexSpace;
    size_t rows = 0, cols = 0;
    bool fortranOrder = false;
};

struct State {
    bool initialized = false;
    std::filesystem::path dir, replayDir;
    Order order = Order::Index;
    ReplayStage replay = ReplayStage::None;
    std::map<std::string, long long> meta;
    std::vector<Artifact> artifacts;
    std::vector<std::string> events;
};

State& state() {
    static State s;
    return s;
}

const State& ready() {
    if (!state().initialized) throw std::runtime_error("controlled validation used before bcp::cv::init()");
    return state();
}

std::string jsonString(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) out << "\\u00" << std::hex << (c >> 4) << (c & 15) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

void writeManifest() {
    const State& s = state();
    const auto tmp = s.dir / "manifest.json.tmp";
    {
        std::ofstream out(tmp);
        out << "{\n  \"schema\": \"bcp-controlled-validation/1\",\n"
            << "  \"diagnostic_only\": true,\n"
            << "  \"order\": " << jsonString(orderName()) << ",\n"
            << "  \"replay_stage\": " << jsonString(replayStageName()) << ",\n"
            << "  \"replay_dir\": " << jsonString(s.replayDir.string()) << ",\n"
            << "  \"index_base\": 0,\n  \"meta\": {";
        bool first = true;
        for (const auto& [k, v] : s.meta) {
            out << (first ? "\n" : ",\n") << "    " << jsonString(k) << ": " << v;
            first = false;
        }
        out << "\n  },\n  \"artifacts\": [";
        for (size_t i = 0; i < s.artifacts.size(); ++i) {
            const Artifact& a = s.artifacts[i];
            out << (i ? ",\n" : "\n") << "    {\"name\": " << jsonString(a.name) << ", \"file\": " << jsonString(a.file)
                << ", \"dtype\": " << jsonString(a.dtype) << ", \"shape\": [" << a.rows << ", " << a.cols << "]"
                << ", \"fortran_order\": " << (a.fortranOrder ? "true" : "false")
                << ", \"stage\": " << jsonString(a.stage) << ", \"index_space\": " << jsonString(a.indexSpace) << "}";
        }
        out << "\n  ],\n  \"events\": [";
        for (size_t i = 0; i < s.events.size(); ++i) out << (i ? ",\n" : "\n") << "    " << jsonString(s.events[i]);
        out << "\n  ]\n}\n";
        if (!out) throw std::runtime_error("cannot write controlled manifest: " + tmp.string());
    }
    std::filesystem::rename(tmp, s.dir / "manifest.json");
}

void writeNpy(const std::filesystem::path& path, const char* descr, size_t rows, size_t cols,
              bool fortranOrder, const void* data, size_t itemSize) {
    std::ostringstream h;
    h << "{'descr': '" << descr << "', 'fortran_order': " << (fortranOrder ? "True" : "False")
      << ", 'shape': (" << rows << ", " << cols << "), }";
    std::string header = h.str();
    const size_t unpadded = 10 + header.size() + 1;
    header += std::string((64 - unpadded % 64) % 64, ' ') + "\n";
    const uint16_t headerLength = static_cast<uint16_t>(header.size());
    const auto tmp = path.string() + ".tmp";
    if (std::filesystem::exists(path) || std::filesystem::exists(tmp))
        throw std::runtime_error("refusing to overwrite controlled artifact: " + path.string());
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write("\x93NUMPY\x01\x00", 8);
        const unsigned char len[2] = {static_cast<unsigned char>(headerLength & 0xff),
                                      static_cast<unsigned char>(headerLength >> 8)};
        out.write(reinterpret_cast<const char*>(len), 2);
        out << header;
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(rows * cols * itemSize));
        if (!out) throw std::runtime_error("cannot write controlled artifact: " + path.string());
    }
    std::filesystem::rename(tmp, path);
}

// Parses the subset of .npy v1 written above (and by NumPy for 2-D arrays).
std::vector<char> readNpy(const std::filesystem::path& path, const char* descr, size_t rows, size_t cols,
                          size_t itemSize, bool& fortranOrder) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("missing replay artifact: " + path.string());
    char magic[8];
    unsigned char len[2];
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(len), 2);
    if (!in || std::memcmp(magic, "\x93NUMPY\x01\x00", 8) != 0)
        throw std::runtime_error("not an .npy v1.0 file: " + path.string());
    std::string header(static_cast<size_t>(len[0] | (len[1] << 8)), '\0');
    in.read(&header[0], static_cast<std::streamsize>(header.size()));
    auto has = [&](const std::string& s) { return header.find(s) != std::string::npos; };
    std::ostringstream shape;
    shape << "'shape': (" << rows << ", " << cols << ")";
    if (!in || !has(std::string("'descr': '") + descr + "'") || !has(shape.str()))
        throw std::runtime_error("replay artifact dtype/shape mismatch for " + path.string() + ": expected " + descr +
                                 " " + shape.str() + ", header " + header);
    if (has("'fortran_order': True")) fortranOrder = true;
    else if (has("'fortran_order': False")) fortranOrder = false;
    else throw std::runtime_error("replay artifact has no fortran_order flag: " + path.string());
    if ((cols && rows > std::numeric_limits<size_t>::max() / cols) ||
        rows * cols > std::numeric_limits<size_t>::max() / itemSize)
        throw std::runtime_error("replay artifact size overflow: " + path.string());
    std::vector<char> data(rows * cols * itemSize);
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("replay artifact payload size mismatch: " + path.string());
    return data;
}

void registerArtifact(Artifact a) {
    auto& list = state().artifacts;
    for (const Artifact& existing : list)
        if (existing.name == a.name) throw std::runtime_error("controlled artifact written twice: " + a.name);
    list.push_back(std::move(a));
    writeManifest();
}

}  // namespace

void init() {
    State& s = state();
    if (s.initialized) return;
    const char* dir = std::getenv("BCP_CONTROLLED_DIR");
    if (!dir || !dir[0]) throw std::runtime_error("GIF_controlled requires BCP_CONTROLLED_DIR (capture root)");
    s.dir = std::filesystem::absolute(dir);
    std::filesystem::create_directories(s.dir);
    if (std::filesystem::exists(s.dir / "manifest.json"))
        throw std::runtime_error("BCP_CONTROLLED_DIR already contains a manifest; use a fresh directory: " + s.dir.string());

    const std::string ord = std::getenv("BCP_CONTROLLED_ORDER") ? std::getenv("BCP_CONTROLLED_ORDER") : "index";
    if (ord == "index" || ord.empty()) s.order = Order::Index;
    else if (ord == "original") s.order = Order::Original;
    else throw std::runtime_error("BCP_CONTROLLED_ORDER must be index or original, got " + ord);

    const std::string stage = std::getenv("BCP_CONTROLLED_REPLAY_STAGE") ? std::getenv("BCP_CONTROLLED_REPLAY_STAGE") : "none";
    if (stage == "none" || stage.empty()) s.replay = ReplayStage::None;
    else if (stage == "basis") s.replay = ReplayStage::Basis;
    else if (stage == "pre-repair") s.replay = ReplayStage::PreRepair;
    else if (stage == "post-mvc") s.replay = ReplayStage::PostMvc;
    else throw std::runtime_error("BCP_CONTROLLED_REPLAY_STAGE must be none, basis, pre-repair or post-mvc, got " + stage);

    const char* replay = std::getenv("BCP_CONTROLLED_REPLAY_DIR");
    if (s.replay != ReplayStage::None) {
        if (!replay || !replay[0]) throw std::runtime_error("BCP_CONTROLLED_REPLAY_STAGE requires BCP_CONTROLLED_REPLAY_DIR");
        s.replayDir = std::filesystem::absolute(replay);
        if (!std::filesystem::is_directory(s.replayDir))
            throw std::runtime_error("BCP_CONTROLLED_REPLAY_DIR is not a directory: " + s.replayDir.string());
        if (std::filesystem::equivalent(s.replayDir, s.dir))
            throw std::runtime_error("BCP_CONTROLLED_REPLAY_DIR must differ from BCP_CONTROLLED_DIR");
    } else if (replay && replay[0]) {
        throw std::runtime_error("BCP_CONTROLLED_REPLAY_DIR given without BCP_CONTROLLED_REPLAY_STAGE");
    }
    s.initialized = true;
    event(std::string("controlled validation: order=") + orderName() + " replay_stage=" + replayStageName());
}

Order order() { return ready().order; }
ReplayStage replayStage() { return ready().replay; }

const char* orderName() { return state().order == Order::Index ? "index" : "original"; }

const char* replayStageName() {
    switch (state().replay) {
    case ReplayStage::None: return "none";
    case ReplayStage::Basis: return "basis";
    case ReplayStage::PreRepair: return "pre-repair";
    case ReplayStage::PostMvc: return "post-mvc";
    }
    return "unknown";
}

void setMeta(const std::string& key, long long value) {
    ready();
    state().meta[key] = value;
    writeManifest();
}

void event(const std::string& text) {
    std::cerr << "BCP_CONTROLLED " << text << std::endl;
    state().events.push_back(text);
    if (state().initialized || !state().dir.empty()) writeManifest();
}

void writeF64(const std::string& name, const std::string& stage, const std::string& indexSpace,
              size_t rows, size_t cols, const double* data, bool fortranOrder) {
    const State& s = ready();
    writeNpy(s.dir / (name + ".npy"), "<f8", rows, cols, fortranOrder, data, sizeof(double));
    registerArtifact({name, name + ".npy", "<f8", stage, indexSpace, rows, cols, fortranOrder});
}

void writeI32(const std::string& name, const std::string& stage, const std::string& indexSpace,
              size_t rows, size_t cols, const int32_t* data) {
    const State& s = ready();
    writeNpy(s.dir / (name + ".npy"), "<i4", rows, cols, false, data, sizeof(int32_t));
    registerArtifact({name, name + ".npy", "<i4", stage, indexSpace, rows, cols, false});
}

std::vector<double> readF64(const std::string& name, size_t rows, size_t cols, bool fortranOrder) {
    const State& s = ready();
    bool fileFortran = false;
    std::vector<char> raw = readNpy(s.replayDir / (name + ".npy"), "<f8", rows, cols, sizeof(double), fileFortran);
    std::vector<double> data(rows * cols);
    if (!raw.empty()) std::memcpy(data.data(), raw.data(), raw.size());
    for (double value : data)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite floating replay artifact: " + name);
    if (fileFortran == fortranOrder) return data;
    std::vector<double> converted(rows * cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            const double v = fileFortran ? data[c * rows + r] : data[r * cols + c];
            (fortranOrder ? converted[c * rows + r] : converted[r * cols + c]) = v;
        }
    return converted;
}

std::vector<int32_t> readI32(const std::string& name, size_t rows, size_t cols) {
    const State& s = ready();
    bool fileFortran = false;
    std::vector<char> raw = readNpy(s.replayDir / (name + ".npy"), "<i4", rows, cols, sizeof(int32_t), fileFortran);
    std::vector<int32_t> data(rows * cols);
    if (!raw.empty()) std::memcpy(data.data(), raw.data(), raw.size());
    if (fileFortran && cols > 1) {
        std::vector<int32_t> converted(rows * cols);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) converted[r * cols + c] = data[c * rows + r];
        return converted;
    }
    return data;
}

}  // namespace bcp::cv
