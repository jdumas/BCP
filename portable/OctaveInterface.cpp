/**
 * Portable GNU Octave + matio implementation of MatlabInterface.
 *
 * Implements the subset of GIF/GIF/Utils/MatlabInterface.h linked by GIF and
 * its tests (raw mxArray, index and sparse-read APIs are omitted or throw) using:
 *   1. A persistent headless GNU Octave child process launched via POSIX
 *      pipe/fork/execvp (no shell interpolation, no GUI).
 *   2. Individual MAT v5 file transfers via matio inside a private per-engine
 *      temporary directory managed with RAII.
 *   3. Default MAT v5 output via save_default_options('-mat7-binary') so
 *      original .m scripts and scipy.io.loadmat work unmodified.
 */

#include "Utils/MatlabInterface.h"

#include <matio.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dirent.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define ERROR_LOG(msg)   std::cerr << "ERROR: "   << msg << std::endl
#define WARNING_LOG(msg) std::cerr << "WARNING: " << msg << std::endl
#define MESSAGE2_LOG(msg) std::cout << msg << std::flush

namespace {

constexpr const char* kTempTransferVar = "BCP_OCTAVE_TRANSFER_TMP__";
constexpr int kDefaultEvalTimeoutMs = 600000; // 10 minutes default per eval

int getEvalTimeoutMs() {
    const char* env = std::getenv("BCP_OCTAVE_TIMEOUT_MS");
    if (env && env[0] != '\0') {
        char* endp = nullptr;
        long val = std::strtol(env, &endp, 10);
        if (endp != env && val > 0 && val <= 86400000L) {
            return static_cast<int>(val);
        }
    }
    return kDefaultEvalTimeoutMs;
}

std::string trimWhitespace(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

// Escape a string for inclusion inside Octave single quotes: ' -> ''
std::string escapeOctaveSingleQuoted(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\'') {
            out += "''";
        } else {
            out += c;
        }
    }
    return out;
}

bool isSimpleIdentifier(const std::string& s) {
    if (s.empty()) return false;
    unsigned char first = static_cast<unsigned char>(s[0]);
    if (!(std::isalpha(first) || first == '_')) return false;
    for (size_t i = 1; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (!(std::isalnum(c) || c == '_')) return false;
    }
    return true;
}

// Validates variable names or struct.field expressions (possibly nested, e.g. GIF.F1).
bool isValidTargetExpression(const std::string& expr) {
    if (expr.empty()) return false;
    size_t start = 0;
    while (start < expr.size()) {
        size_t dot = expr.find('.', start);
        std::string part = (dot == std::string::npos)
                               ? expr.substr(start)
                               : expr.substr(start, dot - start);
        if (!isSimpleIdentifier(part)) {
            return false;
        }
        if (dot == std::string::npos) {
            break;
        }
        start = dot + 1;
        if (start >= expr.size()) {
            return false;
        }
    }
    return true;
}

void removeDirectoryRecursive(const std::string& dirPath) {
    if (dirPath.empty()) return;
    DIR* dir = ::opendir(dirPath.c_str());
    if (dir) {
        struct dirent* entry = nullptr;
        while ((entry = ::readdir(dir)) != nullptr) {
            if (std::strcmp(entry->d_name, ".") == 0 ||
                std::strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            std::string fullPath = dirPath + "/" + entry->d_name;
            struct stat st;
            if (::lstat(fullPath.c_str(), &st) == 0) {
                if (S_ISDIR(st.st_mode)) {
                    removeDirectoryRecursive(fullPath);
                } else {
                    ::unlink(fullPath.c_str());
                }
            }
        }
        ::closedir(dir);
    }
    ::rmdir(dirPath.c_str());
}

struct TempDirRAII {
    std::string path;

    TempDirRAII() {
        const char* baseTmp = std::getenv("TMPDIR");
        if (!baseTmp || baseTmp[0] == '\0') {
            baseTmp = "/tmp";
        }
        std::string tmpl = std::string(baseTmp) + "/bcp-octave-XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        char* created = ::mkdtemp(buf.data());
        if (!created) {
            throw std::runtime_error(std::string("Failed to create temporary directory: ") +
                                     std::strerror(errno));
        }
        ::chmod(created, 0700);
        path = created;
    }

    ~TempDirRAII() {
        if (!path.empty()) {
            removeDirectoryRecursive(path);
            path.clear();
        }
    }

    TempDirRAII(const TempDirRAII&) = delete;
    TempDirRAII& operator=(const TempDirRAII&) = delete;
};

struct MatFileCleanupRAII {
    std::string path;
    explicit MatFileCleanupRAII(std::string p) : path(std::move(p)) {}
    ~MatFileCleanupRAII() {
        if (!path.empty()) {
            ::unlink(path.c_str());
        }
    }
};

struct MatOpenRAII {
    mat_t* fp = nullptr;
    explicit MatOpenRAII(mat_t* f) : fp(f) {}
    ~MatOpenRAII() {
        if (fp) {
            Mat_Close(fp);
            fp = nullptr;
        }
    }
};

struct MatVarRAII {
    matvar_t* var = nullptr;
    explicit MatVarRAII(matvar_t* v) : var(v) {}
    ~MatVarRAII() {
        if (var) {
            Mat_VarFree(var);
            var = nullptr;
        }
    }
};

} // namespace

// Internal state backing the forward-declared `struct engine` in MatlabInterface.h
struct engine {
    pid_t pid = -1;
    int stdin_fd = -1;
    int stdout_fd = -1;
    std::unique_ptr<TempDirRAII> temp_dir;
    uint64_t seq = 0;

    ~engine() {
        closeProcess();
    }

    bool isAlive() {
        if (pid <= 0 || stdin_fd < 0 || stdout_fd < 0) {
            return false;
        }
        int status = 0;
        pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid || (w == -1 && errno == ECHILD)) {
            pid = -1;
            closePipes();
            return false;
        }
        return true;
    }

    std::string nextMatPath(const char* prefix) {
        ++seq;
        std::ostringstream oss;
        oss << temp_dir->path << "/" << prefix << "_" << seq << ".mat";
        return oss.str();
    }

    void closePipes() {
        if (stdin_fd >= 0) {
            ::close(stdin_fd);
            stdin_fd = -1;
        }
        if (stdout_fd >= 0) {
            ::close(stdout_fd);
            stdout_fd = -1;
        }
    }

    void closeProcess() {
        // Close stdin (sending EOF to octave-cli) and stdout without writing to pipe
        // so teardown can never raise SIGPIPE.
        closePipes();
        if (pid > 0) {
            int status = 0;
            bool exited = false;
            for (int i = 0; i < 20; ++i) {
                pid_t w = ::waitpid(pid, &status, WNOHANG);
                if (w == pid || (w == -1 && errno == ECHILD)) {
                    exited = true;
                    break;
                }
                ::usleep(10000); // 10ms
            }
            if (!exited) {
                ::kill(pid, SIGTERM);
                for (int i = 0; i < 20; ++i) {
                    pid_t w = ::waitpid(pid, &status, WNOHANG);
                    if (w == pid || (w == -1 && errno == ECHILD)) {
                        exited = true;
                        break;
                    }
                    ::usleep(10000);
                }
            }
            if (!exited) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &status, 0);
            }
            pid = -1;
        }
    }
};

namespace {

// A dead Octave process loses the whole workspace; only the constructor or an
// explicit GetEngine(true) may start a fresh one.
void throwIfEngineLost(Engine* ep, const char* caller) {
    if (!ep || !ep->isAlive()) {
        std::string msg = std::string("MatlabInterface::") + caller +
                          ": Octave engine is not running; its workspace was lost";
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
}

[[noreturn]] void throwUnsupportedSparseRead(const char* method) {
    throw std::runtime_error(std::string("MatlabInterface::") + method +
                             ": reading sparse matrices is not supported by the Octave backend");
}

bool setNonBlockingAndCloexec(int fd) {
    int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl < 0) return false;
    if (::fcntl(fd, F_SETFL, fl | O_NONBLOCK) < 0) return false;
    if (::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) return false;
    return true;
}

bool writeAllToFd(int fd, const char* data, size_t len, int timeoutMs) {
    size_t written = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (written < len) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }
        int remainMs = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLOUT | POLLERR | POLLHUP;
        pfd.revents = 0;
        int pr = ::poll(&pfd, 1, std::max(1, remainMs));
        if (pr < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (pr == 0) {
            return false;
        }
        ssize_t n = ::write(fd, data + written, len - written);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;
        }
        if (n == 0) {
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

// Execute a command inside the persistent Octave child process using a try/catch +
// unique sentinel protocol. Sentinel tokens are C++-interpolated string literals and
// error reporting uses lasterror(), so user scripts calling `clear` or `clearvars`
// inside `code` cannot clobber the framing state.
int runOctaveCommand(Engine* ep, const std::string& code, std::string& capturedOutput) {
    capturedOutput.clear();
    if (!ep || !ep->isAlive()) {
        ERROR_LOG("Octave engine is not open");
        return 1;
    }

    // Ignore SIGPIPE in case child process died unexpectedly.
    struct sigaction sa_ign, sa_old;
    std::memset(&sa_ign, 0, sizeof(sa_ign));
    sa_ign.sa_handler = SIG_IGN;
    bool restoredSigpipe = (::sigaction(SIGPIPE, &sa_ign, &sa_old) == 0);
    struct SigpipeRestore {
        bool active;
        struct sigaction old_sa;
        ~SigpipeRestore() {
            if (active) {
                ::sigaction(SIGPIPE, &old_sa, nullptr);
            }
        }
    } sigRestore{restoredSigpipe, sa_old};

    uint64_t id = ++ep->seq;
    std::ostringstream sentinelPrefixOss;
    sentinelPrefixOss << "__BCP_OCTAVE_SENTINEL_" << ep->pid << "_" << id << "_";
    const std::string sentinelPrefix = sentinelPrefixOss.str();
    const std::string okToken = sentinelPrefix + "OK__\n";
    const std::string errToken = sentinelPrefix + "ERR__\n";

    std::ostringstream framed;
    framed << "try\n"
           << code << "\n"
           << "fprintf(1, '\\n" << sentinelPrefix << "OK__\\n');\n"
           << "catch\n"
           << "fprintf(2, 'OCTAVE_ERROR: %s\\n', lasterror().message);\n"
           << "fprintf(1, '\\n" << sentinelPrefix << "ERR__\\n');\n"
           << "end_try_catch\n"
           << "fflush(stdout); fflush(stderr);\n";

    const std::string payload = framed.str();
    const int timeoutMs = getEvalTimeoutMs();

    if (!writeAllToFd(ep->stdin_fd, payload.data(), payload.size(), timeoutMs)) {
        ERROR_LOG("Failed to write command to Octave process");
        return 1;
    }

    std::string accum;
    size_t searchFrom = 0;
    char buf[4096];
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

    while (true) {
        size_t okPos = accum.find(okToken, searchFrom);
        if (okPos != std::string::npos) {
            if (okPos > 0 && accum[okPos - 1] == '\n') {
                --okPos;
            }
            capturedOutput = accum.substr(0, okPos);
            return 0;
        }
        size_t errPos = accum.find(errToken, searchFrom);
        if (errPos != std::string::npos) {
            if (errPos > 0 && accum[errPos - 1] == '\n') {
                --errPos;
            }
            capturedOutput = accum.substr(0, errPos);
            return 1;
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            ERROR_LOG("Timed out waiting for Octave command completion");
            capturedOutput = accum;
            return 1;
        }
        int remainMs = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());

        struct pollfd pfd;
        pfd.fd = ep->stdout_fd;
        pfd.events = POLLIN | POLLHUP | POLLERR;
        pfd.revents = 0;

        int pr = ::poll(&pfd, 1, std::max(1, remainMs));
        if (pr < 0) {
            if (errno == EINTR) continue;
            ERROR_LOG("poll() failed reading from Octave: " << std::strerror(errno));
            capturedOutput = accum;
            return 1;
        }
        if (pr == 0) {
            ERROR_LOG("Timed out polling Octave stdout");
            capturedOutput = accum;
            return 1;
        }

        ssize_t n = ::read(ep->stdout_fd, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            ERROR_LOG("read() failed from Octave: " << std::strerror(errno));
            capturedOutput = accum;
            return 1;
        }
        if (n == 0) {
            ERROR_LOG("Octave process closed output pipe unexpectedly");
            capturedOutput = accum;
            return 1;
        }
        // Retain the overlap for a sentinel split across reads, without
        // rescanning all previous output on every chunk.
        searchFrom = accum.size() > errToken.size() ? accum.size() - errToken.size() : 0;
        accum.append(buf, static_cast<size_t>(n));
    }
}

void storeOutputBuffer(char* buffer, const std::string& text) {
    if (!buffer) return;
    size_t copyLen = std::min(text.size(), static_cast<size_t>(MAX_OUTPUT_BUFFER_SIZE - 1));
    if (copyLen > 0) {
        std::memcpy(buffer, text.data(), copyLen);
    }
    buffer[copyLen] = '\0';
}

// Writes a MAT v5 file containing variable `kTempTransferVar` and loads it into `targetName`
// in the Octave workspace (handling both top-level names and struct.field targets).
int transferMatFileToOctave(Engine* ep,
                            char* outputBuf,
                            const std::string& matPath,
                            const char* targetName) {
    if (!targetName) {
        throw std::runtime_error("OctaveInterface: null target variable name in transferMatFileToOctave");
    }
    std::string target = trimWhitespace(targetName);
    if (!isValidTargetExpression(target)) {
        std::string msg = std::string("Invalid Octave variable target expression: '") + targetName + "'";
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    std::string escapedPath = escapeOctaveSingleQuoted(matPath);
    std::ostringstream cmd;
    cmd << "load('" << escapedPath << "', '" << kTempTransferVar << "');\n"
        << target << " = " << kTempTransferVar << ";\n"
        << "clear " << kTempTransferVar << ";";

    std::string out;
    int res = runOctaveCommand(ep, cmd.str(), out);
    storeOutputBuffer(outputBuf, out);
    if (res != 0) {
        std::string msg = "Failed to load variable '" + target + "' into Octave: " + out;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    return 0;
}

// Saves expression `sourceName` from Octave into a temporary MAT v5 file under `kTempTransferVar`.
int saveOctaveVarToMatFile(Engine* ep,
                           char* outputBuf,
                           const char* sourceName,
                           const std::string& matPath) {
    if (!sourceName) {
        throw std::runtime_error("OctaveInterface: null source variable name in saveOctaveVarToMatFile");
    }
    std::string source = trimWhitespace(sourceName);
    if (!isValidTargetExpression(source)) {
        std::string msg = std::string("Invalid Octave variable source expression: '") + sourceName + "'";
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    std::string escapedPath = escapeOctaveSingleQuoted(matPath);
    std::ostringstream cmd;
    cmd << kTempTransferVar << " = " << source << ";\n"
        << "save('-v6', '" << escapedPath << "', '" << kTempTransferVar << "');\n"
        << "clear " << kTempTransferVar << ";";

    std::string out;
    int res = runOctaveCommand(ep, cmd.str(), out);
    storeOutputBuffer(outputBuf, out);
    if (res != 0) {
        std::string msg = "Variable '" + source + "' could not be saved from Octave: " + out;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    return 0;
}

bool readRealDataAsDoubles(matvar_t* var, std::vector<double>& out, size_t expectedCount) {
    out.assign(expectedCount, 0.0);
    if (expectedCount == 0) return true;
    if (!var || !var->data || var->isComplex) return false;

    switch (var->data_type) {
        case MAT_T_DOUBLE: {
            const double* p = static_cast<const double*>(var->data);
            std::copy(p, p + expectedCount, out.begin());
            return true;
        }
        case MAT_T_SINGLE: {
            const float* p = static_cast<const float*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_INT64: {
            const int64_t* p = static_cast<const int64_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_UINT64: {
            const uint64_t* p = static_cast<const uint64_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_INT32: {
            const int32_t* p = static_cast<const int32_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_UINT32: {
            const uint32_t* p = static_cast<const uint32_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_INT16: {
            const int16_t* p = static_cast<const int16_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_UINT16: {
            const uint16_t* p = static_cast<const uint16_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_INT8: {
            const int8_t* p = static_cast<const int8_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        case MAT_T_UINT8: {
            const uint8_t* p = static_cast<const uint8_t*>(var->data);
            for (size_t i = 0; i < expectedCount; ++i) out[i] = static_cast<double>(p[i]);
            return true;
        }
        default:
            return false;
    }
}

bool readComplexOrRealDataAsComplexDoubles(matvar_t* var,
                                           std::vector<std::complex<double>>& out,
                                           size_t expectedCount) {
    out.assign(expectedCount, std::complex<double>(0.0, 0.0));
    if (expectedCount == 0) return true;
    if (!var || !var->data) return false;

    if (!var->isComplex) {
        std::vector<double> realVals;
        if (!readRealDataAsDoubles(var, realVals, expectedCount)) return false;
        for (size_t i = 0; i < expectedCount; ++i) {
            out[i] = std::complex<double>(realVals[i], 0.0);
        }
        return true;
    }

    const mat_complex_split_t* split = static_cast<const mat_complex_split_t*>(var->data);
    if (!split || !split->Re || !split->Im) return false;

    switch (var->data_type) {
        case MAT_T_DOUBLE: {
            const double* re = static_cast<const double*>(split->Re);
            const double* im = static_cast<const double*>(split->Im);
            for (size_t i = 0; i < expectedCount; ++i) {
                out[i] = std::complex<double>(re[i], im[i]);
            }
            return true;
        }
        case MAT_T_SINGLE: {
            const float* re = static_cast<const float*>(split->Re);
            const float* im = static_cast<const float*>(split->Im);
            for (size_t i = 0; i < expectedCount; ++i) {
                out[i] = std::complex<double>(static_cast<double>(re[i]),
                                              static_cast<double>(im[i]));
            }
            return true;
        }
        default:
            return false;
    }
}

} // namespace

////////////////////////////////////////////////////////////
// Explicit instantiations used by GIF and the script-engine tests.

template int MatlabInterface::SetEngineRealMatrix<double>(const char* name, unsigned int m, unsigned int n, const double* vals, bool colmaj);
template int MatlabInterface::SetEngineComplexMatrix<double>(const char* name, unsigned int m, unsigned int n, const std::complex<double>* vals, bool colmaj);
template int MatlabInterface::SetEngineSparseRealMatrix<unsigned int, double>(
    const char* name, unsigned int n, const unsigned int* rowind, const unsigned int* colind, const double* vals, unsigned int nrows, unsigned int ncols);
template int MatlabInterface::SetEngineSparseComplexMatrix<unsigned int, double>(
    const char* name, unsigned int n, const unsigned int* rowind, const unsigned int* colind, const std::complex<double>* vals, unsigned int nrows, unsigned int ncols);
template int MatlabInterface::GetEngineRealMatrix<double>(const char* name, unsigned int m, unsigned int n, double* vals, bool colmaj);
template int MatlabInterface::GetEngineComplexMatrix<double>(const char* name, unsigned int m, unsigned int n, std::complex<double>* vals, bool colmaj);
template int MatlabInterface::GetEngineRealMatrix<double>(const char* name, unsigned int& m, unsigned int& n, std::vector<double>& vals, bool colmaj);
template int MatlabInterface::GetEngineComplexMatrix<double>(const char* name, unsigned int& m, unsigned int& n, std::vector<std::complex<double>>& vals, bool colmaj);

////////////////////////////////////////////////////////////

MatlabInterface::MatlabInterface() : m_ep(nullptr), mOutputStringBuffer(nullptr) {
    mOutputStringBuffer = new char[MAX_OUTPUT_BUFFER_SIZE];
    std::memset(mOutputStringBuffer, 0, MAX_OUTPUT_BUFFER_SIZE);

    std::cout << "Starting Octave engine" << std::endl;

    try {
        EngineOpen();
        EngineSetupUI();
    } catch (...) {
        delete[] mOutputStringBuffer;
        mOutputStringBuffer = nullptr;
        throw;
    }
}

MatlabInterface::~MatlabInterface() {
    EngineClose();
    delete[] mOutputStringBuffer;
    mOutputStringBuffer = nullptr;
}

void MatlabInterface::EngineOpen() {
    if (m_ep) {
        EngineClose();
    }

    std::unique_ptr<Engine> ep(new Engine());
    ep->temp_dir.reset(new TempDirRAII());

    int inPipe[2] = {-1, -1};
    int outPipe[2] = {-1, -1};
    if (::pipe(inPipe) != 0) {
        std::string msg = std::string("Failed to create stdin pipe for Octave: ") + std::strerror(errno);
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    if (::pipe(outPipe) != 0) {
        int err = errno;
        ::close(inPipe[0]);
        ::close(inPipe[1]);
        std::string msg = std::string("Failed to create stdout pipe for Octave: ") + std::strerror(err);
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    const char* exeEnv = std::getenv("BCP_OCTAVE_EXECUTABLE");
    std::string exe = (exeEnv && exeEnv[0] != '\0') ? exeEnv : "octave-cli";

    pid_t pid = ::fork();
    if (pid < 0) {
        int err = errno;
        ::close(inPipe[0]);
        ::close(inPipe[1]);
        ::close(outPipe[0]);
        ::close(outPipe[1]);
        std::string msg = std::string("fork() failed for Octave engine: ") + std::strerror(err);
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    if (pid == 0) {
        // Child process: wire stdin and merged stdout/stderr, then execvp directly (no shell).
        ::dup2(inPipe[0], STDIN_FILENO);
        ::dup2(outPipe[1], STDOUT_FILENO);
        ::dup2(outPipe[1], STDERR_FILENO);

        ::close(inPipe[0]);
        ::close(inPipe[1]);
        ::close(outPipe[0]);
        ::close(outPipe[1]);

        char* const argv[] = {
            const_cast<char*>(exe.c_str()),
            const_cast<char*>("--quiet"),
            const_cast<char*>("--no-gui"),
            const_cast<char*>("--no-history"),
            const_cast<char*>("--no-init-file"),
            const_cast<char*>("--no-window-system"),
            nullptr
        };
        ::execvp(exe.c_str(), argv);
        std::fprintf(stderr, "FATAL: execvp('%s') failed: %s\n", exe.c_str(), std::strerror(errno));
        ::_exit(127);
    }

    // Parent process
    ::close(inPipe[0]);
    ::close(outPipe[1]);

    if (!setNonBlockingAndCloexec(inPipe[1]) || !setNonBlockingAndCloexec(outPipe[0])) {
        int err = errno;
        ::close(inPipe[1]);
        ::close(outPipe[0]);
        ep->pid = pid;
        ep->closeProcess();
        std::string msg = std::string("fcntl() failed configuring Octave pipes: ") + std::strerror(err);
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    ep->pid = pid;
    ep->stdin_fd = inPipe[1];
    ep->stdout_fd = outPipe[0];

    std::ostringstream initCmd;
    initCmd << "more off;\n"
            << "page_screen_output(0);\n"
            << "page_output_immediately(1);\n"
            << "save_default_options('-mat7-binary');";

    std::string initOut;
    if (runOctaveCommand(ep.get(), initCmd.str(), initOut) != 0) {
        std::string msg = "Can't initialize Octave engine (" + exe + "): " + initOut;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    m_ep = ep.release();
}

void MatlabInterface::EngineSetupUI() {
    // Headless Octave: no GUI setup required.
}

void MatlabInterface::EngineClose() {
    if (m_ep) {
        delete m_ep;
        m_ep = nullptr;
    }
}

template <typename T>
int MatlabInterface::SetEngineRealMatrix(const char* name,
                                         unsigned int m,
                                         unsigned int n,
                                         const T* vals,
                                         bool colmaj) {
    if (!m_ep || !m_ep->isAlive() || !name) {
        throw std::runtime_error("Octave engine is not open or null variable name in SetEngineRealMatrix");
    }

    const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::vector<double> colMajorData(total, 0.0);
    if (vals != nullptr) {
        if (colmaj) {
            for (size_t i = 0; i < total; ++i) {
                colMajorData[i] = static_cast<double>(vals[i]);
            }
        } else {
            for (unsigned int j = 0; j < n; ++j) {
                for (unsigned int i = 0; i < m; ++i) {
                    size_t idxM = static_cast<size_t>(j) * m + i;
                    size_t idx = static_cast<size_t>(i) * n + j;
                    colMajorData[idxM] = static_cast<double>(vals[idx]);
                }
            }
        }
    }

    std::string matPath = m_ep->nextMatPath("put_real");
    MatFileCleanupRAII cleanup(matPath);

    mat_t* matfp = Mat_CreateVer(matPath.c_str(), nullptr, MAT_FT_MAT5);
    if (!matfp) {
        std::string msg = "Mat_CreateVer failed for " + matPath;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    MatOpenRAII matClose(matfp);

    size_t dims[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};
    matvar_t* matvar = Mat_VarCreate(kTempTransferVar,
                                     MAT_C_DOUBLE,
                                     MAT_T_DOUBLE,
                                     2,
                                     dims,
                                     colMajorData.data(),
                                     0);
    if (!matvar) {
        std::string msg = std::string("Mat_VarCreate failed for real matrix ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    MatVarRAII varCleanup(matvar);

    if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0) {
        std::string msg = std::string("Mat_VarWrite failed for real matrix ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    Mat_Close(matClose.fp);
    matClose.fp = nullptr;

    return transferMatFileToOctave(m_ep, mOutputStringBuffer, matPath, name);
}

template <typename T>
int MatlabInterface::SetEngineComplexMatrix(const char* name,
                                            unsigned int m,
                                            unsigned int n,
                                            const std::complex<T>* vals,
                                            bool colmaj) {
    if (!m_ep || !m_ep->isAlive() || !name) {
        throw std::runtime_error("Octave engine is not open or null variable name in SetEngineComplexMatrix");
    }

    const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::vector<double> re(total, 0.0);
    std::vector<double> im(total, 0.0);

    if (vals != nullptr) {
        if (colmaj) {
            for (size_t i = 0; i < total; ++i) {
                re[i] = static_cast<double>(vals[i].real());
                im[i] = static_cast<double>(vals[i].imag());
            }
        } else {
            for (unsigned int j = 0; j < n; ++j) {
                for (unsigned int i = 0; i < m; ++i) {
                    size_t idxM = static_cast<size_t>(j) * m + i;
                    size_t idx = static_cast<size_t>(i) * n + j;
                    re[idxM] = static_cast<double>(vals[idx].real());
                    im[idxM] = static_cast<double>(vals[idx].imag());
                }
            }
        }
    }

    std::string matPath = m_ep->nextMatPath("put_cplx");
    MatFileCleanupRAII cleanup(matPath);

    mat_t* matfp = Mat_CreateVer(matPath.c_str(), nullptr, MAT_FT_MAT5);
    if (!matfp) {
        std::string msg = "Mat_CreateVer failed for " + matPath;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    MatOpenRAII matClose(matfp);

    size_t dims[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};
    mat_complex_split_t split = {re.data(), im.data()};
    matvar_t* matvar = Mat_VarCreate(kTempTransferVar,
                                     MAT_C_DOUBLE,
                                     MAT_T_DOUBLE,
                                     2,
                                     dims,
                                     &split,
                                     MAT_F_COMPLEX);
    if (!matvar) {
        std::string msg = std::string("Mat_VarCreate failed for complex matrix ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    MatVarRAII varCleanup(matvar);

    if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0) {
        std::string msg = std::string("Mat_VarWrite failed for complex matrix ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    Mat_Close(matClose.fp);
    matClose.fp = nullptr;

    return transferMatFileToOctave(m_ep, mOutputStringBuffer, matPath, name);
}

template <typename IndexType, typename ValueType>
int MatlabInterface::SetEngineEncodedSparseRealMatrix(const char* name,
                                                      unsigned int n,
                                                      const IndexType* rowind,
                                                      const IndexType* colind,
                                                      const ValueType* vals) {
    if (!m_ep || !m_ep->isAlive() || !name) {
        throw std::runtime_error("Octave engine is not open or null variable name in SetEngineEncodedSparseRealMatrix");
    }

    std::vector<double> colMajorTriplets(static_cast<size_t>(n) * 3, 0.0);
    if (vals != nullptr) {
        assert(rowind != nullptr);
        assert(colind != nullptr);
        for (unsigned int i = 0; i < n; ++i) {
            colMajorTriplets[0 * static_cast<size_t>(n) + i] = static_cast<double>(rowind[i] + 1);
            colMajorTriplets[1 * static_cast<size_t>(n) + i] = static_cast<double>(colind[i] + 1);
            colMajorTriplets[2 * static_cast<size_t>(n) + i] = static_cast<double>(vals[i]);
        }
    }

    return SetEngineRealMatrix<double>(name, n, 3, colMajorTriplets.data(), true);
}

template <typename IndexType, typename ValueType>
int MatlabInterface::SetEngineSparseRealMatrix(const char* name,
                                               unsigned int n,
                                               const IndexType* rowind,
                                               const IndexType* colind,
                                               const ValueType* vals,
                                               unsigned int nrows,
                                               unsigned int ncols) {
    if (!name) {
        throw std::runtime_error("Null variable name in SetEngineSparseRealMatrix");
    }
    std::string target = trimWhitespace(name);
    if (!isValidTargetExpression(target)) {
        std::string msg = std::string("Invalid sparse matrix target name: ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    int res = SetEngineEncodedSparseRealMatrix(target.c_str(), n, rowind, colind, vals);
    if (res != 0) return res;

    std::ostringstream cmd;
    if (ncols == 0 || nrows == 0) {
        cmd << target << " = sparse(" << target << "(:,1), " << target << "(:,2), " << target << "(:,3));";
    } else {
        assert(ncols > 0 && nrows > 0);
        cmd << target << " = sparse(" << target << "(:,1), " << target << "(:,2), " << target << "(:,3), "
            << nrows << ", " << ncols << ");";
    }
    return Eval(cmd.str().c_str());
}

template <typename IndexType, typename ValueType>
int MatlabInterface::SetEngineEncodedSparseComplexMatrix(const char* name,
                                                         unsigned int n,
                                                         const IndexType* rowind,
                                                         const IndexType* colind,
                                                         const std::complex<ValueType>* vals) {
    if (!m_ep || !m_ep->isAlive() || !name) {
        throw std::runtime_error("Octave engine is not open or null variable name in SetEngineEncodedSparseComplexMatrix");
    }

    std::vector<std::complex<double>> colMajorTriplets(static_cast<size_t>(n) * 3,
                                                       std::complex<double>(0.0, 0.0));
    if (vals != nullptr) {
        assert(rowind != nullptr);
        assert(colind != nullptr);
        for (unsigned int i = 0; i < n; ++i) {
            colMajorTriplets[0 * static_cast<size_t>(n) + i] =
                std::complex<double>(static_cast<double>(rowind[i] + 1), 0.0);
            colMajorTriplets[1 * static_cast<size_t>(n) + i] =
                std::complex<double>(static_cast<double>(colind[i] + 1), 0.0);
            colMajorTriplets[2 * static_cast<size_t>(n) + i] =
                std::complex<double>(static_cast<double>(vals[i].real()),
                                     static_cast<double>(vals[i].imag()));
        }
    }

    return SetEngineComplexMatrix<double>(name, n, 3, colMajorTriplets.data(), true);
}

template <typename IndexType, typename ValueType>
int MatlabInterface::SetEngineSparseComplexMatrix(const char* name,
                                                  unsigned int n,
                                                  const IndexType* rowind,
                                                  const IndexType* colind,
                                                  const std::complex<ValueType>* vals,
                                                  unsigned int nrows,
                                                  unsigned int ncols) {
    if (!name) {
        throw std::runtime_error("Null variable name in SetEngineSparseComplexMatrix");
    }
    std::string target = trimWhitespace(name);
    if (!isValidTargetExpression(target)) {
        std::string msg = std::string("Invalid sparse complex matrix target name: ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    int res = SetEngineEncodedSparseComplexMatrix(target.c_str(), n, rowind, colind, vals);
    if (res != 0) return res;

    // Note: in Octave, sparse(i, j, s, m, n) requires index vectors i and j to be real.
    // Because encoded complex triplets store i and j in columns 1 & 2 of a complex matrix,
    // taking real(...) on columns 1 and 2 works identically in both MATLAB and Octave.
    std::ostringstream cmd;
    if (ncols == 0 || nrows == 0) {
        cmd << target << " = sparse(real(" << target << "(:,1)), real(" << target << "(:,2)), "
            << target << "(:,3));";
    } else {
        assert(ncols > 0 && nrows > 0);
        cmd << target << " = sparse(real(" << target << "(:,1)), real(" << target << "(:,2)), "
            << target << "(:,3), " << nrows << ", " << ncols << ");";
    }
    return Eval(cmd.str().c_str());
}

int MatlabInterface::CreateAllZerosSparseMatrix(const char* name, int nRows, int nCols) {
    if (!name) {
        throw std::runtime_error("Null variable name in CreateAllZerosSparseMatrix");
    }
    std::string target = trimWhitespace(name);
    if (!isValidTargetExpression(target)) {
        std::string msg = std::string("Invalid sparse matrix target name: ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }
    std::ostringstream cmd;
    cmd << target << " = sparse(" << nRows << ", " << nCols << ");";
    return Eval(cmd.str().c_str());
}

int MatlabInterface::SetEngineStringArray(const char* name, unsigned int n, const char** val) {
    if (!m_ep || !m_ep->isAlive() || !name || !val) {
        throw std::runtime_error("Octave engine is not open or null argument in SetEngineStringArray");
    }
    std::string target = trimWhitespace(name);
    if (!isValidTargetExpression(target)) {
        std::string msg = std::string("Invalid string array target name: ") + name;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    std::ostringstream cmd;
    if (n == 0) {
        cmd << target << " = '';";
    } else if (n == 1) {
        std::string s = val[0] ? val[0] : "";
        cmd << target << " = '" << escapeOctaveSingleQuoted(s) << "';";
    } else {
        cmd << target << " = char(";
        for (unsigned int i = 0; i < n; ++i) {
            if (i > 0) cmd << ", ";
            std::string s = val[i] ? val[i] : "";
            cmd << "'" << escapeOctaveSingleQuoted(s) << "'";
        }
        cmd << ");";
    }
    return Eval(cmd.str().c_str());
}

template <typename T>
int MatlabInterface::GetEngineRealMatrix(const char* name,
                                         unsigned int m,
                                         unsigned int n,
                                         T* dest,
                                         bool colmaj) {
    assert(dest != nullptr);
    unsigned int actualM = 0;
    unsigned int actualN = 0;
    std::vector<T> vec;
    int res = GetEngineRealMatrix<T>(name, actualM, actualN, vec, colmaj);
    if (res != 0) return res;
    if (actualM != m || actualN != n) {
        std::ostringstream oss;
        oss << "GetEngineRealMatrix: expected size " << m << "x" << n << ". Got "
            << actualM << "x" << actualN << ".";
        ERROR_LOG(oss.str());
        throw std::runtime_error(oss.str());
    }
    std::copy(vec.begin(), vec.end(), dest);
    return 0;
}

template <typename T>
int MatlabInterface::GetEngineComplexMatrix(const char* name,
                                            unsigned int m,
                                            unsigned int n,
                                            std::complex<T>* dest,
                                            bool colmaj) {
    assert(dest != nullptr);
    unsigned int actualM = 0;
    unsigned int actualN = 0;
    std::vector<std::complex<T>> vec;
    int res = GetEngineComplexMatrix<T>(name, actualM, actualN, vec, colmaj);
    if (res != 0) return res;
    if (actualM != m || actualN != n) {
        std::ostringstream oss;
        oss << "GetEngineComplexMatrix: expected size " << m << "x" << n << ". Got "
            << actualM << "x" << actualN << ".";
        ERROR_LOG(oss.str());
        throw std::runtime_error(oss.str());
    }
    std::copy(vec.begin(), vec.end(), dest);
    return 0;
}

template <typename T>
int MatlabInterface::GetEngineRealMatrix(const char* name,
                                         unsigned int& m,
                                         unsigned int& n,
                                         std::vector<T>& dest,
                                         bool colmaj) {
    m = 0;
    n = 0;
    dest.clear();
    if (!m_ep || !m_ep->isAlive() || !name) {
        throw std::runtime_error("Octave engine is not open or null variable name in GetEngineRealMatrix");
    }

    std::string matPath = m_ep->nextMatPath("get_real");
    MatFileCleanupRAII cleanup(matPath);

    saveOctaveVarToMatFile(m_ep, mOutputStringBuffer, name, matPath);

    mat_t* matfp = Mat_Open(matPath.c_str(), MAT_ACC_RDONLY);
    if (!matfp) {
        throw std::runtime_error(std::string("Mat_Open failed reading matrix ") + name);
    }
    MatOpenRAII matClose(matfp);

    matvar_t* var = Mat_VarRead(matfp, kTempTransferVar);
    if (!var || var->rank != 2 || var->class_type == MAT_C_SPARSE) {
        if (var) Mat_VarFree(var);
        throw std::runtime_error(std::string("Mat_VarRead invalid dense matrix for ") + name);
    }
    MatVarRAII varCleanup(var);

    m = static_cast<unsigned int>(var->dims[0]);
    n = static_cast<unsigned int>(var->dims[1]);
    const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);

    std::vector<double> colMajDoubles;
    if (!readRealDataAsDoubles(var, colMajDoubles, total)) {
        m = 0;
        n = 0;
        throw std::runtime_error(std::string("Failed to decode real matrix data for ") + name);
    }

    dest.resize(total);
    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int i = 0; i < m; ++i) {
            size_t idxM = static_cast<size_t>(j) * m + i;
            size_t idx = colmaj ? idxM : (static_cast<size_t>(i) * n + j);
            dest[idx] = static_cast<T>(colMajDoubles[idxM]);
        }
    }
    return 0;
}

template <typename T>
int MatlabInterface::GetEngineComplexMatrix(const char* name,
                                            unsigned int& m,
                                            unsigned int& n,
                                            std::vector<std::complex<T>>& dest,
                                            bool colmaj) {
    m = 0;
    n = 0;
    dest.clear();
    if (!m_ep || !m_ep->isAlive() || !name) {
        throw std::runtime_error("Octave engine is not open or null variable name in GetEngineComplexMatrix");
    }

    std::string matPath = m_ep->nextMatPath("get_cplx");
    MatFileCleanupRAII cleanup(matPath);

    saveOctaveVarToMatFile(m_ep, mOutputStringBuffer, name, matPath);

    mat_t* matfp = Mat_Open(matPath.c_str(), MAT_ACC_RDONLY);
    if (!matfp) {
        throw std::runtime_error(std::string("Mat_Open failed reading complex matrix ") + name);
    }
    MatOpenRAII matClose(matfp);

    matvar_t* var = Mat_VarRead(matfp, kTempTransferVar);
    if (!var || var->rank != 2 || var->class_type == MAT_C_SPARSE) {
        if (var) Mat_VarFree(var);
        throw std::runtime_error(std::string("Mat_VarRead invalid complex matrix for ") + name);
    }
    MatVarRAII varCleanup(var);

    m = static_cast<unsigned int>(var->dims[0]);
    n = static_cast<unsigned int>(var->dims[1]);
    const size_t total = static_cast<size_t>(m) * static_cast<size_t>(n);

    std::vector<std::complex<double>> colMajComplex;
    if (!readComplexOrRealDataAsComplexDoubles(var, colMajComplex, total)) {
        m = 0;
        n = 0;
        throw std::runtime_error(std::string("Failed to decode complex matrix data for ") + name);
    }

    dest.resize(total);
    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int i = 0; i < m; ++i) {
            size_t idxM = static_cast<size_t>(j) * m + i;
            size_t idx = colmaj ? idxM : (static_cast<size_t>(i) * n + j);
            dest[idx] = std::complex<T>(static_cast<T>(colMajComplex[idxM].real()),
                                        static_cast<T>(colMajComplex[idxM].imag()));
        }
    }
    return 0;
}

int MatlabInterface::Eval(const char* matlab_code) {
    if (!matlab_code) {
        throw std::runtime_error("Null command passed to MatlabInterface::Eval");
    }
    throwIfEngineLost(m_ep, "Eval");

    std::string out;
    int res = runOctaveCommand(m_ep, matlab_code, out);
    storeOutputBuffer(mOutputStringBuffer, out);
    if (res != 0) {
        std::ostringstream oss;
        oss << "Error " << res << " running Octave command \"" << matlab_code << "\": " << out;
        ERROR_LOG(oss.str());
        throw std::runtime_error(oss.str());
    }
    return 0;
}

void MatlabInterface::EvalToCout(const char* matlab_code, bool saveCommandString) {
    if (!matlab_code) {
        throw std::runtime_error("Null command passed to MatlabInterface::EvalToCout");
    }
    if (saveCommandString) {
        SetEngineStringArray("LAST_EVAL_TO_COUT", 1, &matlab_code);
        Eval("if ~exist('EVAL_TO_COUT', 'var'); EVAL_TO_COUT={LAST_EVAL_TO_COUT}; end; "
             "EVAL_TO_COUT = [LAST_EVAL_TO_COUT; EVAL_TO_COUT(1:min(99,end))];");
    }

    Eval(matlab_code);

    if (mOutputStringBuffer && mOutputStringBuffer[0] != '\0') {
        MESSAGE2_LOG(mOutputStringBuffer);
    }
}

std::string MatlabInterface::EvalToString(const char* matlab_code, bool saveCommandString) {
    int res = 0;
    return EvalToString(matlab_code, res, saveCommandString);
}

std::string MatlabInterface::EvalToString(const char* matlab_code, int& res, bool saveCommandString) {
    if (!matlab_code) {
        res = 1;
        throw std::runtime_error("Null command passed to MatlabInterface::EvalToString");
    }
    if (saveCommandString) {
        SetEngineStringArray("LAST_EVAL_TO_COUT", 1, &matlab_code);
        Eval("if ~exist('EVAL_TO_COUT', 'var'); EVAL_TO_COUT={LAST_EVAL_TO_COUT}; end; "
             "EVAL_TO_COUT = [LAST_EVAL_TO_COUT; EVAL_TO_COUT(1:min(99,end))];");
    }

    res = Eval(matlab_code);
    return std::string(mOutputStringBuffer ? mOutputStringBuffer : "");
}

int MatlabInterface::AddScriptPath(const char* path) {
    if (!path) {
        throw std::runtime_error("Null path passed to MatlabInterface::AddScriptPath");
    }
    std::string escaped = escapeOctaveSingleQuoted(path);
    std::string cmd = "addpath('" + escaped + "');";
    return Eval(cmd.c_str());
}

MatlabInterface& MatlabInterface::GetEngine(bool restart) {
    static MatlabInterface matlab;
    if (restart) {
        matlab.EngineClose();
        matlab.EngineOpen();
    } else {
        throwIfEngineLost(matlab.m_ep, "GetEngine");
    }
    return matlab;
}

// The current GIF pipeline only sends sparse matrices to Octave. These readers are
// referenced by compiled-but-uncalled MatlabGMMDataExchange getters, so they exist
// only to link and fail loudly if a future caller relies on them.
int MatlabInterface::GetSparseRealMatrix(const char*, std::vector<unsigned int>&, std::vector<unsigned int>&,
                                         std::vector<double>&, unsigned int&, unsigned int&, unsigned int&) {
    throwUnsupportedSparseRead("GetSparseRealMatrix");
}

int MatlabInterface::GetSparseComplexMatrix(const char*, std::vector<unsigned int>&, std::vector<unsigned int>&,
                                            std::vector<std::complex<double>>&, unsigned int&, unsigned int&,
                                            unsigned int&) {
    throwUnsupportedSparseRead("GetSparseComplexMatrix");
}

int MatlabInterface::GetEngineEncodedSparseRealMatrix(const char*, std::vector<unsigned int>&,
                                                      std::vector<unsigned int>&, std::vector<double>&,
                                                      unsigned int&, unsigned int&) {
    throwUnsupportedSparseRead("GetEngineEncodedSparseRealMatrix");
}

int MatlabInterface::GetEngineEncodedSparseComplexMatrix(const char*, std::vector<unsigned int>&,
                                                         std::vector<unsigned int>&,
                                                         std::vector<std::complex<double>>&,
                                                         unsigned int&, unsigned int&) {
    throwUnsupportedSparseRead("GetEngineEncodedSparseComplexMatrix");
}

bool MatlabInterface::GetMatrixDimensions(const char* variableName,
                                          unsigned int& m,
                                          unsigned int& n) {
    m = 0;
    n = 0;
    if (!variableName || variableName[0] == '\0') {
        throw std::runtime_error("Null or empty variableName in GetMatrixDimensions");
    }

    std::string expr = trimWhitespace(variableName);
    if (!isValidTargetExpression(expr)) {
        throw std::runtime_error(std::string("Invalid variable expression in GetMatrixDimensions: ") + variableName);
    }

    throwIfEngineLost(m_ep, "GetMatrixDimensions");

    std::ostringstream cmd;
    cmd << "fprintf(1, '__BCP_DIMS__ %d %d\\n', size(" << expr << ", 1), size(" << expr << ", 2));";

    std::string out;
    int res = runOctaveCommand(m_ep, cmd.str(), out);
    storeOutputBuffer(mOutputStringBuffer, out);
    if (res != 0) {
        std::string msg = "GetMatrixDimensions failed for '" + expr + "': " + out;
        ERROR_LOG(msg);
        throw std::runtime_error(msg);
    }

    const std::string tag = "__BCP_DIMS__";
    size_t pos = out.find(tag);
    if (pos == std::string::npos) {
        throw std::runtime_error("GetMatrixDimensions missing dimension output for '" + expr + "'");
    }

    std::istringstream iss(out.substr(pos + tag.size()));
    long rows = 0;
    long cols = 0;
    if ((iss >> rows >> cols) && rows > 0 && cols > 0) {
        m = static_cast<unsigned int>(rows);
        n = static_cast<unsigned int>(cols);
        return true;
    }
    return false;
}
