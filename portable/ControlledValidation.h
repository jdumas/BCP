// Diagnostic-only capture/replay and canonical ordering for the GIF_controlled
// validation executable. Compiled ONLY when BCP_CONTROLLED_VALIDATION is defined;
// the production GIF target never includes this header.
#pragma once
#ifndef BCP_CONTROLLED_VALIDATION
#error "ControlledValidation.h is only for the GIF_controlled validation target"
#endif

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace bcp::cv {

// BCP_CONTROLLED_ORDER: index (default) sorts by stable CGAL mesh index;
// original keeps the production pointer-hash traversal (historical diagnostics).
enum class Order { Index, Original };
// BCP_CONTROLLED_REPLAY_STAGE: which captured state replaces live values.
enum class ReplayStage { None, Basis, PreRepair, PostMvc };

// Reads and validates the environment once; throws std::runtime_error when
// BCP_CONTROLLED_DIR is missing or a replay request is incomplete.
void init();
Order order();
ReplayStage replayStage();
const char* orderName();
const char* replayStageName();

// Records an integer fact (mesh sizes, DOFs, system dimensions) in the manifest.
void setMeta(const std::string& key, long long value);
// Appends an event line to the manifest and prints it to stderr.
void event(const std::string& text);

// Writes <BCP_CONTROLLED_DIR>/<name>.npy and registers it in manifest.json.
// fortranOrder=true means `data` is column-major (e.g. GMM dense matrices).
void writeF64(const std::string& name, const std::string& stage, const std::string& indexSpace,
              size_t rows, size_t cols, const double* data, bool fortranOrder);
void writeI32(const std::string& name, const std::string& stage, const std::string& indexSpace,
              size_t rows, size_t cols, const int32_t* data);

// Reads <BCP_CONTROLLED_REPLAY_DIR>/<name>.npy, requiring exact dtype and shape.
// The result uses the requested memory order regardless of the file's order.
std::vector<double> readF64(const std::string& name, size_t rows, size_t cols, bool fortranOrder);
std::vector<int32_t> readI32(const std::string& name, size_t rows, size_t cols);

// Same elements as the input set; index-sorted in Order::Index.
template <class Handle, class Set>
std::vector<Handle> ordered(const Set& set) {
    std::vector<Handle> result(set.begin(), set.end());
    if (order() == Order::Index)
        std::sort(result.begin(), result.end(), [](const Handle& a, const Handle& b) { return a->index() < b->index(); });
    return result;
}
// Integer vectors (system row ids) are sorted in Order::Index.
inline void canonicalize(std::vector<int>& values) {
    if (order() == Order::Index) std::sort(values.begin(), values.end());
}

}  // namespace bcp::cv
