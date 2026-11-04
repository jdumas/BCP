#pragma once
#include <algorithm>
#include <cstdint>

namespace bcp {
// Column count for GIF's dense unit right-hand-side batches. These replace
// Panua's selected inverse with inverse-column solves that reuse one factorization.
// The right-hand side B and solution X are both column-major rows x batch
// doubles. Each buffer is limited to 256 MiB, so B+X stays within 512 MiB,
// except when a single column already exceeds the budget. The cap of 32
// columns comes from measuring MKL's multi-RHS solve on a real 457,717-row
// GIF KKT: 16 -> 32 columns gave -8%, while 32 -> 64 gave -9% for twice the memory.
// rows is an LP64 PARDISO dimension (at most INT_MAX), and the arithmetic uses
// 64-bit integers, so rows * 8 bytes cannot overflow.
constexpr std::uint64_t kInverseColumnBufferBytes = std::uint64_t(256) << 20;
constexpr int kMaxInverseColumnBatch = 32;

inline int inverseColumnBatchSize(int rows, int columns) {
    const std::uint64_t columnBytes = std::uint64_t(std::max(rows, 1)) * sizeof(double);
    const std::uint64_t fitting = kInverseColumnBufferBytes / columnBytes;
    const std::uint64_t limit = std::uint64_t(std::max(1, std::min(columns, kMaxInverseColumnBatch)));
    return int(std::max<std::uint64_t>(1, std::min(fitting, limit)));
}
}
