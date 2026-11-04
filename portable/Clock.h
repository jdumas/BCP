#pragma once
#include <chrono>

namespace bcp {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point end, Clock::time_point start) {
    return std::chrono::duration<double>(end-start).count();
}
}
