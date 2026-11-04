#pragma once
#include <cstdlib>
#include <stdexcept>

namespace bcp {
inline void set_environment(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value) != 0) throw std::runtime_error("Cannot set environment variable");
#else
    if (setenv(name, value, 1) != 0) throw std::runtime_error("Cannot set environment variable");
#endif
}
}
