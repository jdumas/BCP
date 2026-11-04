#include "Utils/MatlabInterface.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 2) return 1;
    const auto temp = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(temp);
    setenv("TMPDIR", temp.c_str(), 1);
    setenv("BCP_OCTAVE_EXECUTABLE", "/nonexistent/bcp-octave-startup-test", 1);
    setenv("BCP_OCTAVE_TIMEOUT_MS", "1000", 1);
    bool failed = false;
    try {
        MatlabInterface::GetEngine();
    } catch (const std::exception& error) {
        failed = true;
        std::cout << "Startup failed as expected: " << error.what() << '\n';
    }
    if (!failed || !std::filesystem::is_empty(temp)) return 1;
    std::filesystem::remove(temp);
    return 0;
} catch(const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
