#include "GIF/GIF.h"
#include "Utils/MatlabInterface.h"
#include "Environment.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 12) {
        std::cerr << "Usage: GIF input.obj output.mat run_IDT allow_MVC_fix allow_local_fix "
                     "interior_faces boundary_segment_size curvature_rate outer_rate energy_threshold edge_lengths.mat\n";
        return 1;
    }
#ifdef BCP_DEFAULT_OCTAVE_EXECUTABLE
    if (!std::getenv("BCP_OCTAVE_EXECUTABLE"))
        bcp::set_environment("BCP_OCTAVE_EXECUTABLE", BCP_DEFAULT_OCTAVE_EXECUTABLE);
#endif
    MatlabInterface::GetEngine().AddScriptPath(BCP_MATLAB_SCRIPTS_DIR);
    GIF program;
    const bool success=program.run(
        std::filesystem::absolute(argv[1]).string(), std::filesystem::absolute(argv[2]).string(),
        std::string(argv[3])=="True", std::string(argv[4])=="True", std::string(argv[5])=="True",
        std::string(argv[11]).empty() ? "" : std::filesystem::absolute(argv[11]).string(),
        std::stoi(argv[6]), std::stoi(argv[7]), std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]));
    return success ? 0 : 1;
} catch(const std::exception& e) {
    std::cerr << "GIF: " << e.what() << '\n';
    return 1;
}
