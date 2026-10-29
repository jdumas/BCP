// MATLAB-free MAT v5 front-end for the original PT midpoint-angle computation.
#include "MatFile.h"
#include <chrono>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 4) {
        std::cerr << "Usage: PT <source_matrices.mat> <result.mat> 1\n";
        return 1;
    }
    if (std::string(argv[3]) != "1") throw std::runtime_error("Only midpoint-angle operation 1 is supported");
    const auto start = std::chrono::steady_clock::now();
    bcp::MatFile source(argv[1]);
    const auto f = source.read("f");
    const auto rings = source.read("one_rings");
    const auto integrated = source.read("integrated_angles");
    const auto sums = source.read("all_one_ring_angles_sums");
    if (f.cols() != 3 || rings.cols() == 0 || integrated.rows() != rings.rows() ||
        integrated.cols() != rings.cols() || sums.size() != rings.rows() ||
        (sums.rows() != 1 && sums.cols() != 1) ||
        !f.allFinite() || !rings.allFinite() || !integrated.allFinite() || !sums.allFinite())
        throw std::runtime_error("Invalid parallel-transport matrix shapes or values");
    Eigen::MatrixXd result(f.rows(), 3);
    for (Eigen::Index face = 0; face < f.rows(); ++face) {
        for (int corner = 0; corner < 3; ++corner) {
            const double vi = f(face, corner);
            if (vi < 0 || vi >= rings.rows() || vi != std::floor(vi))
                throw std::runtime_error("Invalid zero-based vertex index");
            const int v = static_cast<int>(vi);
            const double v1 = f(face, (corner + 1) % 3), v2 = f(face, (corner + 2) % 3);
            int i1 = -1, i2 = -1;
            for (Eigen::Index j = 0; j < rings.cols(); ++j) {
                if (rings(v,j) == v1) i1 = static_cast<int>(j);
                if (rings(v,j) == v2) i2 = static_cast<int>(j);
            }
            if (i1 < 0 || i2 < 0) throw std::runtime_error("Face neighbor missing from one-ring");
            double a1 = integrated(v,i1), a2 = integrated(v,i2);
            if (i1 == rings.cols() - 1 && i2 < i1 - 1) a2 = sums.data()[v];
            else if (i2 == rings.cols() - 1 && i1 < i2 - 1) a1 = sums.data()[v];
            result(face,corner) = (a1 + a2) / 2;
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    bcp::MatFile output(argv[2], true);
    output.write("angles_vertices_to_faces", result);
    output.write("total_time", Eigen::MatrixXd::Constant(1,1,seconds));
    std::cout << "Total time: " << seconds << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "PT: " << e.what() << '\n';
    return 1;
}
