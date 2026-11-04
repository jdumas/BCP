// The subset of the PARDISO C API used by BCP, backed by Eigen SparseLU.
// Symmetric inputs use one-based upper-triangular CSR, as in the original code.
// This is a portability backend, not a performance-equivalent PARDISO replacement.
#include "PardisoAPI.h"
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Solver {
    Eigen::SparseMatrix<double> matrix;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    bool analyzed = false;
    bool factorized = false;
};
Eigen::SparseMatrix<double> matrix(int n, int type, const double* a, const int* ia, const int* ja) {
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i)
        for (int k = ia[i] - 1; k < ia[i + 1] - 1; ++k) {
            const int j = ja[k] - 1;
            if (j < 0 || j >= n || !std::isfinite(a[k])) throw std::runtime_error("Invalid CSR matrix");
            entries.emplace_back(i, j, a[k]);
            if ((type == 2 || type == -2) && i != j) entries.emplace_back(j, i, a[k]);
        }
    Eigen::SparseMatrix<double> result(n, n);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}
}

extern "C" {
void bcp_pardisoinit(void* handle, int*, int*, int* iparm, double* dparm, int* error) {
    auto** pt = static_cast<void**>(handle);
    std::fill(pt, pt + 64, nullptr);
    std::fill(iparm, iparm + 64, 0);
    std::fill(dparm, dparm + 64, 0.);
    pt[0] = new Solver;
    *error = 0;
}
void bcp_pardiso(void* handle, int*, int*, int* type, int* phase, int* n,
    double* a, int* ia, int* ja, int*, int* nrhs, int*, int*,
    double* b, double* x, int* error, double*) {
    auto** pt = static_cast<void**>(handle);
    *error = 0;
    if (*phase == -1) { delete static_cast<Solver*>(pt[0]); pt[0] = nullptr; return; }
    try {
        auto* s = static_cast<Solver*>(pt[0]);
        if (!s || *n <= 0) throw std::runtime_error("Uninitialized solver");
        if (*phase == 11 || *phase == 12 || *phase == 13 || *phase == 22) {
            s->matrix = matrix(*n, *type, a, ia, ja);
            if (*phase != 22) {
                s->factorized = false;
                s->lu.analyzePattern(s->matrix);
                s->analyzed = true;
            }
            if (*phase == 11) return;
            if (!s->analyzed) throw std::runtime_error("Missing symbolic analysis");
            s->lu.factorize(s->matrix);
            s->factorized = s->lu.info() == Eigen::Success;
            if (!s->factorized) throw std::runtime_error("SparseLU factorization failed");
        }
        if (*phase == 13 || *phase == 33) {
            if (!s->factorized || *nrhs < 1) throw std::runtime_error("No factors or right-hand sides");
            Eigen::Map<const Eigen::MatrixXd> rhs(b, *n, *nrhs);
            Eigen::Map<Eigen::MatrixXd> solution(x, *n, *nrhs);
            solution = s->lu.solve(rhs);
            if (s->lu.info() != Eigen::Success || !solution.allFinite()) throw std::runtime_error("SparseLU solve failed");
        } else if (*phase != 12 && *phase != 22) throw std::runtime_error("Unsupported PARDISO phase");
    } catch (const std::exception& e) {
        std::cerr << "Eigen linear solver: " << e.what() << '\n';
        *error = -4;
    }
}
}
