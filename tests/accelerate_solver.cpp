// Numerical/lifecycle regression tests against the real macOS framework, not a stub.
#include "PardisoAPI.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct System {
    void* pt[64]{};
    int type, n = 3, solver = 0, ip[64]{}, error = 0, dummy = 0, one = 1;
    double dp[64]{};
    // Upper-triangle CSR with explicit diagonal, including zero diagonals.
    std::vector<int> ia{1, 3, 5, 6}, ja{1, 2, 2, 3, 3};
    std::vector<double> a, b, x;
    explicit System(int kind, std::vector<double> values) : type(kind), a(std::move(values)) {
        bcp_pardisoinit(pt, &type, &solver, ip, dp, &error);
        require(error == 0, "Initialization failed");
    }
    int call(int phase, int nrhs = 0) {
        bcp_pardiso(pt, &one, &one, &type, &phase, &n, a.data(), ia.data(), ja.data(),
                    &dummy, &nrhs, ip, &dummy, b.data(), x.data(), &error, dp);
        return error;
    }
    ~System() { call(-1); }
    void solve(int nrhs) {
        b.resize(n * nrhs);
        x.assign(n * nrhs, 0.0);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) b[n*j+i] = 1.0 + i + 0.25*j;
        const auto original = b;
        require(call(33, nrhs) == 0, "Solve failed");
        require(b == original, "Solve overwrote RHS");
        for (int j = 0; j < nrhs; ++j) {
            std::vector<double> residual(n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int k = ia[i]-1; k < ia[i+1]-1; ++k) {
                    const int col = ja[k]-1;
                    residual[i] += a[k]*x[n*j+col];
                    if ((type == 2 || type == -2) && i != col)
                        residual[col] += a[k]*x[n*j+i];
                }
            for (int i = 0; i < n; ++i)
                require(std::abs(residual[i] - b[n*j+i]) < 1e-10, "Bad solve residual/orientation");
        }
    }
};

void symmetric(int type, std::vector<double> a) {
    System s(type, std::move(a));
    require(s.call(33, 1) != 0, "Solve before factorization accepted");
    require(s.call(22) != 0, "Factorization before analysis accepted");
    require(s.call(11) == 0, "Symbolic analysis failed with dummy nrhs");
    require(s.call(33, 1) != 0, "Symbolic analysis incorrectly enabled solving");
    require(s.call(22) == 0, "Numeric factorization failed with dummy nrhs");
    s.solve(2);
    s.solve(16);
    s.solve(3);
    // Same pattern, different values: reuse symbolic and numeric storage.
    for (double& value : s.a) value *= 2.0;
    require(s.call(22) == 0, "Numeric refactorization failed");
    s.solve(2);
    // In-place solve, also used by compatibility clients.
    s.b = {1, 2, 3}; s.x = s.b;
    require(s.call(33, 1) == 0, "Reference solve failed");
    const auto expected = s.x;
    int phase = 33, rhs = 1;
    bcp_pardiso(s.pt, &s.one, &s.one, &s.type, &phase, &s.n, s.a.data(), s.ia.data(), s.ja.data(),
                &s.dummy, &rhs, s.ip, &s.dummy, s.b.data(), s.b.data(), &s.error, s.dp);
    require(!s.error, "In-place solve failed");
    for (int i = 0; i < s.n; ++i) require(std::abs(s.b[i]-expected[i]) < 1e-10, "Bad in-place solve");
    require(s.call(11) == 0, "Repeated analysis failed");
    require(s.call(33, 1) != 0, "Reanalysis left stale numerical factors");
    require(s.call(12) == 0, "Combined analysis/factorization failed");
    s.solve(2);
    // A changed but still valid pattern must not reuse old symbolic analysis.
    s.ja[1] = 3;
    require(s.call(22) != 0, "Changed pattern accepted without reanalysis");
    require(s.call(33, 1) != 0, "Failed refactorization left stale factors");
    s.a[0] += 5; s.a[2] += 5;
    require(s.call(12) == 0, "Reanalysis of changed pattern failed");
    s.solve(1);
    require(s.call(-1) == 0 && s.pt[0] == nullptr, "Release failed");
    require(s.call(-1) == 0, "Repeated release failed");
}
void general(int type) {
    System s(type, {4, 1, 2, 3, 5, 1, 2});
    s.ia = {1, 3, 6, 8}; s.ja = {1, 3, 1, 2, 3, 2, 3};
    s.b = {1, 2, 3}; s.x.resize(3);
    require(s.call(13, 1) == 0, "Combined general solve failed");
    s.solve(2);
    for (double& value : s.a) value *= 0.5;
    require(s.call(22) == 0, "General refactorization failed");
    s.solve(3);
}
void failures() {
    System s(2, {1, 0, 1, 0, 1});
    require(s.call(12) == 0, "Initial SPD factorization failed");
    s.a.assign(5, 0.0);
    require(s.call(22) != 0, "Singular factorization accepted");
    s.b = {1, 2, 3}; s.x.resize(3);
    require(s.call(33, 1) != 0, "Singular refactor left stale factors");
    s.a = {2, 0, 2, 0, 2};
    require(s.call(22) == 0, "Recovery after failed factorization failed");
    s.solve(1);
    require(s.call(99) != 0, "Unsupported phase accepted");
    require(s.call(33, 0) != 0, "Zero RHS count accepted");
    s.ia[0] = 0;
    require(s.call(11) != 0, "Zero-based CSR accepted");
    require(s.call(33, 1) != 0, "Invalid analysis left stale factors");
    s.call(-1);
    s.type = 6;
    bcp_pardisoinit(s.pt, &s.type, &s.solver, s.ip, s.dp, &s.error);
    require(s.error != 0 && s.pt[0] == nullptr, "Unsupported matrix type accepted");
}
}
int main() try {
    symmetric(2, {4, 1, 4, 1, 3});
    symmetric(-2, {2, 1, -1, 1, 3});
    // Nonsingular indefinite matrix requiring a 2x2 pivot (zero leading diagonal).
    symmetric(-2, {0, 1, 0, 1, 2});
    general(1);
    general(11);
    failures();
    std::cout << "Accelerate sparse solver tests passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
