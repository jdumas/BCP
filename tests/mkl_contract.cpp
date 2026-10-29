// Check translation/lifetime semantics without pretending to test real MKL.
#include "PardisoAPI.h"
#include <mkl_pardiso.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
int analyses = 0, factors = 0, solves = 0, releases = 0;
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}
extern "C" {
void pardisoinit(void* handle, const int* type, int* control) {
    check(*type == -2, "Incorrect matrix type");
    std::fill_n(static_cast<void**>(handle), 64, nullptr);
    std::fill_n(control, 64, 0);
    control[7] = 2;
    control[10] = control[12] = 1;
}
void pardiso(void*, const int*, const int*, const int* type, const int* phase,
             const int* n, const void* values, const int* ia, const int* ja, int* perm,
             const int* nrhs, int* control, const int*, void* rhs, void* solution, int* error) {
    check(*type == -2 && *n == 2, "Wrong MKL matrix dimensions or type");
    check(control[0] == 1 && control[1] == 3 && control[2] == 0 && control[27] == 0 &&
          control[34] == 0 && control[35] == 0, "Panua-only controls leaked into MKL");
    check(*nrhs >= 1 && perm != nullptr, "Invalid RHS count/permutation");
    *error = 0;
    if (*phase == -1) { ++releases; return; }
    check(ia[0] == 1 && ia[1] == 3 && ia[2] == 4 && ja[0] == 1, "CSR indexing changed");
    if (*phase == 11 || *phase == 12 || *phase == 13) ++analyses;
    const auto* a = static_cast<const double*>(values);
    if (*phase == 12 || *phase == 13 || *phase == 22) {
        ++factors;
        if (a[0] == -123) { *error = -4; return; }
    }
    if (*phase == 13 || *phase == 33) {
        ++solves;
        auto* b = static_cast<double*>(rhs);
        auto* x = static_cast<double*>(solution);
        const double determinant = a[0]*a[2]-a[1]*a[1];
        for (int j=0; j<*nrhs; ++j) {
            x[2*j] = (a[2]*b[2*j]-a[1]*b[2*j+1])/determinant;
            x[2*j+1] = (a[0]*b[2*j+1]-a[1]*b[2*j])/determinant;
        }
    }
}
}
int main() try {
    void* pt[64];
    int type=-2, solver=0, ip[64], error=0, phase, n=2, nrhs=0, dummy=0, one=1;
    double dp[64], a[]={2,1,-1}, b[]={3,0,1,-1}, x[4];
    int ia[]={1,3,4}, ja[]={1,2,2};
    bcp_pardisoinit(pt,&type,&solver,ip,dp,&error);
    // Legacy GIF uses all three with meanings incompatible with MKL.
    ip[2]=64; ip[27]=1; ip[35]=1;
    phase=11;
    auto run=[&] { bcp_pardiso(pt,&one,&one,&type,&phase,&n,a,ia,ja,&dummy,
                              &nrhs,ip,&dummy,b,x,&error,dp); };
    run();
    check(!error && analyses==1, "Symbolic analysis failed");
    phase=22; run(); run();
    check(!error && factors==2 && analyses==1, "Symbolic analysis was not reused");
    nrhs=2; phase=33; run();
    check(!error && std::abs(2*x[0]+x[1]-3)<1e-12 && std::abs(x[0]-x[1])<1e-12,
          "Multi-RHS solve translation failed");
    phase=11; run();
    phase=33; run();
    check(error!=0, "Reanalysis did not invalidate old factors");
    phase=22; run();
    check(!error, "Refactorization after reanalysis failed");
    const double unit_rhs[]={1,0,0,1};
    std::copy(unit_rhs,unit_rhs+4,b);
    phase=33; run();
    check(!error && solves==2 && std::abs(x[0]-1.0/3)<1e-12 &&
          std::abs(x[1]-1.0/3)<1e-12 && std::abs(x[3]+2.0/3)<1e-12,
          "Inverse-column solves failed");
    phase=-22; run();
    check(error!=0 && solves==2, "Unsupported Panua selected-inverse phase reached MKL");
    a[0]=-123; phase=22; run();
    check(error==-4, "MKL error was not propagated");
    phase=33; run();
    check(error!=0, "Solve accepted failed factors");
    phase=-1; run();
    check(!error && releases==1 && pt[0]==nullptr, "MKL handle was not released");
    std::cout << "MKL adapter contract passed (stub API; not a real-MKL numerical test)\n";
    return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
