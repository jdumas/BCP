// Intel oneMKL backend for the original callers' private compatibility ABI.
// Keep all MKL controls/handles separate from Panua's differently defined iparm.
#include "PardisoAPI.h"
#include <mkl_pardiso.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

static_assert(std::is_same<MKL_INT, int>::value, "BCP requires oneMKL's LP64 interface");

namespace {
struct Solver {
    std::array<void*, 64> handle{};
    std::array<MKL_INT, 64> control{};
    std::vector<MKL_INT> permutation;
    MKL_INT type = 0;
    MKL_INT n = 0;
    bool analyzed = false;
    bool factorized = false;
};

void run(Solver& s, int phase, double* a, int* ia, int* ja, int nrhs,
         double* rhs, double* solution, int* legacy, int& error, int maxfct, int mnum, int msglvl) {
    // These controls have the same meaning in both APIs. In particular, do NOT
    // copy legacy iparm[2] (threads), [27] (reordering), or [35] (selected inverse).
    for (int i : {7, 10, 12}) s.control[i] = legacy[i];
    MKL_INT p = phase, count = nrhs;
    s.control[27] = 0; // double precision
    s.control[34] = 0; // one-based indices
    s.control[35] = 0; // no Schur complement
    ::pardiso(s.handle.data(), &maxfct, &mnum, &s.type, &p, &s.n, a, ia, ja,
              s.permutation.data(), &count, s.control.data(), &msglvl,
              rhs, solution, &error);
    for (int i : {6, 17, 18}) legacy[i] = s.control[i];
}
}

extern "C" {
void bcp_pardisoinit(void* handle, int* type, int*, int* legacy, double* dparm, int* error) {
    auto** pt = static_cast<void**>(handle);
    std::fill(pt, pt + 64, nullptr);
    std::fill(legacy, legacy + 64, 0);
    std::fill(dparm, dparm + 64, 0.0);
    auto state = std::make_unique<Solver>();
    state->type = *type;
    ::pardisoinit(state->handle.data(), type, state->control.data());
    state->control[0] = 1; // use initialized MKL controls, not automatic defaults
    state->control[1] = 3; // parallel fill-reducing reordering
    state->control[27] = 0;
    state->control[34] = 0;
    state->control[35] = 0;
    for (int i : {7, 10, 12}) legacy[i] = state->control[i];
    pt[0] = state.release();
    *error = 0;
}

void bcp_pardiso(void* handle, int* maxfct, int* mnum, int* type, int* phase, int* n,
                 double* a, int* ia, int* ja, int*, int* nrhs, int* legacy, int* msglvl,
                 double* b, double* x, int* error, double*) {
    auto** pt = static_cast<void**>(handle);
    auto* s = static_cast<Solver*>(pt[0]);
    *error = 0;
    if (!s) { *error = -1; return; }
    if (*phase == -1) {
        if (s->analyzed) run(*s, -1, a, ia, ja, 1, b, x, legacy, *error, *maxfct, *mnum, *msglvl);
        delete s;
        pt[0] = nullptr;
        return;
    }
    if (*n <= 0 || *type != s->type) { *error = -1; return; }
    const bool analyze = *phase == 11 || *phase == 12 || *phase == 13;
    if (analyze) {
        s->factorized = false;
        s->n = *n;
        s->permutation.assign(*n, 0);
    } else if (s->n != *n || !s->analyzed) { *error = -1; return; }

    if (*phase != 11 && *phase != 12 && *phase != 13 && *phase != 22 && *phase != 33) {
        *error = -1;
        return;
    }
    if ((*phase == 33 || *phase == 13) && *nrhs < 1) { *error = -1; return; }
    if (*phase == 33 && !s->factorized) { *error = -1; return; }
    // Old preprocess/reorder calls pass a dummy zero nrhs. MKL still expects a
    // positive count, even when the factorization phase does not access B/X.
    run(*s, *phase, a, ia, ja, (*phase == 33 || *phase == 13) ? *nrhs : 1,
        b, x, legacy, *error, *maxfct, *mnum, *msglvl);
    if (analyze) s->analyzed = true; // also release partially allocated factors on errors
    if (*phase == 12 || *phase == 13 || *phase == 22) s->factorized = *error == 0;
    if (*error) std::cerr << "MKL PARDISO: phase " << *phase << ", error " << *error << '\n';
}
}
