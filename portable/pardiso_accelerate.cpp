// Apple Accelerate Sparse Solvers behind BCP's private PARDISO compatibility ABI.
// Legacy controls belong to Panua; do not pass them to Accelerate.
#include "PardisoAPI.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Solver {
    int type = 0, n = 0;
    bool hasSymbolic = false, hasNumeric = false, factorized = false;
    SparseOpaqueSymbolicFactorization symbolic{};
    SparseOpaqueFactorization_Double numeric{};
    std::vector<int> offsets, columns, rows, source;
    std::vector<long> starts;
    std::vector<double> values;

    bool symmetric() const { return type == 2 || type == -2; }
    void clear() {
        if (hasNumeric) SparseCleanup(numeric);
        if (hasSymbolic) SparseCleanup(symbolic);
        hasNumeric = hasSymbolic = factorized = false;
    }
    ~Solver() { clear(); }

    SparseMatrixStructure structure() {
        SparseAttributes_t attributes{};
        attributes.kind = symmetric() ? SparseSymmetric : SparseOrdinary;
        // Upper-triangle CSR is exactly lower-triangle CSC for a symmetric A.
        attributes.triangle = SparseLowerTriangle;
        return {n, n, starts.data(), rows.data(), attributes, 1};
    }
    void analyze(int size, const int* ia, const int* ja) {
        clear();
        n = size;
        if (!ia || !ja || ia[0] != 1) throw std::runtime_error("Expected one-based CSR");
        for (int i = 0; i < n; ++i) {
            if (ia[i + 1] < ia[i]) throw std::runtime_error("Invalid CSR row offsets");
            for (int k = ia[i] - 1; k < ia[i + 1] - 1; ++k)
                if (ja[k] < 1 || ja[k] > n || (symmetric() && ja[k] < i + 1) ||
                    (k > ia[i] - 1 && ja[k] <= ja[k - 1]))
                    throw std::runtime_error("Expected sorted CSR (upper triangle for symmetric matrices)");
        }
        const int nnz = ia[n] - 1;
        offsets.assign(ia, ia + n + 1);
        columns.assign(ja, ja + nnz);
        starts.assign(n + 1, 0);
        rows.resize(nnz);
        source.clear();
        values.clear();
        if (symmetric()) {
            for (int i = 0; i <= n; ++i) starts[i] = ia[i] - 1;
            for (int k = 0; k < nnz; ++k) rows[k] = ja[k] - 1;
        } else {
            // General matrices must retain their orientation. Build CSC once,
            // saving a value permutation for subsequent numerical factorizations.
            for (int j : columns) ++starts[j];
            for (int j = 0; j < n; ++j) starts[j + 1] += starts[j];
            auto next = starts;
            source.resize(nnz);
            values.resize(nnz);
            for (int i = 0; i < n; ++i)
                for (int k = ia[i] - 1; k < ia[i + 1] - 1; ++k) {
                    const long dest = next[ja[k] - 1]++;
                    rows[dest] = i;
                    source[dest] = k;
                }
        }
        // QR supports general matrices on older macOS too (sparse LU requires 15.5).
        const auto method = type == 2 ? SparseFactorizationCholesky :
                            type == -2 ? SparseFactorizationLDLT : SparseFactorizationQR;
        symbolic = SparseFactor(method, structure());
        hasSymbolic = true; // also clean up a partially constructed object on failure
        if (symbolic.status != SparseStatusOK) throw std::runtime_error("Symbolic analysis failed");
    }
    void factor(double* a, const int* ia, const int* ja) {
        factorized = false;
        if (!hasSymbolic || symbolic.status != SparseStatusOK || !a || !ia || !ja ||
            !std::equal(offsets.begin(), offsets.end(), ia) ||
            !std::equal(columns.begin(), columns.end(), ja))
            throw std::runtime_error("Missing analysis or changed sparsity pattern");
        for (size_t k = 0; k < columns.size(); ++k)
            if (!std::isfinite(a[k])) throw std::runtime_error("Non-finite matrix value");
        double* data = a;
        if (!symmetric()) {
            for (size_t k = 0; k < source.size(); ++k) values[k] = a[source[k]];
            data = values.data();
        }
        SparseMatrix_Double matrix{structure(), data};
        if (hasNumeric && numeric.numericFactorization) {
            SparseRefactor(matrix, &numeric);
        } else {
            if (hasNumeric) SparseCleanup(numeric);
            numeric = SparseFactor(symbolic, matrix);
            hasNumeric = true;
        }
        if (numeric.status != SparseStatusOK) throw std::runtime_error("Numerical factorization failed");
        factorized = true;
    }
    void solve(int nrhs, double* b, double* x) {
        if (!factorized || nrhs < 1 || !b || !x) throw std::runtime_error("Missing factors or right-hand sides");
        const size_t count = static_cast<size_t>(n) * nrhs;
        for (size_t k = 0; k < count; ++k)
            if (!std::isfinite(b[k])) throw std::runtime_error("Non-finite right-hand side");
        DenseMatrix_Double rhs{n, nrhs, n, {}, b}, solution{n, nrhs, n, {}, x};
        if (b == x) SparseSolve(numeric, solution);
        else SparseSolve(numeric, rhs, solution);
        for (size_t k = 0; k < count; ++k)
            if (!std::isfinite(x[k])) throw std::runtime_error("Non-finite solution");
    }
};
}

extern "C" {
void bcp_pardisoinit(void* handle, int* type, int* solver, int* legacy, double* dparm, int* error) {
    auto** pt = static_cast<void**>(handle);
    std::fill(pt, pt + 64, nullptr);
    std::fill(legacy, legacy + 64, 0);
    std::fill(dparm, dparm + 64, 0.0);
    *error = -1;
    if (*solver != 0 || (*type != 2 && *type != -2 && *type != 1 && *type != 11)) return;
    try {
        auto state = std::make_unique<Solver>();
        state->type = *type;
        pt[0] = state.release();
        *error = 0;
    } catch (const std::bad_alloc&) { *error = -2; }
}

void bcp_pardiso(void* handle, int* maxfct, int* mnum, int* type, int* phase, int* n,
                 double* a, int* ia, int* ja, int*, int* nrhs, int*, int*,
                 double* b, double* x, int* error, double*) {
    auto** pt = static_cast<void**>(handle);
    auto* s = static_cast<Solver*>(pt[0]);
    *error = 0;
    if (*phase == -1) { delete s; pt[0] = nullptr; return; }
    if (!s || *n <= 0 || *type != s->type || *maxfct != 1 || *mnum != 1) { *error = -1; return; }
    const bool analyze = *phase == 11 || *phase == 12 || *phase == 13;
    const bool factor = *phase == 12 || *phase == 13 || *phase == 22;
    const bool solve = *phase == 13 || *phase == 33;
    if ((!analyze && !factor && !solve) || (!analyze && s->n != *n) || (solve && *nrhs < 1)) {
        *error = -1;
        return;
    }
    try {
        if (analyze) s->analyze(*n, ia, ja);
        if (factor) s->factor(a, ia, ja);
        if (solve) s->solve(*nrhs, b, x);
    } catch (const std::bad_alloc&) {
        s->factorized = false;
        *error = -2;
    } catch (const std::exception& e) {
        s->factorized = false;
        std::cerr << "Accelerate sparse solver: phase " << *phase << ": " << e.what() << '\n';
        *error = -4;
    }
}
}
