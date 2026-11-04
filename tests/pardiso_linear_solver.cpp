// Class-level regression for GIF's PardisoLinearSolver wrapper: direct
// multi-RHS buffers, GIF-style full/partial unit-column batches, aliasing,
// size/error paths and solver diagnostics. argv[1] names the linked backend.
#include "Utils/PardisoLinearSolver.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

// [K C^T; C 0] with SPD tridiagonal K (k-by-k) and two constraint rows:
// symmetric indefinite with inertia (k positive, 2 negative), like GIF's KKT.
Eigen::MatrixXd kktMatrix(int k) {
    const int n = k + 2;
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < k; ++i) {
        a(i, i) = 4.0 + 0.25 * i;
        if (i + 1 < k) a(i, i + 1) = a(i + 1, i) = -1.0;
    }
    for (int i = 0; i < k; ++i) {
        const int row = k + (2 * i < k ? 0 : 1);
        a(row, i) = a(i, row) = 1.0 + 0.1 * i;
    }
    return a;
}

// Upper triangle with every diagonal entry stored, as GIF::SetElementsForPARDISO does.
Eigen::SparseMatrix<double, Eigen::RowMajor> upperWithDiagonal(const Eigen::MatrixXd& a) {
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = i; j < a.cols(); ++j)
            if (i == j || a(i, j) != 0.0) triplets.emplace_back(i, j, a(i, j));
    Eigen::SparseMatrix<double, Eigen::RowMajor> m(a.rows(), a.cols());
    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();
    return m;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> fullPattern(const Eigen::MatrixXd& a) {
    return a.sparseView();
}

Eigen::Map<const Eigen::MatrixXd> view(const GMMDenseColMatrix& m) {
    return Eigen::Map<const Eigen::MatrixXd>(m.data(), m.nrows(), m.ncols());
}

GMMDenseColMatrix gmmFrom(const Eigen::MatrixXd& m) {
    GMMDenseColMatrix out(m.rows(), m.cols());
    std::copy(m.data(), m.data() + m.size(), out.begin());
    return out;
}

double backwardError(const Eigen::MatrixXd& a, const Eigen::MatrixXd& x, const Eigen::MatrixXd& b) {
    const double denominator = a.norm() * x.norm() + b.norm();
    return denominator == 0.0 ? (a * x - b).norm() : (a * x - b).norm() / denominator;
}

void checkSolution(const Eigen::MatrixXd& a, const GMMDenseColMatrix& x, const Eigen::MatrixXd& b,
                   const std::string& what) {
    check(x.nrows() == size_t(b.rows()) && x.ncols() == size_t(b.cols()), what + ": solution shape changed");
    check(view(x).allFinite(), what + ": non-finite solution");
    const double error = backwardError(a, view(x), b);
    check(error < 1e-13, what + ": backward error " + std::to_string(error));
    check((view(x) - a.fullPivLu().solve(b)).norm() < 1e-10 * (1.0 + view(x).norm()),
          what + ": differs from the dense oracle");
}

void testSymmetricIndefinite(const std::string& backend) {
    const int k = 6, n = k + 2;
    const Eigen::MatrixXd a = kktMatrix(k);
    PardisoLinearSolver solver(-2);
    check(solver.init(true), "KKT init failed");
    check(solver.createPardisoFormatMatrix(upperWithDiagonal(a)), "KKT matrix rejected");
    check(solver.preprocess(), "KKT factorization failed");
    const bool reportsDiagnostics = backend == "MKL" || backend == "PANUA";
    const PardisoSolverDiagnostics factored = solver.diagnostics();
    if (reportsDiagnostics) {
        check(factored.factorNonzeros >= n, "Factor nonzeros were not reported");
        check(factored.positiveEigenvalues == k && factored.negativeEigenvalues == 2,
              "KKT inertia was not reported: +" + std::to_string(factored.positiveEigenvalues) +
              " -" + std::to_string(factored.negativeEigenvalues));
        check(factored.perturbedPivots >= 0 && factored.peakSymbolicMemoryKB >= 0,
              "Invalid pivot/memory diagnostics");
    }

    // Distinct multi-RHS buffers: the RHS must stay intact.
    Eigen::MatrixXd b(n, 3);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < n; ++i) b(i, j) = std::sin(1.0 + i + 3.0 * j);
    GMMDenseColMatrix rhs = gmmFrom(b), x(n, 3);
    check(solver.solve(rhs, x), "Multi-RHS solve failed");
    check(view(rhs) == b, "Solve modified the right-hand side");
    checkSolution(a, x, b, "Multi-RHS solve");
    if (reportsDiagnostics) {
        const PardisoSolverDiagnostics solved = solver.diagnostics();
        check(solved.iterativeRefinementSteps >= 0, "Invalid refinement-step count");
        check(solved.factorNonzeros == factored.factorNonzeros, "Solve changed factor statistics");
    }

    // GIF::calculateHarmonicBasisInPARDISO: unit columns in full and partial batches,
    // reusing the same buffers (resized per batch) and factors.
    const Eigen::MatrixXd inverse = a.inverse();
    const int batchSize = 3;
    GMMDenseColMatrix units, columns;
    for (int first = 0; first < n; first += batchSize) {
        const int count = std::min(batchSize, n - first);
        units.resize(n, count);
        columns.resize(n, count);
        std::fill(units.begin(), units.end(), 0.0);
        for (int j = 0; j < count; ++j) units(first + j, j) = 1.0;
        check(solver.solve(units, columns), "Batched inverse-column solve failed");
        check((view(columns) - inverse.middleCols(first, count)).norm() < 1e-12,
              "Inverse columns differ in batch starting at " + std::to_string(first));
    }

    // Aliased RHS/solution keeps the wrapper's original in-place semantics.
    GMMDenseColMatrix inPlace = gmmFrom(b);
    check(solver.solve(inPlace, inPlace), "Aliased solve failed");
    checkSolution(a, inPlace, b, "Aliased solve");

#ifdef BCP_PARDISO_COMPAT
    // Direct buffers are validated before the solver can read or write them.
    const double sentinel = 7.0;
    GMMDenseColMatrix wrongRows(n + 1, 3), wrongCols(n, 2), shortRhs(n - 1, 3);
    std::fill(wrongRows.begin(), wrongRows.end(), sentinel);
    std::fill(wrongCols.begin(), wrongCols.end(), sentinel);
    check(!solver.solve(rhs, wrongRows), "Accepted a solution with the wrong row count");
    check(!solver.solve(rhs, wrongCols), "Accepted a solution with the wrong column count");
    check(std::all_of(wrongRows.begin(), wrongRows.end(), [&](double v) { return v == sentinel; }) &&
          std::all_of(wrongCols.begin(), wrongCols.end(), [&](double v) { return v == sentinel; }),
          "Rejected solution buffers were modified");
    GMMDenseColMatrix shortX(n - 1, 3);
    check(!solver.solve(shortRhs, shortX), "Accepted a right-hand side with the wrong row count");
    GMMDenseColMatrix noRhs(n, 0), noX(n, 0);
    check(!solver.solve(noRhs, noX), "Accepted zero right-hand sides");
    GMMDenseColMatrix nonFinite = gmmFrom(b);
    nonFinite(2, 1) = std::numeric_limits<double>::quiet_NaN();
    check(!solver.solve(nonFinite, x), "Accepted a non-finite right-hand side");
    // A rejected call leaves the factors usable.
    check(solver.solve(rhs, x), "Solve after rejected calls failed");
    checkSolution(a, x, b, "Solve after rejected calls");
#endif

    // Re-analysis of a different pattern with the same handle must report the
    // new factor statistics, matching a fresh solver.
    const Eigen::MatrixXd larger = kktMatrix(2 * k);
    check(solver.createPardisoFormatMatrix(upperWithDiagonal(larger)), "Larger KKT rejected");
    check(solver.preprocess(), "Larger KKT re-analysis failed");
    PardisoLinearSolver fresh(-2);
    check(fresh.init(true) && fresh.createPardisoFormatMatrix(upperWithDiagonal(larger)) &&
          fresh.preprocess(), "Fresh larger KKT factorization failed");
    if (reportsDiagnostics) {
        check(solver.diagnostics().factorNonzeros == fresh.diagnostics().factorNonzeros &&
              solver.diagnostics().factorNonzeros != factored.factorNonzeros,
              "Re-analysis did not update factor nonzeros");
        check(solver.diagnostics().positiveEigenvalues == 2 * k &&
              solver.diagnostics().negativeEigenvalues == 2, "Re-analysis inertia mismatch");
    }
    Eigen::MatrixXd b2 = Eigen::MatrixXd::Ones(larger.rows(), 2);
    b2.col(1).setLinSpaced(-1.0, 2.0);
    GMMDenseColMatrix rhs2 = gmmFrom(b2), x2(larger.rows(), 2);
    check(solver.solve(rhs2, x2), "Solve after re-analysis failed");
    checkSolution(larger, x2, b2, "Solve after re-analysis");
}

// Nonsymmetric matrix type 11, as in the GIF mean-value repair solve, plus the
// combined analysis/factorization/solve phase.
void testGeneral() {
    const int n = 5;
    Eigen::MatrixXd a(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a(i, j) = i == j ? 5.0 + i : (j == i + 1 ? -1.5 : (j + 2 == i ? 0.75 : 0.0));
    Eigen::MatrixXd b(n, 2);
    b.col(0).setLinSpaced(1.0, 3.0);
    b.col(1) = Eigen::VectorXd::Unit(n, 3);

    PardisoLinearSolver factored(11);
    check(factored.init(false) && factored.createPardisoFormatMatrix(fullPattern(a)) &&
          factored.preprocess(), "General factorization failed");
    GMMDenseColMatrix rhs = gmmFrom(b), x(n, 2);
    check(factored.solve(rhs, x), "General solve failed");
    check(view(rhs) == b, "General solve modified the right-hand side");
    checkSolution(a, x, b, "General solve");

    PardisoLinearSolver oneTime(11);
    check(oneTime.init(false) && oneTime.createPardisoFormatMatrix(fullPattern(a)),
          "General one-time setup failed");
    GMMDenseColMatrix y(n, 2);
    check(oneTime.oneTimeSolve(rhs, y), "General one-time solve failed");
    checkSolution(a, y, b, "General one-time solve");
}
}

int main(int argc, char** argv) try {
    check(argc == 2, "Expected the linked BCP_LINEAR_SOLVER name");
    testSymmetricIndefinite(argv[1]);
    testGeneral();
    std::cout << "PardisoLinearSolver tests passed (" << argv[1] << ")\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
