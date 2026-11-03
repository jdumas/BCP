#pragma once


#include "Utils/GMM_Macros.h"
#include "Eigen/SparseCore"
#include <iostream>
#include "pardiso.h"


struct PardisoSolverDiagnostics
{
	int iterativeRefinementSteps;		/* iparm[6] */
	int perturbedPivots;				/* iparm[13] */
	int peakSymbolicMemoryKB;			/* iparm[14] */
	int permanentSymbolicMemoryKB;		/* iparm[15] */
	int factorizationAndSolveMemoryKB;	/* iparm[16] */
	int factorNonzeros;					/* iparm[17] */
	int positiveEigenvalues;			/* iparm[21], symmetric indefinite only */
	int negativeEigenvalues;			/* iparm[22], symmetric indefinite only */
};

class PardisoLinearSolver
{
public:

	PardisoLinearSolver(int matrixType=11,  int solver_in = 0);
	~PardisoLinearSolver();

	void setMatrixType(int matrixType);
	bool init(bool isHighlyIndefinite=false);
	bool preprocess();
	bool reorder();
	bool numericalFactorization();
	bool selectiveInverse();
	bool getMatrixInGMMformat(GMMSparseRowMatrix& M);
	bool getMatrixInGMMformat(GMMCompressed1RowMatrix& M);
	bool createPardisoFormatMatrix(const GMMCompressed1RowMatrix& M, std::vector<int>& rowsOfElementsToSet, std::vector<int>& colsOfElementsToSet);
	bool createPardisoFormatMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor>& M);//here we assume that the matrix is already upper triangular and that the diagonal elements are set
	bool createPardisoFormatMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor>& M, std::vector<int>& rowsOfElementsToSet, std::vector<int>& colsOfElementsToSet);//here we assume that we need to set elements

	template<class DenseColMatrixType>
	bool solve(DenseColMatrixType& RHS, DenseColMatrixType& X);
	
	template<class DenseColMatrixType>
	bool oneTimeSolve(DenseColMatrixType& RHS, DenseColMatrixType& X);

	// Solver statistics from the most recent PARDISO phase. These iparm output
	// slots have the same meaning in Panua PARDISO and the oneMKL adapter; the
	// Accelerate and Eigen adapters do not report them and leave zeros.
	PardisoSolverDiagnostics diagnostics() const;

private:

	void *pt[64];			/* Internal solver memory pointer pt */
	int mtype;				/* matrix type */
	int solver;				/* solver method */
	int error;				/* error output */
	int iparm[64];			/* Pardiso control parameters */
	double dparm[64];		/* Pardiso control parameters */
	int maxfct;				/* Maximum number of numerical factorizations.  */
	int mnum;				/* Which factorization to use. */
	int msglvl;				/* Print statistical information  */
	/* Sparse csr-compressed sparse row format matrix */
	std::vector<double> a;	/* Nonzero values of the coefficient matrix */
	std::vector<int> ja;	/* column indices of the sparse matrix */
	std::vector<int> ia;	/* ia(i) points to the first column index of row i in the array ja */
	/**/
	std::vector<double> b;	/* Right hand side */
	std::vector<double> x;	/* Solution */
	int n;					/* Number of equations in the sparse linear systems */
	int phase;				/* Which phase to perform. */
	double   ddum;			/* Double dummy */
	int      idum;			/* Integer dummy. */
	int nrhs;				/* Number of right-hand sides */
	int mNumProcessors;		/* Number of processors */
	bool mIsSymmetric;
	int mNonzeros;
	bool mPardisoInitialized;

	template<class DenseColMatrixType>
	bool solveWithFactors(DenseColMatrixType& RHS, DenseColMatrixType& X, const char* failureMessage);

	template<class DenseColMatrixType>
	bool checkDirectSolveBuffers(DenseColMatrixType& RHS, DenseColMatrixType& X);

	template<class DenseColMatrixType>
	bool createRHSvector(DenseColMatrixType& RHS);

	template<class DenseColMatrixType>
	bool updateX(DenseColMatrixType& X);
};



inline PardisoSolverDiagnostics PardisoLinearSolver::diagnostics() const
{
	return {iparm[6], iparm[13], iparm[14], iparm[15], iparm[16], iparm[17], iparm[21], iparm[22]};
}

template<class DenseColMatrixType>
bool PardisoLinearSolver::solve(DenseColMatrixType& RHS, DenseColMatrixType& X)
{
	/* -------------------------------------------------------------------- */
	/* ..  Back substitution and iterative refinement.                      */
	/* -------------------------------------------------------------------- */
	phase = 33;
	//Number of performed iterative refinement steps=iparm[6]
	return solveWithFactors(RHS, X, "\nERROR during solution: ");
}

template<class DenseColMatrixType>
bool PardisoLinearSolver::oneTimeSolve(DenseColMatrixType& RHS, DenseColMatrixType& X)
{
	phase = 13;	/*Reordering And Symbolic Factorization*/
	//Number of performed iterative refinement steps=iparm[6]
	return solveWithFactors(RHS, X, "ERROR during oneTimeSolve: ");
}

template<class DenseColMatrixType>
bool PardisoLinearSolver::solveWithFactors(DenseColMatrixType& RHS, DenseColMatrixType& X, const char* failureMessage)
{
#ifdef BCP_PARDISO_COMPAT
	// Distinct contiguous column-major GMM buffers are passed to the solver
	// directly. An aliased call keeps the original wrapper's internal copies,
	// because the vendor solvers require distinct B and X arrays.
	if (&RHS != &X) {
		if (!checkDirectSolveBuffers(RHS, X))
			return false;
		pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &a[0], &ia[0], &ja[0], &idum, &nrhs, iparm, &msglvl, RHS.data(), X.data(), &error, dparm);
		if (error != 0) {
			std::cout << failureMessage << error << std::endl;
			return false;
		}
		return true;
	}
#endif
	if (!createRHSvector(RHS))
		return false;

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &a[0], &ia[0], &ja[0], &idum, &nrhs, iparm, &msglvl, &b[0], &x[0], &error, dparm);
	if (error != 0) {
		std::cout << failureMessage << error << std::endl;
		return false;
	}

	if (!updateX(X))
		return false;

	return true;
}

template<class DenseColMatrixType>
bool PardisoLinearSolver::checkDirectSolveBuffers(DenseColMatrixType& RHS, DenseColMatrixType& X)
{
	const size_t rows = static_cast<size_t>(n);
	const size_t columns = RHS.ncols();
	if (n <= 0 || RHS.nrows() != rows || X.nrows() != rows || X.ncols() != columns ||
		RHS.size() != rows * columns || X.size() != rows * columns) {
		std::cout << "\nERROR right hand side or solution dimensions do not match the " << n
			<< "-row matrix: RHS " << RHS.nrows() << "x" << RHS.ncols()
			<< ", X " << X.nrows() << "x" << X.ncols() << std::endl;
		error = -1;
		return false;
	}
	nrhs = static_cast<int>(columns);

	//check vector
	pardiso_chkvec(&n, &nrhs, RHS.data(), &error);
	if (error != 0) {
		std::cout << "\nERROR right hand side: " << error << std::endl;
		return false;
	}
	return true;
}

template<class DenseColMatrixType>
bool PardisoLinearSolver::createRHSvector(DenseColMatrixType& RHS)
{
	nrhs = RHS.ncols();

	b = RHS;
	x.resize(n*nrhs);

	//check vector
	pardiso_chkvec(&n, &nrhs, &b[0], &error);
	if (error != 0) {
		std::cout << "\nERROR right hand side: " << error << std::endl;
		return false;
	}
	return true;
}

template<class DenseColMatrixType>
bool PardisoLinearSolver::updateX(DenseColMatrixType& X)
{
	X.assign(x.begin(), x.end());
	return true;
}
