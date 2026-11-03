#!/usr/bin/env python3
"""Independent backward-error check for BCP KKT / MVC solver dumps.

Diagnostic dumps are produced only when BCP_COMPARE_KKT_DIR is set (outside
timed runs). Each pass directory contains:

* matrix.mtx    MatrixMarket coordinate real general: the FULL matrix A.
* metadata.json kind ("kkt" or "mvc"), index_base (must be 0),
                selected_columns, selected_rows and pass identity/dimensions.
* kkt: batch_*.mat, each with X (N-by-k solution) and columns (k zero-based
  unit right-hand-side column indices). B is rebuilt as sparse unit columns,
  never as a dense inverse.
* mvc: solution.mat with the actual right-hand sides B and solution X.

For every solution the checker reports the normalized backward error
    ||A X - B||_F / (||A||_F ||X||_F + ||B||_F)
(0 when numerator and denominator are both 0, infinity when only the
denominator is 0) and compares X with an independent SciPy SuperLU solve.
For KKT batches the selected_rows x columns entries consumed by GIF are
compared separately. Output is JSON; the exit status is non-zero if any check
exceeds its tolerance or a dump is malformed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

DEFAULT_BACKWARD_TOLERANCE = 1e-10
DEFAULT_SOLUTION_TOLERANCE = 1e-8


class DumpError(ValueError):
    """A dump directory does not follow the shared diagnostic format."""


def normalized_backward_error(residual_norm, matrix_norm, solution_norm, rhs_norm):
    denominator = matrix_norm * solution_norm + rhs_norm
    if denominator == 0.0:
        return 0.0 if residual_norm == 0.0 else float("inf")
    return residual_norm / denominator


def relative_difference(actual, reference):
    scale = np.linalg.norm(reference)
    difference = np.linalg.norm(actual - reference)
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / scale


def zero_based_indices(values, name, limit):
    array = np.asarray(values, dtype=float).ravel()
    indices = array.astype(np.int64)
    if not np.array_equal(indices, array):
        raise DumpError(f"{name} must contain integers")
    if indices.size and (indices.min() < 0 or indices.max() >= limit):
        raise DumpError(f"{name} must be zero-based indices below {limit}")
    return indices


def read_matrix(directory):
    path = directory / "matrix.mtx"
    if not path.is_file():
        raise DumpError(f"missing {path}")
    matrix = scipy.sparse.csr_matrix(scipy.io.mmread(str(path)), dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise DumpError(f"matrix is {matrix.shape[0]}x{matrix.shape[1]}, not square")
    return matrix


def read_metadata(directory, size):
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    kind = metadata.get("kind")
    if kind not in ("kkt", "mvc"):
        raise DumpError(f"metadata kind must be 'kkt' or 'mvc', got {kind!r}")
    if metadata.get("index_base") != 0:
        raise DumpError("metadata index_base must be 0")
    for key in ("rows", "cols", "n"):
        if key in metadata and int(metadata[key]) != size:
            raise DumpError(f"metadata {key}={metadata[key]} does not match the {size}x{size} matrix")
    return metadata


def load_mat(path, names):
    try:
        data = scipy.io.loadmat(str(path))
    except NotImplementedError as error:  # MAT v7.3 / HDF5
        raise DumpError(f"{path.name}: unsupported MAT format ({error})") from error
    missing = [name for name in names if name not in data]
    if missing:
        raise DumpError(f"{path.name}: missing {', '.join(missing)}")
    return data


def dense_columns(value, rows, name):
    array = np.asarray(value.toarray() if scipy.sparse.issparse(value) else value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != rows:
        raise DumpError(f"{name} must have {rows} rows, got shape {array.shape}")
    return array


def check_solution(matrix, matrix_norm, factor, rhs, solution, tolerances):
    rhs_dense = rhs.toarray() if scipy.sparse.issparse(rhs) else rhs
    residual = matrix @ solution - rhs_dense
    rhs_norms = np.linalg.norm(rhs_dense, axis=0)
    column_errors = [
        normalized_backward_error(np.linalg.norm(residual[:, j]), matrix_norm,
                                  np.linalg.norm(solution[:, j]), rhs_norms[j])
        for j in range(solution.shape[1])
    ]
    result = {
        "rhs_columns": int(solution.shape[1]),
        "backward_error": normalized_backward_error(
            np.linalg.norm(residual), matrix_norm, np.linalg.norm(solution), np.linalg.norm(rhs_norms)),
        "max_column_backward_error": max(column_errors, default=0.0),
        "finite": bool(np.isfinite(solution).all()),
    }
    ok = result["finite"] and result["backward_error"] <= tolerances["backward"] and \
        result["max_column_backward_error"] <= tolerances["backward"]
    if factor is not None:
        reference = factor.solve(rhs_dense)
        result["independent_backward_error"] = normalized_backward_error(
            np.linalg.norm(matrix @ reference - rhs_dense), matrix_norm,
            np.linalg.norm(reference), np.linalg.norm(rhs_norms))
        result["independent_relative_difference"] = relative_difference(solution, reference)
        ok = ok and result["independent_relative_difference"] <= tolerances["solution"]
        result["_reference"] = reference
    result["ok"] = bool(ok)
    return result


def check_pass(directory, tolerances, independent):
    matrix = read_matrix(directory)
    size = matrix.shape[0]
    metadata = read_metadata(directory, size)
    selected_rows = zero_based_indices(metadata.get("selected_rows", []), "selected_rows", size)
    selected_columns = zero_based_indices(metadata.get("selected_columns", []), "selected_columns", size)
    matrix_norm = scipy.sparse.linalg.norm(matrix)
    report = {
        "path": str(directory),
        "kind": metadata["kind"],
        "metadata": metadata,
        "n": int(size),
        "nnz": int(matrix.nnz),
        "matrix_frobenius_norm": float(matrix_norm),
        "solutions": [],
    }
    if metadata["kind"] == "kkt":
        asymmetry = scipy.sparse.linalg.norm(matrix - matrix.T)
        report["symmetry_relative_error"] = 0.0 if matrix_norm == 0.0 else float(asymmetry / matrix_norm)
    factor = scipy.sparse.linalg.splu(matrix.tocsc()) if independent else None

    ok = report.get("symmetry_relative_error", 0.0) <= tolerances["backward"]
    if metadata["kind"] == "kkt":
        batches = sorted(directory.glob("batch_*.mat"))
        if not batches:
            raise DumpError(f"{directory}: no batch_*.mat files")
        solved_columns = []
        for path in batches:
            data = load_mat(path, ("X", "columns"))
            solution = dense_columns(data["X"], size, "X")
            columns = zero_based_indices(data["columns"], "columns", size)
            if columns.size != solution.shape[1]:
                raise DumpError(f"{path.name}: {columns.size} columns for {solution.shape[1]} solutions")
            rhs = scipy.sparse.csc_matrix(
                (np.ones(columns.size), (columns, np.arange(columns.size))), shape=(size, columns.size))
            result = check_solution(matrix, matrix_norm, factor, rhs, solution, tolerances)
            reference = result.pop("_reference", None)
            if reference is not None and selected_rows.size:
                difference = np.abs(solution[selected_rows] - reference[selected_rows])
                scale = max(np.abs(reference[selected_rows]).max(initial=0.0), np.finfo(float).tiny)
                result["selected_entries_max_abs_difference"] = float(difference.max(initial=0.0))
                result["selected_entries_relative_difference"] = float(difference.max(initial=0.0) / scale)
                result["ok"] = result["ok"] and \
                    result["selected_entries_relative_difference"] <= tolerances["solution"]
            result["file"] = path.name
            report["solutions"].append(result)
            ok = ok and result["ok"]
            solved_columns.extend(columns.tolist())
        if selected_columns.size and sorted(solved_columns) != sorted(selected_columns.tolist()):
            report["column_coverage_error"] = "batch columns do not match metadata selected_columns"
            ok = False
    else:
        path = directory / "solution.mat"
        if not path.is_file():
            raise DumpError(f"missing {path}")
        data = load_mat(path, ("B", "X"))
        rhs = dense_columns(data["B"], size, "B")
        solution = dense_columns(data["X"], size, "X")
        if rhs.shape != solution.shape:
            raise DumpError(f"B shape {rhs.shape} does not match X shape {solution.shape}")
        result = check_solution(matrix, matrix_norm, factor, rhs, solution, tolerances)
        result.pop("_reference", None)
        result["file"] = path.name
        report["solutions"].append(result)
        ok = ok and result["ok"]
    report["ok"] = bool(ok)
    return report


def pass_directories(root):
    if (root / "metadata.json").is_file():
        return [root]
    directories = sorted(path.parent for path in root.glob("*/metadata.json"))
    if not directories:
        raise DumpError(f"{root}: no pass directories with metadata.json")
    return directories


def check_dump(root, backward_tolerance=DEFAULT_BACKWARD_TOLERANCE,
               solution_tolerance=DEFAULT_SOLUTION_TOLERANCE, independent=True):
    tolerances = {"backward": backward_tolerance, "solution": solution_tolerance}
    report = {"root": str(root), "norm": "frobenius", "independent_solver":
              "scipy.sparse.linalg.splu" if independent else None,
              "tolerances": tolerances, "passes": []}
    for directory in pass_directories(Path(root)):
        try:
            report["passes"].append(check_pass(directory, tolerances, independent))
        except (DumpError, OSError, ValueError, RuntimeError) as error:
            report["passes"].append({"path": str(directory), "ok": False, "error": str(error)})
    report["ok"] = all(entry["ok"] for entry in report["passes"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path, help="BCP_COMPARE_KKT_DIR root or one pass directory")
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    parser.add_argument("--backward-tolerance", type=float, default=DEFAULT_BACKWARD_TOLERANCE)
    parser.add_argument("--solution-tolerance", type=float, default=DEFAULT_SOLUTION_TOLERANCE,
                        help="relative difference allowed against the independent solve")
    parser.add_argument("--no-independent-solve", action="store_true",
                        help="report backward errors only (skip the SuperLU factorization)")
    args = parser.parse_args(argv)
    try:
        report = check_dump(args.dump, args.backward_tolerance, args.solution_tolerance,
                            not args.no_independent_solve)
    except DumpError as error:
        report = {"root": str(args.dump), "ok": False, "error": str(error), "passes": []}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
