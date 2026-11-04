"""Tests for scripts/check_kkt.py on synthetic dumps in the shared diagnostic format."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io
import scipy.sparse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_kkt", ROOT/"scripts/check_kkt.py")
check_kkt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_kkt)


def kkt_matrix(k=6):
    """[K C^T; C 0]: SPD tridiagonal K and two constraint rows (symmetric indefinite)."""
    n = k + 2
    a = np.zeros((n, n))
    for i in range(k):
        a[i, i] = 4.0 + 0.25 * i
        if i + 1 < k:
            a[i, i + 1] = a[i + 1, i] = -1.0
        row = k + (0 if 2 * i < k else 1)
        a[row, i] = a[i, row] = 1.0 + 0.1 * i
    return a


def write_pass(directory, matrix, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(str(directory/"matrix.mtx"), scipy.sparse.coo_matrix(matrix), symmetry="general")
    (directory/"metadata.json").write_text(json.dumps(metadata))


def write_kkt_pass(directory, matrix, columns, batch=3, perturb=None, metadata=None):
    n = matrix.shape[0]
    write_pass(directory, matrix, metadata or {
        "kind": "kkt", "index_base": 0, "n": n, "pass": "harmonic_basis",
        "selected_columns": list(columns), "selected_rows": [0, 2, 3]})
    for number, first in enumerate(range(0, len(columns), batch)):
        chunk = np.asarray(columns[first:first + batch])
        rhs = np.zeros((n, chunk.size))
        rhs[chunk, np.arange(chunk.size)] = 1.0
        solution = np.linalg.solve(matrix, rhs)
        if perturb is not None and number == 0:
            solution[perturb] += 1e-3
        # Stored as a MATLAB-style column of doubles, as the C++ dump writes it.
        scipy.io.savemat(str(directory/f"batch_{number:04d}.mat"),
                         {"X": solution, "columns": chunk.reshape(-1, 1).astype(float)})


class CheckKktTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def test_backward_error_handles_zero_denominator(self):
        self.assertEqual(check_kkt.normalized_backward_error(0.0, 0.0, 0.0, 0.0), 0.0)
        self.assertEqual(check_kkt.normalized_backward_error(1.0, 0.0, 5.0, 0.0), float("inf"))
        self.assertAlmostEqual(check_kkt.normalized_backward_error(1.0, 2.0, 3.0, 4.0), 0.1)

    def test_accurate_kkt_batches_pass_with_unit_rhs(self):
        matrix = kkt_matrix()
        write_kkt_pass(self.root/"pass_0", matrix, [6, 7, 1, 4])
        report = check_kkt.check_dump(self.root)
        self.assertTrue(report["ok"], report)
        entry = report["passes"][0]
        self.assertEqual((entry["kind"], entry["n"]), ("kkt", 8))
        self.assertEqual(entry["symmetry_relative_error"], 0.0)
        self.assertEqual([s["rhs_columns"] for s in entry["solutions"]], [3, 1])
        for solution in entry["solutions"]:
            self.assertLess(solution["backward_error"], 1e-14)
            self.assertLess(solution["independent_relative_difference"], 1e-12)
            self.assertIn("selected_entries_relative_difference", solution)

    def test_inaccurate_solution_fails(self):
        write_kkt_pass(self.root, kkt_matrix(), [6, 7], perturb=(0, 0))
        report = check_kkt.check_dump(self.root)
        self.assertFalse(report["ok"])
        self.assertGreater(report["passes"][0]["solutions"][0]["backward_error"], 1e-6)

    def test_selected_column_coverage_is_checked(self):
        matrix = kkt_matrix()
        write_kkt_pass(self.root, matrix, [6, 7], metadata={
            "kind": "kkt", "index_base": 0, "selected_columns": [6, 7, 5], "selected_rows": []})
        report = check_kkt.check_dump(self.root)
        self.assertFalse(report["ok"])
        self.assertIn("column_coverage_error", report["passes"][0])

    def test_nonsymmetric_mvc_solution(self):
        matrix = np.array([[5.0, -1.5, 0.0], [0.0, 6.0, -1.5], [0.75, 0.0, 7.0]])
        rhs = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        directory = self.root/"mvc"
        write_pass(directory, matrix, {"kind": "mvc", "index_base": 0, "selected_columns": [],
                                       "selected_rows": [], "pass": "mean_value_repair"})
        scipy.io.savemat(str(directory/"solution.mat"), {"B": rhs, "X": np.linalg.solve(matrix, rhs)})
        report = check_kkt.check_dump(self.root)
        self.assertTrue(report["ok"], report)
        self.assertNotIn("symmetry_relative_error", report["passes"][0])

    def test_malformed_dumps_are_reported_not_accepted(self):
        matrix = kkt_matrix()
        write_kkt_pass(self.root/"a_one_based", matrix, [6], metadata={
            "kind": "kkt", "index_base": 1, "selected_columns": [6], "selected_rows": []})
        directory = self.root/"b_mismatch"
        write_pass(directory, matrix, {"kind": "kkt", "index_base": 0})
        scipy.io.savemat(str(directory/"batch_0000.mat"),
                         {"X": np.zeros((8, 2)), "columns": np.array([[6.0]])})
        report = check_kkt.check_dump(self.root)
        self.assertFalse(report["ok"])
        errors = [entry.get("error", "") for entry in report["passes"]]
        self.assertIn("index_base", errors[0])
        self.assertIn("1 columns for 2 solutions", errors[1])

    def test_command_line_writes_json_and_exit_status(self):
        write_kkt_pass(self.root/"pass", kkt_matrix(), [6, 7])
        output = self.root/"report.json"
        self.assertEqual(check_kkt.main([str(self.root), "--output", str(output)]), 0)
        self.assertTrue(json.loads(output.read_text())["ok"])
        self.assertEqual(check_kkt.main([str(self.root/"missing"), "--output", str(output)]), 1)
        self.assertFalse(json.loads(output.read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
