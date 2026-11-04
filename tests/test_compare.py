"""Tests for scripts/compare.py: mode contracts, rigid alignment, traces, KKT diagnostics, and exit gates."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
compare = importlib.util.module_from_spec(spec)
sys.modules["compare"] = compare
spec.loader.exec_module(compare)


class CompareHarnessTests(unittest.TestCase):
    def test_reference_mode_sets_cpu_native_without_cuda_probe(self):
        probe_called = False

        def forbidden_probe():
            nonlocal probe_called
            probe_called = True
            return False, "should not be called"

        mode = compare.resolve_mode_spec(
            "reference",
            sys_platform="linux",
            solver="MKL",
            cuda_probe=forbidden_probe,
        )
        self.assertFalse(probe_called)
        self.assertEqual(mode.device, "cpu")
        self.assertEqual(mode.gif_device, "cpu")
        self.assertEqual(mode.gif_backend, "native")
        self.assertEqual(mode.solver, "MKL")
        self.assertIn("reference", mode.claim_scope)

    def test_reference_mode_supports_macos_accelerate(self):
        mode = compare.resolve_mode_spec("reference", sys_platform="darwin", solver="ACCELERATE")
        self.assertEqual(mode.device, "cpu")
        self.assertEqual(mode.gif_device, "cpu")
        self.assertEqual(mode.gif_backend, "native")
        self.assertEqual(mode.solver, "ACCELERATE")

    def test_parity_mode_requires_linux_mkl_and_working_cuda(self):
        mode = compare.resolve_mode_spec(
            "parity",
            sys_platform="linux",
            solver="MKL",
            cuda_probe=lambda: (True, "NVIDIA RTX 5000 Ada"),
        )
        self.assertEqual(mode.device, "cuda:0")
        self.assertEqual(mode.gif_device, "gpu")
        self.assertEqual(mode.gif_backend, "torch")
        self.assertEqual(mode.solver, "MKL")

        with self.assertRaisesRegex(RuntimeError, "requires Linux"):
            compare.resolve_mode_spec(
                "parity", sys_platform="darwin", solver="MKL", cuda_probe=lambda: (True, "GPU")
            )
        with self.assertRaisesRegex(RuntimeError, "requires MKL"):
            compare.resolve_mode_spec(
                "parity", sys_platform="linux", solver="EIGEN", cuda_probe=lambda: (True, "GPU")
            )
        with self.assertRaisesRegex(RuntimeError, "requires working CUDA"):
            compare.resolve_mode_spec(
                "parity",
                sys_platform="linux",
                solver="MKL",
                cuda_probe=lambda: (False, "CUDA driver unavailable"),
            )

    def test_execution_environment_preserves_caller_threads_unless_overridden(self):
        mode = compare.ModeSpec(
            mode="parity",
            device="cuda:0",
            gif_device="gpu",
            gif_backend="torch",
            solver="MKL",
            claim_scope="test",
        )
        base = {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        env = compare.build_execution_environment(
            mode,
            base_env=base,
            timeout_seconds=2400.0,
            octave_timeout_ms=2400000,
        )
        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        self.assertEqual(env["MKL_NUM_THREADS"], "8")
        self.assertEqual(env["BCP_DEVICE"], "cuda:0")
        self.assertEqual(env["BCP_GIF_DEVICE"], "gpu")
        self.assertEqual(env["BCP_GIF_BACKEND"], "torch")
        self.assertEqual(env["BCP_NATIVE_TIMEOUT_SECONDS"], "2400.0")
        self.assertEqual(env["BCP_OCTAVE_TIMEOUT_MS"], "2400000")
        self.assertNotIn("BCP_COMPARE_KKT_DIR", env)
        self.assertNotIn("BCP_COMPARE_TRACE_PATH", env)

        env_explicit = compare.build_execution_environment(
            mode,
            base_env=base,
            threads=4,
            kkt_dir=Path("/tmp/kkt"),
            trace_path=Path("/tmp/trace"),
            stop_after_basis=True,
        )
        self.assertEqual(env_explicit["OMP_NUM_THREADS"], "4")
        self.assertEqual(env_explicit["MKL_NUM_THREADS"], "4")
        self.assertEqual(env_explicit["BCP_COMPARE_KKT_DIR"], str(Path("/tmp/kkt").resolve()))
        self.assertEqual(env_explicit["BCP_COMPARE_TRACE_PATH"], str(Path("/tmp/trace").resolve()))
        self.assertEqual(env_explicit["BCP_COMPARE_STOP_AFTER_BASIS"], "1")

    def test_run_single_mode_pipeline_omits_redundant_outer_subprocess_timeout(self):
        mode = compare.ModeSpec(
            mode="reference",
            device="cpu",
            gif_device="cpu",
            gif_backend="native",
            solver="MKL",
            claim_scope="test",
        )
        fake_proc = mock.Mock(stdout="Total time: 0.1\n", stderr="", returncode=0)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(compare.subprocess, "run", return_value=fake_proc) as run_mock, \
             mock.patch("igl.read_triangle_mesh", return_value=(np.zeros((3, 3)), np.array([[0, 1, 2]]))):
            mesh = Path(tmp) / "in.obj"
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            compare.run_single_mode_pipeline(
                mode, mesh, Path(tmp) / "work", timeout_seconds=1234.0, stop_after_basis=True
            )
            kwargs = run_mock.call_args.kwargs
            self.assertNotIn("timeout", kwargs)
            self.assertEqual(kwargs["env"]["BCP_NATIVE_TIMEOUT_SECONDS"], "1234.0")

    def test_rigid_alignment_recovers_rotation_and_translation_without_scale_or_reflection(self):
        rng = np.random.default_rng(123)
        ref = rng.standard_normal((25, 2))
        theta = 0.65
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        shift = np.array([3.25, -1.75])
        cand = ref @ rot.T + shift

        aligned, info = compare.align_uv_rigid(cand, ref)
        np.testing.assert_allclose(aligned, ref, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(info["rotation_det"], 1.0, places=12)
        self.assertEqual(info["scale_applied"], 1.0)
        self.assertFalse(info["reflection_allowed"])
        self.assertLess(info["max_vertex_l2_error"], 1e-12)

        scaled = 2.5 * ref
        _, scaled_info = compare.align_uv_rigid(scaled, ref)
        self.assertGreater(scaled_info["max_vertex_l2_error"], 0.5)

        reflected = ref.copy()
        reflected[:, 0] *= -1.0
        _, refl_info = compare.align_uv_rigid(reflected, ref)
        self.assertAlmostEqual(refl_info["rotation_det"], 1.0, places=12)
        self.assertGreater(refl_info["max_vertex_l2_error"], 0.1)

    def test_parse_gif_diagnostics_and_compare_traces_rejects_empty_nan_and_confounded(self):
        sample_log = """
GIF evaluator backend: torch (cuda:0)
GIF compute device: gpu
Symbolic and Numerical factorization: 0.0125
Harmonic basis inverse-column solves: 0.0045
Subspace construction took 0.031
BCP_TRACE_STEP remesh=1 step=1 iter=1 E=4.50000000000000000 Eprev=5.00000000000000000 normG=1.25000000000000000 normG1=0.75000000000000000 normD=0.50000000000000000 t=0.80000000000000004 tMax=1.00000000000000000 stepNorm=0.40000000000000002 foldovers=0
BCP_TRACE_REMESH remesh=1 newtonIters=1 initE=5.00000000000000000 finalE=4.50000000000000000 rate=0.10000000000000001
Newton and back substitution took 0.089
Total time: 0.125
Final Esd = 2.03125
"""
        parsed = compare.parse_gif_diagnostics(sample_log)
        self.assertEqual(parsed["gif_compute_device"], "gpu")
        self.assertEqual(parsed["gif_evaluator_backend"], "torch (cuda:0)")
        self.assertAlmostEqual(parsed["factorization_seconds"], 0.0125)
        self.assertAlmostEqual(parsed["basis_solve_seconds"], 0.0045)
        self.assertAlmostEqual(parsed["subspace_seconds"], 0.031)
        self.assertAlmostEqual(parsed["newton_and_backsub_seconds"], 0.089)
        self.assertAlmostEqual(parsed["total_gif_seconds"], 0.125)
        self.assertAlmostEqual(parsed["final_esd_logged"], 2.03125)
        self.assertEqual(len(parsed["steps"]), 1)
        self.assertEqual(len(parsed["remeshes"]), 1)

        perturbed_log = sample_log.replace("E=4.50000000000000000", "E=4.50000000010000000")
        parsed_2 = compare.parse_gif_diagnostics(perturbed_log)
        trace_cmp = compare.compare_traces(parsed, parsed_2, rtol=1e-7, atol=1e-9)
        self.assertTrue(trace_cmp["within_envelope"])
        self.assertTrue(trace_cmp["strict_trajectory_parity_certified"])

        # Confounded comparison must NOT certify strict trajectory parity even when within envelope.
        confounded = compare.compare_traces(parsed, parsed_2, confounded_inputs=True)
        self.assertTrue(confounded["confounded_by_preprocessing"])
        self.assertFalse(confounded["strict_trajectory_parity_certified"])

        # Empty traces must fail (never accept [] == []).
        empty_cmp = compare.compare_traces({"steps": [], "remeshes": []}, {"steps": [], "remeshes": []})
        self.assertFalse(empty_cmp["non_empty"])
        self.assertFalse(empty_cmp["within_envelope"])
        self.assertFalse(empty_cmp["strict_trajectory_parity_certified"])

        # NaN/Inf fields must fail (never pass via NaN > tol == False).
        nan_log = sample_log.replace("normG=1.25000000000000000", "normG=nan")
        nan_cmp = compare.compare_traces(parsed, compare.parse_gif_diagnostics(nan_log))
        self.assertFalse(nan_cmp["finite"])
        self.assertFalse(nan_cmp["within_envelope"])
        self.assertFalse(nan_cmp["strict_trajectory_parity_certified"])

        diverged_step_log = sample_log.replace("normD=0.50000000000000000", "normD=0.55000000000000000")
        trace_bad_step = compare.compare_traces(parsed, compare.parse_gif_diagnostics(diverged_step_log))
        self.assertFalse(trace_bad_step["within_envelope"])
        self.assertFalse(trace_bad_step["strict_trajectory_parity_certified"])

    def test_normalized_backward_error_and_kkt_dump_evaluation_and_exit_codes(self):
        try:
            import scipy.io as sio
            import scipy.sparse as sp
        except ImportError:
            self.skipTest("SciPy not available")

        zero_mat = np.zeros((2, 2))
        zero_vec = np.zeros((2, 1))
        self.assertEqual(compare.normalized_backward_error(zero_mat, zero_vec, zero_vec), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kkt_dir = root / "pass_0000_kkt"
            kkt_dir.mkdir()
            a_dense = np.array(
                [
                    [4.0, -1.0, 0.0, 1.0],
                    [-1.0, 3.0, -1.0, 0.0],
                    [0.0, -1.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            sio.mmwrite(str(kkt_dir / "matrix.mtx"), sp.coo_matrix(a_dense), symmetry="general")
            cols = np.array([[2], [3]], dtype=np.float64)
            b = np.zeros((4, 2), dtype=np.float64)
            b[2, 0] = 1.0
            b[3, 1] = 1.0
            x = np.linalg.solve(a_dense, b)
            sio.savemat(str(kkt_dir / "batch_0000.mat"), {"X": x, "columns": cols})
            (kkt_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "kind": "kkt",
                        "pass_id": "pass_0000_kkt",
                        "index_base": 0,
                        "mtype": -2,
                        "rows": 4,
                        "cols": 4,
                        "n": 4,
                        "stored_nnz": 9,
                        "num_dof": 2,
                        "batch_size": 2,
                        "batch_count": 1,
                        "cones_constraints_start_row": 2,
                        "selected_columns": [2, 3],
                        "selected_rows": [0, 1],
                    }
                ),
                encoding="utf-8",
            )

            mvc_dir = root / "pass_0001_mvc"
            mvc_dir.mkdir()
            mvc_dense = np.array([[1.0, -0.3, 0.0], [-0.2, 1.0, -0.4], [0.0, -0.5, 1.0]], dtype=np.float64)
            sio.mmwrite(str(mvc_dir / "matrix.mtx"), sp.coo_matrix(mvc_dense), symmetry="general")
            b_mvc = np.array([[1.0, 0.5], [0.2, -1.0], [-0.5, 0.25]], dtype=np.float64)
            x_mvc = np.linalg.solve(mvc_dense, b_mvc)
            sio.savemat(str(mvc_dir / "solution.mat"), {"B": b_mvc, "X": x_mvc})
            (mvc_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "kind": "mvc",
                        "pass_id": "pass_0001_mvc",
                        "index_base": 0,
                        "mtype": 11,
                        "rows": 3,
                        "cols": 3,
                        "n": 3,
                        "stored_nnz": 7,
                        "num_dof": 2,
                        "batch_size": 2,
                        "batch_count": 1,
                        "cones_constraints_start_row": 0,
                        "selected_columns": [],
                        "selected_rows": [],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                rc_ok = compare.main(["--check-kkt-dir", str(root)])
            self.assertEqual(rc_ok, 0)

            # Corrupt solution X so backward error fails; main must return non-zero (1).
            sio.savemat(str(mvc_dir / "solution.mat"), {"B": b_mvc, "X": x_mvc + 5.0})
            with redirect_stdout(io.StringIO()):
                rc_bad = compare.main(["--check-kkt-dir", str(root)])
            self.assertEqual(rc_bad, 1)

    def test_gates_verify_actual_runtime_placement(self):
        for mode, device, backend in (
            ("reference", "cpu", "native"),
            ("parity", "gpu", "torch (cuda:0, inprocess)"),
        ):
            run = {"uv_metrics": {"locally_injective": True}, "gif_diagnostics": {
                "gif_compute_device": device, "gif_evaluator_backend": backend}}
            report = {"runs": {mode: run}}
            self.assertTrue(compare.evaluate_report_gates(report)[0])
            for field, failure in (
                ("gif_compute_device", "unexpected_gif_device"),
                ("gif_evaluator_backend", "unexpected_gif_backend"),
            ):
                with self.subTest(mode=mode, field=field):
                    bad = {"runs": {mode: dict(run, gif_diagnostics={
                        **run["gif_diagnostics"], field: "wrong"})}}
                    ok, failures = compare.evaluate_report_gates(bad)
                    self.assertFalse(ok)
                    self.assertIn(f"{mode}:{failure}", failures)
                    del bad["runs"][mode]["gif_diagnostics"][field]
                    self.assertFalse(compare.evaluate_report_gates(bad)[0])
            if mode == "parity":
                # A legacy worker label or CPU Torch evaluation is not parity.
                for wrong in ("torch (cuda:0)", "torch (cpu, inprocess)"):
                    run["gif_diagnostics"]["gif_evaluator_backend"] = wrong
                    self.assertFalse(compare.evaluate_report_gates(report)[0])

    def test_report_gates_and_nonzero_exit_on_failed_uv_trace_or_kkt(self):
        good_run = {
            "mode": "reference",
            "uv_metrics": {"locally_injective": True},
            "gif_diagnostics": {
                "gif_compute_device": "cpu",
                "gif_evaluator_backend": "native",
                "steps": [
                    {
                        "remesh": 1,
                        "step": 1,
                        "iter": 1,
                        "foldovers": 0,
                        "E": 2.1,
                        "Eprev": 2.2,
                        "normG": 0.1,
                        "normG1": 0.1,
                        "normD": 0.05,
                        "t": 1.0,
                        "tMax": 1.0,
                        "stepNorm": 0.05,
                    }
                ],
                "remeshes": [{"remesh": 1, "newtonIters": 1, "initE": 2.2, "finalE": 2.1, "rate": 0.04}],
            },
            "kkt_diagnostics": {"ok": True},
        }
        report_ok = compare.build_comparison_report(
            mode="reference",
            input_mesh=Path("mesh.obj"),
            runs={"reference": good_run},
            require_trace=True,
            require_kkt=True,
        )
        self.assertTrue(report_ok["ok"])
        self.assertEqual(report_ok["gate_failures"], [])
        self.assertFalse(report_ok["original_matlab_panua_runtime_parity_certified"])

        # Missing trace when --trace was requested must fail gate.
        bad_trace_run = dict(good_run, gif_diagnostics={"steps": [], "remeshes": []})
        report_bad_trace = compare.build_comparison_report(
            mode="reference",
            input_mesh=Path("mesh.obj"),
            runs={"reference": bad_trace_run},
            require_trace=True,
        )
        self.assertFalse(report_bad_trace["ok"])
        self.assertIn("reference:missing_or_nonfinite_trace", report_bad_trace["gate_failures"])

        # Failed UV comparison or failed matched controller replay must fail gate and return 1 from main.
        report_bad_replay = compare.build_comparison_report(
            mode="both",
            input_mesh=Path("mesh.obj"),
            runs={"reference": good_run, "parity": good_run},
            reference_vs_parity={"passed": False},
            matched_controller_replay={"ok": False},
        )
        self.assertFalse(report_bad_replay["ok"])
        self.assertIn("reference_vs_parity:uv_or_energy_mismatch", report_bad_replay["gate_failures"])
        self.assertIn("matched_controller_replay:failed", report_bad_replay["gate_failures"])

        # Verify main() returns non-zero when a requested gate fails.
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(compare, "run_single_mode_pipeline", return_value=bad_trace_run), \
             mock.patch.object(compare, "run_matched_controller_replay", return_value={"ok": True}), \
             redirect_stdout(io.StringIO()):
            mesh = Path(tmp) / "disk.obj"
            compare.generate_curved_disk_mesh(mesh, grid_size=4)
            rc = compare.main(["--mode", "reference", "--input", str(mesh), "--trace"])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
