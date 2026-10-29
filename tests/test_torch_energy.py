"""Tests for BCP/GIF_and_CM/torch_energy.py (port of symmetricDirichletEnergy.m)."""

import os
import sys
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
except ImportError:  # Native environments without NumPy/PyTorch skip these tests.
    np = None
    torch = None

if torch is not None:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BCP", "GIF_and_CM"))
    from torch_energy import TorchEnergy


def reference_symmetric_dirichlet(x_small, J_fz, J_fbz, area, fixed_indices, fixed_values, spd, exclude_flips):
    """Literal NumPy transcription of symmetricDirichletEnergy.m (one-based fixed indices)."""
    fixed = np.asarray(fixed_indices, dtype=np.int64).reshape(-1) - 1
    free = np.setdiff1d(np.arange(x_small.size + fixed.size), fixed)
    x = np.zeros(x_small.size + fixed.size)
    x[free] = x_small.reshape(-1)
    x[fixed] = np.asarray(fixed_values, dtype=np.float64).reshape(-1)
    n = x.size // 2
    xc = x[:n] + 1j * x[n:]
    J_fz = np.asarray(J_fz, dtype=np.complex128)
    J_fbz = np.asarray(J_fbz, dtype=np.complex128)
    area = np.asarray(area, dtype=np.float64).reshape(-1)
    fz, fbz = J_fz @ xc, J_fbz @ xc
    abs_fz, abs_fbz = np.abs(fz), np.abs(fbz)
    SIGMA, sigma = abs_fz + abs_fbz, abs_fz - abs_fbz
    foldovers = np.nonzero(sigma < 0)[0]
    if exclude_flips:
        keep = sigma >= 0
        J_fz, J_fbz, fz, fbz, area = J_fz[keep], J_fbz[keep], fz[keep], fbz[keep], area[keep]
        abs_fz, abs_fbz, SIGMA, sigma = abs_fz[keep], abs_fbz[keep], SIGMA[keep], sigma[keep]
    E = 0.5 * area @ (SIGMA**2 + SIGMA**-2.0 + sigma**2 + sigma**-2.0)
    grad_h_SIGMA = area * ((SIGMA**4 - 1) / SIGMA**3)
    grad_h_sigma = area * ((sigma**4 - 1) / sigma**3)
    grad_abs_fz = np.conj(J_fz) * (fz / abs_fz)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        fbz_direction = fbz / abs_fbz
    fbz_direction[abs_fbz == 0] = 1
    grad_abs_fbz = np.conj(J_fbz) * fbz_direction[:, None]
    grad_SIGMA_x = grad_abs_fz + grad_abs_fbz
    grad_sigma_x = grad_abs_fz - grad_abs_fbz
    G_complex = grad_SIGMA_x.T @ grad_h_SIGMA + grad_sigma_x.T @ grad_h_sigma
    G = np.concatenate((G_complex.real, G_complex.imag))[free]

    def split(m):
        return np.concatenate((m.real, m.imag), axis=1)

    gS, gs, gfz, gfbz = split(grad_SIGMA_x), split(grad_sigma_x), split(grad_abs_fz), split(grad_abs_fbz)
    H1 = gS.T @ (gS * (area * (1 + 3 * SIGMA**-4.0))[:, None]) + gs.T @ (gs * (area * (1 + 3 * sigma**-4.0))[:, None])
    Ar1 = np.concatenate((J_fz.real, -J_fz.imag), axis=1)
    Ar2 = np.concatenate((J_fz.imag, J_fz.real), axis=1)
    Br1 = np.concatenate((J_fbz.real, -J_fbz.imag), axis=1)
    Br2 = np.concatenate((J_fbz.imag, J_fbz.real), axis=1)
    alpha = grad_h_SIGMA + grad_h_sigma
    beta = grad_h_SIGMA - grad_h_sigma
    if spd:
        alpha = np.maximum(alpha, 0)
    s = (alpha / abs_fz)[:, None]
    HA = Ar1.T @ (Ar1 * s) + Ar2.T @ (Ar2 * s) - gfz.T @ (gfz * s)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = beta / abs_fbz
    s[abs_fbz == 0] = 1
    s = s[:, None]
    HB = Br1.T @ (Br1 * s) + Br2.T @ (Br2 * s) - gfbz.T @ (gfbz * s)
    H = (H1 + (HA + HB))[np.ix_(free, free)]
    return {"foldovers": foldovers + 1, "fz": fz, "fbz": fbz, "E": E, "G": G, "H": H}


def make_problem(seed, face_count=7, complex_count=4, fbz_ratio=0.3, fixed_indices=(2, 8)):
    """Random dense problem with |fbz| = fbz_ratio * |fz| per face at the returned x."""
    rng = np.random.default_rng(seed)
    J_fz = rng.standard_normal((face_count, complex_count)) + 1j * rng.standard_normal((face_count, complex_count))
    J_fbz = rng.standard_normal((face_count, complex_count)) + 1j * rng.standard_normal((face_count, complex_count))
    x_full = rng.standard_normal(2 * complex_count)
    xc = x_full[:complex_count] + 1j * x_full[complex_count:]
    ratio = np.broadcast_to(np.asarray(fbz_ratio, dtype=np.float64), (face_count,))
    J_fbz = J_fbz * (ratio * np.abs(J_fz @ xc) / np.abs(J_fbz @ xc))[:, None]
    area = rng.uniform(0.5, 1.5, (face_count, 1))
    fixed = np.asarray(fixed_indices, dtype=np.float64).reshape(-1, 1)
    fixed_zero_based = fixed.astype(np.int64).reshape(-1) - 1
    fixed_values = x_full[fixed_zero_based].reshape(-1, 1)
    free = np.setdiff1d(np.arange(2 * complex_count), fixed_zero_based)
    x_small = x_full[free].reshape(-1, 1)
    return {"J_fz": J_fz, "J_fbz": J_fbz, "area": area, "fixed_indices": fixed, "fixed_values": fixed_values}, x_small


def prepared_engine(problem):
    engine = TorchEnergy("cpu")
    engine.prepare(problem["J_fz"], problem["J_fbz"], problem["area"], problem["fixed_indices"], problem["fixed_values"])
    return engine


@unittest.skipIf(torch is None, "NumPy/PyTorch not installed")
class TorchEnergyTest(unittest.TestCase):
    def assert_matches_reference(self, problem, x, spd, exclude_flips):
        out = prepared_engine(problem).evaluate(x, spd=spd, exclude_flips=exclude_flips)
        ref = reference_symmetric_dirichlet(x, problem["J_fz"], problem["J_fbz"], problem["area"],
                                            problem["fixed_indices"], problem["fixed_values"], spd, exclude_flips)
        np.testing.assert_array_equal(out["foldovers"].reshape(-1), ref["foldovers"])
        np.testing.assert_allclose(out["fz"].reshape(-1), ref["fz"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out["fbz"].reshape(-1), ref["fbz"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out["E"].reshape(()), ref["E"], rtol=1e-12)
        np.testing.assert_allclose(out["G"].reshape(-1), ref["G"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(out["H"], ref["H"], rtol=1e-11, atol=1e-10)
        return out

    def finite_difference_check(self, problem, x, exclude_flips):
        engine = prepared_engine(problem)
        out = engine.evaluate(x, spd=False, exclude_flips=exclude_flips)
        step = 1e-6
        fd_G = np.zeros(x.size)
        fd_H = np.zeros((x.size, x.size))
        for k in range(x.size):
            plus, minus = x.copy(), x.copy()
            plus[k] += step
            minus[k] -= step
            out_plus = engine.evaluate(plus, spd=False, exclude_flips=exclude_flips)
            out_minus = engine.evaluate(minus, spd=False, exclude_flips=exclude_flips)
            np.testing.assert_array_equal(out_plus["foldovers"], out["foldovers"])
            np.testing.assert_array_equal(out_minus["foldovers"], out["foldovers"])
            fd_G[k] = (out_plus["E"][0, 0] - out_minus["E"][0, 0]) / (2 * step)
            fd_H[:, k] = (out_plus["G"][:, 0] - out_minus["G"][:, 0]) / (2 * step)
        G_scale = max(1.0, np.abs(out["G"]).max())
        H_scale = max(1.0, np.abs(out["H"]).max())
        np.testing.assert_allclose(out["G"][:, 0], fd_G, rtol=0, atol=1e-6 * G_scale)
        np.testing.assert_allclose(out["H"], fd_H, rtol=0, atol=1e-6 * H_scale)
        np.testing.assert_allclose(out["H"], out["H"].T, rtol=0, atol=1e-12 * H_scale)

    def test_gradient_and_unprojected_hessian_match_finite_differences(self):
        problem, x = make_problem(seed=1)
        self.finite_difference_check(problem, x, exclude_flips=True)
        self.finite_difference_check(problem, x, exclude_flips=False)

    def test_finite_differences_with_excluded_foldovers(self):
        problem, x = make_problem(seed=2, fbz_ratio=[0.3, 1.5, 0.2, 0.4, 2.0, 0.5, 0.1])
        self.assertEqual(prepared_engine(problem).evaluate(x)["foldovers"].reshape(-1).tolist(), [2.0, 5.0])
        self.finite_difference_check(problem, x, exclude_flips=True)

    def test_finite_differences_with_included_foldovers(self):
        problem, x = make_problem(seed=3, fbz_ratio=[0.3, 1.5, 0.2, 0.4, 2.0, 0.5, 0.1])
        self.finite_difference_check(problem, x, exclude_flips=False)

    def test_finite_differences_without_fixed_variables(self):
        problem, x = make_problem(seed=4, fixed_indices=np.zeros((0, 1)))
        self.assertEqual(x.shape, (8, 1))
        self.finite_difference_check(problem, x, exclude_flips=True)

    def test_matches_literal_numpy_reference_including_spd_projection(self):
        problem, x = make_problem(seed=5, fbz_ratio=[0.3, 1.5, 0.2, 0.9, 2.0, 0.5, 0.1])
        # Shrink the map so some faces have alpha < 0 and the SPD clamp is active.
        for scale in (1.0, 0.3):
            scaled = dict(problem, fixed_values=problem["fixed_values"] * scale)
            for spd in (True, False):
                for exclude_flips in (True, False):
                    self.assert_matches_reference(scaled, x * scale, spd, exclude_flips)
        scaled = dict(problem, fixed_values=problem["fixed_values"] * 0.3)
        engine = prepared_engine(scaled)
        self.assertFalse(np.allclose(engine.evaluate(x * 0.3, spd=True)["H"], engine.evaluate(x * 0.3, spd=False)["H"]))

    def test_fixed_variables_select_free_entries_of_full_problem(self):
        problem, x = make_problem(seed=6)
        out = prepared_engine(problem).evaluate(x, spd=True)
        fixed = problem["fixed_indices"].astype(np.int64).reshape(-1) - 1
        free = np.setdiff1d(np.arange(8), fixed)
        x_full = np.zeros(8)
        x_full[free] = x.reshape(-1)
        x_full[fixed] = problem["fixed_values"].reshape(-1)
        full = prepared_engine(dict(problem, fixed_indices=np.zeros((0, 1)), fixed_values=np.zeros((0, 1))))
        out_full = full.evaluate(x_full.reshape(-1, 1), spd=True)
        np.testing.assert_allclose(out["E"], out_full["E"], rtol=1e-14)
        np.testing.assert_allclose(out["G"], out_full["G"][free], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out["H"], out_full["H"][np.ix_(free, free)], rtol=1e-12, atol=1e-12)

    def test_zero_fbz_uses_unit_direction_and_unit_hb_multiplier(self):
        # Single face whose fbz row is non-zero but annihilates x exactly.
        xc = np.array([1 + 2j, 3 - 1j])
        a = np.array([[2 - 1j, 0.5 + 1j]])
        b = np.array([[xc[1], -xc[0]]])
        area = np.array([[1.25]])
        x = np.concatenate((xc.real, xc.imag)).reshape(-1, 1)
        engine = TorchEnergy("cpu")
        engine.prepare(a, b, area, np.zeros((0, 1)), np.zeros((0, 1)))
        for spd in (True, False):
            out = engine.evaluate(x, spd=spd)
            self.assertEqual(out["fbz"][0, 0], 0)
            fz = (a @ xc)[0]
            abs_fz = abs(fz)
            ghS = ghs = area[0, 0] * (abs_fz**4 - 1) / abs_fz**3
            alpha, beta = ghS + ghs, ghS - ghs
            if spd:
                alpha = max(alpha, 0.0)
            grad_abs_fz = np.conj(a[0]) * fz / abs_fz
            grad_abs_fbz = np.conj(b[0]) * 1  # direction 1 where |fbz| == 0
            gfz = np.concatenate((grad_abs_fz.real, grad_abs_fz.imag))
            gfbz = np.concatenate((grad_abs_fbz.real, grad_abs_fbz.imag))
            expected_G = (gfz + gfbz) * ghS + (gfz - gfbz) * ghs
            h = area[0, 0] * (1 + 3 * abs_fz**-4)
            Ar1 = np.concatenate((a[0].real, -a[0].imag))
            Ar2 = np.concatenate((a[0].imag, a[0].real))
            Br1 = np.concatenate((b[0].real, -b[0].imag))
            Br2 = np.concatenate((b[0].imag, b[0].real))
            HA = alpha / abs_fz * (np.outer(Ar1, Ar1) + np.outer(Ar2, Ar2) - np.outer(gfz, gfz))
            HB = 1.0 * (np.outer(Br1, Br1) + np.outer(Br2, Br2) - np.outer(gfbz, gfbz))  # multiplier 1
            expected_H = h * np.outer(gfz + gfbz, gfz + gfbz) + h * np.outer(gfz - gfbz, gfz - gfbz) + HA + HB
            self.assertTrue(np.all(np.isfinite(out["H"])))
            np.testing.assert_allclose(out["G"].reshape(-1), expected_G, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(out["H"], expected_H, rtol=1e-12, atol=1e-12)
            self.assertGreater(np.abs(HB).max(), 0.1)

    def test_foldover_filtering(self):
        ratios = [0.3, 1.5, 0.2, 0.4, 2.0, 0.5, 0.1]
        problem, x = make_problem(seed=7, fbz_ratio=ratios)
        engine = prepared_engine(problem)
        excluded = engine.evaluate(x, exclude_flips=True)
        included = engine.evaluate(x, exclude_flips=False)
        self.assertEqual(excluded["foldovers"].shape, (2, 1))
        np.testing.assert_array_equal(excluded["foldovers"], [[2.0], [5.0]])
        np.testing.assert_array_equal(included["foldovers"], [[2.0], [5.0]])
        keep = np.array([0, 2, 3, 5, 6])
        self.assertEqual(excluded["fz"].shape, (5, 1))
        self.assertEqual(included["fz"].shape, (7, 1))
        np.testing.assert_array_equal(excluded["fz"], included["fz"][keep])
        np.testing.assert_array_equal(excluded["fbz"], included["fbz"][keep])
        kept_problem = dict(problem, J_fz=problem["J_fz"][keep], J_fbz=problem["J_fbz"][keep], area=problem["area"][keep])
        kept = prepared_engine(kept_problem).evaluate(x, exclude_flips=False)
        self.assertEqual(kept["foldovers"].shape, (0, 1))
        for key in ("E", "G", "H"):
            np.testing.assert_allclose(excluded[key], kept[key], rtol=1e-12, atol=1e-12)
        self.assertGreater(abs(included["E"][0, 0] - excluded["E"][0, 0]), 1e-3)

    def test_all_flipped(self):
        problem, x = make_problem(seed=8, fbz_ratio=2.0)
        engine = prepared_engine(problem)
        excluded = engine.evaluate(x, exclude_flips=True)
        np.testing.assert_array_equal(excluded["foldovers"].reshape(-1), np.arange(1, 8))
        self.assertEqual(excluded["fz"].shape, (0, 1))
        self.assertEqual(excluded["fbz"].shape, (0, 1))
        np.testing.assert_array_equal(excluded["E"], [[0.0]])
        np.testing.assert_array_equal(excluded["G"], np.zeros((6, 1)))
        np.testing.assert_array_equal(excluded["H"], np.zeros((6, 6)))
        included = engine.evaluate(x, exclude_flips=False)
        self.assertEqual(included["fz"].shape, (7, 1))
        self.assertTrue(np.all(np.isfinite(included["H"])))
        self.assert_matches_reference(problem, x, spd=True, exclude_flips=False)

    def test_prepare_replaces_cache_and_copies_inputs(self):
        problem_a, x_a = make_problem(seed=9)
        problem_b, x_b = make_problem(seed=10, face_count=5, complex_count=3, fixed_indices=(1,))
        engine = prepared_engine(problem_a)
        first_a = engine.evaluate(x_a)
        problem_a["J_fz"][:] = 0  # cached device data must not alias caller arrays
        problem_a["area"][:] = 0
        np.testing.assert_array_equal(engine.evaluate(x_a)["H"], first_a["H"])
        engine.prepare(problem_b["J_fz"], problem_b["J_fbz"], problem_b["area"],
                       problem_b["fixed_indices"], problem_b["fixed_values"])
        out_b = engine.evaluate(x_b)
        fresh_b = prepared_engine(problem_b).evaluate(x_b)
        self.assertEqual(out_b["H"].shape, (5, 5))
        for key in out_b:
            np.testing.assert_array_equal(out_b[key], fresh_b[key])
        with self.assertRaises(ValueError):
            engine.evaluate(x_a)

    def test_real_inputs_and_one_dimensional_columns(self):
        problem, x = make_problem(seed=11)
        real_problem = dict(problem, J_fz=problem["J_fz"].real.copy(), J_fbz=problem["J_fbz"].real.copy())
        complex_problem = dict(real_problem, J_fz=real_problem["J_fz"] + 0j, J_fbz=real_problem["J_fbz"] + 0j)
        out_real = prepared_engine(real_problem).evaluate(x)
        out_complex = prepared_engine(complex_problem).evaluate(x)
        for key in out_real:
            np.testing.assert_array_equal(out_real[key], out_complex[key])
        engine = TorchEnergy("cpu")
        engine.prepare(problem["J_fz"], problem["J_fbz"], problem["area"].reshape(-1),
                       problem["fixed_indices"].reshape(-1).astype(np.int32), problem["fixed_values"].reshape(-1))
        out_flat = engine.evaluate(x.reshape(-1))
        out_column = prepared_engine(problem).evaluate(x)
        for key in out_flat:
            np.testing.assert_array_equal(out_flat[key], out_column[key])

    def test_output_shapes_and_dtypes(self):
        problem, x = make_problem(seed=12)
        out = prepared_engine(problem).evaluate(x)
        self.assertEqual(set(out), {"foldovers", "fz", "fbz", "E", "G", "H"})
        self.assertEqual(out["foldovers"].shape, (0, 1))
        self.assertEqual(out["foldovers"].dtype, np.float64)
        self.assertEqual((out["fz"].shape, out["fz"].dtype), ((7, 1), np.complex128))
        self.assertEqual((out["fbz"].shape, out["fbz"].dtype), ((7, 1), np.complex128))
        self.assertEqual((out["E"].shape, out["E"].dtype), ((1, 1), np.float64))
        self.assertEqual((out["G"].shape, out["G"].dtype), ((6, 1), np.float64))
        self.assertEqual((out["H"].shape, out["H"].dtype), ((6, 6), np.float64))

    def test_invalid_configuration_is_rejected(self):
        problem, x = make_problem(seed=13)
        engine = TorchEnergy("cpu")
        with self.assertRaises(RuntimeError):
            engine.evaluate(x)

        def prepare(**overrides):
            args = dict(problem, **overrides)
            engine.prepare(args["J_fz"], args["J_fbz"], args["area"], args["fixed_indices"], args["fixed_values"])

        with self.assertRaises(ValueError):
            prepare(J_fbz=problem["J_fbz"][:, :3])
        with self.assertRaises(ValueError):
            prepare(area=problem["area"][:5])
        with self.assertRaises(ValueError):
            prepare(fixed_indices=np.array([[0.0], [3.0]]))
        with self.assertRaises(ValueError):
            prepare(fixed_indices=np.array([[1.0], [9.0]]))
        with self.assertRaises(ValueError):
            prepare(fixed_indices=np.array([[1.5], [3.0]]))
        with self.assertRaises(ValueError):
            prepare(fixed_indices=np.array([[3.0], [3.0]]))
        with self.assertRaises(ValueError):
            prepare(fixed_values=np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            prepare(J_fz=np.full_like(problem["J_fz"], np.nan))
        prepare()
        with self.assertRaises(ValueError):
            engine.evaluate(np.zeros((5, 1)))
        with self.assertRaises(TypeError):
            engine.evaluate(x, spd=1)
        with self.assertRaises(ValueError):
            TorchEnergy("gpu")

    def test_selected_hessian_is_exact_full_submatrix(self):
        problem, x = make_problem(seed=15)
        full = prepared_engine(problem).evaluate(x)
        selection = np.array([[6], [1], [3]])
        engine = TorchEnergy('cpu')
        engine.prepare(problem['J_fz'], problem['J_fbz'], problem['area'],
                       problem['fixed_indices'], problem['fixed_values'], selection)
        selected = engine.evaluate(x)
        indices = selection[:, 0] - 1
        np.testing.assert_array_equal(selected['H'], full['H'][np.ix_(indices, indices)])
        np.testing.assert_array_equal(selected['G'], full['G'])
        with self.assertRaises(ValueError):
            engine.prepare(problem['J_fz'], problem['J_fbz'], problem['area'],
                           problem['fixed_indices'], problem['fixed_values'], [0])

    def test_cpu_selection_does_not_probe_cuda(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("CUDA probed for device='cpu'")

        problem, x = make_problem(seed=14)
        with mock.patch.object(torch.cuda, "is_available", forbidden), \
                mock.patch.object(torch.cuda, "init", forbidden), \
                mock.patch.object(torch.cuda, "device_count", forbidden), \
                mock.patch.object(torch.cuda, "current_device", forbidden):
            engine = prepared_engine(problem)
            engine.evaluate(x)
        self.assertEqual(engine.device, "cpu")

    def test_explicit_cuda_initialization_error_propagates_without_fallback(self):
        sentinel = RuntimeError("sentinel CUDA initialization failure")

        def failing_init():
            raise sentinel

        with mock.patch.object(torch.cuda, "init", failing_init):
            with self.assertRaises(RuntimeError) as caught:
                TorchEnergy("cuda:0")
        self.assertIs(caught.exception, sentinel)

    def test_auto_selects_cpu_when_cuda_unavailable(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(TorchEnergy("auto").device, "cpu")


if __name__ == "__main__":
    unittest.main()
