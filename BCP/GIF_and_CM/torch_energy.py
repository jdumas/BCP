"""PyTorch port of GIF/GIF/MatlabScripts/symmetricDirichletEnergy.m.

The energy, gradient and Hessian are the analytic expressions of the MATLAB
reference, evaluated in float64/complex128 on a selected torch device.  The
Jacobians, areas and fixed-variable data are uploaded once by
``TorchEnergy.prepare`` and reused by every ``TorchEnergy.evaluate`` call, which
only uploads the free coordinates and downloads the results. An optional
one-based ``hessian_indices`` argument to prepare selects a Hessian sub-block
on the device after the full calculation, preserving its entries exactly while
avoiding a full-Hessian download. GPU caches are released before a new scaffold
is allocated; conjugates are lazy views rather than additional matrix copies.

Conventions preserved from the MATLAB reference:

* The full coordinate vector is ``[real coordinates; imaginary coordinates]``
  (length ``2 * N``); free coordinates fill ``setdiff(1:2N, FixedIndices)`` in
  ascending order and fixed coordinates take ``FixedValues``.
* ``foldovers`` are the one-based faces with ``|fz| - |fbz| < 0``, reported
  before any filtering.  ``exclude_flips=True`` corresponds to ``i == 0``
  (fold-over rows removed from every term); ``exclude_flips=False`` corresponds
  to ``i ~= 0`` (all rows kept).
* Where ``|fbz| == 0`` the fbz direction is ``1`` and the HB multiplier is ``1``.
* The complex gradient uses the non-conjugate transpose (``.'``).
* ``spd=True`` applies ``alpha = max(alpha, 0)`` (MATLAB/Octave ``max`` maps NaN
  to 0).  No other regularization is applied.
"""

import numpy as np
import torch

__all__ = ["TorchEnergy"]


def _select_device(device):
    if not isinstance(device, str):
        raise TypeError(f"device must be a string ('cpu', 'cuda[:index]' or 'auto'), got {type(device).__name__}")
    if device == "cpu":
        # Deliberately no torch.cuda call: CPU selection must not probe CUDA.
        return torch.device("cpu")
    if device == "auto":
        if torch.cuda.is_available():
            return _initialize_cuda(torch.device("cuda", 0))
        return torch.device("cpu")
    if device == "cuda" or device.startswith("cuda:"):
        try:
            requested = torch.device(device)
        except RuntimeError as error:
            raise ValueError(f"invalid CUDA device string {device!r}: {error}") from error
        return _initialize_cuda(requested)
    raise ValueError(f"device must be 'cpu', 'cuda[:index]' or 'auto', got {device!r}")


def _initialize_cuda(requested):
    # Any CUDA initialization error propagates unchanged; there is no CPU fallback.
    torch.cuda.init()
    index = requested.index if requested.index is not None else torch.cuda.current_device()
    selected = torch.device("cuda", index)
    torch.zeros(1, dtype=torch.float64, device=selected)
    torch.cuda.synchronize(selected)
    return selected


def _real_float64(name, value):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a numeric array, got dtype {array.dtype}")
    if np.iscomplexobj(array):
        if np.any(array.imag != 0):
            raise ValueError(f"{name} must be real")
        array = array.real
    return array.astype(np.float64)


def _column(name, array, length):
    """Accept shape (length, 1) or (length,); an empty column may also be (0, 0)."""
    if length == 0 and array.size == 0:
        return array.reshape(0)
    if array.shape in ((length, 1), (length,)):
        return array.reshape(length)
    raise ValueError(f"{name} must have shape ({length}, 1), got {array.shape}")


class TorchEnergy:
    def __init__(self, device="auto"):
        self._torch_device = _select_device(device)
        self.device = str(self._torch_device)
        self._prepared = None

    @torch.inference_mode()
    def prepare(self, J_fz, J_fbz, area, fixed_indices, fixed_values, hessian_indices=None):
        J_fz = np.asarray(J_fz)
        J_fbz = np.asarray(J_fbz)
        for name, matrix in (("J_fz", J_fz), ("J_fbz", J_fbz)):
            if not np.issubdtype(matrix.dtype, np.number):
                raise TypeError(f"{name} must be a numeric array, got dtype {matrix.dtype}")
            if matrix.ndim != 2:
                raise ValueError(f"{name} must be a 2-D matrix, got shape {matrix.shape}")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be finite")
        if J_fz.shape != J_fbz.shape:
            raise ValueError(f"J_fz and J_fbz shapes differ: {J_fz.shape} vs {J_fbz.shape}")
        face_count, complex_count = J_fz.shape
        if face_count == 0 or complex_count == 0:
            raise ValueError(f"Jacobians must be non-empty, got shape {J_fz.shape}")
        coordinate_count = 2 * complex_count

        area = _column("area", _real_float64("area", area), face_count)
        if not np.all(np.isfinite(area)):
            raise ValueError("area must be finite")

        fixed_indices = _real_float64("fixed_indices", fixed_indices)
        fixed_values = _real_float64("fixed_values", fixed_values)
        fixed_count = fixed_indices.size
        fixed_indices = _column("fixed_indices", fixed_indices, fixed_count)
        fixed_values = _column("fixed_values", fixed_values, fixed_count)
        if not np.all(np.isfinite(fixed_indices)) or np.any(fixed_indices != np.round(fixed_indices)):
            raise ValueError("fixed_indices must be integers")
        if np.any(fixed_indices < 1) or np.any(fixed_indices > coordinate_count):
            raise ValueError(f"fixed_indices must be one-based indices in [1, {coordinate_count}]")
        fixed_zero_based = fixed_indices.astype(np.int64) - 1
        if np.unique(fixed_zero_based).size != fixed_count:
            raise ValueError("fixed_indices must be unique")
        if not np.all(np.isfinite(fixed_values)):
            raise ValueError("fixed_values must be finite")
        free_zero_based = np.setdiff1d(np.arange(coordinate_count, dtype=np.int64), fixed_zero_based)

        selected_hessian = None
        if hessian_indices is not None:
            selected_hessian = _real_float64("hessian_indices", hessian_indices)
            selected_hessian = _column("hessian_indices", selected_hessian, selected_hessian.size)
            if (not np.all(np.isfinite(selected_hessian)) or
                np.any(selected_hessian != np.round(selected_hessian)) or
                np.any(selected_hessian < 1) or np.any(selected_hessian > free_zero_based.size) or
                np.unique(selected_hessian).size != selected_hessian.size):
                raise ValueError("hessian_indices must be unique one-based free-coordinate indices")
            selected_hessian = selected_hessian.astype(np.int64) - 1

        # All validation is complete. Release old device buffers before allocating
        # the next scaffold's cache, rather than holding two full GPU caches.
        self._prepared = None
        device = self._torch_device
        J_fz_t = torch.tensor(J_fz, dtype=torch.complex128, device=device)
        J_fbz_t = torch.tensor(J_fbz, dtype=torch.complex128, device=device)
        free_t = torch.tensor(free_zero_based, dtype=torch.int64, device=device)

        def free_columns(matrix):
            return matrix.index_select(1, free_t).contiguous()

        # Ar1/Ar2/Br1/Br2 of the reference, restricted to free columns: entry (i, j)
        # of every Hessian term depends only on columns i and j, so the free
        # sub-block H(free, free) can be formed directly.
        prepared = {
            "complex_count": complex_count,
            "free_count": free_zero_based.size,
            "J_fz": J_fz_t,
            "J_fbz": J_fbz_t,
            "Ar1": free_columns(torch.cat((J_fz_t.real, -J_fz_t.imag), dim=1)),
            "Ar2": free_columns(torch.cat((J_fz_t.imag, J_fz_t.real), dim=1)),
            "Br1": free_columns(torch.cat((J_fbz_t.real, -J_fbz_t.imag), dim=1)),
            "Br2": free_columns(torch.cat((J_fbz_t.imag, J_fbz_t.real), dim=1)),
            "area": torch.tensor(area, dtype=torch.float64, device=device),
            "hessian_indices": None if selected_hessian is None else torch.tensor(selected_hessian, dtype=torch.int64, device=device),
            "free_indices": free_t,
            "fixed_indices": torch.tensor(fixed_zero_based, dtype=torch.int64, device=device),
            "fixed_values": torch.tensor(fixed_values, dtype=torch.float64, device=device),
        }
        self._prepared = prepared

    @torch.inference_mode()
    def evaluate(self, x, spd=True, exclude_flips=True):
        prepared = self._prepared
        if prepared is None:
            raise RuntimeError("TorchEnergy.prepare must be called before evaluate")
        for name, flag in (("spd", spd), ("exclude_flips", exclude_flips)):
            if not isinstance(flag, (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool, got {type(flag).__name__}")
        free_count = prepared["free_count"]
        x = _column("x", _real_float64("x", x), free_count)

        device = self._torch_device
        n = prepared["complex_count"]
        free_indices = prepared["free_indices"]

        # Fixed-variable expansion.
        x_small = torch.tensor(x, dtype=torch.float64, device=device)
        x_full = torch.zeros(2 * n, dtype=torch.float64, device=device)
        x_full[free_indices] = x_small
        x_full[prepared["fixed_indices"]] = prepared["fixed_values"]
        x_complex = torch.complex(x_full[:n], x_full[n:])

        J_fz = prepared["J_fz"]
        J_fbz = prepared["J_fbz"]
        fz = J_fz @ x_complex
        fbz = J_fbz @ x_complex
        abs_fz = fz.abs()
        abs_fbz = fbz.abs()
        SIGMA = abs_fz + abs_fbz
        sigma = abs_fz - abs_fbz
        foldover_mask = sigma < 0
        foldovers = torch.nonzero(foldover_mask).reshape(-1)

        # Lazy conjugate views avoid two extra persistent complex matrix copies.
        J_fz_conj = J_fz.conj()
        J_fbz_conj = J_fbz.conj()
        Ar1, Ar2 = prepared["Ar1"], prepared["Ar2"]
        Br1, Br2 = prepared["Br1"], prepared["Br2"]
        area = prepared["area"]
        if exclude_flips and foldovers.numel() > 0:
            keep = torch.nonzero(~foldover_mask).reshape(-1)
            J_fz_conj = J_fz_conj.index_select(0, keep)
            J_fbz_conj = J_fbz_conj.index_select(0, keep)
            Ar1, Ar2 = Ar1.index_select(0, keep), Ar2.index_select(0, keep)
            Br1, Br2 = Br1.index_select(0, keep), Br2.index_select(0, keep)
            area = area.index_select(0, keep)
            fz, fbz = fz.index_select(0, keep), fbz.index_select(0, keep)
            abs_fz, abs_fbz = abs_fz.index_select(0, keep), abs_fbz.index_select(0, keep)
            SIGMA, sigma = SIGMA.index_select(0, keep), sigma.index_select(0, keep)

        # Energy.
        E = 0.5 * (area @ (SIGMA.pow(2) + SIGMA.pow(-2) + sigma.pow(2) + sigma.pow(-2)))

        # Gradient.
        grad_h_SIGMA = area * ((SIGMA.pow(4) - 1) / SIGMA.pow(3))
        grad_h_sigma = area * ((sigma.pow(4) - 1) / sigma.pow(3))
        grad_abs_fz = J_fz_conj * (fz / abs_fz).unsqueeze(1)
        fbz_zero = abs_fbz == 0
        fbz_direction = torch.where(fbz_zero, torch.ones_like(fbz), fbz / abs_fbz)
        grad_abs_fbz = J_fbz_conj * fbz_direction.unsqueeze(1)

        def real_free(complex_matrix):
            # [real(.) imag(.)] restricted to the free coordinates.
            return torch.cat((complex_matrix.real, complex_matrix.imag), dim=1).index_select(1, free_indices)

        grad_abs_fz_real = real_free(grad_abs_fz)
        grad_abs_fbz_real = real_free(grad_abs_fbz)
        grad_SIGMA_x_real = grad_abs_fz_real + grad_abs_fbz_real
        grad_sigma_x_real = grad_abs_fz_real - grad_abs_fbz_real
        # G_complex = grad_SIGMA_x.' * grad_h_SIGMA + grad_sigma_x.' * grad_h_sigma with a
        # non-conjugate transpose and real weights, so [real(G_complex); imag(G_complex)]
        # equals the real/imaginary column blocks contracted with the same weights.
        G = grad_SIGMA_x_real.T @ grad_h_SIGMA + grad_sigma_x_real.T @ grad_h_sigma

        # Hessian.
        hess_h11 = area * (1 + 3 * SIGMA.pow(-4))
        hess_h22 = area * (1 + 3 * sigma.pow(-4))
        D_g1g1 = grad_SIGMA_x_real.T @ (grad_SIGMA_x_real * hess_h11.unsqueeze(1))
        D_g2g2 = grad_sigma_x_real.T @ (grad_sigma_x_real * hess_h22.unsqueeze(1))
        H1 = D_g1g1 + D_g2g2

        alpha = grad_h_SIGMA + grad_h_sigma
        beta = grad_h_SIGMA - grad_h_sigma
        if spd:
            alpha = torch.where(alpha > 0, alpha, torch.zeros_like(alpha))

        scalar_for_mult = (alpha / abs_fz).unsqueeze(1)
        HA = Ar1.T @ (Ar1 * scalar_for_mult) + Ar2.T @ (Ar2 * scalar_for_mult)
        HA = HA - grad_abs_fz_real.T @ (grad_abs_fz_real * scalar_for_mult)

        scalar_for_mult = torch.where(fbz_zero, torch.ones_like(beta), beta / abs_fbz).unsqueeze(1)
        HB = Br1.T @ (Br1 * scalar_for_mult) + Br2.T @ (Br2 * scalar_for_mult)
        HB = HB - grad_abs_fbz_real.T @ (grad_abs_fbz_real * scalar_for_mult)

        H2 = HA + HB
        H = H1 + H2
        if prepared["hessian_indices"] is not None:
            selection = prepared["hessian_indices"]
            H = H.index_select(0, selection).index_select(1, selection)

        return {
            "foldovers": (foldovers + 1).to(torch.float64).reshape(-1, 1).cpu().numpy(),
            "fz": fz.reshape(-1, 1).cpu().numpy(),
            "fbz": fbz.reshape(-1, 1).cpu().numpy(),
            "E": E.reshape(1, 1).cpu().numpy(),
            "G": G.reshape(-1, 1).cpu().numpy(),
            "H": H.cpu().numpy(),
        }
