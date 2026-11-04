#!/usr/bin/env python3
"""Comparison and diagnostic harness for license-free reference vs Linux/CUDA parity.

Provides two explicit execution modes:
- reference: original .m logic through Octave on CPU with MKL/Accelerate solver.
  Clearly labeled numerical reference, not original-placement performance.
- parity: Linux x86-64 with CUDA for Python/Torch preprocessing+prolongation and
  GIF energy evaluation, plus MKL CPU sparse solves. No auto/CPU fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "BCP", ROOT / "BCP/utils", ROOT / "BCP/GIF_and_CM", ROOT / "scripts"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

STEP_TRACE_RE = re.compile(
    r"^BCP_TRACE_STEP\s+remesh=(?P<remesh>\d+)\s+step=(?P<step>\d+)\s+iter=(?P<iter>\d+)\s+"
    r"E=(?P<E>\S+)\s+Eprev=(?P<Eprev>\S+)\s+normG=(?P<normG>\S+)\s+normG1=(?P<normG1>\S+)\s+"
    r"normD=(?P<normD>\S+)\s+t=(?P<t>\S+)\s+tMax=(?P<tMax>\S+)\s+stepNorm=(?P<stepNorm>\S+)\s+"
    r"foldovers=(?P<foldovers>\d+)\s*$"
)
REMESH_TRACE_RE = re.compile(
    r"^BCP_TRACE_REMESH\s+remesh=(?P<remesh>\d+)\s+newtonIters=(?P<newtonIters>\d+)\s+"
    r"initE=(?P<initE>\S+)\s+finalE=(?P<finalE>\S+)\s+rate=(?P<rate>\S+)\s*$"
)
STEP_FLOAT_KEYS = ("E", "Eprev", "normG", "normG1", "normD", "t", "tMax", "stepNorm")
COMPARE_STEP_KEYS = ("E", "normG", "normG1", "normD", "t", "stepNorm")
REMESH_FLOAT_KEYS = ("initE", "finalE", "rate")
STAGE_PATTERNS = {
    "subspace_seconds": re.compile(r"Subspace construction took\s+([0-9eE.+\-]+)"),
    "factorization_seconds": re.compile(r"Symbolic and Numerical factorization:\s+([0-9eE.+\-]+)"),
    "basis_solve_seconds": re.compile(r"(?:Harmonic basis inverse-column solves|Selected inverse):\s+([0-9eE.+\-]+)"),
    "newton_and_backsub_seconds": re.compile(r"Newton and back substitution took\s+([0-9eE.+\-]+)"),
    "total_gif_seconds": re.compile(r"Total time:\s+([0-9eE.+\-]+)"),
    "final_esd_logged": re.compile(r"Final Esd =\s+([0-9eE.+\-]+)"),
}
GIF_DEVICE_RE = re.compile(r"GIF compute device:\s+(\w+)")
GIF_BACKEND_RE = re.compile(r"GIF evaluator backend:\s+([^\r\n]+)")


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    device: str
    gif_device: str
    gif_backend: str
    solver: str
    claim_scope: str


def _default_cuda_probe() -> Tuple[bool, str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() returned False"
        torch.cuda.init()
        return True, torch.cuda.get_device_properties(torch.device("cuda:0")).name
    except Exception as exc:
        return False, str(exc)


def resolve_mode_spec(
    mode: str,
    *,
    sys_platform: Optional[str] = None,
    solver: Optional[str] = None,
    cuda_probe: Optional[Callable[[], Tuple[bool, str]]] = None,
) -> ModeSpec:
    plat = sys_platform if sys_platform is not None else sys.platform
    eff_solver = (solver or os.environ.get("BCP_LINEAR_SOLVER") or ("ACCELERATE" if plat == "darwin" else "MKL")).upper()
    if mode == "reference":
        if eff_solver not in ("MKL", "ACCELERATE"):
            raise ValueError(f"reference mode requires MKL or ACCELERATE, got {eff_solver}")
        return ModeSpec("reference", "cpu", "cpu", "native", eff_solver,
                        "license_free_numerical_reference_cpu_not_original_gpu_placement")
    if mode == "parity":
        if not str(plat).startswith("linux"):
            raise RuntimeError(f"parity mode requires Linux x86-64 with CUDA and MKL, got '{plat}'")
        if eff_solver != "MKL":
            raise RuntimeError(f"parity mode requires MKL CPU sparse solver; '{eff_solver}' not allowed")
        ok, detail = (cuda_probe or _default_cuda_probe)()
        if not ok:
            raise RuntimeError(f"parity mode requires working CUDA for preprocessing and GIF (no CPU fallback): {detail}")
        return ModeSpec("parity", "cuda:0", "gpu", "torch", "MKL",
                        "linux_cuda_mkl_port_placement_not_certified_matlab_panua_runtime_parity")
    raise ValueError(f"Unsupported mode '{mode}'; choose 'reference', 'parity', or 'both'")


def build_execution_environment(
    spec: ModeSpec,
    *,
    base_env: Optional[Mapping[str, str]] = None,
    timeout_seconds: float = 1800.0,
    octave_timeout_ms: int = 1800000,
    threads: Optional[int] = None,
    kkt_dir: Optional[Path] = None,
    trace_path: Optional[Path] = None,
    stop_after_basis: bool = False,
) -> Dict[str, str]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if octave_timeout_ms <= 0 or octave_timeout_ms > 86400000:
        raise ValueError("octave_timeout_ms must be in (0, 86400000]")
    env = dict(os.environ if base_env is None else base_env)
    env.update({
        "BCP_DEVICE": spec.device,
        "BCP_GIF_DEVICE": spec.gif_device,
        "BCP_GIF_BACKEND": spec.gif_backend,
        "BCP_LINEAR_SOLVER": spec.solver,
        "BCP_PYTHON_EXECUTABLE": sys.executable,
        "BCP_NATIVE_TIMEOUT_SECONDS": str(float(timeout_seconds)),
        "BCP_OCTAVE_TIMEOUT_MS": str(int(octave_timeout_ms)),
    })
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive when specified")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(int(threads))
    for key, val in (
        ("BCP_COMPARE_KKT_DIR", str(Path(kkt_dir).resolve()) if kkt_dir else None),
        ("BCP_COMPARE_TRACE_PATH", str(Path(trace_path).resolve()) if trace_path else None),
        ("BCP_COMPARE_STOP_AFTER_BASIS", "1" if stop_after_basis else None),
    ):
        if val is not None:
            env[key] = val
        else:
            env.pop(key, None)
    return env


def collect_runtime_metadata(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    active = os.environ if env is None else env
    meta: Dict[str, Any] = {
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "thread_env": {k: active.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_THREADING_LAYER", "VECLIB_MAXIMUM_THREADS")},
        "timeout_env": {k: active.get(k) for k in ("BCP_NATIVE_TIMEOUT_SECONDS", "BCP_OCTAVE_TIMEOUT_MS")},
        "numpy_version": np.__version__,
        "octave_executable": active.get("BCP_OCTAVE_EXECUTABLE") or shutil.which("octave-cli") or shutil.which("octave"),
    }
    for mod_name, key in (("scipy", "scipy_version"), ("igl", "igl_version")):
        try:
            meta[key] = getattr(__import__(mod_name), "__version__", "installed")
        except Exception:
            meta[key] = None
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_version"] = torch.version.cuda
        meta["cuda_device_name"] = torch.cuda.get_device_name(0) if meta["cuda_available"] else None
    except Exception:
        meta.update({"torch_version": None, "cuda_available": False, "cuda_version": None})
    return meta


def parse_gif_diagnostics(text: str) -> Dict[str, Any]:
    steps, remeshes = [], []
    for raw in text.splitlines():
        line = raw.strip()
        m_step = STEP_TRACE_RE.match(line)
        if m_step:
            entry = {k: int(m_step.group(k)) for k in ("remesh", "step", "iter", "foldovers")}
            entry.update({k: float(m_step.group(k)) for k in STEP_FLOAT_KEYS})
            steps.append(entry)
            continue
        m_rem = REMESH_TRACE_RE.match(line)
        if m_rem:
            entry = {k: int(m_rem.group(k)) for k in ("remesh", "newtonIters")}
            entry.update({k: float(m_rem.group(k)) for k in REMESH_FLOAT_KEYS})
            remeshes.append(entry)
    dev, backend = GIF_DEVICE_RE.findall(text), GIF_BACKEND_RE.findall(text)
    out: Dict[str, Any] = {
        "gif_compute_device": dev[-1].strip() if dev else None,
        "gif_evaluator_backend": backend[-1].strip() if backend else None,
        "steps": steps,
        "remeshes": remeshes,
    }
    for key, pat in STAGE_PATTERNS.items():
        found = pat.findall(text)
        out[key] = float(found[-1]) if found else None
    return out


def align_uv_rigid(uv_candidate: np.ndarray, uv_reference: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align candidate UVs to reference UVs using ONLY proper 2D rigid motion (SO(2) + translation)."""
    cand, ref = np.asarray(uv_candidate, dtype=np.float64), np.asarray(uv_reference, dtype=np.float64)
    if cand.shape != ref.shape or cand.ndim != 2 or cand.shape[1] != 2 or cand.shape[0] == 0:
        raise ValueError(f"Expected non-empty matching (V, 2) UV arrays, got {cand.shape} vs {ref.shape}")
    if not (np.isfinite(cand).all() and np.isfinite(ref).all()):
        raise ValueError("UV arrays must contain only finite values")
    cand_mean, ref_mean = cand.mean(axis=0), ref.mean(axis=0)
    cand_c, ref_c = cand - cand_mean, ref - ref_mean
    u, _, vt = np.linalg.svd(cand_c.T @ ref_c)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rot = u @ vt
    aligned = cand_c @ rot + ref_mean
    diff = aligned - ref
    max_l2 = float(np.linalg.norm(diff, axis=1).max())
    scale = max(1.0, float(np.linalg.norm(ref_c, axis=1).max(initial=0.0)))
    return aligned, {
        "rotation_matrix": rot.tolist(),
        "rotation_det": float(np.linalg.det(rot)),
        "rotation_angle_rad": float(math.atan2(rot[0, 1], rot[0, 0])),
        "translation": (ref_mean - cand_mean @ rot).tolist(),
        "scale_applied": 1.0,
        "reflection_allowed": False,
        "rms_error": float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))),
        "max_abs_error": float(np.abs(diff).max()),
        "max_vertex_l2_error": max_l2,
        "relative_max_l2_error": max_l2 / scale,
    }


def compute_uv_metrics(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> Dict[str, Any]:
    from compute_stats import compute_stats_from_V
    v, f, uv = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), np.asarray(uvs, dtype=np.float64)
    if uv.shape != (v.shape[0], 2) or not np.isfinite(uv).all():
        raise ValueError(f"Invalid or non-finite UV array of shape {uv.shape}")
    e_sd, p9999, p999, p99, p95, w5, w1, max_e, k_mean, flips = compute_stats_from_V(v, f, uv)
    a, b = uv[f[:, 1]] - uv[f[:, 0]], uv[f[:, 2]] - uv[f[:, 0]]
    non_pos = int(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] <= 0.0))
    return {
        "vertex_count": int(v.shape[0]), "face_count": int(f.shape[0]),
        "e_sd": float(e_sd), "e_sd_95": float(p95), "e_sd_99": float(p99),
        "e_sd_99_9": float(p999), "e_sd_99_99": float(p9999),
        "e_sd_worst_5": float(w5), "e_sd_worst_1": float(w1), "e_sd_max": float(max_e),
        "k_mean": float(k_mean), "flips_sigma": int(flips),
        "non_positive_oriented_faces": non_pos,
        "locally_injective": bool(int(flips) == 0 and non_pos == 0),
    }


def _trace_entries_finite(steps: Sequence[Mapping[str, Any]], remeshes: Sequence[Mapping[str, Any]]) -> bool:
    return (
        all(all(k in s for k in COMPARE_STEP_KEYS) and all(math.isfinite(float(s[k])) for k in STEP_FLOAT_KEYS if k in s) for s in steps)
        and all("finalE" in r and "initE" in r and all(math.isfinite(float(r[k])) for k in REMESH_FLOAT_KEYS if k in r) for r in remeshes)
    )


def compare_traces(
    reference_trace: Mapping[str, Any],
    candidate_trace: Mapping[str, Any],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    confounded_inputs: bool = False,
) -> Dict[str, Any]:
    ref_steps, cand_steps = list(reference_trace.get("steps", [])), list(candidate_trace.get("steps", []))
    ref_rem, cand_rem = list(reference_trace.get("remeshes", [])), list(candidate_trace.get("remeshes", []))
    non_empty = bool(ref_steps and cand_steps and ref_rem and cand_rem)
    finite = _trace_entries_finite(ref_steps, ref_rem) and _trace_entries_finite(cand_steps, cand_rem)
    step_count_match = non_empty and len(ref_steps) == len(cand_steps)
    remesh_count_match = non_empty and len(ref_rem) == len(cand_rem)

    max_rel_diffs, max_abs_diffs = {k: 0.0 for k in COMPARE_STEP_KEYS}, {k: 0.0 for k in COMPARE_STEP_KEYS}
    foldover_mismatches = 0
    within_envelope = bool(non_empty and finite and step_count_match and remesh_count_match)

    for rs, cs in zip(ref_steps, cand_steps):
        if int(rs.get("foldovers", 0)) != int(cs.get("foldovers", 0)):
            foldover_mismatches += 1
            within_envelope = False
        for key in COMPARE_STEP_KEYS:
            rv, cv = float(rs[key]), float(cs[key])
            if not (math.isfinite(rv) and math.isfinite(cv)):
                max_abs_diffs[key] = max_rel_diffs[key] = float("inf")
                within_envelope = False
                continue
            diff = abs(cv - rv)
            max_abs_diffs[key] = max(max_abs_diffs[key], diff)
            max_rel_diffs[key] = max(max_rel_diffs[key], diff / max(1.0, abs(rv)))
            if not (diff <= atol + rtol * abs(rv)):
                within_envelope = False

    remesh_iters_match = remesh_count_match
    for rr, cr in zip(ref_rem, cand_rem):
        if int(rr.get("newtonIters", 0)) != int(cr.get("newtonIters", 0)):
            remesh_iters_match = within_envelope = False
        for key in ("initE", "finalE"):
            rv, cv = float(rr[key]), float(cr[key])
            if not (math.isfinite(rv) and math.isfinite(cv)) or not (abs(cv - rv) <= atol + rtol * abs(rv)):
                within_envelope = False

    return {
        "non_empty": non_empty, "finite": finite,
        "confounded_by_preprocessing": bool(confounded_inputs),
        "reference_step_count": len(ref_steps), "candidate_step_count": len(cand_steps),
        "step_count_match": step_count_match,
        "reference_remesh_count": len(ref_rem), "candidate_remesh_count": len(cand_rem),
        "remesh_count_match": remesh_count_match, "remesh_iters_match": remesh_iters_match,
        "foldover_mismatches": foldover_mismatches,
        "max_abs_diffs": max_abs_diffs, "max_rel_diffs": max_rel_diffs,
        "rtol": rtol, "atol": atol,
        "within_envelope": within_envelope,
        "strict_trajectory_parity_certified": bool(within_envelope and not confounded_inputs),
    }


def _load_check_kkt_module() -> Any:
    try:
        import check_kkt
        return check_kkt
    except ImportError:
        for path in (ROOT / "scripts" / "check_kkt.py", ROOT.parent / "native" / "scripts" / "check_kkt.py"):
            if path.is_file():
                spec = importlib.util.spec_from_file_location("check_kkt", path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["check_kkt"] = mod
                    spec.loader.exec_module(mod)
                    return mod
    raise RuntimeError("scripts/check_kkt.py is required for KKT diagnostic checks")


def normalized_backward_error(matrix: Any, x: np.ndarray, b: np.ndarray) -> float:
    import scipy.sparse as sp
    ck = _load_check_kkt_module()
    x_arr = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    b_arr = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    res_norm = float(np.linalg.norm(matrix @ x_arr - b_arr, ord="fro"))
    a_norm = float(sp.linalg.norm(matrix, ord="fro") if sp.issparse(matrix) else np.linalg.norm(np.asarray(matrix, dtype=np.float64), ord="fro"))
    return float(ck.normalized_backward_error(res_norm, a_norm, float(np.linalg.norm(x_arr, ord="fro")), float(np.linalg.norm(b_arr, ord="fro"))))


def evaluate_kkt_directory(root_dir: Path, **kwargs: Any) -> Dict[str, Any]:
    report = _load_check_kkt_module().check_dump(Path(root_dir), **kwargs)
    sols = [sol for p in report.get("passes", []) for sol in p.get("solutions", [])]
    report["pass_count"] = len(report.get("passes", []))
    report["max_backward_error"] = max((max(float(s.get("backward_error", float("inf"))), float(s.get("max_column_backward_error", float("inf")))) for s in sols), default=0.0)
    report["max_selected_entry_rel_diff"] = max((float(s["selected_entries_relative_difference"]) for s in sols if "selected_entries_relative_difference" in s), default=0.0)
    return report


def run_matched_controller_replay(
    *, candidate_device: str, octave_executable: Optional[str] = None,
    timeout_seconds: float = 180.0, rtol: float = 1e-7, atol: float = 1e-9,
) -> Dict[str, Any]:
    octave = octave_executable or os.environ.get("BCP_OCTAVE_EXECUTABLE") or shutil.which("octave-cli") or shutil.which("octave")
    if not octave:
        raise RuntimeError("Octave executable required for matched controller replay")
    env = dict(os.environ, BCP_PYTHON_EXECUTABLE=sys.executable, BCP_TORCH_REFERENCE_DEVICE=candidate_device, BCP_COMPARE_TRACE_PATH="1")
    env.setdefault("BCP_TORCH_MODULE_DIR", str(ROOT / "BCP" / "GIF_and_CM"))
    if not env.get("BCP_TORCH_NATIVE_DIR"):
        bin_dir = env.get("BCP_BIN_DIR")
        candidates = ([Path(bin_dir).parent, Path(bin_dir)] if bin_dir else []) + [
            ROOT / "build" / "pixi-mkl-octave",
            ROOT / "build" / "pixi-accelerate-octave",
            ROOT / "build" / "pixi-accelerate-octave-source",
        ]
        for candidate in candidates:
            if (candidate / "bcp_torch_energy_native.oct").is_file():
                env["BCP_TORCH_NATIVE_DIR"] = str(candidate.resolve())
                break
    proc = subprocess.run(
        [octave, "--no-gui", "--quiet", str(ROOT / "tests" / "controller_reference.m")],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds, check=True,
    )
    sections = ((proc.stdout or "") + "\n" + (proc.stderr or "")).split("BCP_REPLAY_MODE=")
    ref_diag = parse_gif_diagnostics(next((s for s in sections if s.startswith("native")), ""))
    cand_diag = parse_gif_diagnostics(next((s for s in sections if s.startswith("torch")), ""))
    cmp_res = compare_traces(ref_diag, cand_diag, rtol=rtol, atol=atol, confounded_inputs=False)
    return {"ok": bool(cmp_res["strict_trajectory_parity_certified"]), "candidate_device": candidate_device, "comparison": cmp_res}


def generate_curved_disk_mesh(path: Path, grid_size: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    import igl
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    xy = np.linspace(-1.0, 1.0, grid_size)
    vertices = np.array([[x, y, 0.15 * (x * x + y * y)] for y in xy for x in xy], dtype=np.float64)
    faces = np.array([
        tri for j in range(grid_size - 1) for i in range(grid_size - 1)
        for a in (j * grid_size + i,) for tri in ([a, a + 1, a + grid_size + 1], [a, a + grid_size + 1, a + grid_size])
    ], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    igl.write_triangle_mesh(str(path), vertices, faces)
    return vertices, faces


def load_uvs_from_file(path: Path, expected_vertices: Optional[int] = None) -> np.ndarray:
    import scipy.io as sio
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"UV file not found: {path}")
    if path.suffix.lower() == ".mat":
        data = sio.loadmat(str(path))
        for key in ("uvs_ordered", "uvs", "pred_V", "UV"):
            if key in data:
                uv = np.asarray(data[key], dtype=np.float64)[:, :2]
                break
        else:
            raise KeyError(f"No recognized UV key in MAT file {path}")
    else:
        vt_rows = [[float(p[1]), float(p[2])] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("vt ") for p in (line.split(),)]
        v_rows = [[float(p[1]), float(p[2])] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("v ") for p in (line.split(),)]
        if not (vt_rows or v_rows):
            raise ValueError(f"No vt or v coordinates found in OBJ file {path}")
        uv = np.asarray(vt_rows if vt_rows else v_rows, dtype=np.float64)
    if expected_vertices is not None and uv.shape != (expected_vertices, 2):
        raise ValueError(f"UV shape {uv.shape} from {path} does not match {(expected_vertices, 2)}")
    return uv


def compare_uv_outputs(
    vertices: np.ndarray, faces: np.ndarray, reference_uv: np.ndarray, candidate_uv: np.ndarray,
    *, energy_rtol: float = 1e-5, energy_atol: float = 1e-6, uv_max_l2_tol: float = 1e-4,
) -> Dict[str, Any]:
    ref_m, cand_m = compute_uv_metrics(vertices, faces, reference_uv), compute_uv_metrics(vertices, faces, candidate_uv)
    _, align = align_uv_rigid(candidate_uv, reference_uv)
    e_diff = abs(cand_m["e_sd"] - ref_m["e_sd"])
    energy_ok = bool(math.isfinite(e_diff) and e_diff <= (energy_atol + energy_rtol * abs(ref_m["e_sd"])))
    uv_ok = bool(math.isfinite(align["relative_max_l2_error"]) and align["relative_max_l2_error"] <= uv_max_l2_tol)
    inj_ok = bool(cand_m["locally_injective"] == ref_m["locally_injective"] and cand_m["flips_sigma"] == ref_m["flips_sigma"])
    return {
        "reference_metrics": ref_m, "candidate_metrics": cand_m, "rigid_alignment": align,
        "e_sd_abs_diff": float(e_diff), "e_sd_rel_diff": float(e_diff / max(1.0, abs(ref_m["e_sd"]))),
        "energy_within_tolerance": energy_ok, "uv_within_tolerance": uv_ok,
        "injectivity_match": inj_ok, "passed": bool(energy_ok and uv_ok and inj_ok),
    }


def run_single_mode_pipeline(
    spec: ModeSpec, input_mesh: Path, work_dir: Path, *,
    idt: bool = False, target_faces: Optional[int] = None,
    timeout_seconds: float = 1800.0, octave_timeout_ms: int = 1800000,
    threads: Optional[int] = None, collect_traces: bool = False,
    collect_kkt: bool = False, stop_after_basis: bool = False,
) -> Dict[str, Any]:
    import igl
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_obj = work_dir / f"output_{spec.mode}{'_idt' if idt else ''}.obj"
    trace_marker = work_dir / "trace.enabled" if collect_traces else None
    kkt_dir = work_dir / "kkt_dump" if collect_kkt else None
    if kkt_dir is not None:
        kkt_dir.mkdir(parents=True, exist_ok=True)

    env = build_execution_environment(
        spec, timeout_seconds=timeout_seconds, octave_timeout_ms=octave_timeout_ms,
        threads=threads, kkt_dir=kkt_dir, trace_path=trace_marker, stop_after_basis=stop_after_basis,
    )
    cmd = [
        sys.executable, str(ROOT / "scripts" / "parameterize.py"),
        str(Path(input_mesh).resolve()), str(output_obj.resolve()),
        "--method", "bcp", "--device", spec.device, "--gif-device", spec.gif_device, "--gif-backend", spec.gif_backend,
    ]
    if target_faces is not None:
        cmd += ["--target-faces", str(int(target_faces))]
    if idt:
        cmd.append("--idt")

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start_wall = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    wall_seconds = time.perf_counter() - start_wall
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    stage_match = re.findall(r"BCP stage times:\s*\(([^)]+)\)", combined)
    stage_times = None
    if stage_match:
        parts = [float(x.strip()) for x in stage_match[-1].split(",") if x.strip()]
        if len(parts) == 4:
            stage_times = {
                "simplification_seconds": parts[0], "cm_seconds": parts[1],
                "prolongation_seconds": parts[2], "gif_seconds": parts[3], "pipeline_sum_seconds": sum(parts),
            }

    vertices, faces = igl.read_triangle_mesh(str(Path(input_mesh).resolve()))
    uv_array = load_uvs_from_file(output_obj, expected_vertices=vertices.shape[0]) if (not stop_after_basis and output_obj.is_file()) else None
    return {
        "mode": spec.mode, "claim_scope": spec.claim_scope,
        "device": spec.device, "gif_device": spec.gif_device, "gif_backend": spec.gif_backend, "solver": spec.solver,
        "idt": bool(idt), "wall_seconds": float(wall_seconds),
        "measurement_scope": "validation_with_diagnostic_io" if collect_traces or collect_kkt else "benchmark_wall",
        "children_maxrss_kb": int(max(rusage_before.ru_maxrss, rusage_after.ru_maxrss)),
        "stage_times": stage_times, "gif_diagnostics": parse_gif_diagnostics(combined),
        "uv_metrics": compute_uv_metrics(vertices, faces, uv_array) if uv_array is not None else None,
        "kkt_diagnostics": evaluate_kkt_directory(kkt_dir) if (kkt_dir and kkt_dir.is_dir()) else None,
        "output_obj": str(output_obj) if output_obj.is_file() else None, "uv_array": uv_array,
    }


def evaluate_report_gates(
    report: Mapping[str, Any], *, require_trace: bool = False, require_kkt: bool = False, stop_after_basis: bool = False,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    runs = report.get("runs", {})
    if not runs:
        failures.append("no_runs_executed")
    for mode_name, run_data in runs.items():
        diag = run_data.get("gif_diagnostics") or {}
        expected_device = "gpu" if mode_name == "parity" else "cpu"
        expected_backend = "torch (cuda:0, inprocess)" if mode_name == "parity" else "native"
        if diag.get("gif_compute_device") != expected_device:
            failures.append(f"{mode_name}:unexpected_gif_device")
        if diag.get("gif_evaluator_backend") != expected_backend:
            failures.append(f"{mode_name}:unexpected_gif_backend")
        if not stop_after_basis and not (run_data.get("uv_metrics") or {}).get("locally_injective", False):
            failures.append(f"{mode_name}:uv_not_locally_injective")
        if require_trace and not stop_after_basis:
            diag = run_data.get("gif_diagnostics") or {}
            steps, rem = diag.get("steps", []), diag.get("remeshes", [])
            if not steps or not rem or not _trace_entries_finite(steps, rem):
                failures.append(f"{mode_name}:missing_or_nonfinite_trace")
        if require_kkt and not (run_data.get("kkt_diagnostics") or {}).get("ok", False):
            failures.append(f"{mode_name}:kkt_diagnostics_failed")
    if report.get("reference_vs_parity") is not None and not report["reference_vs_parity"].get("passed", False):
        failures.append("reference_vs_parity:uv_or_energy_mismatch")
    trace_cmp = report.get("trace_comparison")
    if trace_cmp is not None:
        if not trace_cmp.get("non_empty", False) or not trace_cmp.get("finite", False):
            failures.append("trace_comparison:empty_or_nonfinite")
        elif not trace_cmp.get("confounded_by_preprocessing", False) and not trace_cmp.get("within_envelope", False):
            failures.append("trace_comparison:exceeded_strict_envelope")
    if report.get("matched_controller_replay") is not None and not report["matched_controller_replay"].get("ok", False):
        failures.append("matched_controller_replay:failed")
    for mode_name, g_cmp in (report.get("golden_comparison") or {}).items():
        if not g_cmp.get("passed", False):
            failures.append(f"golden_comparison:{mode_name}_failed")
    return len(failures) == 0, failures


def build_comparison_report(
    *, mode: str, input_mesh: Path, runs: Dict[str, Dict[str, Any]],
    golden_comparison: Optional[Dict[str, Any]] = None, reference_vs_parity: Optional[Dict[str, Any]] = None,
    trace_comparison: Optional[Dict[str, Any]] = None, matched_controller_replay: Optional[Dict[str, Any]] = None,
    require_trace: bool = False, require_kkt: bool = False, stop_after_basis: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    has_golden = golden_comparison is not None
    report: Dict[str, Any] = {
        "schema_version": 1, "requested_mode": mode, "input_mesh": str(Path(input_mesh).resolve()),
        "original_matlab_panua_runtime_parity_certified": False, "golden_baseline_supplied": bool(has_golden),
        "parity_claim_statement": (
            "Golden baseline comparison included; runtime speed parity still requires same-hardware licensed execution."
            if has_golden else
            "No licensed MATLAB/Panua golden baseline supplied; this report validates license-free numerical reference and/or Linux/CUDA port placement only, and does NOT certify original MATLAB/Panua runtime parity."
        ),
        "runtime_metadata": collect_runtime_metadata(env),
        "runs": {k: {ik: iv for ik, iv in v.items() if ik != "uv_array"} for k, v in runs.items()},
        "reference_vs_parity": reference_vs_parity, "trace_comparison": trace_comparison,
        "matched_controller_replay": matched_controller_replay, "golden_comparison": golden_comparison,
    }
    report["ok"], report["gate_failures"] = evaluate_report_gates(
        report, require_trace=require_trace, require_kkt=require_kkt, stop_after_basis=stop_after_basis,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "parity", "both"), default="reference")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--golden-uv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--target-faces", type=int, help="Override simplification; input meshes keep the original policy by default (generated fixture: 80)")
    parser.add_argument("--idt", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--octave-timeout-ms", type=int, default=1800000)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--kkt-diagnostics", action="store_true")
    parser.add_argument("--stop-after-basis", action="store_true")
    parser.add_argument("--check-kkt-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.check_kkt_dir is not None:
        report = evaluate_kkt_directory(args.check_kkt_dir)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if report.get("ok", False) else 1

    specs = [resolve_mode_spec(m) for m in (("reference", "parity") if args.mode == "both" else (args.mode,))]

    def _execute(base_work: Path) -> Dict[str, Any]:
        import igl
        if args.input is not None:
            if not args.input.is_file():
                parser.error(f"Input mesh not found: {args.input}")
            mesh_path = args.input.resolve()
            vertices, faces = igl.read_triangle_mesh(str(mesh_path))
        else:
            mesh_path = base_work / "generated_curved_disk.obj"
            vertices, faces = generate_curved_disk_mesh(mesh_path, grid_size=args.grid_size)

        target_faces = args.target_faces
        if args.input is None and target_faces is None:
            target_faces = 80
        runs = {
            spec.mode: run_single_mode_pipeline(
                spec, mesh_path, base_work / spec.mode, idt=args.idt, target_faces=target_faces,
                timeout_seconds=args.timeout, octave_timeout_ms=args.octave_timeout_ms, threads=args.threads,
                collect_traces=args.trace, collect_kkt=args.kkt_diagnostics, stop_after_basis=args.stop_after_basis,
            )
            for spec in specs
        }
        ref_vs_parity, trace_comp, replay_report = None, None, None
        if "reference" in runs and "parity" in runs:
            if runs["reference"].get("uv_array") is not None and runs["parity"].get("uv_array") is not None:
                ref_vs_parity = compare_uv_outputs(vertices, faces, runs["reference"]["uv_array"], runs["parity"]["uv_array"])
            if args.trace and not args.stop_after_basis:
                trace_comp = compare_traces(runs["reference"]["gif_diagnostics"], runs["parity"]["gif_diagnostics"], confounded_inputs=True)
                replay_report = run_matched_controller_replay(candidate_device="cuda:0")
        elif args.trace and not args.stop_after_basis:
            replay_report = run_matched_controller_replay(candidate_device="cuda:0" if "parity" in runs else "cpu")

        golden_comp = None
        if args.golden_uv is not None:
            golden_uv = load_uvs_from_file(args.golden_uv, expected_vertices=vertices.shape[0])
            golden_comp = {m: compare_uv_outputs(vertices, faces, golden_uv, r["uv_array"]) for m, r in runs.items() if r.get("uv_array") is not None}

        return build_comparison_report(
            mode=args.mode, input_mesh=mesh_path, runs=runs, golden_comparison=golden_comp,
            reference_vs_parity=ref_vs_parity, trace_comparison=trace_comp, matched_controller_replay=replay_report,
            require_trace=args.trace, require_kkt=args.kkt_diagnostics, stop_after_basis=args.stop_after_basis,
        )

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        report = _execute(args.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="bcp-compare-") as tmp:
            report = _execute(Path(tmp))

    output_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
