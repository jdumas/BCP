#!/usr/bin/env python3
"""Opt-in, I/O-heavy source-backed validation; never a production benchmark.

Captures a canonical native GIF run, freezes its used basis and original
controller states, and replays the unchanged controller using native .m and
Torch on identical arrays. Historical golden checks are not weakened or
relabelled by this additional diagnostic.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
KERNEL_RTOL = 2e-9  # Same scaled envelope as tests/torch_energy_reference.m.
TRACE_RTOL = 1e-7
TRACE_ATOL = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_fingerprint() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "torch", "meshlib", "libigl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    conda = {}
    for path in (Path(sys.prefix) / "conda-meta").glob("*.json"):
        item = json.loads(path.read_text())
        if item.get("name") in {"mkl", "libblas", "liblapack", "libopenblas", "octave", "libgomp", "python"}:
            conda[item["name"]] = {k: item.get(k) for k in ("version", "build")}
    controls = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "MKL_DYNAMIC", "OMP_DYNAMIC", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE")
    return {"python": sys.version, "prefix": sys.prefix, "packages": packages, "conda": conda,
            "environment": {key: os.environ.get(key) for key in controls}}


def contained_file(folder: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe artifact path: {name}")
    path = folder / relative
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"Artifact escapes capture directory: {name}")
    return path


def encode_report(report: dict[str, Any]) -> str:
    """Keep completed non-finite numerical outcomes as failures, not JSON/IO errors."""
    nonfinite = []
    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(path)
            return None
        if isinstance(value, dict):
            return {key: clean(item, path + "." + key) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, path + f"[{index}]") for index, item in enumerate(value)]
        return value
    safe = clean(report, "report")
    if nonfinite:
        report["nonfinite_numerical_values"] = nonfinite
        report["ok"] = report["primary_checks_passed"] = False
        safe.update(nonfinite_numerical_values=nonfinite, ok=False, primary_checks_passed=False)
    return json.dumps(safe, indent=2, allow_nan=False) + "\n"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Controller instrumentation seam changed: {old!r}")
    return text.replace(old, new)


def instrument_controller(source: str) -> str:
    """Only insert calls; do not rewrite any equation, branch or tolerance."""
    result = "bcp_controlled_capture_state();\n" + source
    for expression, stage in (
        ("    [R,p]=chol(H1);", "chol_primary"),
        ("        [R,p]=chol(H1 + eye(size(H1)));", "chol_shifted"),
        ("        [R,p]=chol(eye(size(H1)));", "chol_identity"),
        ("    iter=iter+1;", "step"),
    ):
        result = replace_once(result, expression, expression + f"\n    bcp_controlled_trace_state('{stage}');")
    for condition, reason in (
        ("    if sum(temp_arr1 .* temp_arr1+temp_arr2 .* temp_arr2 > 0.8*radius_squared)\n        break;", "radius"),
        ("    if norm(G)<gradTol\n        break;", "gradient_tolerance"),
        ("    if (t*norm(d)<stepTol)\n        break;", "step_tolerance"),
        ("        if (EitersCounter >= numConvIters)\n            break;", "energy_counter"),
    ):
        indent = "            " if reason == "energy_counter" else "        "
        replacement = condition.replace(indent + "break;", indent + f"bcp_controlled_trace_state('stop','{reason}');\n" + indent + "break;")
        result = replace_once(result, condition, replacement)
    return result


def create_overlay(folder: Path, native_module: Path) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=False)
    original = ROOT / "GIF/GIF/MatlabScripts/iterationOfNewtonWithScaffold.m"
    generated = folder / original.name
    generated.write_text(instrument_controller(original.read_text()), encoding="utf-8")
    for name in ("bcp_controlled_capture_state.m", "bcp_controlled_trace_state.m", "bcp_controlled_kernel_state.m"):
        shutil.copyfile(ROOT / "tests" / name, folder / name)
    if not native_module.is_file():
        raise FileNotFoundError(f"In-process bridge required: {native_module}")
    (folder / native_module.name).symlink_to(native_module.resolve())
    return {"original_controller": sha256(original), "instrumented_controller": sha256(generated)}


@contextmanager
def environment(values: Mapping[str, str]):
    old = os.environ.copy()
    try:
        os.environ.update({k: str(v) for k, v in values.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def checked_native(executable: Path, arguments: list[Any], env: Mapping[str, str], log_dir: Path, timeout: float, resume: bool = False):
    marker = log_dir / "run-complete.json"
    configuration = {"executable_sha256": sha256(executable), "arguments": list(map(str, arguments)),
                     "environment": dict(env), "runtime": runtime_fingerprint()}
    if resume and marker.is_file():
        completed = json.loads(marker.read_text())
        if completed["configuration"] != configuration:
            raise ValueError(f"Completed stage configuration changed: {log_dir}")
        inventory = {str(p.relative_to(log_dir)) for p in log_dir.rglob("*") if p.is_file() and p != marker}
        if inventory != set(completed["files"]):
            raise ValueError(f"Completed stage file inventory changed: {log_dir}")
        for name, digest in completed["files"].items():
            if sha256(contained_file(log_dir, name)) != digest:
                raise ValueError(f"Completed stage integrity failure: {log_dir / name}")
        return subprocess.CompletedProcess([str(executable), *map(str, arguments)], 0,
            (log_dir / "stdout.log").read_text(), (log_dir / "stderr.log").read_text())
    if resume and log_dir.exists() and any(log_dir.iterdir()):
        raise ValueError(f"Incomplete stage retained at {log_dir}; use a fresh work directory rather than overwriting evidence")
    sys.path.insert(0, str(ROOT / "BCP/utils"))
    from process_utils import run_native
    log_dir.mkdir(parents=True, exist_ok=True)
    with environment(env), redirect_stdout(io.StringIO()):
        try:
            result = run_native(executable, arguments, timeout)
        except BaseException as error:
            (log_dir / "execution-error.log").write_text(str(error), encoding="utf-8")
            raise
    (log_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")
    completed = {"configuration": configuration, "files": {str(p.relative_to(log_dir)): sha256(p)
                 for p in sorted(log_dir.rglob("*")) if p.is_file() and p != marker}}
    marker.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    return result


def freeze_inputs(source: Path, destination: Path) -> dict[str, Any]:
    """Copy explicit immutable inputs; never create hard links to operator data."""
    destination.mkdir(parents=True, exist_ok=False)
    required = ("frozen-source.obj", "frozen-edge-lengths.mat")
    optional = ("coarse.npz", "coarse-uv.mat", "capture.json")
    files = {}
    for name in required + optional:
        original = source / name
        if not original.is_file():
            if name in required:
                raise FileNotFoundError(original)
            continue
        target = destination / name
        shutil.copyfile(original, target)
        files[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    lengths = sio.loadmat(destination / required[1])["edge_lengths"]
    if lengths.ndim != 2 or lengths.shape[1] != 3 or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Invalid frozen edge lengths")
    return {"files": files, "face_count": int(len(lengths))}


def capture_pipeline_inputs(mesh: Path, destination: Path, device: str, target_faces: int | None):
    """Run unchanged preprocessing once; intercept only the final GIF launch."""
    import igl
    sys.path[:0] = [str(ROOT / p) for p in ("BCP/GIF_and_CM", "BCP/utils", "BCP")]
    os.environ["BCP_DEVICE"] = device
    import MeshProcessor
    import run_GIF_CM as pipeline
    import run_GIF_utils as gif
    MeshProcessor.TARGET_SIMPLIFICATION_FACES_COUNT = target_faces
    destination.mkdir(parents=True, exist_ok=False)
    old_simplification, old_cm, old_gif = pipeline.run_simplification, pipeline.run_CM, gif.run_GIF
    capture = {}

    class Captured(BaseException):
        pass

    def simplify(*args, **kwargs):
        result = old_simplification(*args, **kwargs)
        np.savez(destination / "coarse.npz", V=result[0], F=result[1], closest_faces=result[2], barycentric=result[3])
        return result

    def cm(*args, **kwargs):
        result = old_cm(*args, **kwargs)
        sio.savemat(destination / "coarse-uv.mat", {"uvs": result[0]})
        capture.update(cm_iterations=result[2], coarse_energy=result[3])
        return result

    def intercept(*args, **kwargs):
        vertices, faces = args[:2]
        igl.write_triangle_mesh(str(destination / "frozen-source.obj"), vertices, faces)
        sio.savemat(destination / "frozen-edge-lengths.mat", {"edge_lengths": np.asarray(args[-2])})
        capture.update(gif_controls=list(args[4:9]), face_count=len(faces))
        raise Captured

    pipeline.run_simplification, pipeline.run_CM, gif.run_GIF = simplify, cm, intercept
    try:
        vertices, faces = igl.read_triangle_mesh(str(mesh))
        try:
            pipeline.run_GIF_and_CM_with_simplification(vertices, faces)
        except Captured:
            pass
        else:
            raise RuntimeError("Pipeline did not reach the expected GIF boundary")
    finally:
        pipeline.run_simplification, pipeline.run_CM, gif.run_GIF = old_simplification, old_cm, old_gif
    (destination / "capture.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")


def array_difference(reference: Any, candidate: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        return {"passed": False, "exact": False, "reason": "shape_mismatch", "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number):
        equal = bool(np.array_equal(a, b))
        return {"passed": equal, "exact": equal}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    maximum = float(np.max(np.abs(b - a), initial=0.0)) if finite else None
    scale = max(1.0, float(np.max(np.abs(a), initial=0.0))) if finite else None
    return {"passed": bool(finite and maximum <= atol + rtol * scale), "exact": bool(finite and np.array_equal(a, b)),
            "finite": finite, "max_abs_difference": maximum, "scaled_max_difference": maximum / scale if finite else None,
            "rtol": rtol, "atol": atol}


def load_struct(path: Path, key: str) -> dict[str, Any]:
    data = sio.loadmat(path, simplify_cells=True)
    value = data.get(key)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Invalid {key} schema in {path}")
    return value


def compare_controller_runs(reference: Path, candidate: Path) -> dict[str, Any]:
    a_files = sorted(reference.glob("remesh_*_step_*.mat"))
    b_files = sorted(candidate.glob("remesh_*_step_*.mat"))
    if not a_files or not b_files:
        return {"passed": False, "exact": False, "reason": "missing_step_records"}
    report: dict[str, Any] = {"passed": len(a_files) == len(b_files), "exact": len(a_files) == len(b_files),
                              "reference_steps": len(a_files), "candidate_steps": len(b_files), "steps": [],
                              "first_arithmetic_difference": None, "first_envelope_failure": None, "first_branch_difference": None}
    fields = ("x_before", "fz", "fbz", "E", "G1", "H1", "d1", "tMax", "t", "x_after", "primary_chol_p", "fallback")
    kernel = {"fz", "fbz", "E", "G1", "H1"}
    branches = {"primary_chol_p", "fallback"}
    for a_path, b_path in zip(a_files, b_files):
        a, b = load_struct(a_path, "state"), load_struct(b_path, "state")
        entry = {"reference_file": a_path.name, "candidate_file": b_path.name, "fields": {}}
        for field in fields:
            if field not in a or field not in b:
                metric = {"passed": False, "exact": False, "reason": "missing_field"}
            else:
                metric = array_difference(a[field], b[field], rtol=KERNEL_RTOL if field in kernel else (0 if field in branches else TRACE_RTOL),
                                          atol=0 if field in kernel or field in branches else TRACE_ATOL)
            entry["fields"][field] = metric
            location = {"step": a_path.name, "field": field, **metric}
            if not metric["exact"] and report["first_arithmetic_difference"] is None:
                report["first_arithmetic_difference"] = location
            if not metric["passed"] and report["first_envelope_failure"] is None:
                report["first_envelope_failure"] = location
            if field in branches and not metric["exact"] and report["first_branch_difference"] is None:
                report["first_branch_difference"] = location
            report["passed"] = report["passed"] and metric["passed"]
            report["exact"] = report["exact"] and metric["exact"]
        report["steps"].append(entry)
    if len(a_files) != len(b_files) and report["first_branch_difference"] is None:
        report["first_branch_difference"] = {"reason": "different_iteration_counts", "reference": len(a_files), "candidate": len(b_files)}
    terminals = []
    for folder in (reference, candidate):
        prefix = (a_files if folder == reference else b_files)[-1].stem.rsplit("_step_", 1)[0]
        stop = folder / (prefix + "_stop.mat")
        terminals.append(load_struct(stop, "terminal") if stop.is_file() else {"reason": "iteration_cap"})
    report["stop_reasons"] = [v["reason"] for v in terminals]
    if terminals[0]["reason"] != terminals[1]["reason"]:
        report["passed"] = report["exact"] = False
        if report["first_branch_difference"] is None:
            report["first_branch_difference"] = {"reason": "different_stop_reasons", "values": report["stop_reasons"]}
    return report


def read_native_manifest(folder: Path) -> dict[str, Any]:
    value = json.loads((folder / "manifest.json").read_text())
    if value.get("schema") != "bcp-controlled-validation/1" or value.get("diagnostic_only") is not True:
        raise ValueError("Missing controlled-native capability/provenance marker")
    names = set()
    nonfinite_diagnostics = {}
    for artifact in value.get("artifacts", []):
        name, file = artifact["name"], artifact["file"]
        if name in names or Path(file).name != file or Path(name).name != name:
            raise ValueError("Invalid/duplicate native artifact name")
        names.add(name)
        array = np.load(contained_file(folder, file), allow_pickle=False, mmap_mode="r")
        if array.dtype.str != artifact["dtype"] or list(array.shape) != artifact["shape"]:
            raise ValueError(f"Native artifact dtype/shape mismatch: {name}")
        if not np.isfinite(array).all():
            if name == "outlier_ranking_values":
                nonfinite_diagnostics[name] = int(np.count_nonzero(~np.isfinite(array)))
            else:
                raise ValueError(f"Non-finite native replay/stage artifact: {name}")
    if not names:
        raise ValueError("Controlled manifest has no captured artifacts")
    value["nonfinite_diagnostic_values"] = nonfinite_diagnostics
    return value


def compare_native_runs(reference: Path, candidate: Path) -> dict[str, Any]:
    a, b = read_native_manifest(reference), read_native_manifest(candidate)
    right = {entry["name"]: entry for entry in b["artifacts"]}
    result: dict[str, Any] = {"exact": a["meta"] == b["meta"], "meta_equal": a["meta"] == b["meta"],
                              "artifacts": {}, "first_difference": None}
    for artifact in a["artifacts"]:
        name = artifact["name"]
        if name not in right:
            metric = {"passed": False, "exact": False, "reason": "missing_artifact"}
        else:
            left_data = np.load(reference / artifact["file"], allow_pickle=False, mmap_mode="r")
            right_data = np.load(candidate / right[name]["file"], allow_pickle=False, mmap_mode="r")
            metric = array_difference(left_data, right_data, rtol=0, atol=0)
        result["artifacts"][name] = metric
        result["exact"] = result["exact"] and metric["exact"]
        if not metric["exact"] and result["first_difference"] is None:
            result["first_difference"] = {"artifact": name, "stage": artifact["stage"], **metric}
    extra = sorted(set(right) - {entry["name"] for entry in a["artifacts"]})
    if extra:
        result["exact"] = False
        result["additional_candidate_artifacts"] = extra
    return result


def replay_controller(snapshot: Path, output: Path, mode: str, device: str, octave: Path, overlay: Path, timeout: float, resume: bool = False):
    output.mkdir(parents=True, exist_ok=resume)
    env = {"BCP_CONTROLLED_ROOT": str(ROOT), "BCP_CONTROLLED_SNAPSHOT": str(snapshot),
           "BCP_CONTROLLED_RESULT": str(output / "result.mat"), "BCP_CONTROLLED_OVERLAY": str(overlay),
           "BCP_CONTROLLED_EVALUATOR": mode, "BCP_CONTROLLED_DEVICE": device,
           "BCP_CONTROLLED_TRACE_DIR": str(output), "BCP_CONTROLLED_SNAPSHOT_DIR": "",
           "BCP_COMPARE_TRACE_PATH": "", "BCP_TORCH_NATIVE_DIR": str(overlay),
           "BCP_TORCH_MODULE_DIR": str(ROOT / "BCP/GIF_and_CM")}
    checked_native(octave, ["--no-gui", "--quiet", ROOT / "tests/controlled_controller_replay.m"], env, output, timeout, resume)
    return load_struct(output / "result.mat", "result")


def verify_inputs(folder: Path, manifest: Mapping[str, Any]):
    for name, metadata in manifest["files"].items():
        if Path(name).name != name or sha256(contained_file(folder, name)) != metadata["sha256"]:
            raise ValueError(f"Frozen input integrity failure: {name}")


def compare_snapshots(reference: Path, candidate: Path) -> dict[str, Any]:
    a, b = load_struct(reference, "snapshot"), load_struct(candidate, "snapshot")
    fields = {}
    for group in ("gif", "vars"):
        for name in sorted(set(a[group]) | set(b[group])):
            if name not in a[group] or name not in b[group]:
                fields[group + "." + name] = {"passed": False, "exact": False, "reason": "missing_field"}
            else:
                fields[group + "." + name] = array_difference(a[group][name], b[group][name], rtol=0, atol=0)
    return {"exact": all(v["exact"] for v in fields.values()), "fields": fields}


def validate_snapshot(snapshot: Path):
    value = load_struct(snapshot, "snapshot")
    gif, variables = value.get("gif", {}), value.get("vars", {})
    required = {"TJ_fz", "TJ_fbz", "TArea", "FixedIndices", "FixedValues", "ScaffoldUVonCones"}
    if not required.issubset(gif):
        raise ValueError("Snapshot lacks required GIF arrays")
    required_vars = {"x_initial", "numDOF", "numDOF1", "numDOFScaffold", "totVars", "almostAllVars", "iter", "timesRemeshed"}
    if not required_vars.issubset(variables):
        raise ValueError("Snapshot lacks required controller variables")
    # Original Octave/MATLAB size/iteration/index expressions use doubles. Do not
    # silently accept integer captures that change the native bridge index ABI.
    for key in required_vars - {"x_initial"}:
        field = np.asarray(variables[key])
        if field.size != 1 or field.dtype != np.dtype("float64") or not np.isfinite(field).all():
            raise ValueError(f"Captured {key} must be a finite double scalar")
    j, b = np.asarray(gif["TJ_fz"]), np.asarray(gif["TJ_fbz"])
    if j.ndim != 2 or j.shape != b.shape or not np.isfinite(j).all() or not np.isfinite(b).all():
        raise ValueError("Invalid captured Jacobians")
    if j.dtype not in (np.dtype("float64"), np.dtype("complex128")) or b.dtype not in (np.dtype("float64"), np.dtype("complex128")):
        raise ValueError("Captured Jacobians must retain double precision")
    if np.size(gif["TArea"]) != j.shape[0] or not np.isfinite(gif["TArea"]).all():
        raise ValueError("Invalid captured areas")
    for key in ("TArea", "FixedIndices", "FixedValues", "ScaffoldUVonCones"):
        if np.asarray(gif[key]).dtype != np.dtype("float64") or not np.isfinite(gif[key]).all():
            raise ValueError(f"Captured GIF.{key} must be finite double data")
    x = np.asarray(variables["x_initial"])
    if x.dtype != np.dtype("float64") or not np.isfinite(x).all():
        raise ValueError("Captured coordinates must retain finite double precision")
    for name in ("numDOF", "numDOF1", "numDOFScaffold", "totVars", "almostAllVars", "iter", "timesRemeshed"):
        value = float(variables[name])
        if value < 0 or value != math.floor(value):
            raise ValueError(f"Captured {name} must be a nonnegative integer-valued double")
    if int(variables["numDOFScaffold"]) != j.shape[1] or int(variables["totVars"]) != np.size(variables["x_initial"]):
        raise ValueError("Captured controller dimensions disagree")
    if int(variables["numDOF1"]) + int(variables["numDOF"]) != j.shape[1] or float(variables["numDOF"]) < 1:
        raise ValueError("Captured scaffold split disagrees")
    fixed = np.asarray(gif["FixedIndices"]).reshape(-1)
    if np.any(fixed != np.floor(fixed)) or np.any(fixed < 1) or np.any(fixed > 2 * j.shape[1]) or len(np.unique(fixed)) != len(fixed):
        raise ValueError("Invalid captured fixed-coordinate indices")
    if np.size(gif["FixedValues"]) != len(fixed) or 2 * j.shape[1] - len(fixed) != np.size(x):
        raise ValueError("Captured fixed/free coordinate dimensions disagree")
    if int(variables["almostAllVars"]) != np.size(x) - int(variables["numDOF"]):
        raise ValueError("Captured reduced-coordinate split disagrees")
    if np.size(gif["ScaffoldUVonCones"]) != np.size(x):
        raise ValueError("Captured scaffold coordinates disagree")


def run(args) -> dict[str, Any]:
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()) and not args.resume:
        raise ValueError("Use a new empty --work-dir or --resume; existing captures will not be overwritten")
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": 1, "measurement_scope": "controlled_validation_with_diagnostic_io",
                              "reference_kind": "license_free_original_m_equations", "original_runtime_parity_certified": False,
                              "historical_golden_gates_modified": False, "production_defaults_modified": False,
                              "order": args.order, "device": args.device, "kernel_scaled_tolerance": KERNEL_RTOL,
                              "trace_rtol": TRACE_RTOL, "trace_atol": TRACE_ATOL, "runs": {}, "ok": False}
    try:
        native = args.native_bin.resolve()
        if not native.is_file() or native.name != "GIF_controlled":
            raise ValueError("An explicitly built GIF_controlled validation executable is required")
        octave_name = args.octave or os.environ.get("BCP_OCTAVE_EXECUTABLE") or shutil.which("octave-cli") or shutil.which("octave")
        if not octave_name:
            raise ValueError("Octave is required; no runtime fallback")
        octave = Path(octave_name).resolve()
        module = native.parent.parent / "bcp_torch_energy_native.oct"
        session_path = work / "session.json"
        requested_input = {"kind": "frozen" if args.frozen_input_dir else "mesh",
                           "path": str((args.frozen_input_dir or args.input).resolve()), "target_faces": args.target_faces}
        report["runtime"] = runtime_fingerprint()
        matlab_sources = {p.name: sha256(p) for p in sorted((ROOT / "GIF/GIF/MatlabScripts").glob("*.m"))}
        gate_sources = {name: sha256(ROOT / name) for name in ("scripts/compare.py", "BCP/GIF_and_CM/compute_stats.py",
                       "BCP/utils/func_utils.py", "BCP/utils/process_utils.py")}
        expected_overlay = {"iterationOfNewtonWithScaffold.m", "bcp_torch_energy_native.oct",
                            "bcp_controlled_capture_state.m", "bcp_controlled_trace_state.m", "bcp_controlled_kernel_state.m"}
        if args.resume:
            if not session_path.is_file():
                raise ValueError("Resume requires a valid session.json provenance file")
            session = json.loads(session_path.read_text())
            if session.get("requested_input") != requested_input:
                raise ValueError("Resume input selection differs from the captured session")
            if session["order"] != args.order or session["device"] != args.device:
                raise ValueError("Resume order/device differ from the captured session")
            if session["runtime"] != report["runtime"]:
                raise ValueError("Runtime packages/thread settings changed since capture")
            hashes = session["source_hashes"]
            if {p.name for p in (work / "overlay").iterdir()} != expected_overlay:
                raise ValueError("Validation overlay file inventory changed")
            original = ROOT / "GIF/GIF/MatlabScripts/iterationOfNewtonWithScaffold.m"
            actual = {"original_controller": sha256(original), "instrumented_controller": sha256(work / "overlay" / original.name),
                      "torch_energy": sha256(ROOT / "BCP/GIF_and_CM/torch_energy.py"), "native_executable": sha256(native),
                      "native_bridge": sha256(module), "replay_script": sha256(ROOT / "tests/controlled_controller_replay.m"),
                      "controlled_driver": sha256(Path(__file__)), "kernel_probe_driver": sha256(ROOT / "scripts/controlled_kernel.py"),
                      "matlab_sources": matlab_sources, "gate_sources": gate_sources}
            for name in ("bcp_controlled_capture_state.m", "bcp_controlled_trace_state.m", "bcp_controlled_kernel_state.m"):
                actual[name] = sha256(work / "overlay" / name)
                if actual[name] != sha256(ROOT / "tests" / name):
                    raise ValueError("Validation helper changed since capture")
            if actual != hashes:
                raise ValueError("Source/binary provenance changed since capture")
            report["source_hashes"] = hashes
            inputs = session["inputs"]
        else:
            report["source_hashes"] = create_overlay(work / "overlay", module)
            report["source_hashes"]["torch_energy"] = sha256(ROOT / "BCP/GIF_and_CM/torch_energy.py")
            report["source_hashes"]["native_executable"] = sha256(native)
            report["source_hashes"]["native_bridge"] = sha256(module)
            report["source_hashes"]["replay_script"] = sha256(ROOT / "tests/controlled_controller_replay.m")
            report["source_hashes"]["controlled_driver"] = sha256(Path(__file__))
            report["source_hashes"]["kernel_probe_driver"] = sha256(ROOT / "scripts/controlled_kernel.py")
            report["source_hashes"]["matlab_sources"] = matlab_sources
            report["source_hashes"]["gate_sources"] = gate_sources
            for name in ("bcp_controlled_capture_state.m", "bcp_controlled_trace_state.m", "bcp_controlled_kernel_state.m"):
                report["source_hashes"][name] = sha256(ROOT / "tests" / name)
            if args.frozen_input_dir:
                inputs = freeze_inputs(args.frozen_input_dir.resolve(), work / "inputs")
            else:
                capture_pipeline_inputs(args.input.resolve(), work / "inputs", args.device, args.target_faces)
                inputs = {"files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (work / "inputs").iterdir() if p.is_file()}}
            session = {"schema_version": 1, "order": args.order, "device": args.device, "source_hashes": report["source_hashes"], "inputs": inputs, "runtime": report["runtime"], "requested_input": requested_input}
            session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        report["inputs"] = inputs
        verify_inputs(work / "inputs", inputs)
        common = {"BCP_CONTROLLED_ORDER": args.order, "BCP_GIF_BACKEND": "native", "BCP_GIF_DEVICE": "cpu",
                  "BCP_TORCH_NATIVE_DIR": str(work / "overlay"), "BCP_TORCH_MODULE_DIR": str(ROOT / "BCP/GIF_and_CM"),
                  "BCP_COMPARE_TRACE_PATH": "", "BCP_COMPARE_KKT_DIR": "", "BCP_COMPARE_STOP_AFTER_BASIS": ""}
        capture = work / "native_capture"
        env = {**common, "BCP_CONTROLLED_DIR": str(capture), "BCP_CONTROLLED_REPLAY_STAGE": "none",
               "BCP_CONTROLLED_REPLAY_DIR": "", "BCP_CONTROLLED_SNAPSHOT_DIR": str(capture / "controller_states"),
               "BCP_CONTROLLED_TRACE_DIR": str(capture / "controller_steps")}
        command = [work / "inputs/frozen-source.obj", capture / "output.mat", "False", "True", "True", -1, -1, .001, .1, 0.0,
                   work / "inputs/frozen-edge-lengths.mat"]
        checked_native(native, command, env, capture, args.timeout, args.resume)
        # The native target writes capability/provenance metadata; never trust a production executable's ignored flags.
        report["native_capture_manifest"] = read_native_manifest(capture)
        if args.fresh_native_repeat:
            fresh = work / "native_fresh_repeat"
            fresh_env = {**env, "BCP_CONTROLLED_DIR": str(fresh),
                         "BCP_CONTROLLED_SNAPSHOT_DIR": str(fresh / "controller_states"),
                         "BCP_CONTROLLED_TRACE_DIR": str(fresh / "controller_steps")}
            fresh_command = list(command); fresh_command[1] = fresh / "output.mat"
            checked_native(native, fresh_command, fresh_env, fresh, args.timeout, args.resume)
            report["native_fresh_repeat"] = compare_native_runs(capture, fresh)
            report["native_fresh_repeat_exact"] = report["native_fresh_repeat"]["exact"]
        else:
            report["native_fresh_repeat_exact"] = None
        report["native_replays"] = {}
        for stage in args.native_replay_stages:
            repeated = work / ("native_replay_" + stage)
            replay_env = {**common, "BCP_CONTROLLED_DIR": str(repeated), "BCP_CONTROLLED_REPLAY_STAGE": stage,
                          "BCP_CONTROLLED_REPLAY_DIR": str(capture),
                          "BCP_CONTROLLED_SNAPSHOT_DIR": "", "BCP_CONTROLLED_TRACE_DIR": ""}
            replay_command = list(command)
            replay_command[1] = repeated / "output.mat"
            checked_native(native, replay_command, replay_env, repeated, args.timeout, args.resume)
            report["native_replays"][stage] = compare_native_runs(capture, repeated)
        candidate_capture = work / "torch_capture"
        candidate_env = {**common, "BCP_CONTROLLED_DIR": str(candidate_capture), "BCP_CONTROLLED_REPLAY_STAGE": "basis",
                         "BCP_CONTROLLED_REPLAY_DIR": str(capture), "BCP_GIF_BACKEND": "torch",
                         "BCP_GIF_DEVICE": "gpu" if args.device.startswith("cuda") else "cpu",
                         "BCP_CONTROLLED_SNAPSHOT_DIR": str(candidate_capture / "controller_states"),
                         "BCP_CONTROLLED_TRACE_DIR": str(candidate_capture / "controller_steps")}
        candidate_command = list(command)
        candidate_command[1] = candidate_capture / "output.mat"
        actual = checked_native(native, candidate_command, candidate_env, candidate_capture, args.timeout, args.resume)
        expected_backend = f"torch ({args.device}, inprocess)"
        if f"GIF evaluator backend: {expected_backend}" not in ((actual.stdout or "") + "\n" + (actual.stderr or "")):
            raise ValueError("Candidate evaluator did not report the requested in-process device")
        report["pipeline_native_vs_torch"] = compare_native_runs(capture, candidate_capture)
        import igl
        from compare import compare_uv_outputs
        vertices, faces = igl.read_triangle_mesh(str(work / "inputs/frozen-source.obj"))
        report["pipeline_uv_comparison"] = compare_uv_outputs(vertices, faces,
            np.load(capture / "uv_final.npy", allow_pickle=False), np.load(candidate_capture / "uv_final.npy", allow_pickle=False))
        snapshots = sorted((capture / "controller_states").glob("remesh_*.mat"))
        if not snapshots:
            raise ValueError("No controller snapshots captured")
        report["captured_remeshes"] = len(snapshots)
        if args.max_remeshes is not None:
            snapshots = snapshots[:args.max_remeshes]
        report["tested_remeshes"] = len(snapshots)
        report["all_captured_remeshes_tested"] = len(snapshots) == report["captured_remeshes"]
        checks = {}
        for snapshot in snapshots:
            validate_snapshot(snapshot)
            candidate_snapshot = candidate_capture / "controller_states" / snapshot.name
            input_comparison = compare_snapshots(snapshot, candidate_snapshot) if candidate_snapshot.is_file() else {"exact": False, "reason": "candidate_remesh_missing"}
            folder = work / "replays" / snapshot.stem
            native_result = replay_controller(snapshot, folder / "native", "native", args.device, octave, work / "overlay", args.timeout, args.resume)
            repeat_result = replay_controller(snapshot, folder / "native-repeat", "native", args.device, octave, work / "overlay", args.timeout, args.resume)
            torch_result = replay_controller(snapshot, folder / "torch", "torch", args.device, octave, work / "overlay", args.timeout, args.resume)
            repeat = compare_controller_runs(folder / "native", folder / "native-repeat")
            candidate = compare_controller_runs(folder / "native", folder / "torch")
            checks[snapshot.stem] = {"snapshot_sha256": sha256(snapshot), "pipeline_snapshot_inputs": input_comparison, "native_repeat": repeat, "native_vs_torch": candidate,
                "final_energy": {"native": float(native_result["energy"]), "native_repeat": float(repeat_result["energy"]), "torch": float(torch_result["energy"])}}
            report["runs"] = checks
            report["completed_controller_remeshes"] = len(checks)
            report["native_repeat_exact"] = all(v["native_repeat"]["exact"] for v in checks.values())
            report["controller_envelopes_passed"] = all(v["native_vs_torch"]["passed"] for v in checks.values())
            report["native_replays_exact"] = all(v["exact"] for v in report["native_replays"].values())
            report["primary_checks_passed"] = bool(len(checks) == len(snapshots) and report["native_repeat_exact"] and
                (not args.fresh_native_repeat or report["native_fresh_repeat_exact"]) and
                report["native_replays_exact"] and report["controller_envelopes_passed"] and report["pipeline_uv_comparison"]["passed"] and
                not report["native_capture_manifest"]["nonfinite_diagnostic_values"])
            if args.kernel_isolation:
                try:
                    from controlled_kernel import isolate
                    checks[snapshot.stem]["kernel_isolation"] = isolate(snapshot, work / "kernel_diagnostics" / snapshot.stem,
                        args.device, octave, module, args.timeout, args.resume)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    checks[snapshot.stem]["kernel_isolation"] = {"diagnostic_only": True, "error": str(error)}
                    report["kernel_isolation_failed"] = True
            report["runs"] = checks
            (work / "report.json").write_text(encode_report(report), encoding="utf-8")
        report["runs"] = checks
        report["native_repeat_exact"] = all(v["native_repeat"]["exact"] for v in checks.values())
        report["controller_envelopes_passed"] = all(v["native_vs_torch"]["passed"] for v in checks.values())
        report["native_replays_exact"] = all(v["exact"] for v in report["native_replays"].values())
        report["requested_diagnostics_succeeded"] = not report.get("kernel_isolation_failed", False)
        report["ok"] = bool(report["primary_checks_passed"] and report["requested_diagnostics_succeeded"])
        # Whole-pipeline exactness is descriptive, not substituted for the unchanged
        # historical UV/energy gates or the controlled controller envelopes.
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report["execution_error"] = str(error)
    output = work / "report.json"
    if args.resume and "execution_error" in report and output.exists():
        import uuid
        report["preserved_prior_report"] = str(output)
        output = work / ("resume-error-" + uuid.uuid4().hex + ".json")
    report["report_file"] = str(output)
    output.write_text(encode_report(report), encoding="utf-8")
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Capture unchanged BCP preprocessing once")
    inputs.add_argument("--frozen-input-dir", type=Path, help="Directory with frozen-source.obj and frozen-edge-lengths.mat")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, default=Path(os.environ.get("BCP_BIN_DIR", ROOT / "build/pixi-mkl-octave/bin")) / "GIF_controlled")
    parser.add_argument("--octave")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--order", choices=("index", "original"), default="index", help="Validation executable only; production defaults untouched")
    parser.add_argument("--target-faces", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-remeshes", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse only integrity-checked completed stages; never overwrite incomplete evidence")
    parser.add_argument("--kernel-isolation", action=argparse.BooleanOptionalAction, default=True,
                        help="Diagnose first evaluations with frozen primitives; diagnostic execution failures return 2 while preserving independent primary gates")
    parser.add_argument("--fresh-native-repeat", action=argparse.BooleanOptionalAction, default=True,
                        help="Check a separate canonical native execution before frozen-state replay")
    parser.add_argument("--native-replay-stages", nargs="*", choices=("basis", "pre-repair", "post-mvc"),
                        default=["basis", "pre-repair", "post-mvc"])
    return parser



def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if args.target_faces is not None and args.target_faces < 1:
        parser.error("--target-faces must be positive")
    if args.max_remeshes is not None and args.max_remeshes < 1:
        parser.error("--max-remeshes must be positive")
    report = run(args)
    summary = {k: report.get(k) for k in ("ok", "execution_error", "measurement_scope", "order", "device", "captured_remeshes", "tested_remeshes", "all_captured_remeshes_tested", "native_repeat_exact", "native_fresh_repeat_exact", "native_replays_exact", "controller_envelopes_passed", "primary_checks_passed", "requested_diagnostics_succeeded")}
    summary["report"] = report["report_file"]
    summary["first_divergences"] = {key: {"arithmetic": value["native_vs_torch"]["first_arithmetic_difference"],
        "envelope": value["native_vs_torch"]["first_envelope_failure"], "branch": value["native_vs_torch"]["first_branch_difference"]}
        for key, value in report["runs"].items()}
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 2 if "execution_error" in report or report.get("kernel_isolation_failed") else (0 if report["ok"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
