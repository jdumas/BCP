#!/usr/bin/env python3
"""Run CM or the complete BCP pipeline without changing source paths."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "BCP", ROOT / "BCP/utils", ROOT / "BCP/GIF_and_CM"):
    sys.path.insert(0, str(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Consistently oriented triangle disk (OBJ)")
    parser.add_argument("output", type=Path, help="Output OBJ with per-vertex texture coordinates")
    parser.add_argument("--method", choices=("cm", "bcp"), default="cm")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), help="PyTorch tensor device")
    parser.add_argument("--gif-device", choices=("auto", "cpu", "gpu"), help="GIF evaluator device")
    parser.add_argument("--gif-backend", choices=("auto", "native", "torch"), help="GIF energy evaluator backend")
    parser.add_argument("--target-faces", type=int)
    parser.add_argument("--idt", action="store_true", help="Use the intrinsic-Delaunay GIF fix")
    args = parser.parse_args()
    if args.device:
        os.environ["BCP_DEVICE"] = args.device
    if args.gif_device:
        os.environ["BCP_GIF_DEVICE"] = args.gif_device
    if args.gif_backend:
        os.environ["BCP_GIF_BACKEND"] = args.gif_backend
    os.environ.setdefault("BCP_PYTHON_EXECUTABLE", sys.executable)
    from GIF_and_CM import consts
    required = [consts.ROOT_PATH_CM_EXE]
    if args.method == "bcp":
        required += [consts.ROOT_PATH_GIF_EXE, consts.ROOT_PATH_PARALLEL_TRANSPORT_EXE]
    for binary in required:
        if not Path(binary).is_file():
            parser.error(f"Missing native executable: {binary}. See README.md for build instructions.")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.target_faces is not None and args.target_faces < 1:
        parser.error("--target-faces must be positive")

    import igl
    from func_utils import write_obj_with_UVs
    vertices, faces = igl.read_triangle_mesh(str(args.input.resolve()))
    if args.method == "cm":
        from run_CM_utils import run_CM
        uvs, seconds, iterations, energy = run_CM(
            vertices, faces, consts.ROOT_PATH_CM_EXE, consts.ROOT_DIR_CACHE_CM)
        print(f"CM: {seconds:.6g}s, {iterations:g} iterations, E_SD={energy:.6g}")
    else:
        import MeshProcessor
        MeshProcessor.TARGET_SIMPLIFICATION_FACES_COUNT = args.target_faces
        from run_GIF_CM import (
            run_GIF_and_CM_with_simplification,
            run_GIF_and_CM_with_simplification_with_IDT,
        )
        run = run_GIF_and_CM_with_simplification_with_IDT if args.idt else run_GIF_and_CM_with_simplification
        from GIF_and_CM.device import DEVICE
        print(f"PyTorch compute device: {DEVICE}", flush=True)
        result = run(vertices, faces)
        uvs = result[4]
        print(f"BCP stage times: {result[:4]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_obj_with_UVs(str(args.output), vertices, faces, uvs)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
