#!/usr/bin/env python3
"""Cross-platform Pixi task driver: one interpreter/environment, no shell quoting."""
import argparse
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, env=None):
    print("+", subprocess.list2cmdline(list(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def build_source_octave(env):
    command = [sys.executable, ROOT / "scripts" / "build_octave.py"]
    print("+", subprocess.list2cmdline(list(map(str, command))), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True,
                            stdout=subprocess.PIPE, text=True)
    executable = Path(result.stdout.strip())
    if not executable.is_absolute() or not executable.is_file():
        raise RuntimeError("The source builder did not return an installed octave-cli; refusing a prebuilt fallback")
    return str(executable)


def runtime_environment(build, engine, octave=None):
    env = os.environ.copy()
    env["BCP_BIN_DIR"] = str(build / "bin")
    env["BCP_PYTHON_EXECUTABLE"] = sys.executable
    if octave:
        env["BCP_OCTAVE_EXECUTABLE"] = octave
    # MATLAB DLLs must also be on PATH for a directly linked Windows executable.
    matlab_root = os.environ.get("Matlab_ROOT_DIR")
    cache = build / "CMakeCache.txt"
    if engine == "matlab" and not matlab_root and cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("Matlab_ROOT_DIR:"):
                matlab_root = line.split("=", 1)[1]
                break
    if engine == "matlab" and matlab_root:
        if sys.platform == "darwin":
            arch = "maca64" if platform.machine() == "arm64" else "maci64"
        else:
            arch = "win64" if os.name == "nt" else "glnxa64"
        runtime = Path(matlab_root) / "bin" / arch
        env["PATH"] = str(Path(matlab_root) / "bin") + os.pathsep + str(runtime) + os.pathsep + env.get("PATH", "")
        if os.name != "nt":
            library_path = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
            env[library_path] = str(runtime) + os.pathsep + env.get(library_path, "")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "build", "test", "parameterize"))
    parser.add_argument("--engine", choices=("native", "matlab", "octave"),
                        default={"matlab": "matlab", "octave": "octave", "octave-source": "octave"}.get(os.environ.get("PIXI_ENVIRONMENT_NAME"), "native"))
    parser.add_argument("--solver", choices=("ACCELERATE", "MKL", "EIGEN", "PANUA"),
                        default="ACCELERATE" if sys.platform == "darwin" else "MKL")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Require both CUDA tensors and the GIF GPU evaluator")
    parser.add_argument("--gif-backend", choices=("auto", "native", "torch"))
    args, extra = parser.parse_known_args()
    if extra and args.command != "parameterize":
        parser.error(f"Unexpected arguments: {extra}")
    if args.gpu and args.command != "test":
        parser.error("--gpu is a test mode; parameterize accepts --device and --gif-device")
    if (args.full or args.gpu) and args.engine == "native":
        parser.error("--full/--gpu requires a MATLAB or Octave pipeline environment")
    source_octave = args.engine == "octave" and os.environ.get("PIXI_ENVIRONMENT_NAME") == "octave-source"
    suffix = "-source" if source_octave else ""
    build = ROOT / "build" / f"pixi-{args.solver.lower()}-{args.engine}{suffix}"
    env = runtime_environment(build, args.engine)
    octave_executable = None
    # Pixi/Conda puts C/C++ packages under Library on Windows.
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
    prefixes = [prefix, prefix / "Library"]
    inherited = os.environ.get("CMAKE_PREFIX_PATH")
    if inherited:
        prefixes += [Path(p) for p in inherited.split(os.pathsep) if p]
    # Configure/build in this process so --solver/--engine also apply to all
    # prerequisite steps; Pixi does not forward task arguments to depends-on.
    if args.engine == "octave":
        if source_octave:
            if sys.platform != "darwin":
                parser.error("The optional octave-source environment is macOS-only")
            octave_executable = build_source_octave(env)
        else:
            requested = env.get("BCP_OCTAVE_EXECUTABLE")
            octave_executable = shutil.which(requested) if requested else (
                shutil.which("octave-cli") or shutil.which("octave"))
            if not octave_executable:
                parser.error("Octave is missing. Use the Pixi octave environment; no runtime test will be silently skipped.")
            octave_executable = str(Path(octave_executable).resolve())
        env["BCP_OCTAVE_EXECUTABLE"] = octave_executable
    if args.command in ("configure", "build", "test", "parameterize"):
        command = [
            "cmake", "-S", ROOT, "-B", build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={';'.join(map(str, prefixes))}",
            f"-DBCP_LINEAR_SOLVER={args.solver}",
            f"-DBCP_BUILD_GIF={'OFF' if args.engine == 'native' else 'ON'}",
            f"-DBCP_SCRIPT_ENGINE={args.engine.upper() if args.engine != 'native' else 'MATLAB'}",
            "-DBCP_BUILD_TESTS=ON",
            f"-DBCP_TEST_PIPELINE={'ON' if args.full or args.engine != 'native' else 'OFF'}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if octave_executable:
            command.append(f"-DBCP_OCTAVE_EXECUTABLE={octave_executable}")
        if args.engine == "matlab" and os.environ.get("Matlab_ROOT_DIR"):
            command.append(f"-DMatlab_ROOT_DIR={os.environ['Matlab_ROOT_DIR']}")
        run(command, env)
        # Initial FindMatlab discovery may have just populated Matlab_ROOT_DIR.
        env = runtime_environment(build, args.engine, octave_executable)
    if args.command == "configure":
        return
    run(["cmake", "--build", build, "--config", "Release", "--parallel", "4"], env)
    if args.command == "build":
        return
    if args.gif_backend:
        env["BCP_GIF_BACKEND"] = args.gif_backend
    if args.command == "test":
        if args.gpu:
            env["BCP_DEVICE"] = "cuda:0"
            env["BCP_GIF_DEVICE"] = "gpu"
            env["BCP_TORCH_REFERENCE_DEVICE"] = "cuda:0"
            print(f"Required devices: PyTorch=cuda:0, GIF={env['BCP_GIF_DEVICE']}", flush=True)
        run(["ctest", "--test-dir", build, "-C", "Release", "--output-on-failure", "--no-tests=error"], env)
    else:
        if args.engine != "native":
            extra = ["--method", "bcp", *extra]
        run([sys.executable, "scripts/parameterize.py", *extra], env)


if __name__ == "__main__":
    main()
