#!/usr/bin/env python3
"""Opt-in, project-local macOS Octave build. Never modifies the Conda package.

Run with `pixi run -e octave-source build-octave`. stdout is reserved for the
installed octave-cli path (consumed by pixi_tasks.py); progress goes to stderr.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "10.3.0"
ARCHIVE_SHA256 = "2fcb38dc062e440f1e06c069bbca840ed46dcc8f983e473e1558fcc38384ee6b"
ARCHIVE_URL = f"https://ftp.gnu.org/gnu/octave/octave-{VERSION}.tar.gz"
PATCH = ROOT / "scripts" / "patches" / f"octave-{VERSION}-macos-getcwd.patch"
FLAGS = (
    "--enable-shared", "--disable-static", "--disable-docs", "--disable-java",
    "--disable-openmp", "--without-qt", "--without-fltk", "--with-x=no",
    "--without-opengl", "--without-framework-carbon", "--without-magick",
    "--without-portaudio", "--without-sndfile",
    "--without-sundials_nvecserial", "--without-sundials_ida", "--without-sundials_core",
    "--without-sundials_sunlinsolklu",
    "--without-qrupdate", "--without-pcre2", "--enable-readline", "--enable-rapidjson",
    "--with-blas=-lblas", "--with-lapack=-llapack",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_environment(prefix):
    env = os.environ.copy()
    for key, default in (("CC", "cc"), ("CXX", "c++"), ("FC", "gfortran")):
        env.setdefault(key, default)
    env.setdefault("F77", env["FC"])
    for key in ("CFLAGS", "CXXFLAGS", "FFLAGS", "FCFLAGS"):
        env.setdefault(key, "-O2 -g0")
    include, lib = shlex.quote(str(prefix / "include")), shlex.quote(str(prefix / "lib"))
    env["CPPFLAGS"] = f"-I{include} " + env.get("CPPFLAGS", "")
    env["LDFLAGS"] = f"-L{lib} -Wl,-rpath,{lib} " + env.get("LDFLAGS", "")
    env["PKG_CONFIG_LIBDIR"] = os.pathsep.join(str(prefix / p) for p in ("lib/pkgconfig", "share/pkgconfig"))
    return env


def build_spec(prefix, env):
    # Rebuild if sources, patch, builder, toolchain flags, or Conda packages change.
    keys = ("CC", "CXX", "FC", "F77", "CFLAGS", "CXXFLAGS", "FFLAGS", "FCFLAGS",
            "CPPFLAGS", "LDFLAGS", "SDKROOT", "CONDA_BUILD_SYSROOT", "MACOSX_DEPLOYMENT_TARGET")
    return {
        "version": VERSION, "archive_sha256": ARCHIVE_SHA256,
        "patch_sha256": digest(PATCH), "builder_sha256": digest(__file__),
        "prefix": str(prefix), "machine": platform.machine(), "flags": list(FLAGS),
        "environment": {key: env.get(key) for key in keys},
        "packages": sorted(p.name for p in (prefix / "conda-meta").glob("*.json")),
        "tools": {key: shutil.which(shlex.split(env[key])[0]) for key in ("CC", "CXX", "FC")},
    }


def download_archive(path):
    if path.exists():
        if digest(path) != ARCHIVE_SHA256:
            raise RuntimeError(f"Cached Octave archive checksum mismatch: {path}; remove it explicitly before retrying")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Octave source] Downloading {ARCHIVE_URL}", file=sys.stderr, flush=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        pending = Path(stream.name)
        try:
            with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > 256 * 1024 * 1024:
                        raise RuntimeError("Octave source download exceeded 256 MiB")
                    stream.write(chunk)
            stream.close()
            if digest(pending) != ARCHIVE_SHA256:
                raise RuntimeError("Downloaded Octave archive checksum mismatch")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def extract_archive(archive, destination):
    """Reject traversal, special files and escaping links before extracting."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != f"octave-{VERSION}":
                raise RuntimeError(f"Unsafe Octave archive path: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError(f"Unsupported Octave archive entry: {member.name}")
        if sum(member.size for member in members) > 1024 * 1024 * 1024:
            raise RuntimeError("Unpacked Octave archive exceeds 1 GiB")
        # Python 3.11 in Pixi includes the backported data extraction filter.
        source.extractall(destination, members=members, filter="data")


def run_logged(command, cwd, env, log):
    print(f"[Octave source] {shlex.join(map(str, command))}\n  log: {log}", file=sys.stderr, flush=True)
    with log.open("w") as output:
        subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def smoke(executable, root, env, log):
    check_env = dict(env, BCP_EXPECTED_CWD=str(root.resolve()))
    code = ("assert(strcmp(pwd(), getenv('BCP_EXPECTED_CWD'))); "
            "A = sparse([2,1;1,3]); b = [1;2]; assert(norm(A*(A\\b)-b) < 1e-12); "
            "decoded = jsondecode(jsonencode(struct('value',7))); assert(decoded.value == 7); "
            "disp('BCP patched Octave: cwd, JSON and sparse solve passed');")
    # No cwd relocation or permission changes: exercise the original workspace.
    run_logged([executable, "--no-gui", "--quiet", "--no-history", "--no-init-file",
                "--no-site-file", "--eval", code], root, check_env, log)


@contextmanager
def source_lock(base):
    # The macOS-only builder uses an OS-released lock, including on interruption.
    import fcntl
    base.mkdir(parents=True, exist_ok=True)
    with (base / "build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another Octave source build is running; wait for its result") from error
        yield


def ensure_octave(root, jobs):
    if sys.platform != "darwin":
        raise RuntimeError("The optional patched Octave source build is macOS-only")
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != "octave-source":
        raise RuntimeError("Use `pixi run -e octave-source build-octave`; no source build is selected by default")
    with source_lock(root / "build" / "octave-source"):
        return build_octave(root, jobs)


def build_octave(root, jobs):
    prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
    env = build_environment(prefix)
    for name in (env["CC"], env["CXX"], env["FC"], "make", "patch", "pkg-config", "perl", "bison", "gperf"):
        if not shutil.which(shlex.split(name)[0]):
            raise RuntimeError(f"Missing source-build tool: {name}")
    spec = build_spec(prefix, env)
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    base = root / "build" / "octave-source"
    work = base / identity
    install = work / "install"
    executable = install / "bin" / "octave-cli"
    receipt = install / "bcp-build.json"
    if receipt.is_file() and executable.is_file() and os.access(executable, os.X_OK):
        if json.loads(receipt.read_text()) == spec:
            print(f"[Octave source] Reusing validated build: {executable}", file=sys.stderr)
            return executable
    # A repair must not leave an old success receipt usable after a failed smoke.
    receipt.unlink(missing_ok=True)
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    archive = base / "downloads" / f"octave-{VERSION}.tar.gz"
    download_archive(archive)
    sources = work / "source"
    ready = sources / "bcp-patched.json"
    source = sources / f"octave-{VERSION}"
    if not ready.is_file():
        if sources.exists():
            shutil.rmtree(sources)
        sources.mkdir()
        extract_archive(archive, sources)
        run_logged(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", PATCH],
                   source, env, logs / "patch.log")
        ready.write_text(json.dumps({"patch_sha256": digest(PATCH)}))
    build = work / "build"
    build.mkdir(exist_ok=True)
    configured = build / "bcp-configured.json"
    if not configured.is_file():
        run_logged([source / "configure", f"--prefix={install}", *FLAGS,
                    f"--with-hdf5-includedir={prefix / 'include'}",
                    f"--with-hdf5-libdir={prefix / 'lib'}"], build, env, logs / "configure.log")
        configured.write_text(json.dumps(spec, sort_keys=True))
    run_logged(["make", f"-j{jobs}"], build, env, logs / "build.log")
    run_logged(["make", "install"], build, env, logs / "install.log")
    smoke(executable, root, env, logs / "smoke.log")
    # A failed build/install/runtime check can never create a reusable receipt.
    pending_receipt = receipt.with_suffix(".json.tmp")
    pending_receipt.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    pending_receipt.replace(receipt)
    return executable


def positive_jobs(value):
    number = int(value)
    if not 1 <= number <= 64:
        raise argparse.ArgumentTypeError("build jobs must be between 1 and 64")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=positive_jobs, default=os.environ.get("BCP_OCTAVE_BUILD_JOBS", "4"))
    args = parser.parse_args()
    try:
        executable = ensure_octave(ROOT, args.jobs)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"[Octave source] {error}\nSee the command log above. No prebuilt fallback was selected.\n")
    print(executable)


if __name__ == "__main__":
    main()
