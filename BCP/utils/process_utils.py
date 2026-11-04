"""Checked, shell-free execution shared by CM, PT and GIF wrappers."""
import math
import os
from pathlib import Path
import signal
import subprocess


def _kill_process_tree(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def run_native(executable, arguments, timeout):
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Native executable not found: {executable}. Build it with CMake first.")
    override = os.environ.get("BCP_NATIVE_TIMEOUT_SECONDS")
    if override is not None:
        timeout = float(override)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("BCP_NATIVE_TIMEOUT_SECONDS must be finite and positive")
    command = [str(executable), *map(str, arguments)]
    # Inherit the caller's thread policy. In particular, do not serialize MKL.
    with subprocess.Popen(
        command,
        cwd=executable.parent,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _kill_process_tree(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from error
        except BaseException:
            _kill_process_tree(process)
            process.wait()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode:
        raise RuntimeError(
            f"{executable.name} exited with code {result.returncode}\n"
            f"{result.stdout or ''}\n{result.stderr or ''}"
        )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result
