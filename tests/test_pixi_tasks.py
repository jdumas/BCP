"""Verify environment/task routing without installing or mocking numerical success."""
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pixi_tasks", ROOT/"scripts/pixi_tasks.py")
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


class PixiTaskTests(unittest.TestCase):
    def invoke(self, args, environment="default", octave=True, platform="linux"):
        with tempfile.TemporaryDirectory() as work, patch.object(tasks, "ROOT", Path(work)), \
             patch.dict(os.environ, {"PIXI_ENVIRONMENT_NAME": environment, "CONDA_PREFIX": work}, clear=True), \
             patch.object(sys, "argv", ["pixi_tasks.py", *args]), \
             patch.object(tasks.sys, "platform", platform), \
             patch.object(tasks.shutil, "which", return_value="octave-cli" if octave else None), \
             patch.object(tasks.subprocess, "run") as process, redirect_stdout(io.StringIO()):
            tasks.main()
            return process.call_args_list

    def test_native_config_is_mkl_not_silent_fallback(self):
        calls = self.invoke(["configure"])
        command = calls[0].args[0]
        self.assertIn("-DBCP_LINEAR_SOLVER=MKL", command)
        self.assertIn("-DBCP_BUILD_GIF=OFF", command)
        self.assertIn("-DBCP_TEST_PIPELINE=OFF", command)
        self.assertIn(f"-DPython3_EXECUTABLE={sys.executable}", command)

    def test_macos_defaults_to_accelerate(self):
        for environment in ("default", "octave"):
            calls = self.invoke(["test"], environment, platform="darwin")
            self.assertIn("-DBCP_LINEAR_SOLVER=ACCELERATE", calls[0].args[0])
            engine = "native" if environment == "default" else "octave"
            for call in calls:
                self.assertTrue(any(f"pixi-accelerate-{engine}" in str(arg) for arg in call.args[0]))

    def test_source_environment_explicitly_builds_and_routes_patched_octave(self):
        executable = "/private/bcp-source/install/bin/octave-cli"
        with patch.object(tasks, "build_source_octave", return_value=executable) as build:
            calls = self.invoke(["test"], "octave-source", platform="darwin")
        build.assert_called_once()
        self.assertIn(f"-DBCP_OCTAVE_EXECUTABLE={executable}", calls[0].args[0])
        self.assertIn("-DBCP_SCRIPT_ENGINE=OCTAVE", calls[0].args[0])
        self.assertIn("-DBCP_TEST_PIPELINE=ON", calls[0].args[0])
        for call in calls:
            self.assertEqual(call.kwargs["env"]["BCP_OCTAVE_EXECUTABLE"], executable)
            self.assertTrue(any("pixi-accelerate-octave-source" in str(arg) for arg in call.args[0]))

    def test_prebuilt_environment_never_builds_octave_from_source(self):
        with patch.object(tasks, "build_source_octave") as build:
            calls = self.invoke(["configure"], "octave", platform="darwin")
        build.assert_not_called()
        self.assertTrue(any(arg.startswith("-DBCP_OCTAVE_EXECUTABLE=") for arg in calls[0].args[0]))
        self.assertNotIn("-source", calls[0].kwargs["env"]["BCP_BIN_DIR"])

    def test_source_build_failure_does_not_fall_back_to_prebuilt(self):
        with patch.object(tasks, "build_source_octave", side_effect=RuntimeError("source build failed")):
            with self.assertRaisesRegex(RuntimeError, "source build failed"):
                self.invoke(["test"], "octave-source", platform="darwin")

    def test_source_environment_is_not_supported_on_linux(self):
        with patch.object(tasks, "build_source_octave") as build:
            with self.assertRaises(SystemExit) as result:
                self.invoke(["configure"], "octave-source")
        self.assertEqual(result.exception.code, 2)
        build.assert_not_called()

    def test_windows_keeps_mkl_default(self):
        calls = self.invoke(["configure"], platform="win32")
        self.assertIn("-DBCP_LINEAR_SOLVER=MKL", calls[0].args[0])

    def test_macos_respects_explicit_solver(self):
        calls = self.invoke(["configure", "--solver", "EIGEN"], platform="darwin")
        self.assertIn("-DBCP_LINEAR_SOLVER=EIGEN", calls[0].args[0])

    def test_macos_matlab_runtime_paths(self):
        for machine, arch in (("arm64", "maca64"), ("x86_64", "maci64")):
            with patch.object(tasks.sys, "platform", "darwin"), \
                 patch.object(tasks.platform, "machine", return_value=machine), \
                 patch.dict(os.environ, {"Matlab_ROOT_DIR": "/Applications/MATLAB.app"}, clear=True):
                env = tasks.runtime_environment(Path("nonexistent-build"), "matlab")
                self.assertEqual(env["DYLD_LIBRARY_PATH"], f"/Applications/MATLAB.app/bin/{arch}" + os.pathsep)
                self.assertNotIn("LD_LIBRARY_PATH", env)

    def test_full_environment_requires_integration_gate(self):
        for environment, engine in (("matlab","MATLAB"), ("octave","OCTAVE")):
            calls = self.invoke(["configure"], environment)
            command = calls[0].args[0]
            self.assertIn(f"-DBCP_SCRIPT_ENGINE={engine}", command)
            self.assertIn("-DBCP_BUILD_GIF=ON", command)
            self.assertIn("-DBCP_TEST_PIPELINE=ON", command)

    def test_no_octave_is_a_configuration_failure(self):
        with self.assertRaises(SystemExit) as result:
            self.invoke(["configure"], "octave", octave=False)
        self.assertEqual(result.exception.code, 2)

    def test_solver_override_applies_to_configure_build_and_test(self):
        calls = self.invoke(["test", "--solver", "EIGEN"])
        self.assertEqual(len(calls), 3)
        self.assertIn("-DBCP_LINEAR_SOLVER=EIGEN", calls[0].args[0])
        for call in calls:
            self.assertTrue(any("pixi-eigen-native" in str(arg) for arg in call.args[0]))
        self.assertIn("--no-tests=error", calls[-1].args[0])

    def test_gpu_test_requires_both_gpu_stages(self):
        for environment in ("matlab", "octave"):
            calls = self.invoke(["test", "--gpu"], environment)
            env = calls[-1].kwargs["env"]
            self.assertEqual(env["BCP_DEVICE"], "cuda:0")
            self.assertEqual(env["BCP_GIF_DEVICE"], "gpu")
            self.assertEqual(env["BCP_TORCH_REFERENCE_DEVICE"], "cuda:0")

    def test_torch_backend_override_is_forwarded_to_pipeline(self):
        calls = self.invoke(["test", "--gif-backend", "torch"], "octave")
        self.assertEqual(calls[-1].kwargs["env"]["BCP_GIF_BACKEND"], "torch")
        self.assertEqual(calls[-1].kwargs["env"]["BCP_PYTHON_EXECUTABLE"], sys.executable)

    def test_gpu_test_requires_a_pipeline_environment(self):
        with self.assertRaises(SystemExit) as result:
            self.invoke(["test", "--gpu"])
        self.assertEqual(result.exception.code, 2)

    def test_parameterize_preserves_literal_arguments_and_environment(self):
        calls = self.invoke(["parameterize", "input path.obj", "output path.obj"], "octave")
        command = calls[-1].args[0]
        self.assertEqual(command[-4:], ["--method","bcp","input path.obj","output path.obj"])
        self.assertIn("pixi-mkl-octave", calls[-1].kwargs["env"]["BCP_BIN_DIR"])


if __name__ == "__main__":
    unittest.main()
