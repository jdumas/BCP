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
    def invoke(self, args, environment="default", octave=True):
        with tempfile.TemporaryDirectory() as work, patch.object(tasks, "ROOT", Path(work)), \
             patch.dict(os.environ, {"PIXI_ENVIRONMENT_NAME": environment, "CONDA_PREFIX": work}, clear=True), \
             patch.object(sys, "argv", ["pixi_tasks.py", *args]), \
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
