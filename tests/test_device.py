"""Device policy tests: do not require or fake a successful real CUDA run."""
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1]/"BCP/GIF_and_CM/device.py"


class DeviceTests(unittest.TestCase):
    def load(self, requested, available=False, failure=None):
        def device(value):
            return types.SimpleNamespace(type=value.split(":")[0], index=0)
        cuda = types.SimpleNamespace(
            is_available=Mock(return_value=available),
            init=Mock(side_effect=failure),
            get_device_properties=Mock(),
        )
        torch = types.SimpleNamespace(device=device, cuda=cuda)
        spec = importlib.util.spec_from_file_location("bcp_device_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"BCP_DEVICE":requested}, clear=True), patch.dict("sys.modules", {"torch":torch}):
            spec.loader.exec_module(module)
        return module, cuda

    def test_cpu_does_not_probe_cuda(self):
        module, cuda = self.load("cpu")
        self.assertEqual(module.DEVICE.type, "cpu")
        cuda.is_available.assert_not_called()
        cuda.init.assert_not_called()

    def test_auto_can_select_cpu(self):
        module, cuda = self.load("auto", available=False)
        self.assertEqual(module.DEVICE.type, "cpu")
        cuda.init.assert_not_called()

    def test_auto_gpu_is_initialized_and_validated(self):
        module, cuda = self.load("auto", available=True)
        self.assertEqual(module.DEVICE.type, "cuda")
        cuda.init.assert_called_once()
        cuda.get_device_properties.assert_called_once()

    def test_required_cuda_does_not_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, "requires working CUDA: driver error 304"):
            self.load("cuda:0", failure=RuntimeError("driver error 304"))

    def test_unsupported_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported BCP_DEVICE"):
            self.load("mps")


if __name__ == "__main__":
    unittest.main()
