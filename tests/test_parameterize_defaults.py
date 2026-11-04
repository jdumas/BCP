"""Full-pipeline defaults preserve original GPU placement without silent fallback."""
import importlib.util,os,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('parameterize_defaults',ROOT/'scripts/parameterize.py')
parameterize=importlib.util.module_from_spec(spec);spec.loader.exec_module(parameterize)

class ParameterizeDefaultsTests(unittest.TestCase):
    def test_primary_platforms_require_cuda_and_gpu(self):
        for platform in ('linux','win32'):
            with patch.dict(os.environ,{},clear=True),patch.object(sys,'platform',platform):
                parameterize.apply_compute_defaults('bcp')
                self.assertEqual(os.environ['BCP_DEVICE'],'cuda:0')
                self.assertEqual(os.environ['BCP_GIF_DEVICE'],'gpu')
    def test_explicit_cpu_and_auto_requests_are_preserved(self):
        for device,gif in [('cpu','cpu'),('auto','auto')]:
            with patch.dict(os.environ,{'BCP_DEVICE':device,'BCP_GIF_DEVICE':gif},clear=True):
                parameterize.apply_compute_defaults('bcp')
                self.assertEqual(os.environ['BCP_DEVICE'],device);self.assertEqual(os.environ['BCP_GIF_DEVICE'],gif)
    def test_macos_keeps_cpu_default(self):
        with patch.dict(os.environ,{},clear=True),patch.object(sys,'platform','darwin'):
            parameterize.apply_compute_defaults('bcp')
            self.assertEqual(os.environ['BCP_DEVICE'],'cpu');self.assertEqual(os.environ['BCP_GIF_DEVICE'],'cpu')
    def test_cm_has_no_gpu_default_or_probe(self):
        with patch.dict(os.environ,{},clear=True):
            parameterize.apply_compute_defaults('cm')
            self.assertNotIn('BCP_DEVICE',os.environ);self.assertNotIn('BCP_GIF_DEVICE',os.environ)

if __name__=='__main__':unittest.main()
