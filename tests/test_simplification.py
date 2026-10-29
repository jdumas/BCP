"""Regression checks for the original simplification budget and packing path."""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"BCP"))
sys.path.insert(0, str(ROOT/"BCP/GIF_and_CM"))
try:
    import numpy as np
    import MeshProcessor as processor
except ImportError:
    processor = None


@unittest.skipIf(processor is None, "Optional mesh pipeline dependencies not installed")
class SimplificationTests(unittest.TestCase):
    def check_budget(self, face_count, expected):
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
        triangles = np.array([[0,1,2]], dtype=np.int32)
        mesh = processor.mrmeshnumpy.meshFromFacesVerts(triangles, vertices)
        settings = processor.mrmeshpy.DecimateSettings()
        faces = types.SimpleNamespace(shape=(face_count,3), copy=lambda: triangles.copy())
        with tempfile.TemporaryDirectory() as work, \
             patch.object(processor, "ROOT_DIR_CACHE_MESHLIB", work), \
             patch.object(processor.mrmeshnumpy, "meshFromFacesVerts", return_value=mesh) as create, \
             patch.object(processor.mrmeshpy, "DecimateSettings", return_value=settings), \
             patch.object(processor.mrmeshpy, "decimateMesh") as decimate:
            result = processor.simplify_mesh(faces, vertices)
        create.assert_called_once()
        decimate.assert_called_once()
        self.assertEqual(settings.maxDeletedFaces, expected)
        self.assertEqual(result[0].shape, vertices.shape)
        self.assertEqual(result[1].shape, triangles.shape)

    def test_fractional_target_is_not_floored_before_budget(self):
        self.check_budget(1_000_001, 990000)

    def test_small_mesh_keeps_original_pack_and_roundtrip_path(self):
        self.check_budget(1, -9999)


if __name__ == "__main__":
    unittest.main()
