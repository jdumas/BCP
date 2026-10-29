"""Boundary-mask regression checks; no native solve or device initialization."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for folder in ("BCP", "BCP/utils", "BCP/GIF_and_CM"):
    sys.path.insert(0, str(ROOT / folder))
try:
    import numpy as np
    import compute_parallel_transport_intrinsic as transport
except ImportError:
    transport = None


@unittest.skipIf(transport is None, "Optional mesh pipeline dependencies not installed")
class ParallelTransportBoundaryTests(unittest.TestCase):
    def boundary_mask(self, vertices, faces):
        class Captured(Exception):
            pass

        mask = None

        def capture(v, f, lengths, is_boundary, **kwargs):
            nonlocal mask
            mask = is_boundary.copy()
            raise Captured()

        with patch.object(transport, "get_vertex_one_rings_intrinsic_vectorized", side_effect=capture):
            with self.assertRaises(Captured):
                transport.compute_parallel_transport_intrinsic_vertex_to_face(
                    vertices, faces, None)
        return mask

    def test_disk_interior_is_not_a_boundary_vertex(self):
        vertices = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,0,0.1]])
        faces = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        np.testing.assert_array_equal(
            self.boundary_mask(vertices, faces), [True, True, True, True, False])

    def test_all_boundary_loops_are_classified(self):
        vertices = np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],
                             [-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
        faces = np.array([[i, (i+1)%4, (i+1)%4+4] for i in range(4)] +
                         [[i, (i+1)%4+4, i+4] for i in range(4)])
        np.testing.assert_array_equal(self.boundary_mask(vertices, faces), np.ones(8, dtype=bool))


if __name__ == "__main__":
    unittest.main()
