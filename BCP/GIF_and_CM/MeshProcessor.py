import time
import warnings
from typing import Tuple

from uuid import uuid4
from pathlib import Path

import os
import numpy as np
import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy
import open3d as o3d

from GIF_and_CM.consts import ROOT_DIR_CACHE_MESHLIB

warnings.filterwarnings("ignore")

TARGET_SIMPLIFICATION_FACES_COUNT = None


def run_simplification(verts: np.ndarray, faces: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    # This function returns the v, f of the simplified mesh, and KD-tree mapping between the triangles centroids of the original mesh to the simplified mesh

    print("Applying simplification!")
    simplified_v, simplified_f, simplification_time = simplify_mesh(faces, verts, max_error=1e20, target_simplification_faces_count=TARGET_SIMPLIFICATION_FACES_COUNT)

    start_correspondence_time = time.time()
    original_mesh_centroids = compute_centroids(verts, faces)
    simplified_v_t = o3d.core.Tensor(simplified_v, dtype=o3d.core.Dtype.Float32)
    simplified_f_t = o3d.core.Tensor(simplified_f, dtype=o3d.core.Dtype.Int32)
    mesh = o3d.t.geometry.TriangleMesh(simplified_v_t, simplified_f_t)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    Q = o3d.core.Tensor(original_mesh_centroids.astype(np.float32))
    res = scene.compute_closest_points(Q)
    closest_points = res["points"].numpy()
    closest_triangles_indices = res["primitive_ids"].numpy()

    tris = simplified_v[simplified_f[closest_triangles_indices]]  # shape (N,3,3)
    A = tris[:, 0, :]  # (N,3)
    B = tris[:, 1, :]
    C = tris[:, 2, :]

    # Edge vectors
    v0 = B - A  # (N,3)
    v1 = C - A
    v2 = closest_points - A

    # Dot products (elementwise for each row)
    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01

    barycentric_coordinates = np.zeros_like(original_mesh_centroids)
    barycentric_coordinates[:, 1] = (d11 * d20 - d01 * d21) / denom
    barycentric_coordinates[:, 2] = (d00 * d21 - d01 * d20) / denom
    barycentric_coordinates[:, 0] = 1.0 - barycentric_coordinates[:, 1] - barycentric_coordinates[:, 2]
    end_correspondence_time = time.time()
    correspondence_time = end_correspondence_time - start_correspondence_time

    total_time = correspondence_time + simplification_time

    return simplified_v, simplified_f, closest_triangles_indices.astype(np.int32), barycentric_coordinates, total_time



def simplify_mesh(faces, verts, max_error=0.05, target_simplification_faces_count=None):
    if target_simplification_faces_count is None:
        target_simplification_faces_count = max(1e4, 0.01 * faces.shape[0])
    if target_simplification_faces_count < 1:
        raise ValueError("Target simplification face count must be positive")
    mesh = mrmeshnumpy.meshFromFacesVerts(faces, verts)
    start_simplification_time = time.time()
    mesh.packOptimally()
    settings = mrmeshpy.DecimateSettings()
    settings.maxDeletedFaces = int(
        faces.shape[0] - target_simplification_faces_count)  # simplified version has at least 10,000 faces
    settings.maxError = max_error  # Maximum error when decimation stops
    settings.subdivideParts = 16
    mrmeshpy.decimateMesh(mesh, settings)
    end_simplification_time = time.time()
    simplification_time = end_simplification_time - start_simplification_time

    # In order to extract the simplification result, meshlib requires saving it as .obj file and then load it
    Path(ROOT_DIR_CACHE_MESHLIB).mkdir(parents=True, exist_ok=True)
    cur_time = uuid4().hex
    tmp_obj_file = ROOT_DIR_CACHE_MESHLIB + "/" + cur_time + ".obj"
    start_simplification_time = time.time()
    mrmeshpy.saveMesh(mesh, tmp_obj_file)
    end_simplification_time = time.time()
    print(end_simplification_time - start_simplification_time)
    simplified_mesh = mrmeshpy.loadMesh(tmp_obj_file)
    os.remove(tmp_obj_file)
    simplified_v = mrmeshnumpy.getNumpyVerts(simplified_mesh)
    simplified_f = mrmeshnumpy.getNumpyFaces(simplified_mesh.topology)
    return simplified_v, simplified_f, simplification_time


def compute_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # Get the vertex coordinates for each triangle
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]

    # Compute centroids
    centroids = (A + B + C) / 3.0
    return centroids
