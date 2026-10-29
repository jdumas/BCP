import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from process_utils import run_native

import igl
import numpy as np
import openmesh as om
import torch
from scipy.io import savemat, loadmat

from GIF_and_CM.consts import ROOT_PATH_PARALLEL_TRANSPORT_EXE, ROOT_DIR_CACHE_PARALLEL_TRANSPORT
from compute_edge_lengths import compute_edge_lengths
from intrinsic_mollification_eps import intrinsic_mollification_eps
from edge_length_map import construct_edge_length_map



def get_vertex_one_rings_intrinsic_vectorized(v, f, edge_lengths, is_boundary_vert, eps=0):
    """
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    returns:
        one_rings: (#V, #neighbors)
            for each vert in the mesh, the indices of its neighboring faces
        all_one_ring_angles: (#V, #neighbors)
            for each vert in the mesh, the angles of its neighboring faces (in CCW order)
            note that the number of neighbors varies based on topology
    """
    mesh = om.TriMesh(v, f)

    edge_length_map = construct_edge_length_map(v, f, edge_lengths)
    axis_edge_indices = [np.nan for i in range(v.shape[0])]

    one_ring_verts = mesh.vertex_vertex_indices()[:, ::-1]
    n_one_ring_verts = np.sum(np.where(one_ring_verts == -1, 0, 1), axis=1)
    mask = one_ring_verts != -1
    max_one_ring_size = one_ring_verts.shape[-1]
    angles = np.empty(one_ring_verts.shape)

    for i in reversed(range(max_one_ring_size)):
        """
        one_ring_verts - np.array [N, max(len(one_rings))]:
            -1 -1    4 7 0 3   ==>  4 4   4 7 0 3
            -1 -1 -1   3 2 6   ==>  3 3 3   3 2 6
            -1 -1    7 9 8 1   ==>  7 7   7 9 8 1
            ...
        """
        one_ring_verts[:, i] = np.where(
            one_ring_verts[:, i] == -1, one_ring_verts[:, (i + 1) % max_one_ring_size], one_ring_verts[:, i]
        )

    already_assigned_axis_edge_indices = np.zeros(v.shape[0], dtype=np.bool_)
    for i in range(max_one_ring_size):
        # periodic assumption for angles around manifold one ring
        a_len = np.maximum(
            edge_length_map[np.arange(v.shape[0]), one_ring_verts[:, i]],
            edge_length_map[one_ring_verts[:, i], np.arange(v.shape[0])],
        )
        b_len = np.maximum(
            edge_length_map[np.arange(v.shape[0]), one_ring_verts[:, (i + 1) % max_one_ring_size]],
            edge_length_map[one_ring_verts[:, (i + 1) % max_one_ring_size], np.arange(v.shape[0])],
        )
        c_len = np.maximum(
            edge_length_map[one_ring_verts[:, i], one_ring_verts[:, (i + 1) % max_one_ring_size]],
            edge_length_map[one_ring_verts[:, (i + 1) % max_one_ring_size], one_ring_verts[:, i]],
        )
        theta = np.arccos(np.clip((a_len**2 + b_len**2 - c_len**2) / (2 * a_len * b_len), -1.0, 1.0))
        angles[:, i] = theta
        axis_edge_indices = np.where(
            np.logical_and(a_len > eps, np.logical_not(already_assigned_axis_edge_indices)), i, axis_edge_indices
        )
        already_assigned_axis_edge_indices[a_len > eps] = True
    axis_edge_indices = axis_edge_indices.astype(int)

    mask_boundary = mask.copy()
    mask_boundary[:, -1] = False
    boundary_vert_angles = np.where(mask_boundary, angles, 0)
    boundary_vert_angles[:, -1] = (2 * np.pi) - np.sum(np.where(mask_boundary, angles, 0), axis=1)

    all_one_ring_angles = np.where(is_boundary_vert[:, None], boundary_vert_angles, np.where(mask, angles, 0))
    total_interior_angles = np.sum(np.array(all_one_ring_angles), axis=1)

    axis_edge_vert_idx = one_ring_verts.flatten()[axis_edge_indices + np.arange(v.shape[0]) * one_ring_verts.shape[1]]
    axis_edge_vertices = v[axis_edge_vert_idx]
    one_ring_vertices = np.where(mask, one_ring_verts, -1)

    return (
        one_ring_vertices,
        all_one_ring_angles,
        total_interior_angles,
        axis_edge_indices,
        axis_edge_vertices,
        n_one_ring_verts,
    )


def compute_parallel_transport_intrinsic_vertex_to_face(
    v, f, frames_faces
):
    """
    Parameters
    ----------
    l: np.array of size [#F, 3]
        list of edge lengths. each row of l represents the edges of face i, as [l_ij, l_jk, l_ki]
    intrinsic_mollification
        compute and add small constant eps to all intrinsic edge lengths to satisfy triangle inequality
    Returns
    -------

    """
    start_time = time.time()
    boundary_loop = igl.boundary_loop(f)
    transition_angles = np.zeros_like(f).astype(np.float64)  # ordering: [[i->j, j->k, k->i], ...]

    boundary_verts = []
    boundary_loops = igl.all_boundary_loop(f) if hasattr(igl, "all_boundary_loop") else [boundary_loop]
    for loop in boundary_loops:
        for i in range(len(loop)):
            boundary_verts.append((loop[i]))
    is_boundary_vert = np.isin(np.arange(v.shape[0]), boundary_verts)

    edge_lengths = compute_edge_lengths(v, f)
    eps = intrinsic_mollification_eps(v, f, edge_lengths)
    edge_lengths = edge_lengths + eps
    (
        one_rings,
        all_one_ring_angles,
        total_interior_angles,
        axis_edge_indices,
        axis_edge_vertices,
        n_one_ring_verts,
    ) = get_vertex_one_rings_intrinsic_vectorized(
        v, f, edge_lengths, is_boundary_vert, eps=eps
    )

    integrated_angles = np.zeros_like(all_one_ring_angles).astype(np.float64)

    for j in range(1, integrated_angles.shape[1]):
        integrated_angles[:, j] = integrated_angles[:, j - 1] + all_one_ring_angles[:, j - 1]

    end_time = time.time()

    one_rings_time = end_time - start_time

    angles_vertices_to_faces, middle_angle_time = compute_middle_angle(f, one_rings, integrated_angles, all_one_ring_angles)


    # Compute the coordinates of the incenter (weighted average)
    start_time = time.time()
    incenter = (
        edge_lengths[:, 1][:, np.newaxis] * v[f[:, 0]]
        + edge_lengths[:, 2][:, np.newaxis] * v[f[:, 1]]
        + edge_lengths[:, 0][:, np.newaxis] * v[f[:, 2]]
    ) / edge_lengths.sum(axis=1)[:, np.newaxis]
    incenter_to_v_vectors = v[f, :] - incenter[:, np.newaxis]
    # incenter_to_v_vectors = v[f, :] - v[np.roll(f, shift=-1, axis=1), :]

    local_edge_vectors = np.einsum(
        "abc, acd->abd", frames_faces[:, :2, :], incenter_to_v_vectors.transpose(0, 2, 1)
    ).transpose(0, 2, 1)
    local_edges_complex = local_edge_vectors[:, :, 0] + 1j * local_edge_vectors[:, :, 1]
    angles_faces_to_vertices = np.remainder(np.angle(local_edges_complex) + np.pi, 2 * np.pi) - np.pi

    for i in range(3):
        vi_idx = f[:, i]

        vi_is_boundary_vert = np.isin(vi_idx, boundary_verts)  # Is vi a boundary vertex

        vi_start = (
            np.arange(f.shape[0]) * all_one_ring_angles.shape[1] + axis_edge_indices[vi_idx]
        )  # The index of the last vertex around vi, in the one-ring of vi, also corresponding to the index of the x-axis edge in the local frame, adjusted for flattening the faces list

        # vi_e_ij_angle = angles_vertices_to_faces[:, i] - integrated_angles[vi_idx].flatten()[vi_start]  # The difference between the angle to eij and the angle to the last vertex around vi, which is the x-axis

        vi_e_ij_angle = np.where(
            vi_is_boundary_vert,
            angles_vertices_to_faces[:, i] - integrated_angles[vi_idx].flatten()[vi_start],
            angles_vertices_to_faces[:, i],
        )  # normalize angle

        vi_e_ij_angle = np.where(
            vi_is_boundary_vert,
            vi_e_ij_angle,
            vi_e_ij_angle * ((2 * np.pi) / np.sum(all_one_ring_angles[vi_idx], axis=1)),
        )  # normalize angle

        transition_angle = (np.pi + angles_faces_to_vertices[:, i]) - vi_e_ij_angle
        transition_angles[:, i] = transition_angle

    complex_rotations = np.exp((0 + 1j) * transition_angles)
    end_time = time.time()

    v2f_angles_time = end_time - start_time

    total_time = one_rings_time + middle_angle_time + v2f_angles_time

    return complex_rotations, total_time

def compute_middle_angle(f, one_rings, integrated_angles, all_one_ring_angles):
    Path(ROOT_DIR_CACHE_PARALLEL_TRANSPORT).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pt-", dir=ROOT_DIR_CACHE_PARALLEL_TRANSPORT) as work:
        source = str(Path(work) / "source.mat")
        result = str(Path(work) / "result.mat")
        savemat(source, {
            "f": f.astype(np.float64),
            "one_rings": one_rings.astype(np.float64),
            "integrated_angles": integrated_angles.astype(np.float64),
            "all_one_ring_angles": all_one_ring_angles.astype(np.float64),
            "all_one_ring_angles_sums": all_one_ring_angles.sum(axis=1, keepdims=True).astype(np.float64),
        })
        run_native(ROOT_PATH_PARALLEL_TRANSPORT_EXE, [source, result, 1], 300)
        data = loadmat(result)
        return data["angles_vertices_to_faces"], float(np.real(data["total_time"][0, 0]))


def interpolate_faces_dimensional_data_to_vertices_torch(faces, vals, complex_rotations_f_to_v, W):
    """
    Accumulates weighted per-face vectors to vertices.

    Args:
        faces (LongTensor): (f, 3) face indices.
        vals (Tensor): (f, m) face vectors.
        complex_rotations_f_to_v (Tensor): (f, 3) weights for each vertex in each face.

    Returns:
        Tensor: (n_vertices, m) accumulated (or averaged) vertex values.
    """
    n_vertices = faces.max() + 1
    device = vals.device
    f, m = vals.shape

    V_sum = torch.zeros((n_vertices, m), dtype=vals.dtype, device=device)
    V_weight = torch.zeros((n_vertices,), dtype=W.dtype, device=device)

    for j in range(3):  # for each corner of the face
        idx = faces[:, j]                   # vertex indices (f,)
        v2f_j = complex_rotations_f_to_v[:, j].unsqueeze(1)      # (f, 1)
        wj = W.unsqueeze(1)                 # (f, 1)
        contrib = wj * v2f_j * vals               # (f, m)

        V_sum.index_add_(0, idx, contrib)
        V_weight.index_add_(0, idx, wj.squeeze(1))

    V_count = V_weight.unsqueeze(1)  # avoid division by zero
    V_sum = V_sum / V_count

    return V_sum
