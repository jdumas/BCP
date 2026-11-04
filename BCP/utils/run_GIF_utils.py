import time
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import igl
import torch
import scipy.io as sio

from GIF_and_CM.device import DEVICE
from process_utils import run_native

TIMEOUT_SECONDS = 300


def save_data_for_GIF(vertices, faces, cache_path, edge_lengths=None):
    cache = Path(cache_path).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    prefix = cache / uuid4().hex
    source = str(prefix) + ".obj"
    igl.write_triangle_mesh(source, vertices, faces)
    lengths = ""
    if edge_lengths is not None:
        lengths = str(prefix) + "_edge_lengths.mat"
        sio.savemat(lengths, {"edge_lengths": edge_lengths.astype(np.float64)})
    return source, str(prefix) + "_uvs.mat", lengths


def run_GIF(
    vertices,
    faces,
    exe_path,
    cache_path,
    interior_faces: int = -1,
    boundary_segment_size: int = -1,
    curvature_meta_vertices_rate: float = 0.001,
    outer_termination_condition_rate: float = 0.1,
    energy_related_termination_condition: float = 2.0,
    edge_lengths=None,
    allow_IDT_fix=False,
):
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="gif-", dir=cache_path) as work:
        source, result, lengths = save_data_for_GIF(vertices, faces, work, edge_lengths)
        flags = ["True", "False", "False"] if allow_IDT_fix else ["False", "True", "True"]
        run_native(exe_path, [
            source, result, *flags, interior_faces, boundary_segment_size,
            curvature_meta_vertices_rate, outer_termination_condition_rate,
            energy_related_termination_condition, lengths,
        ], TIMEOUT_SECONDS)
        data = sio.loadmat(result)
        return [
            data["uvs_ordered"],
            data["flipsData"],
            data["IDT_outputs"].reshape(-1),
            data["parametersMatrix"][4, 0],
            tuple(data["parametersMatrix"][i, 0] for i in range(15, 19)),
            data["mAverageEdgeLengths"],
        ]


def compute_edges_vectors(vertices, faces):
    # This function computes the edge vectors.
    # The returned matrix is of size #F x 3 x 3:
    # edges_vectors[f_ind, edge_ind, :] is the vector of the edge_ind edge in the f_ind face.
    # The order of the faces in edges_vectors is the same order of faces in th faces matrix.
    # The edges order is 01, 12, 20
    faces_vertices = np.zeros((faces.shape[0], 3, 3))
    faces_vertices[:, 0, :] = vertices[faces[:, 0]]
    faces_vertices[:, 1, :] = vertices[faces[:, 1]]
    faces_vertices[:, 2, :] = vertices[faces[:, 2]]
    edges_vectors = np.stack(
        (
            faces_vertices[:, 1, :] - faces_vertices[:, 0, :],
            faces_vertices[:, 2, :] - faces_vertices[:, 1, :],
            faces_vertices[:, 0, :] - faces_vertices[:, 2, :],
        ),
        axis=1,
    )
    return edges_vectors


def compute_EL_from_metric_tensors(metric_tensors, vertices, faces, local_bases=None):
    edges_vectors = torch.from_numpy(compute_edges_vectors(vertices, faces)).to(DEVICE)
    if local_bases is not None:
        edge_vectors_projected = torch.einsum(
            "abc, acd->abd", local_bases[:, :2, :], edges_vectors.transpose(2, 1)
        ).transpose(2, 1)
    else:
        edge_vectors_projected = edges_vectors
    edges_vectors_mul = torch.einsum("adb, adc->adbc", edge_vectors_projected, edge_vectors_projected)
    metric_tensors_rep = metric_tensors.unsqueeze(1).repeat(1, 3, 1, 1)
    edge_lengths = torch.sqrt((metric_tensors_rep * edges_vectors_mul).sum(3).sum(2))
    return edge_lengths


def compute_UVs_from_metric_tensors_GIF(
    metric_tensors,
    vertices,
    faces,
    exe_path,
    cache_path,
    interior_faces: int,
    boundary_segment_size: int,
    curvature_meta_vertices_rate: float,
    outer_termination_condition_rate: float,
    energy_related_termination_condition: float,
    allow_IDT_fix=True,
    local_basis=None,
):
    start = time.time()
    edge_lengths = compute_EL_from_metric_tensors(metric_tensors, vertices, faces, local_basis)
    end = time.time()
    edge_lengths_computation_time = end - start
    print(f"Edge lengths computation time: {edge_lengths_computation_time}")

    assert compute_triangle_inequality_satisfying_rate(edge_lengths) == 1, (
        "Triangle inequality is not satisfied after fixing the predicted metric tensor!"
    )
    res_GIF = run_GIF(
        vertices, faces, exe_path, cache_path, interior_faces,
        boundary_segment_size, curvature_meta_vertices_rate,
        outer_termination_condition_rate, energy_related_termination_condition,
        edge_lengths.cpu().numpy(), allow_IDT_fix,
    )

    res_GIF[3] = res_GIF[3] + edge_lengths_computation_time
    return res_GIF


def compute_triangle_inequality_satisfying_rate(edge_lengths):
    triangle_inequality_criteria = torch.full_like(edge_lengths, False)
    triangle_inequality_criteria[:, 0] = edge_lengths[:, 0] + edge_lengths[:, 1] > edge_lengths[:, 2]
    triangle_inequality_criteria[:, 1] = edge_lengths[:, 1] + edge_lengths[:, 2] > edge_lengths[:, 0]
    triangle_inequality_criteria[:, 2] = edge_lengths[:, 2] + edge_lengths[:, 0] > edge_lengths[:, 1]
    triangle_inequality_holds = triangle_inequality_criteria.all(axis=1)
    if triangle_inequality_holds.sum() == triangle_inequality_holds.shape[0]:
        return 1
    triangle_inequality_holds_rate = triangle_inequality_holds.sum() / triangle_inequality_holds.shape[0]
    return triangle_inequality_holds_rate
