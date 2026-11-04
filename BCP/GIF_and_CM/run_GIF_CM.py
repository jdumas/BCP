import time

import igl
import numpy as np
import torch
from GIF_and_CM.device import DEVICE

from GIF_and_CM.consts import ROOT_PATH_CM_EXE, ROOT_DIR_CACHE_CM, ROOT_PATH_GIF_EXE, ROOT_DIR_CACHE_GIF, \
    INTERIOR_FACES_IN_GIF, BOUNDARY_SEGMENT_SIZE_IN_GIF, CURVATURE_META_VERTICES_RATE_IN_GIF, \
    OUTER_TERMINATION_CONDITION_RATE_IN_GIF, ENERGY_RELATED_TERMINATION_CONDITION_IN_GIF
from MeshProcessor import run_simplification
from utils import build_face_tangent_frames
from compute_parallel_transport_intrinsic import \
    compute_parallel_transport_intrinsic_vertex_to_face, interpolate_faces_dimensional_data_to_vertices_torch
from run_CM_utils import run_CM
from run_GIF_utils import compute_UVs_from_metric_tensors_GIF



def get_gt_mu(f, s_v, t_v, frames_faces):
    f, s_v, t_v = f.to(DEVICE), s_v.to(DEVICE), t_v.to(DEVICE)
    f_s_v = torch.zeros((f.shape[0], 2, 3), dtype=torch.float64).to(DEVICE)
    f_s_v[:, 0, :] = s_v[f[:, 1]] - s_v[f[:, 0]]
    f_s_v[:, 1, :] = s_v[f[:, 2]] - s_v[f[:, 0]]
    f_t_v = torch.zeros((f.shape[0], 2, 3), dtype=torch.float64).to(DEVICE)
    f_t_v[:, 0, :] = t_v[f[:, 1]] - t_v[f[:, 0]]
    f_t_v[:, 1, :] = t_v[f[:, 2]] - t_v[f[:, 0]]
    local_basis = frames_faces[:, :2, :]
    f_s_v_local = torch.einsum("abc, acd -> abd", f_s_v, torch.transpose(local_basis, 2, 1))
    v1 = f_s_v_local[:, 0, 0] + 1j * f_s_v_local[:, 0, 1]
    v2 = f_s_v_local[:, 1, 0] + 1j * f_s_v_local[:, 1, 1]
    v1_t = f_t_v[:, 0, 0] + 1j * f_t_v[:, 0, 1]
    v2_t = f_t_v[:, 1, 0] + 1j * f_t_v[:, 1, 1]
    mu = (v1 * v2_t - v2 * v1_t) / (v1_t * v2.conj() - v2_t * v1.conj())
    faces_mu_2d = torch.stack((mu.real, mu.imag), dim=1)
    return faces_mu_2d


def low_res_to_high_res_mu_via_averaged_z_axis(frames_high_res: torch.Tensor, frames_low_res: torch.Tensor, simplified_to_original_ind: np.ndarray):
    cur_frames_low_res_all = frames_low_res[simplified_to_original_ind, :, :]

    z_axis_high_res_all = frames_high_res[:, 2, :]
    z_axis_low_res_all = cur_frames_low_res_all[:, 2, :]
    mean_z_all = (z_axis_high_res_all + z_axis_low_res_all) / 2
    mean_z_all = mean_z_all / torch.linalg.norm(mean_z_all, axis=1).reshape((-1, 1)).repeat(1, 3)
    proj_x_high_res_all = frames_high_res[:, 0, :] - torch.einsum("ab, ab -> a", frames_high_res[:, 0, :],
                                                                  mean_z_all).reshape(-1, 1).repeat(
        (1, 3)) * mean_z_all
    proj_x_low_res_all = cur_frames_low_res_all[:, 0, :] - torch.einsum("ab, ab -> a",
                                                                        cur_frames_low_res_all[:, 0, :],
                                                                        mean_z_all).reshape(-1, 1).repeat(
        (1, 3)) * mean_z_all

    proj_x_high_res_all = proj_x_high_res_all / torch.linalg.norm(proj_x_high_res_all, axis=1).reshape(
        (-1, 1)).repeat(1, 3)
    proj_x_low_res_all = proj_x_low_res_all / torch.linalg.norm(proj_x_low_res_all, axis=1).reshape((-1, 1)).repeat(
        1, 3)

    sign_all = torch.sign(
        torch.einsum("ab, ab -> a", mean_z_all, torch.cross(proj_x_low_res_all, proj_x_high_res_all)))
    angle_all = -sign_all * torch.acos(
        torch.einsum("ab, ab -> a", proj_x_high_res_all, proj_x_low_res_all).clip(-1, 1))

    angle_all[torch.isnan(angle_all)] = 0

    return torch.exp(1j * angle_all)


def low_res_to_high_res_mu_via_planes(frames_high_res: torch.Tensor, frames_low_res: torch.Tensor, intersection_vector: torch.Tensor, simplified_to_original_ind: np.ndarray):
    x1 = torch.sum(intersection_vector * frames_low_res[simplified_to_original_ind][:, 0, :], dim=1)
    y1 = torch.sum(intersection_vector * frames_low_res[simplified_to_original_ind][:, 1, :], dim=1)
    x2 = torch.sum(intersection_vector * frames_high_res[:, 0, :], dim=1)
    y2 = torch.sum(intersection_vector * frames_high_res[:, 1, :], dim=1)
    low_res_plane_angle = torch.arctan2(y1, x1)
    high_res_plane_angle = torch.arctan2(y2, x2)
    return torch.exp(1j * (high_res_plane_angle - low_res_plane_angle))



def low_res_to_high_res_mu_general(frames_high_res: torch.Tensor, frames_low_res: torch.Tensor, low_res_v: torch.tensor, low_res_f: np.ndarray, high_res_v: torch.tensor, high_res_f: np.ndarray, simplified_to_original_ind: np.ndarray):
    n_high_res = torch.cross(high_res_v[high_res_f[:, 0]] - high_res_v[high_res_f[:, 1]], high_res_v[high_res_f[:, 2]] - high_res_v[high_res_f[:, 1]])
    low_res_f_expanded = low_res_f[simplified_to_original_ind]
    n_low_res = torch.cross(low_res_v[low_res_f_expanded[:, 0]] - low_res_v[low_res_f_expanded[:, 1]], low_res_v[low_res_f_expanded[:, 2]] - low_res_v[low_res_f_expanded[:, 1]])

    n_high_res = n_high_res / torch.linalg.norm(n_high_res, dim=1).reshape(-1, 1)
    n_low_res = n_low_res / torch.linalg.norm(n_low_res, dim=1).reshape(-1, 1)

    # Direction of intersection line
    d = torch.cross(n_high_res, n_low_res)

    d_norm = torch.linalg.norm(d, dim=1)
    planes_parallel_ind = torch.argwhere(d_norm < 1e-2).reshape(-1)
    planes_intersecting_ind = torch.argwhere(d_norm >= 1e-2).reshape(-1)
    faces_mu_planes_parallel = low_res_to_high_res_mu_via_averaged_z_axis(frames_high_res[planes_parallel_ind],
                                                                          frames_low_res, simplified_to_original_ind[
                                                                              planes_parallel_ind.cpu().numpy()])

    faces_mu_planes_intersecting = low_res_to_high_res_mu_via_planes(frames_high_res[planes_intersecting_ind], frames_low_res, d[planes_intersecting_ind], simplified_to_original_ind[planes_intersecting_ind.cpu().numpy()])

    res_rotations = torch.zeros(frames_high_res.shape[0], dtype=torch.complex128).to(DEVICE)
    res_rotations[planes_parallel_ind] = faces_mu_planes_parallel
    res_rotations[planes_intersecting_ind] = faces_mu_planes_intersecting

    return res_rotations


def run_GIF_and_CM_with_simplification(v, f):
    # Simplify the mesh
    simplified_v, simplified_f, simplified_to_original, barycentric_coordinates, simplification_time = run_simplification(
        v, f)

    uvs_from_CM, CM_time, CM_iters, simplification_Esd = run_CM(simplified_v, simplified_f, ROOT_PATH_CM_EXE, ROOT_DIR_CACHE_CM)
    # Build faces frames
    start_time1 = time.time()
    simplified_mesh_frames_faces = build_face_tangent_frames(torch.tensor(simplified_v), torch.tensor(simplified_f))
    original_mesh_frames_faces = build_face_tangent_frames(torch.tensor(v), torch.tensor(f))
    simplified_gt_mu = get_gt_mu(torch.from_numpy(simplified_f), torch.from_numpy(simplified_v),
                                 torch.from_numpy(np.hstack((uvs_from_CM, np.zeros((uvs_from_CM.shape[0], 1))))),
                                 simplified_mesh_frames_faces)
    faces_mu_low_res = simplified_gt_mu[:, 0] + 1j * simplified_gt_mu[:, 1]
    rotations_simplified_to_original = low_res_to_high_res_mu_general(original_mesh_frames_faces,
                                                                      simplified_mesh_frames_faces,
                                                                      torch.from_numpy(simplified_v).to(DEVICE),
                                                                      simplified_f,
                                                                      torch.from_numpy(v).to(DEVICE),
                                                                      f,
                                                                      simplified_to_original)

    end_time1 = time.time()
    faces_mu, smoothing_time = smooth_mu(simplified_v, simplified_f, faces_mu_low_res, simplified_mesh_frames_faces,
                                             barycentric_coordinates, simplified_to_original)

    start_time2 = time.time()
    faces_mu = (rotations_simplified_to_original ** 2) * faces_mu
    faces_mu_abs = faces_mu.abs()
    normalization_abs = 0.99 / faces_mu_abs
    above_0_99_mu_ind = faces_mu_abs > 0.99
    faces_mu[above_0_99_mu_ind] = faces_mu[above_0_99_mu_ind] * normalization_abs[above_0_99_mu_ind]
    pred_M = torch.zeros((faces_mu.shape[0], 2, 2)).to(DEVICE)
    sq_abs_plus1_1 = 1 + faces_mu.abs() ** 2
    pred_M[:, 0, 0] = sq_abs_plus1_1 + 2 * faces_mu.real
    pred_M[:, 0, 1] = 2 * faces_mu.imag
    pred_M[:, 1, 0] = pred_M[:, 0, 1]
    pred_M[:, 1, 1] = sq_abs_plus1_1 - 2 * faces_mu.real
    end_time2 = time.time()
    mt_computation_time = end_time1 - start_time1 + end_time2 - start_time2 + smoothing_time

    res_GIF = compute_UVs_from_metric_tensors_GIF(
        pred_M.squeeze(),
        v,
        f,
        ROOT_PATH_GIF_EXE,
        ROOT_DIR_CACHE_GIF,
        INTERIOR_FACES_IN_GIF,
        BOUNDARY_SEGMENT_SIZE_IN_GIF,
        CURVATURE_META_VERTICES_RATE_IN_GIF,
        OUTER_TERMINATION_CONDITION_RATE_IN_GIF,
        ENERGY_RELATED_TERMINATION_CONDITION_IN_GIF,
        False,
        original_mesh_frames_faces.to(torch.float64),
    )

    pred_V, flipsData, IDT_stats, time_GIF, edge_compatibility_data, _ = res_GIF

    return simplification_time, CM_time, mt_computation_time, time_GIF, pred_V, flipsData, CM_iters, simplified_f.shape[0], simplification_Esd, None




def run_GIF_and_CM_with_simplification_with_IDT(v, f):
    # Simplify the mesh
    simplified_v, simplified_f, simplified_to_original, barycentric_coordinates, simplification_time = run_simplification(
        v, f)

    uvs_from_CM, CM_time, CM_iters, simplification_Esd = run_CM(simplified_v, simplified_f, ROOT_PATH_CM_EXE, ROOT_DIR_CACHE_CM)

    # Build faces frames
    start_time1 = time.time()
    simplified_mesh_frames_faces = build_face_tangent_frames(torch.tensor(simplified_v), torch.tensor(simplified_f))
    original_mesh_frames_faces = build_face_tangent_frames(torch.tensor(v), torch.tensor(f))
    simplified_gt_mu = get_gt_mu(torch.from_numpy(simplified_f), torch.from_numpy(simplified_v),
                                 torch.from_numpy(np.hstack((uvs_from_CM, np.zeros((uvs_from_CM.shape[0], 1))))),
                                 simplified_mesh_frames_faces)
    faces_mu_low_res = simplified_gt_mu[:, 0] + 1j * simplified_gt_mu[:, 1]
    rotations_simplified_to_original = low_res_to_high_res_mu_general(original_mesh_frames_faces,
                                                                      simplified_mesh_frames_faces,
                                                                      torch.from_numpy(simplified_v).to(DEVICE),
                                                                      simplified_f,
                                                                      torch.from_numpy(v).to(DEVICE),
                                                                      f,
                                                                      simplified_to_original)
    end_time1 = time.time()
    faces_mu, smoothing_time = smooth_mu(simplified_v, simplified_f, faces_mu_low_res, simplified_mesh_frames_faces, barycentric_coordinates, simplified_to_original)
    start_time2 = time.time()
    faces_mu = (rotations_simplified_to_original ** 2) * faces_mu

    faces_mu_abs = faces_mu.abs()
    normalization_abs = 0.99 / faces_mu_abs
    above_0_99_mu_ind = faces_mu_abs > 0.99
    faces_mu[above_0_99_mu_ind] = faces_mu[above_0_99_mu_ind] * normalization_abs[above_0_99_mu_ind]
    pred_M = torch.zeros((faces_mu.shape[0], 2, 2)).to(DEVICE)
    sq_abs_plus1_1 = 1 + faces_mu.abs() ** 2
    pred_M[:, 0, 0] = sq_abs_plus1_1 + 2 * faces_mu.real
    pred_M[:, 0, 1] = 2 * faces_mu.imag
    pred_M[:, 1, 0] = pred_M[:, 0, 1]
    pred_M[:, 1, 1] = sq_abs_plus1_1 - 2 * faces_mu.real
    end_time2 = time.time()
    mt_computation_time = end_time1 - start_time1 + end_time2 - start_time2 + smoothing_time

    res_GIF = compute_UVs_from_metric_tensors_GIF(
        pred_M.squeeze(),
        v,
        f,
        ROOT_PATH_GIF_EXE,
        ROOT_DIR_CACHE_GIF,
        INTERIOR_FACES_IN_GIF,
        BOUNDARY_SEGMENT_SIZE_IN_GIF,
        CURVATURE_META_VERTICES_RATE_IN_GIF,
        OUTER_TERMINATION_CONDITION_RATE_IN_GIF,
        ENERGY_RELATED_TERMINATION_CONDITION_IN_GIF,
        True,
        original_mesh_frames_faces.to(torch.float64),
    )

    pred_V, flipsData, IDT_stats, time_GIF, edge_compatibility_data, _ = res_GIF

    return simplification_time, CM_time, mt_computation_time, time_GIF, pred_V, flipsData, CM_iters, simplified_f.shape[0], simplification_Esd, IDT_stats

def smooth_mu(simplified_v, simplified_f, faces_mu_low_res, simplified_mesh_frames_faces, barycentric_coordinates, simplified_to_original):
    s_verts_to_faces_transport, t1 = compute_parallel_transport_intrinsic_vertex_to_face(simplified_v, simplified_f, simplified_mesh_frames_faces.cpu().numpy())
    start_time = time.time()
    s_faces_torch = torch.from_numpy(simplified_f).to(DEVICE)
    barycentric_coordinates_torch = torch.from_numpy(barycentric_coordinates).to(DEVICE)
    s_verts_to_faces_transport_torch = torch.from_numpy(s_verts_to_faces_transport).to(DEVICE)
    vertices_mu = interpolate_faces_dimensional_data_to_vertices_torch(
        s_faces_torch,
        faces_mu_low_res.reshape(-1, 1),
        1 / s_verts_to_faces_transport_torch ** 2,
        torch.from_numpy(igl.doublearea(simplified_v, simplified_f)).to(DEVICE)
    )
    faces_mu = barycentric_coordinates_torch[:, 0] * vertices_mu[simplified_f[simplified_to_original, 0]].reshape(-1) * s_verts_to_faces_transport_torch[simplified_to_original, 0] ** 2 +\
               barycentric_coordinates_torch[:, 1] * vertices_mu[simplified_f[simplified_to_original, 1]].reshape(-1) * s_verts_to_faces_transport_torch[simplified_to_original, 1] ** 2 +\
               barycentric_coordinates_torch[:, 2] * vertices_mu[simplified_f[simplified_to_original, 2]].reshape(-1) * s_verts_to_faces_transport_torch[simplified_to_original, 2] ** 2
    end_time = time.time()
    total_time = end_time - start_time + t1

    return faces_mu, total_time
