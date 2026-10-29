import torch
from GIF_and_CM.device import DEVICE


def build_face_tangent_frames(verts: torch.Tensor, faces: torch.Tensor) -> torch:
    """
    compute per-face reference coordinate frames
    expected tensors on CPU, returns a tensor on the selected device
    """
    verts = verts.to(DEVICE)
    faces = faces.to(DEVICE)
    tri_edge_1 = verts[faces[:, 1], :] - verts[faces[:, 0], :]
    tri_edge_2 = verts[faces[:, 2], :] - verts[faces[:, 0], :]
    axis_x = tri_edge_1
    axis_n = torch.cross(tri_edge_1, tri_edge_2)
    axis_y = torch.cross(axis_n, axis_x)
    # normalize
    axis_x = axis_x / torch.linalg.norm(axis_x, axis=1).unsqueeze(1)
    axis_y = axis_y / torch.linalg.norm(axis_y, axis=1).unsqueeze(1)
    axis_n = axis_n / torch.linalg.norm(axis_n, axis=1).unsqueeze(1)
    frames = torch.stack((axis_x, axis_y, axis_n), dim=-2)
    return frames.to(torch.float64)