import numpy as np

COMPARISON_TOLERANCE = 1e-13


def extract_k(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray):
    paramc = uvs[:, 0] + 1j * uvs[:, 1]

    i_1 = faces[:, 0]
    i_2 = faces[:, 1]
    i_3 = faces[:, 2]

    p1 = vertices[i_1, :]
    p2 = vertices[i_2, :]
    p3 = vertices[i_3, :]

    u1 = p3 - p2
    u2 = p1 - p2

    cross_prod = np.sqrt(np.sum((np.cross(u1, u2, axis=1) ** 2), axis=1))

    u1_length = np.sqrt(np.sum((u1**2), axis=1))
    dot = np.sum((u1 * u2), axis=1)

    triangleAreas = cross_prod / 2

    e1 = u1_length
    e3 = -(dot + 1j * cross_prod) / u1_length
    e2 = -e1 - e3

    t1 = -1j * e1
    t2 = -1j * e2
    t3 = -1j * e3

    f1 = paramc[i_1]
    f2 = paramc[i_2]
    f3 = paramc[i_3]

    fzbarAbs = abs((f1 * t1 + f2 * t2 + f3 * t3) / (4 * triangleAreas))
    fzAbs = abs((f1 * np.conj(t1) + f2 * np.conj(t2) + f3 * np.conj(t3)) / (4 * triangleAreas))

    validIndices = fzAbs != 0

    k = np.zeros((faces.shape[0], 1))
    k[fzAbs == 0, 0] = 1
    k[validIndices, 0] = fzbarAbs[validIndices] / fzAbs[validIndices]
    k = k

    sigma_1 = fzAbs + fzbarAbs
    sigma_2 = fzAbs - fzbarAbs

    return (
        k,
        triangleAreas.reshape((-1, 1)),
        sigma_1.reshape((-1, 1)),
        sigma_2.reshape((-1, 1)),
    )


def write_obj_with_UVs(filename, vertices, faces, uvs):
    # This function saves an obj file with UVs, with double precision for the UVs
    with open(filename, "w") as file:
        # Write vertices
        for vertex in vertices:
            file.write(f"v {' '.join(f'{coord:.16f}' for coord in vertex)}\n")

        # Write texture coordinates
        for uv_coord in uvs:
            file.write(f"vt {' '.join(f'{coord:.16f}' for coord in uv_coord)}\n")

        # Write faces
        for face in faces:
            # OBJ format uses 1-based indexing for vertices and texture coordinates
            file.write(f"f {' '.join(f'{face[i] + 1}/{face[i] + 1}' for i in range(len(face)))}\n")

