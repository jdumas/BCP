"""CM wrappers using the original MAT protocol, with no Windows shell."""
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import igl
import numpy as np
import scipy.io as sio

from process_utils import run_native

TIMEOUT_SECONDS = 3600


def save_data_for_CM(vertices, faces, cache_path, init_UVs=None):
    cache = Path(cache_path).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    prefix = cache / uuid4().hex
    source = str(prefix) + ".obj"
    igl.write_triangle_mesh(source, vertices, faces)
    initial = None
    if init_UVs is not None:
        initial = str(prefix) + "_init.obj"
        igl.write_triangle_mesh(initial, np.column_stack((init_UVs, np.zeros(len(init_UVs)))), faces)
    return source, str(prefix) + "_res.mat", str(prefix) + "_meta.txt", initial


def _run_CM(vertices, faces, exe_path, cache_path, *, energy=None, seconds=None, initial=None):
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cm-", dir=cache_path) as work:
        source, result, meta, init_file = save_data_for_CM(vertices, faces, work, initial)
        arguments = [source, "", result, meta]
        if energy is not None or seconds is not None or initial is not None:
            arguments.append(2 if energy is None else energy)
        if seconds is not None or initial is not None:
            arguments.append(100000 if seconds is None else seconds)
        if init_file is not None:
            arguments.append(init_file)
        run_native(exe_path, arguments, TIMEOUT_SECONDS)
        uvs = sio.loadmat(result)["uvs"][:, :2]
        with open(meta, encoding="utf-8") as file:
            metadata = dict(line.strip().split(": ", 1) for line in file if ": " in line)
        return (
            uvs,
            float(metadata["Total time consumption"]),
            float(metadata["Newton iteration count"]),
            float(metadata["Final E_SD"]),
        )


def run_CM(V, F, exe_path, cache_path):
    return _run_CM(V, F, exe_path, cache_path)


def run_CM_with_target_energy(V, F, target_energy, exe_path, cache_path):
    return _run_CM(V, F, exe_path, cache_path, energy=target_energy)


def run_CM_with_target_runtime(V, F, target_time, exe_path, cache_path):
    return _run_CM(V, F, exe_path, cache_path, seconds=target_time)


def run_CM_with_customized_init(V, F, init_UVs, exe_path, cache_path):
    return _run_CM(V, F, exe_path, cache_path, initial=init_UVs)
