"""Live integration gate: generate a disk and run the complete BCP CLI.

No dependency/runtime skips are allowed here. Run inside a full Pixi environment.
This is a numerical smoke test, not a claim of paper-scale performance parity.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import igl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    n = 12
    xy = np.linspace(-1, 1, n)
    vertices = np.array([[x,y,0.15*(x*x+y*y)] for y in xy for x in xy], dtype=np.float64)
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            faces += [[a,a+1,a+n+1],[a,a+n+1,a+n]]
    faces = np.array(faces, dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="bcp full pipeline ") as work:
        source = Path(work)/"input.obj"
        igl.write_triangle_mesh(str(source), vertices, faces)
        for idt in (False, True):
            output = Path(work)/f"output_{idt}.obj"
            command = [
                sys.executable, str(ROOT/"scripts/parameterize.py"),
                str(source), str(output), "--method", "bcp",
                "--target-faces", "80", "--device", os.environ.get("BCP_DEVICE", "cpu"),
                "--gif-device", os.environ.get("BCP_GIF_DEVICE", "cpu"),
                "--gif-backend", os.environ.get("BCP_GIF_BACKEND", "auto"),
            ]
            if idt:
                command.append("--idt")
            subprocess.run(command, cwd=ROOT, check=True, timeout=600)
            uv = np.array([
                list(map(float, line.split()[1:3]))
                for line in output.read_text().splitlines() if line.startswith("vt ")
            ])
            if uv.shape != (len(vertices),2) or not np.isfinite(uv).all():
                raise AssertionError("Invalid complete-pipeline UV output")
            a = uv[faces[:,1]]-uv[faces[:,0]]
            b = uv[faces[:,2]]-uv[faces[:,0]]
            if not np.all(a[:,0]*b[:,1]-a[:,1]*b[:,0]>0):
                raise AssertionError("Complete pipeline produced flipped/collapsed triangles")
    print("Both complete BCP variants passed on the generated curved disk")


if __name__ == "__main__":
    main()
