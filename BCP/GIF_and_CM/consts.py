"""Repository-relative defaults; override paths without modifying source files."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = Path(os.environ.get("BCP_BIN_DIR", REPO_ROOT / "build" / "bin")).resolve()
CACHE_DIR = Path(os.environ.get("BCP_CACHE_DIR", REPO_ROOT / ".cache" / "bcp")).resolve()
EXE_SUFFIX = ".exe" if os.name == "nt" else ""

ROOT_PATH_CM_EXE = os.environ.get("BCP_CM_EXE", str(BIN_DIR / ("Parameterization_cmd" + EXE_SUFFIX)))
ROOT_PATH_GIF_EXE = os.environ.get("BCP_GIF_EXE", str(BIN_DIR / ("GIF" + EXE_SUFFIX)))
ROOT_PATH_PARALLEL_TRANSPORT_EXE = os.environ.get("BCP_PT_EXE", str(BIN_DIR / ("PT" + EXE_SUFFIX)))
ROOT_DIR_CACHE_CM = str(CACHE_DIR / "cm")
ROOT_DIR_CACHE_GIF = str(CACHE_DIR / "gif")
ROOT_DIR_CACHE_MESHLIB = str(CACHE_DIR / "meshlib")
ROOT_DIR_CACHE_PARALLEL_TRANSPORT = str(CACHE_DIR / "pt")

INTERIOR_FACES_IN_GIF = -1
BOUNDARY_SEGMENT_SIZE_IN_GIF = -1
CURVATURE_META_VERTICES_RATE_IN_GIF = 0.001
OUTER_TERMINATION_CONDITION_RATE_IN_GIF = 0.1
ENERGY_RELATED_TERMINATION_CONDITION_IN_GIF = 0.0
