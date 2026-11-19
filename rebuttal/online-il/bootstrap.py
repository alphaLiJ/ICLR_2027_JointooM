"""Repository sources and the isolated online CUDA extension."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repository_paths import REPO, input_path, source_paths

ROOT = SOURCE = REPO
NATIVE = input_path('JOINTLOOM_NATIVE_DIR', Path(__file__).resolve().parent / 'native/src')


def activate():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    source_paths(native=NATIVE)


activate()
