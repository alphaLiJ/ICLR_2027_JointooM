"""Local CAMAR adaptation; use existing frozen policy and CUDA runtime sources."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from repository_paths import REPO, input_path, source_paths

ROOT = FROZEN = REPO
ONLINE = REPO / 'rebuttal/online-il'

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
source_paths(native=input_path('JOINTLOOM_NATIVE_DIR', HERE / 'native_optimized/src'),
             extra=(ONLINE, HERE / 'third_party/CAMAR/src'))
