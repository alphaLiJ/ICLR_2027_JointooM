"""Isolated reference dependencies and native extension; no global environment edits."""
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(os.environ['JOINTLOOM_REPOSITORY']) / 'rebuttal')
                if 'JOINTLOOM_REPOSITORY' in os.environ else str(WORK.parent))
from repository_paths import REPO, ARTIFACTS as ALL_ARTIFACTS, PYTHON, input_path, source_paths, installed_pogema

ROOT = REPO
ARTIFACTS = input_path('JOINTLOOM_RL_ARTIFACTS', ALL_ARTIFACTS / 'costtracer-rl')
REFERENCE = input_path('JOINTLOOM_COSTTRACER_REFERENCE', ARTIFACTS.parent / 'costtracer-reference')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
POGEMA_SITE = installed_pogema('1.2.2')
source_paths(native=input_path('JOINTLOOM_NATIVE_DIR', REPO / 'rebuttal/online-rl/native/src'),
             extra=(POGEMA_SITE, ARTIFACTS / 'deps'))
