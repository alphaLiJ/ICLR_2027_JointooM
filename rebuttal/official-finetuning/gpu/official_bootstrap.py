"""Repository-local MAGAT+ sources and explicit external input locations."""
from pathlib import Path
import os
import sys
import resource

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ['JOINTLOOM_REPOSITORY']) / 'rebuttal')
                if 'JOINTLOOM_REPOSITORY' in os.environ else str(WORK.parents[1]))
from repository_paths import REPO, ARTIFACTS, PYTHON, input_path, source_paths, installed_pogema

PAPER = SOURCE = REPO
ART = input_path('JOINTLOOM_MAGAT_ARTIFACTS', ARTIFACTS / 'magat-official')
UPSTREAM = input_path('JOINTLOOM_MAGAT_UPSTREAM', REPO / 'rebuttal/official-finetuning/upstream-learning')
ONLINE = REPO / 'rebuttal/online-il'
DESKTOP = input_path('JOINTLOOM_REPORT_DIR', ART / 'reports' / WORK.name)
LACAM_SOURCE = input_path('JOINTLOOM_LACAM_SOURCE', REPO / 'src/real_expert_alg/lacam')
LACAM_LIBRARY = input_path('JOINTLOOM_LACAM_LIBRARY', LACAM_SOURCE / 'liblacam.so')
CHECKPOINT = input_path('JOINTLOOM_MAGAT_CHECKPOINT', ART / 'checkpoints/magat_plus_pretrained/195.pt')

allowed = os.sched_getaffinity(0) & set(range(16))
if allowed:
    os.sched_setaffinity(0, allowed)
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
_target = 65536 if _hard == resource.RLIM_INFINITY else min(65536, _hard)
if _soft < _target:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'
POGEMA_SITE = installed_pogema('1.3.0')
source_paths(native=input_path('JOINTLOOM_NATIVE_DIR', ONLINE / 'native/src'),
             upstream=UPSTREAM, extra=(POGEMA_SITE, ONLINE))
import magat_plus
magat_plus.__path__ = [str(UPSTREAM / 'magat_plus')]
