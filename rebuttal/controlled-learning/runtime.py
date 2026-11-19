"""Isolated DAgger experiment; reuse the verified official MAGAT+ environment."""
from pathlib import Path
import json
import os
import sys
import time

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ['JOINTLOOM_REPOSITORY']) / 'rebuttal')
                if 'JOINTLOOM_REPOSITORY' in os.environ else str(WORK.parent))
from repository_paths import REPO, ARTIFACTS, PYTHON, input_path
PAPER = REPO
PREVIOUS = REPO / 'rebuttal/official-finetuning/gpu'
sys.path.insert(0, str(PREVIOUS))
import official_bootstrap as legacy

OPTIMIZED_NATIVE = input_path('JOINTLOOM_NATIVE_DIR', REPO / 'rebuttal/gpu-control/native_optimized/src')
sys.path.insert(0, str(OPTIMIZED_NATIVE))
ART = input_path('JOINTLOOM_DAGGER_ARTIFACTS', ARTIFACTS / 'magat-controlled')
DESKTOP = input_path('JOINTLOOM_REPORT_DIR', ART / 'reports')


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def publish(stage, **fields):
    value = dict(stage=stage, updated=time.time(), pid=os.getpid(), **fields)
    for directory in (ART, DESKTOP):
        json_write(directory / 'status.json', value)
    (DESKTOP / 'progress.txt').write_text('\n'.join(f'{k}: {v}' for k, v in value.items()) + '\n')


def configuration(agents=None):
    from configuration import configuration as original
    args = original()
    if os.environ.get('JOINTLOOM_DAGGER_MAP_DIR'):
        args.map_dir = os.environ['JOINTLOOM_DAGGER_MAP_DIR']
    if agents is not None:
        args.num_agents = f'{agents}+{agents}'
    return args


def load_model(checkpoint=None, seed=0):
    import random
    import numpy as np
    import torch
    from magat_plus.magat.agents import get_model
    args = configuration()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, kwargs = get_model(args, torch.device('cuda'))
    path = checkpoint or args.pretrain_weights_path
    model.load_state_dict(torch.load(path, map_location='cuda', weights_only=False))
    model.eval()
    return model, kwargs


def task_config(seed, agents):
    from grid_config_generator import grid_config_generator_factory
    return grid_config_generator_factory(configuration(agents))(int(seed), map_id=0)


def packed_state(observations):
    import numpy as np
    a = len(observations)
    rows = np.zeros((a, 8), dtype=np.uint16)
    rows[:, 1] = np.arange(a)
    rows[:, 2:4] = np.asarray([o['global_xy'] for o in observations])
    rows[:, 4:6] = np.asarray([o['global_target_xy'] for o in observations])
    rows[:, 7] = np.asarray([o['agents'][5, 5] > 0 for o in observations], dtype=np.uint16) * 2
    return rows
