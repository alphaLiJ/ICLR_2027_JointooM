"""A compact mmap task bank shared by real SF and JointLoom runs."""
import bootstrap
import json
from pathlib import Path
import numpy as np
from tasks import write_task_bank

SF_ROOT=bootstrap.ARTIFACTS/'sample-factory-native'
_CACHE={}


def bank_path(seed,num_envs=16,bank_episodes=64):
    if bank_episodes!=64:
        return SF_ROOT/'tasks'/f'bank-{bank_episodes}'/f'envs-{num_envs}'/f'seed-{seed}'
    base=SF_ROOT/'tasks' if num_envs==16 else SF_ROOT/'tasks'/f'envs-{num_envs}'
    return base/f'seed-{seed}'


def prepare(seed,num_envs=16,bank_episodes=64,progress=None):
    root=bank_path(seed,num_envs,bank_episodes)
    if (root/'tasks.json').exists():return root
    return write_task_bank(root,seed=seed,envs=num_envs,bank=bank_episodes,progress=progress)


def load(seed,env_id=None,num_envs=16,bank_episodes=64):
    key=(seed,num_envs,bank_episodes)
    if key not in _CACHE:
        root=bank_path(seed,num_envs,bank_episodes)
        data=json.loads((root/'tasks.json').read_text())
        for k in ('positions','goals','streams'):data[k]=np.load(root/(k+'.npy'),mmap_mode='r')
        data['grids']=[np.load(root/f'grid-{e}.npy',mmap_mode='r') for e in range(data['envs'])]
        _CACHE[key]=data
    tasks=_CACHE[key]
    if env_id is None:return tasks
    e=int(env_id)
    return dict(tasks,grids=[tasks['grids'][e]],positions=tasks['positions'][:,e:e+1],
                goals=tasks['goals'][:,e:e+1],streams=tasks['streams'][:,e:e+1],
                names=[tasks['names'][e]],envs=1)
