import bootstrap
import json
import mmap
import os
from pathlib import Path
import numpy as np
import yaml
from scipy.ndimage import label
from configuration import SPEC

TRAIN_MAPS = SPEC['environment']['train_map_ids']


def map_cells(name,catalog):
    grid=np.array([[c in '#@T' for c in row] for row in catalog[name].splitlines()],np.int32)
    components,count=label(grid==0)
    sizes=np.bincount(components.ravel());sizes[0]=0
    cells=np.argwhere(components==sizes.argmax()).astype(np.int16)
    grid=np.where(components==sizes.argmax(),0,1).astype(np.int32)
    return grid,cells


def make_tasks(seed=0, envs=16, agents=128, bank=SPEC['environment']['task_bank_episodes'], horizon=512, map_ids=None):
    catalog = yaml.safe_load((bootstrap.ROOT/'maps/maps.yaml').read_text())
    names = map_ids or TRAIN_MAPS
    grids, positions, goals, streams, chosen = [],[],[],[],[]
    rng = np.random.default_rng(seed+10000)
    for e in range(envs):
        name = names[e%len(names)]
        grid,cells=map_cells(name,catalog)
        assert len(cells)>=2*agents
        pairs=np.stack([cells[rng.choice(len(cells),2*agents,replace=False)] for _ in range(bank)])
        positions.append(pairs[:,:agents]);goals.append(pairs[:,agents:])
        # Independent uniform component goals, as Pogema 1.2.2 generate_new_target.
        streams.append(cells[rng.integers(len(cells),size=(bank,agents,horizon+1))])
        grids.append(grid);chosen.append(name)
    return dict(grids=grids, positions=np.stack(positions,axis=1), goals=np.stack(goals,axis=1),
                streams=np.stack(streams,axis=1), names=chosen, horizon=horizon,
                seed=seed, agents=agents, envs=envs)


def write_task_bank(folder,seed=0,envs=16,agents=128,bank=2048,horizon=512,map_ids=None,
                    chunk_episodes=64,progress=None):
    """Preserve make_tasks RNG order while bounding resident memory during generation.

    The final NPY layout is unchanged. Flush and release each generated chunk
    so a large bank is neither duplicated by stack nor accumulated as dirty RAM.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    catalog=yaml.safe_load((bootstrap.ROOT/'maps/maps.yaml').read_text())
    names=map_ids or TRAIN_MAPS;chosen=[];rng=np.random.default_rng(seed+10000)
    positions=np.lib.format.open_memmap(folder/'positions.npy',mode='w+',dtype=np.int16,shape=(bank,envs,agents,2))
    goals=np.lib.format.open_memmap(folder/'goals.npy',mode='w+',dtype=np.int16,shape=(bank,envs,agents,2))
    streams=np.lib.format.open_memmap(folder/'streams.npy',mode='w+',dtype=np.int16,shape=(bank,envs,agents,horizon+1,2))
    fd=os.open(folder/'streams.npy',os.O_RDONLY)
    try:
        for e in range(envs):
            name=names[e%len(names)];grid,cells=map_cells(name,catalog)
            assert len(cells)>=2*agents
            pairs=np.stack([cells[rng.choice(len(cells),2*agents,replace=False)] for _ in range(bank)])
            positions[:,e]=pairs[:,:agents];goals[:,e]=pairs[:,agents:]
            for begin in range(0,bank,chunk_episodes):
                end=min(begin+chunk_episodes,bank)
                chunk=cells[rng.integers(len(cells),size=(end-begin,agents,horizon+1))]
                streams[begin:end,e]=chunk;del chunk
                streams.flush()
                streams._mmap.madvise(mmap.MADV_DONTNEED)
                os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                if progress:progress(dict(env_id=e,envs=envs,episodes_written=end,bank_episodes=bank,
                    fraction=(e*bank+end)/(envs*bank)))
            np.save(folder/f'grid-{e}.npy',grid);chosen.append(name)
        positions.flush();goals.flush()
    finally:
        os.close(fd)
        for array in (positions,goals,streams):array._mmap.close()
    metadata=dict(names=chosen,horizon=horizon,seed=seed,agents=agents,envs=envs,
        task_bank_episodes=bank,preparation='bounded-memory direct NPY mmap; original RNG draw order',
        chunk_episodes=chunk_episodes)
    tmp=folder/'tasks.json.tmp';tmp.write_text(json.dumps(metadata,indent=2));tmp.replace(folder/'tasks.json')
    return folder


def subset(tasks, ids):
    return dict(tasks, grids=[tasks['grids'][i] for i in ids],
                positions=tasks['positions'][:,ids].copy(),goals=tasks['goals'][:,ids].copy(),
                streams=tasks['streams'][:,ids].copy(), names=[tasks['names'][i] for i in ids],envs=len(ids))
