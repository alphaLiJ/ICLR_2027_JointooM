"""Official CPU input conversion at sample access; raw source files only.

Contexts retain raw mmaps and lazily generated map/goal distances, never a
frame's expanded observation, graph, edges or model input.
"""
import official_bootstrap as bootstrap
from collections import OrderedDict
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch


class RawCase:
    def __init__(self,directory,args,dataset_kwargs):
        from pogema import GridConfig
        from pogema.grid import Grid
        from magat_plus.runtime_data_generation import get_runtime_data_generator
        path=Path(directory)
        self.compact=np.load(path/'compact.npy',mmap_mode='r')
        self.terminated=np.load(path/'terminated.npy',mmap_mode='r')
        self.grid_source=np.load(path/'grid.npy',mmap_mode='r')
        metadata=json.loads((path/'result.json').read_text())
        if not metadata.get('occupancy_in_record',False):
            raise ValueError('Raw trajectory lacks its Pogema occupancy context: '+directory)
        cfg=GridConfig(**json.loads((path/'grid_config.json').read_text()))
        self.grid=Grid(cfg)
        self.env=SimpleNamespace(grid=self.grid,num_agents=cfg.num_agents,grid_config=cfg)
        self.generator=get_runtime_data_generator(cfg,args,dataset_kwargs)

    def build(self,step):
        rows=self.compact[step]
        self.grid.positions_xy=rows[:,2:4].astype(int).tolist()
        self.grid.finishes_xy=rows[:,4:6].astype(int).tolist()
        self.grid.positions.fill(0)
        visible=(rows[:,7]&2)!=0
        positions=rows[:,2:4].astype(np.int64)
        self.grid.positions[positions[visible,0],positions[visible,1]]=1
        observations=[dict(obstacles=self.grid.get_obstacles_for_agent(i),
            agents=self.grid.get_positions(i),global_xy=self.grid.positions_xy[i],
            global_target_xy=self.grid.finishes_xy[i]) for i in range(len(rows))]
        data=self.generator(observations,self.env)
        data.y=torch.from_numpy(rows[:,6].astype(np.int64))
        data.terminated=torch.from_numpy(np.asarray(self.terminated[step],dtype=bool).copy())
        return data


class OnlineGraphDataset(torch.utils.data.Dataset):
    active_args=None
    active_stats_directory=None

    @classmethod
    def configure(cls,args,stats_directory=None):
        cls.active_args=args;cls.active_stats_directory=stats_directory

    def __init__(self,dense_dataset,**kwargs):
        from configuration import configuration
        self.frames=list(dense_dataset[0])
        self.args=self.active_args or configuration()
        self.stats_directory=self.active_stats_directory
        self.dataset_kwargs={k:kwargs[k] for k in ('use_edge_attr','use_edge_attr_for_messages') if k in kwargs}
        self.phase='unspecified';self.owner=None;self.contexts=OrderedDict()
        self.graphs=0;self.agents=0;self.build_seconds=0.;self.last_stats=0.

    def __len__(self):return len(self.frames)

    def __getitem__(self,index):
        started=time.perf_counter();pid=os.getpid()
        if self.owner!=pid:
            self.owner=pid;self.contexts=OrderedDict()
            self.graphs=0;self.agents=0;self.build_seconds=0.;self.last_stats=0.
            if self.stats_directory:
                from multiprocessing.util import Finalize
                self._finalizer=Finalize(None,self.flush,exitpriority=10)
        frame=self.frames[index]
        if frame.directory in self.contexts:
            case=self.contexts.pop(frame.directory)
        else:
            case=RawCase(frame.directory,self.args,self.dataset_kwargs)
        self.contexts[frame.directory]=case
        if len(self.contexts)>4:self.contexts.popitem(last=False)
        data=case.build(frame.step)
        self.graphs+=1;self.agents+=len(data.y);self.build_seconds+=time.perf_counter()-started
        if self.stats_directory and (self.graphs%64==0 or time.perf_counter()-self.last_stats>15):
            self.flush()
        return data

    def flush(self):
        if not self.stats_directory or not self.owner:return
        directory=Path(self.stats_directory);directory.mkdir(parents=True,exist_ok=True)
        value=dict(pid=self.owner,phase=self.phase,graphs_built=self.graphs,agents_built=self.agents,
            cpu_build_seconds=self.build_seconds,cached_raw_cases=len(self.contexts),
            source_files=['compact.npy','terminated.npy','grid.npy','grid_config.json','result.json'],
            per_frame_model_inputs_cached=False)
        path=directory/f'{self.owner}-{self.phase}.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
        self.last_stats=time.perf_counter()
