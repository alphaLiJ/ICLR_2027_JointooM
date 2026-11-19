"""Direct fixed-policy task metrics; no solver search or checkpoint selection."""
import numpy as np
import torch
import yaml
from scipy.ndimage import label
from bootstrap import SOURCE
from policies import MagatPolicy
from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator


def evaluation_maps(seed):
    catalog=yaml.safe_load((SOURCE/'maps/maps.yaml').read_text());maps=[]
    for name in ['test-mazes-s40_wc4_od30','test-mazes-s41_wc5_od50','test-mazes-s42_wc7_od30','test-mazes-s43_wc2_od45']:
        grid=np.array([[c in '#@T' for c in row] for row in catalog[name].strip().splitlines()],dtype=np.uint8)
        maps.append(('maze',name,grid))
    for i in range(4):
        grid=(np.random.default_rng(seed+i).random((65,65))<.2).astype(np.uint8)
        maps.append(('random',f'random-e4-seed{seed+i}',grid))
    for i in range(4):
        grid=np.zeros((65,65),dtype=np.uint8)
        for x in range(3+i%2,59,8+i//2):
            for y in range(3+i//2,57,12-i%2):grid[x:x+4,y:y+8]=1
        maps.append(('warehouse',f'warehouse-e4-shelves65-v{i}',grid))
    return maps


def make_evaluation_tasks(config,agents):
    grids=[];positions=[];goals=[];metadata=[]
    maps=evaluation_maps(config['task_seed'])
    if agents==512:maps=[m for m in maps if m[0]=='maze']
    for mi,(family,name,grid) in enumerate(maps):
        components,_=label(grid==0);sizes=np.bincount(components.ravel());sizes[0]=0
        cells=np.argwhere(components==sizes.argmax()).astype(np.int16)
        for task in range(config['tasks_per_map']):
            rng=np.random.default_rng(np.random.SeedSequence([config['task_seed'],agents,mi,task]))
            ids=rng.choice(len(cells),2*agents,replace=False)
            grids.append(grid);positions.append(cells[ids[:agents]]);goals.append(cells[ids[agents:]])
            metadata.append(dict(family=family,map_name=name,task=task,agents=agents))
    return np.stack(grids),np.stack(positions),np.stack(goals),metadata


@torch.no_grad()
def evaluate_checkpoint(checkpoint,tasks,config):
    import grid_world_cpp as ext
    grids,positions,goals,metadata=tasks;n,a=positions.shape[:2];device=config['device'];horizon=config['horizon']
    native,extents=stack_grids_for_compiled_simulator(grids,device=device)
    sim=ext.GridWorldSimulator(native,a,0,native.numel(),config['task_seed'],'standard_mapf',horizon+1)
    sim.set_real_map_extents(extents);sim.pyg_builder_mode='local_gather';sim.pyg_local_gather_impl='auto'
    sim.load_state(torch.as_tensor(positions,device=device),torch.as_tensor(goals,device=device),
        torch.zeros((n,a),dtype=torch.uint8,device=device),torch.zeros(n,dtype=torch.int32,device=device))
    policy=MagatPolicy(checkpoint,device)
    from native_extent_inputs import NativeExtentBatchBuilder
    policy.builder=NativeExtentBatchBuilder([g.shape for g in grids],a,device)
    first=torch.full((n,a),-1,dtype=torch.int32,device=device)
    for step in range(1,horizon+1):
        sim.update_derived_state()
        action=policy.act_resident(sim,a,max(1,config['policy_nodes_per_batch']//a))
        sim.update_actions(action);sim.step_sim_only()
        reached=(sim.cur_x==sim.goal_x)&(sim.cur_y==sim.goal_y)
        first=torch.where((first<0)&reached,step,first)
    first=first.cpu().numpy();rows=[]
    for m,times in zip(metadata,first):
        success=bool(np.all(times>=0));arrived=times[times>=0]
        rows.append(dict(**m,success=success,arrived_agents=len(arrived),arrival_fraction=len(arrived)/a,
            end_step=int(arrived.max()) if success else horizon,
            soc=int(arrived.sum()) if success else None,makespan=int(arrived.max()) if success else None,
            mean_arrival_time_if_arrived=float(arrived.mean()) if len(arrived) else None,
            horizon_capped_arrival_sum=int(np.where(times>=0,times,horizon).sum())))
    return rows,first


def summarize_quality(rows):
    summaries=[]
    keys=sorted(set((r['method'],r['agents'],r['family']) for r in rows))
    for method,agents,family in keys:
        group=[r for r in rows if (r['method'],r['agents'],r['family'])==(method,agents,family)]
        summaries.append(dict(method=method,agents=agents,family=family,instances=len(group),
            successes=sum(r['success'] for r in group),success_rate=float(np.mean([r['success'] for r in group])),
            mean_arrival_fraction=float(np.mean([r['arrival_fraction'] for r in group])),
            per_seed={str(s):dict(success_rate=float(np.mean([r['success'] for r in group if r['seed']==s])),
                arrival_fraction=float(np.mean([r['arrival_fraction'] for r in group if r['seed']==s]))) for s in sorted(set(r['seed'] for r in group))}))
    return summaries
