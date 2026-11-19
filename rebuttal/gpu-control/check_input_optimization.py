"""Reference-kernel and policy parity for the optimized observation/input path."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import os
from pathlib import Path
ARTIFACT=input_path('JOINTLOOM_GPU_ARTIFACTS', ARTIFACTS/'gpu-control')
os.environ.setdefault('JOINTLOOM_NATIVE_DIR',str(ARTIFACT/'native'))
import bootstrap
import argparse
import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
from paper_config_compare import configure,tasks,PaperCUDA
from input_optimization import PrefixBatchBuilder


def fixture(agents,envs,seed):
    rng=np.random.default_rng(seed)
    grids=(rng.random((envs,128,128))<(.0 if seed==0 else .2)).astype(np.uint8)
    positions=[];goals=[]
    for grid in grids:
        special=np.array([[0,0],[127,127],[0,127],[127,0],[73,116]],dtype=np.int16)
        grid[special[:,0],special[:,1]]=0
        grid[48,56]=0
        free=np.argwhere(grid==0)
        free=free[~np.any(np.all(free[:,None]==special[None],axis=-1),axis=1)]
        choice=rng.choice(len(free),2*agents,replace=False)
        p=free[choice[:agents]].astype(np.int16)
        g=free[choice[agents:]].astype(np.int16)
        count=min(agents,len(special));p[:count]=special[:count]
        if agents>=5:g[4]=[48,56]  # exact negative half-tie projection
        if agents>=3:g[2]=p[2]     # initial arrived/stationary context
        positions.append(p);goals.append(g)
    p=np.stack(positions);g=np.stack(goals)
    return SimpleNamespace(grids=grids,positions=p,goals=g,arrived=np.all(p==g,axis=-1).astype(np.uint8),
                           num_envs=envs,num_agents=agents,horizon=4096)


def compare_build(env):
    s=env.sim
    s.pyg_local_gather_impl='async_sm89plus'
    raw=env.build()
    raw=env.builder.view_stateful_outputs(s,materialize_edges=True)
    reference=copy.copy(raw)
    reference.x=raw.x.clone();reference.edge_index=raw.edge_index.clone();reference.edge_attr=raw.edge_attr.clone()
    s.pyg_local_gather_impl='warp_cooperative'
    s.build_magat_plus_inputs()
    optimized=PrefixBatchBuilder().view_stateful_outputs(s,materialize_edges=True)
    error=float((reference.x-optimized.x).abs().max().item())
    torch.testing.assert_close(reference.x,optimized.x,rtol=0,atol=1e-6)
    torch.testing.assert_close(reference.edge_index,optimized.edge_index,rtol=0,atol=0)
    torch.testing.assert_close(reference.edge_attr,optimized.edge_attr,rtol=0,atol=0)
    return reference,optimized,error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();configure()
    import grid_world_cpp
    report=dict(extension=grid_world_cpp.__file__,input_cases=[],policy=[])
    with torch.no_grad():
        for agents,envs,seed in [(1,1,0),(3,2,1),(16,3,2),(128,2,3),(256,2,4),(512,2,5)]:
            env=PaperCUDA(fixture(agents,envs,seed))
            a,b,error=compare_build(env)
            report['input_cases'].append(dict(agents=agents,envs=envs,seed=seed,
                max_input_error=error,edges=int(a.edge_attr.shape[0]),graph_equal=True))
            del a,b,env
        # Exact original N256 point, including unreachable-goal and arrived flags.
        env=PaperCUDA(tasks(256));a,b,error=compare_build(env)
        report['input_cases'].append(dict(agents=256,envs=256,source='original frozen input',
            max_input_error=error,edges=int(a.edge_attr.shape[0]),graph_equal=True))
        del a,b,env
        torch.cuda.empty_cache()
        from aligned_tasks import PAPER_CHECKPOINT
        from policies import MagatPolicy
        batch=tasks(16);batch.horizon=4096
        env=PaperCUDA(batch);policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda')
        for step in range(8):
            a,b,error=compare_build(env)
            la=policy.model(a.x,a);lb=policy.model(b.x,b)
            torch.testing.assert_close(la,lb,rtol=1e-4,atol=1e-5)
            aa=la.argmax(-1);ab=lb.argmax(-1)
            torch.testing.assert_close(aa,ab,rtol=0,atol=0)
            report['policy'].append(dict(step=step,max_input_error=error,
                max_logit_error=float((la-lb).abs().max().item()),actions_equal=True))
            env.step(ab.reshape(16,256).to(torch.uint8))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
