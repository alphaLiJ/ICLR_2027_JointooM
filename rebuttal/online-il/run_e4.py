"""E4 fixed-last-checkpoint quality for existing matched-wall training cohorts."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import bootstrap
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from evaluation import make_evaluation_tasks,evaluate_checkpoint,summarize_quality


def checkpoint_list():
    root=input_path('JOINTLOOM_LEGACY_ARTIFACTS', ARTIFACTS/'legacy');rows=[]
    for seed in range(3):
        for method in ['conventional_sync','proposed_async','cpu_cached_prefetch']:
            if method=='cpu_cached_prefetch':
                run=root/'jointloom-strong-cpu-learning-20260921-v1/runs'/f'seed-{seed}'
                cohort='September cached-prefetch logging'
            else:
                run=root/'s2-matched-wall-multiseed-20260812/formal'/f'seed-{seed}'/method
                cohort='August matched-wall'
            result=json.loads((run/'result.json').read_text())
            rows.append(dict(method=method,seed=seed,checkpoint=str(run/'checkpoints/ckpt_latest.pt'),
                training_result=str(run/'result.json'),cohort=cohort,
                optimizer_updates=result.get('num_optimizer_steps'),training_wall_seconds=result.get('total_wall_s')))
    return rows


def run(config,out,max_checkpoints=None):
    if (out/'instances.csv').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,set(config['cpu_affinity']))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoints=checkpoint_list()
    if max_checkpoints is not None:checkpoints=checkpoints[:max_checkpoints]
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (out/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2))
    tasks={a:make_evaluation_tasks(config,a) for a in config['agent_counts']}
    for a,(grids,positions,goals,metadata) in tasks.items():
        np.savez_compressed(out/f'tasks-a{a}.npz',grids=grids,positions=positions,goals=goals)
        (out/f'tasks-a{a}.json').write_text(json.dumps(metadata,indent=2))
    rows=[];started=time.perf_counter()
    for ck in checkpoints:
        for a,task in tasks.items():
            current,first=evaluate_checkpoint(ck['checkpoint'],task,config)
            current=[dict(**ck,**r) for r in current];rows.extend(current)
            np.save(out/f"first-arrival-{ck['method']}-seed{ck['seed']}-a{a}.npy",first)
            with (out/'instances.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            print(json.dumps(dict(method=ck['method'],seed=ck['seed'],agents=a,instances=len(current),
                successes=sum(r['success'] for r in current),arrival_fraction=float(np.mean([r['arrival_fraction'] for r in current])))),flush=True)
    summary=dict(config=config,evaluation_wall_seconds=time.perf_counter()-started,checkpoints=checkpoints,
        results=summarize_quality(rows),
        scope='fixed last checkpoints; common held-out tasks; standard MAPF with frozen agents and soft collisions; argmax policy',
        cohorts='August sync/JointLoom matched-wall; September cached-prefetch follows same protocol/validation, not identical full training records',
        cost_metrics='SoC/makespan only for successful tasks; horizon-capped sum separately named')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['evaluate'])
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/e4.yaml')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--max-checkpoints',type=int)
    p.add_argument('--horizon',type=int);p.add_argument('--tasks-per-map',type=int)
    args=p.parse_args();c=yaml.safe_load(args.config.read_text())
    for arg in ['horizon','tasks_per_map']:
        if getattr(args,arg) is not None:c[arg]=getattr(args,arg)
    run(c,args.out,args.max_checkpoints)
