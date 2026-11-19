"""Finite 256-step policy quality for the paired order-learning checkpoints."""
import bootstrap
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from config_sampler import ORDERS
from evaluation import make_evaluation_tasks,evaluate_checkpoint,summarize_quality
from run_il_order import MAP_NAMES,load_maps
from tasks import make_tasks


def run(root,out,seeds=(0,1,2)):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out)
    os.sched_setaffinity(0,set(range(16)));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=dict(device='cuda:0',task_seed=20261109,tasks_per_map=8,horizon=256,policy_nodes_per_batch=2048,
        training_seeds=list(seeds))
    tasks={a:make_evaluation_tasks(config,a) for a in [32,128,512]}
    train_grids=[];train_positions=[];train_goals=[];metadata=[]
    for mi,name in enumerate(MAP_NAMES):
        batch=make_tasks(name,512,8,20261119+mi)
        train_grids.extend([batch.grid]*8);train_positions.extend(batch.positions);train_goals.extend(batch.goals)
        metadata.extend([dict(family='train-maze-new-tasks',map_name=name,task=i,agents=512) for i in range(8)])
    old=tasks[512]
    tasks[512]=(np.concatenate((old[0],np.stack(train_grids))),np.concatenate((old[1],np.stack(train_positions))),
        np.concatenate((old[2],np.stack(train_goals))),old[3]+metadata)
    for a,task in tasks.items():
        np.savez_compressed(out/f'tasks-a{a}.npz',grids=task[0],positions=task[1],goals=task[2])
        (out/f'tasks-a{a}.json').write_text(json.dumps(task[3],indent=2))
    results=[];started=time.perf_counter()
    for seed in seeds:
        for order in ORDERS:
            directory=root/f'seed-{seed}'/order
            training=json.loads((directory/'result.json').read_text())
            if training['optimizer_updates']!=100000:raise ValueError('quality requires the common fixed final budget')
            for a,task in tasks.items():
                current,arrivals=evaluate_checkpoint(str(directory/'checkpoint.pt'),task,config)
                current=[dict(method=order,seed=seed,training_result=str(directory/'result.json'),**r) for r in current]
                results.extend(current)
                np.save(out/f'arrivals-{order}-seed{seed}-a{a}.npy',arrivals)
                with (out/'instances.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
                print(json.dumps(dict(seed=seed,order=order,agents=a,instances=len(current),successes=sum(r['success'] for r in current))),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(config=config,wall_seconds=time.perf_counter()-started,
        models=len(seeds)*len(ORDERS),instances=len(results),
        results=summarize_quality(results),scope='same finite256 tasks; fixed final100k models; native-extent inputs; selected training seeds explicitly reported; no lifelong closure metric'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args();run(args.models,args.out,args.seeds)
