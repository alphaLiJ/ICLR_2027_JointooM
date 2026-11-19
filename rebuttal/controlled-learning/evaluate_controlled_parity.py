"""Fresh heldout tasks through both actual input paths, fixed final models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import runtime
import argparse
from pathlib import Path
import csv
import json
import time
import numpy as np
import torch
from collector import EnvironmentPool, Inputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=runtime.ART/'controlled-parity')
    parser.add_argument('--desktop',type=Path,default=runtime.DESKTOP/'identical-learning-controlled-comparison')
    parser.add_argument('--seed-base',type=int,default=1_737_000_000)
    parser.add_argument('--tasks',type=int,default=64)
    parser.add_argument('--training-envs',type=int,default=16)
    parser.add_argument('--expected-updates',type=int,default=64)
    parser.add_argument('--agents',type=int,default=64)
    parser.add_argument('--reference-run-root',type=Path,default=None)
    args=parser.parse_args()
    base=args.base
    out=base/'evaluation';out.mkdir(exist_ok=False)
    desktop=args.desktop
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    seed_base=args.seed_base
    runtime.json_write(out/'protocol.json',dict(agents=args.agents,tasks=args.tasks,seed_base=seed_base,
        horizon=512,selection=f'fixed final after {args.expected_updates} updates; no test-based selection',
        input_paths=['cpu','jointloom'],new_tasks=True,
        costs='official final-arrival SoC and makespan on successful tasks'))
    specifications=[('official-pretrained','cpu',None)]
    for seed in (0,1,2):
        if args.reference_run_root:
            name=f'reference256-cpu-n{args.training_envs}-s{seed}'
            path=args.reference_run_root/f'timing-cpu-n{args.training_envs}-s{seed}'/'last.pt'
            specifications.append((name,'cpu',path))
        for backend in ('cpu','jointloom'):
            name=f'timing-{backend}-n{args.training_envs}-s{seed}'
            specifications.append((name,backend,base/name/'last.pt'))
    model,kwargs=runtime.load_model()
    records=[];summaries=[];comparisons=[];reference_traces={};reference_records={}
    started=time.perf_counter()
    for name,backend,path in specifications:
        weights=path or runtime.configuration().pretrain_weights_path
        model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=False));model.eval()
        pool=EnvironmentPool(backend,args.agents,16,8,kwargs)
        inputs=Inputs(backend,pool,np.asarray(runtime.task_config(seed_base,args.agents).map,dtype=np.uint8))
        selected=[];all_traces={}
        try:
            for begin in range(0,args.tasks,16):
                seeds=[seed_base+i for i in range(begin,begin+16)]
                pool.reset(seeds)
                done=np.zeros(16,dtype=bool);costs=np.ones((16,args.agents));cursor=0;traces=[]
                for step in range(512):
                    data=inputs.build()
                    with torch.no_grad():logits=model(data.x,data)
                    actions=pool.sample_actions(logits)
                    traces.append(actions.astype(np.uint8))
                    pool.step(actions,full=step<511)
                    events=pool.completed_episodes[cursor:];cursor=len(pool.completed_episodes)
                    for event in events:
                        e=event['env']
                        if done[e]:continue
                        done[e]=True
                        row=dict(checkpoint=name,input_path=backend,case_id=begin+e,
                            seed=seeds[e],agents=args.agents,success=event['success'],agent_arrival=event['arrival'],
                            soc=float(costs[e].sum()) if event['success'] else None,
                            makespan=step+1 if event['success'] else None)
                        selected.append(row);records.append(row)
                    for e in range(16):
                        if not done[e]:costs[e,pool.arrays['terminated'][e]==0]=step+2
                    if done.all():break
                if not done.all():raise RuntimeError('heldout episode did not finish by official horizon')
                all_traces[str(begin)]=np.asarray(traces,dtype=np.uint8)
                runtime.publish('controlled_parity_heldout_evaluation',status='running',
                    checkpoint=name,input_path=backend,completed_tasks=begin+16,
                    total_tasks=args.tasks,completed_checkpoints=len(summaries),total_checkpoints=len(specifications))
            selected.sort(key=lambda v:v['case_id'])
            np.savez(out/(name+'-actions.npz'),**all_traces)
            successful=[r for r in selected if r['success']]
            row=dict(checkpoint=name,input_path=backend,tasks=args.tasks,successes=len(successful),
                success_rate=len(successful)/args.tasks,mean_agent_arrival=float(np.mean([r['agent_arrival'] for r in selected])),
                soc_mean=float(np.mean([r['soc'] for r in successful])) if successful else None,
                makespan_mean=float(np.mean([r['makespan'] for r in successful])) if successful else None)
            summaries.append(row)
            if '-cpu-' in name:
                reference_traces[name]=all_traces;reference_records[name]=selected
            if backend=='jointloom':
                other=name.replace('-jointloom-','-cpu-')
                actions_equal=all(np.array_equal(a,reference_traces[other][k]) for k,a in all_traces.items())
                strip=lambda r:{k:v for k,v in r.items() if k not in ('checkpoint','input_path')}
                metrics_equal=[strip(v) for v in selected]==[strip(v) for v in reference_records[other]]
                comparison=dict(cpu=other,jointloom=name,actions_identical=actions_equal,
                    action_count=sum(v.size for v in all_traces.values()),
                    per_task_metrics_identical=metrics_equal,identical=actions_equal and metrics_equal)
                comparisons.append(comparison)
                if not comparison['identical']:
                    runtime.json_write(out/'divergence.json',comparison)
                    raise RuntimeError('actual-input-path heldout evaluation differs')
            value=dict(status='running',summaries=summaries,pairs=comparisons)
            runtime.json_write(out/'summary.json',value)
            runtime.json_write(desktop/'independent-task-evaluation.json',value)
            print(row,flush=True)
        finally:pool.close()
    for name,rows in [('instances.csv',records),('summary.csv',summaries)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    value=dict(status='complete',tasks=args.tasks,agents=args.agents,summaries=summaries,pairs=comparisons,
        all_pairs_identical=all(p['identical'] for p in comparisons),
        elapsed_seconds=time.perf_counter()-started)
    runtime.json_write(out/'summary.json',value)
    runtime.json_write(desktop/'independent-task-evaluation.json',value)
    runtime.publish('controlled_parity_evaluation_complete',status='running',artifact_root=str(base),
        all_pairs_identical=value['all_pairs_identical'],next_stage='report')


if __name__=='__main__':main()
