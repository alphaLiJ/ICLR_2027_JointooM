"""Same-task CAMAR discrete adaptation versus the existing JointLoom path."""
import bootstrap
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch


def configure():
    os.sched_setaffinity(0,set(range(16)))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def child(args):
    configure()
    import jax
    import flax
    from aligned_tasks import load_paper_tasks,AlignedResident,PAPER_INPUT,PAPER_CHECKPOINT
    from policies import MagatPolicy
    from native_extent_inputs import NativeExtentBatchBuilder
    from runtime import CamarRuntime

    out=args.out.resolve()
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    tasks=load_paper_tasks(args.envs)
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda') if args.phase=='complete' else None
    if policy:
        policy.builder=NativeExtentBatchBuilder([g.shape for g in tasks.grids],256,'cuda')
    if args.path=='camar_discrete':
        env=CamarRuntime(tasks.grids,tasks.positions,tasks.goals,tasks.initial_arrived,horizon=4096)
        reset=env.reset
        synchronize=env.synchronize
        snapshot=env.result
    else:
        env=AlignedResident(tasks,'standard_mapf',horizon=4096)
        reset=env.reset
        synchronize=torch.cuda.synchronize
        snapshot=env.result

    index=0
    if args.phase=='transition':
        actions=np.random.default_rng(20261109).integers(0,5,size=(args.steps+4,args.envs,256),dtype=np.uint8)
        # Both paths receive an identical preloaded device action trace.
        device_actions=jax.device_put(actions) if args.path=='camar_discrete' else torch.as_tensor(actions,device='cuda')

    @torch.inference_mode()
    def window(steps):
        nonlocal index
        for _ in range(steps):
            if args.phase=='complete':
                if args.path=='camar_discrete':
                    data=env.build()
                else:
                    env.sim.update_derived_state()
                    data=policy.builder.build_from_stateful(env.sim,materialize_edges=False)
                action=policy.act(data).reshape(args.envs,256)
            else:
                action=device_actions[index]
            if args.path=='camar_discrete':env.step(action)
            else:
                env.sim.update_actions(action)
                env.sim.step_sim_only()
            index+=1

    # Compile/warm exact functions and model shape. Cache creation is measured
    # again after resetting, independently of compilation and model warmup.
    window(3)
    synchronize()
    reset();index=0
    synchronize()
    setup_seconds=time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    metrics={}
    for phase,steps in [('first',1),('after_one',1),('sustained',args.steps)]:
        synchronize()
        start=time.perf_counter()
        window(steps)
        synchronize()
        seconds=time.perf_counter()-start
        metrics[phase]=dict(steps=steps,wall_seconds=seconds,
            batch_decision_ms=1000*seconds/steps,
            joint_transitions_per_second=steps*args.envs/seconds)
        print(json.dumps(dict(path=args.path,phase=args.phase,segment=phase,seconds=seconds)),flush=True)
    state=snapshot()
    np.savez_compressed(out/'final-state.npz',**state)
    result=dict(path=args.path,phase=args.phase,envs=args.envs,agents=256,steps=args.steps,
        maps_source=str(PAPER_INPUT),map_shapes=list(tasks.grids.shape),
        task_protocol=tasks.source_stats,mode='standard_mapf',horizon=4096,
        checkpoint=str(PAPER_CHECKPOINT) if policy else None,
        metrics=metrics,setup_compile_warmup_seconds=setup_seconds,
        precision='float32; TF32 disabled',cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        jax_version=jax.__version__,flax_version=flax.__version__,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        jax_memory_stats=jax.devices()[0].memory_stats(),
        final_arrived_agents=int(np.asarray(state['arrived']).sum()),
        baseline_label='CAMAR 0.4.0 with author-migrated discrete MAPF dynamics and MAGAT inputs',
        timer='boundary-synchronized wall time; no per-step host tensor copies; first two decisions separate',
        scope=('same policy/checkpoint; independent closed-loop actions; not a task-quality comparison'
               if policy else 'same preloaded action trace; no observations, energy maps or policy in transition timer'))
    (out/'result.json').write_text(json.dumps(result,indent=2))


def compare(args):
    args.out.mkdir(parents=True,exist_ok=True)
    phases=['transition','complete'] if args.phase=='both' else [args.phase]
    rows=[];runs=[]
    for phase in phases:
        for repeat in range(args.repetitions):
            paths=['camar_discrete','jointloom'] if repeat%2==0 else ['jointloom','camar_discrete']
            pair={}
            for path in paths:
                out=args.out/phase/f'repeat-{repeat}'/path
                if not (out/'result.json').exists():
                    subprocess.run([sys.executable,__file__,'child','--phase',phase,'--path',path,
                        '--envs',str(args.envs),'--steps',str(args.steps),'--out',str(out)],check=True)
                pair[path]=json.loads((out/'result.json').read_text())
                runs.append(pair[path])
            snapshots={p:np.load(args.out/phase/f'repeat-{repeat}'/p/'final-state.npz') for p in paths}
            equal={k:bool(np.array_equal(snapshots['camar_discrete'][k],snapshots['jointloom'][k]))
                   for k in ('positions','goals','arrived')}
            differing_positions=int(np.any(snapshots['camar_discrete']['positions']!=snapshots['jointloom']['positions'],axis=-1).sum())
            for segment in ('first','after_one','sustained'):
                a=pair['camar_discrete']['metrics'][segment]['batch_decision_ms']
                b=pair['jointloom']['metrics'][segment]['batch_decision_ms']
                rows.append(dict(phase=phase,repeat=repeat,segment=segment,
                    camar_batch_ms=a,jointloom_batch_ms=b,speedup=a/b,
                    positions_equal=equal['positions'],goals_equal=equal['goals'],
                    arrived_equal=equal['arrived'],differing_final_agent_positions=differing_positions))
            summary=dict(envs=args.envs,agents=256,steps=args.steps,rows=rows,runs=runs)
            (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
            if phase=='transition' and not all(equal.values()):
                raise AssertionError('shared-action transition mismatch; see saved states')
            print(json.dumps(dict(phase=phase,repeat=repeat,final_state_equal=equal,
                sustained_speedup=rows[-1]['speedup'])),flush=True)
    with (args.out/'pairs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# CAMAR discrete adaptation: same-task comparison','',
        'Author-adapted CAMAR, not an upstream discrete MAPF benchmark. RTX 5080;',
        f'A256, N{args.envs}, original 128x128 map pool, common reachable initial tasks.',
        'Fixed goals; same MAGAT checkpoint for complete decisions. Three timing',
        'segments exclude compilation and model warmup; first contains cold goal maps.',
        'Transition-only receives the same device action trace. Complete decisions',
        'run each policy loop independently; final-state differences are reported.','',
        '| Scope | Segment | CAMAR batch ms (median) | JointLoom batch ms (median) | Median paired speedup |',
        '|---|---|---:|---:|---:|']
    for phase in phases:
        for segment in ('first','after_one','sustained'):
            selected=[r for r in rows if r['phase']==phase and r['segment']==segment]
            text.append(f"| {phase} | {segment} | {statistics.median(r['camar_batch_ms'] for r in selected):.6f} | {statistics.median(r['jointloom_batch_ms'] for r in selected):.6f} | {statistics.median(r['speedup'] for r in selected):.4f} |")
    text+=['','## State checks','']
    for r in rows:
        if r['segment']=='sustained':
            text.append(f"- {r['phase']} repeat {r['repeat']}: positions={r['positions_equal']}, goals={r['goals_equal']}, arrived={r['arrived_equal']}; differing positions={r['differing_final_agent_positions']}.")
    (args.out/'RESULTS.md').write_text('\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['compare','child'])
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--phase',choices=['transition','complete','both'],default='both')
    p.add_argument('--path',choices=['camar_discrete','jointloom'])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='child':
        if args.path is None or args.phase=='both':p.error('child needs --path and one phase')
        child(args)
    else:compare(args)


if __name__=='__main__':main()
