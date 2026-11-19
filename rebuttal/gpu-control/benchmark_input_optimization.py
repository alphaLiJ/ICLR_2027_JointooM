"""Fresh-process input and complete-decision comparison on original N256 tasks."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PATHS=['legacy_jax','camar_discrete','cuda_original','cuda_optimized']


def child(args):
    if args.path.startswith('cuda_'):
        variant = 'native_optimized/src' if args.path=='cuda_optimized' else '../online-il/native/src'
        key = 'JOINTLOOM_OPTIMIZED_NATIVE_DIR' if args.path=='cuda_optimized' else 'JOINTLOOM_ORIGINAL_NATIVE_DIR'
        os.environ['JOINTLOOM_NATIVE_DIR'] = os.environ.get(key, str(Path(__file__).resolve().parent/variant))
    else:
        os.environ.pop('JOINTLOOM_NATIVE_DIR',None)
    import bootstrap
    import numpy as np
    import torch
    from paper_config_compare import configure,tasks,make_env
    from aligned_tasks import PAPER_INPUT,PAPER_CHECKPOINT
    from policies import MagatPolicy
    from expert.fixed_magat_plus_runtime import PyGBatchBuilder
    configure();batch=tasks(args.envs)
    if args.path=='cuda_optimized':
        from input_optimization import OptimizedPaperCUDA
        env=OptimizedPaperCUDA(batch)
    else:env=make_env('jointloom' if args.path=='cuda_original' else args.path,batch)
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda')
    dtype=torch.uint8 if args.path.startswith('cuda_') else torch.int32

    def ready_input():
        data=env.build()
        if data.edge_index is None:
            data.edge_index,data.edge_attr=PyGBatchBuilder._materialize_valid_edges(
                data.edge_index_storage,data.edge_attr_storage,data.num_edges)
        return data

    def complete():
        data=env.build()
        action=policy.model(data.x,data).argmax(-1).reshape(args.envs,256).to(dtype)
        env.step(action)

    with torch.no_grad():
        for _ in range(3):complete();env.synchronize()
        env.reset();complete();env.synchronize()
        if args.path.startswith('cuda_'):
            s=env.sim
            warm=(torch.stack((s.cur_x,s.cur_y),-1).to(torch.int16).clone(),
                  torch.stack((s.goal_x,s.goal_y),-1).to(torch.int16).clone(),
                  s.arrived.clone(),s.step_counts.clone())
            def restore():
                s.load_state(*warm);env.needs_energy=False
        else:
            warm=env.state;key=getattr(env,'key',None)
            def restore():
                env.state=warm
                if key is not None:env.key=key

        restore();data=ready_input();env.synchronize()
        cache_before=env.cache_sizes() if hasattr(env,'cache_sizes') else {}
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(args.input_iterations):data=ready_input()
        env.synchronize();input_seconds=time.perf_counter()-start
        edge_count=int(data.edge_attr.shape[0])
        decision_rows=[]
        for index in range(args.decision_iterations):
            restore();env.synchronize()
            start=time.perf_counter();complete();env.synchronize()
            decision_rows.append(1000*(time.perf_counter()-start))
        cache_after=env.cache_sizes() if hasattr(env,'cache_sizes') else {}
        snapshot=env.result()
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'final-state.npz',**snapshot)
    result=dict(path=args.path,envs=args.envs,agents=256,edge_count=edge_count,
        input_iterations=args.input_iterations,input_total_seconds=input_seconds,
        input_batch_ms=input_seconds*1000/args.input_iterations,
        decision_iterations=args.decision_iterations,decision_rows_ms=decision_rows,
        decision_batch_ms=statistics.median(decision_rows),
        cache_entries_before=cache_before,cache_entries_after=cache_after,
        checkpoint=str(PAPER_CHECKPOINT),input_source=str(PAPER_INPUT),
        task_protocol='original frozen arrays; fixed goals; max_episode_steps=2; original second-decision input',
        warmup='three full decisions plus reset/one advance; initial and post-DLPack variants compiled outside timer',
        timer='input includes ready node/graph views; complete includes input, unchanged model, action and transition; state restoration outside timer',
        precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        optimized_prefix_metadata_bytes_per_build=8 if args.path=='cuda_optimized' else 0)
    if args.path.startswith('cuda_'):
        import grid_world_cpp
        result.update(extension=grid_world_cpp.__file__,resolved_builder=env.sim.resolved_pyg_builder_impl)
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='decision_rows_ms'}),flush=True)


def compare(args):
    import numpy as np
    args.out.mkdir(parents=True,exist_ok=True);runs=[];pairs=[]
    for repeat in range(args.repetitions):
        paths=PATHS if repeat%2==0 else list(reversed(PATHS))
        if repeat==2:paths=PATHS[2:]+PATHS[:2]
        current={}
        for path in paths:
            output=args.out/f'repeat-{repeat}'/path
            if not (output/'result.json').exists():
                env=dict(os.environ);env.pop('JOINTLOOM_NATIVE_DIR',None)
                subprocess.run([sys.executable,__file__,'child','--path',path,'--envs',str(args.envs),
                    '--input-iterations',str(args.input_iterations),'--decision-iterations',str(args.decision_iterations),
                    '--out',str(output)],check=True,env=env)
            current[path]=json.loads((output/'result.json').read_text());runs.append(current[path])
        snapshots={p:np.load(args.out/f'repeat-{repeat}'/p/'final-state.npz') for p in paths}
        for path in PATHS[:-1]:
            equal={k:bool(np.array_equal(snapshots[path][k],snapshots['cuda_optimized'][k]))
                   for k in snapshots['cuda_optimized'].files}
            row=dict(repeat=repeat,reference=path,
                input_speedup=current[path]['input_batch_ms']/current['cuda_optimized']['input_batch_ms'],
                decision_speedup=current[path]['decision_batch_ms']/current['cuda_optimized']['decision_batch_ms'],
                final_state_equal=equal)
            pairs.append(row);print(json.dumps(row),flush=True)
        (args.out/'summary.json').write_text(json.dumps(dict(envs=args.envs,runs=runs,pairs=pairs),indent=2))
        if not all(all(row['final_state_equal'].values()) for row in pairs):
            raise AssertionError('Final states differ; investigate before claiming a speedup')
    text=['# Optimized CUDA input comparison','',
        f'A256/N{args.envs}, original frozen tasks and checkpoint; proper JIT warmup;',
        'cuDNN TF32 enabled consistently; fixed original second-decision input.',
        f'{args.repetitions} fresh-process repetitions, {args.input_iterations} input builds per process and',
        f'{args.decision_iterations} separately timed complete decisions per process.','',
        '| Path | Median input batch ms | Median complete-decision batch ms |',
        '|---|---:|---:|']
    for path in PATHS:
        selected=[r for r in runs if r['path']==path]
        text.append(f"| {path} | {statistics.median(r['input_batch_ms'] for r in selected):.6f} | {statistics.median(r['decision_batch_ms'] for r in selected):.6f} |")
    text+=['','| Reference / optimized CUDA | Median paired input speedup | Median paired complete speedup |',
        '|---|---:|---:|']
    for path in PATHS[:-1]:
        selected=[r for r in pairs if r['reference']==path]
        text.append(f"| {path} | {statistics.median(r['input_speedup'] for r in selected):.4f} | {statistics.median(r['decision_speedup'] for r in selected):.4f} |")
    text+=['','All canonical final-state fields agree in all pairs.',
        'Optimized edge materialization includes one 8-byte device scalar read.',
        'These are fully warmed repeated second-decision measurements, not a training or task-quality result.']
    (args.out/'RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['compare','child'])
    p.add_argument('--path',choices=PATHS)
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--input-iterations',type=int,default=64)
    p.add_argument('--decision-iterations',type=int,default=12)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='child':child(args)
    else:compare(args)
