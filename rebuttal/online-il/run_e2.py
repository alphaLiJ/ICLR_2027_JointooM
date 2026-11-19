"""E2: full lifelong CPU pipeline versus resident JointLoom, fresh paired processes."""
import bootstrap
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from tasks import make_tasks
from envs import CpuPool,ResidentLifelong
from inputs import HostTransfer,combine_inputs
from policies import MagatPolicy


def cpu_window(pool,transfers,policy,steps,agents):
    # Each process owns an independent environment group. CPU prepares its next
    # input while the GPU runs other groups, without changing the policy snapshot.
    host=dict(source_wait=0.,h2d_enqueue=0.,policy_and_action_readback=0.)
    for c in pool.conns:c.send(("build",None))
    for step in range(steps):
        for c,transfer in zip(pool.conns,transfers):
            t=time.perf_counter();arrays=pool.receive(c);host['source_wait']+=time.perf_counter()-t
            t=time.perf_counter()
            data=transfer.send(arrays)
            host['h2d_enqueue']+=time.perf_counter()-t;t=time.perf_counter()
            actions=policy.act(data).reshape(-1,agents).cpu().numpy()
            host['policy_and_action_readback']+=time.perf_counter()-t
            c.send(("step",(actions,step+1<steps)))
    for c in pool.conns:pool.receive(c)
    return dict(parent_host_ranges=host)


def cpu_batched_window(pool,transfer,policy,steps,agents):
    # An alternative strong reference: amortize inference over all environments,
    # allowing workers to parallelize transition/build rather than overlap it.
    host=dict(source_wait=0.,h2d_enqueue=0.,policy_and_action_readback=0.)
    for c in pool.conns:c.send(('build',None))
    for step in range(steps):
        t=time.perf_counter()
        parts=[pool.receive(c) for c in pool.conns]
        arrays=combine_inputs(parts,pool.env_batch*agents)
        host['source_wait']+=time.perf_counter()-t
        t=time.perf_counter();data=transfer.send(arrays)
        host['h2d_enqueue']+=time.perf_counter()-t
        t=time.perf_counter();actions=policy.act(data).reshape(-1,agents).cpu().numpy()
        host['policy_and_action_readback']+=time.perf_counter()-t
        for i,c in enumerate(pool.conns):
            c.send(('step',(actions[i*pool.env_batch:(i+1)*pool.env_batch],step+1<steps)))
    for c in pool.conns:pool.receive(c)
    return dict(parent_host_ranges=host)


def gpu_window(env,policy,steps,agents,env_batch):
    bins=torch.empty(steps//128,dtype=torch.int64,device=env.device)
    for step in range(steps):
        if step:env.sim.update_derived_state()
        actions=policy.act_resident(env.sim,agents,env_batch)
        env.sim.update_actions(actions)
        env.sim.step_sim_only()
        if (step+1)%128==0:bins[(step+1)//128-1].copy_(env.sim.goal_cursors.sum())
    return dict(goal_bins_device=bins)


def gpu_profile(env,policy,agents,env_batch,steps):
    """Separate synchronized diagnostic; never use its sum as formal throughput."""
    rows=[]
    def timed(fn):
        torch.cuda.synchronize();t=time.perf_counter();result=fn();torch.cuda.synchronize()
        return result,time.perf_counter()-t
    for _ in range(steps):
        _,derived=timed(env.sim.update_derived_state)
        data,builder=timed(lambda:policy.builder.build_from_stateful(env.sim))
        actions,model=timed(lambda:policy.act_env_batches(data,agents,env_batch))
        def transition():
            env.sim.update_actions(actions);env.sim.step_sim_only()
        _,move=timed(transition)
        rows.append(dict(derived_seconds=derived,builder_seconds=builder,
            model_and_batching_seconds=model,action_transition_seconds=move,
            changed_goals=int(env.sim.goal_changed_flags.sum().item())))
    return dict(scope='separate synchronized diagnostic after warmup',rows=rows,
        mean_seconds={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith('_seconds')})


def run_child(config,path,out):
    import grid_world_cpp as ext
    os.sched_setaffinity(0,set(config['cpu_affinity']))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(config['seed'])
    began=time.perf_counter()
    tasks=make_tasks(config['map_name'],config['agents'],config['envs'],config['seed'])
    policy=MagatPolicy(config['checkpoint'],config['device'])
    if config.get('native_extent_inputs',False):
        from native_extent_inputs import NativeExtentBatchBuilder
        policy.builder=NativeExtentBatchBuilder([tasks.grid.shape]*config['envs'],config['agents'],config['device'])
    group_envs=config['envs']//config['workers']
    scheduling=config.get('scheduling','pipelined')
    env_batch=config['envs'] if scheduling=='batched' else group_envs
    if path=='jointloom' and config.get('gpu_env_batch') is not None:
        env_batch=config['gpu_env_batch']
        if not 1<=env_batch<=config['envs']:
            raise ValueError('gpu_env_batch must be between 1 and envs')
    pool=None
    try:
        if path in ('cpu_pipeline','pogema_original'):
            pool=CpuPool(tasks,config['workers'],config['cpu_affinity'],
                backend='original' if path=='pogema_original' else 'optimized')
            if scheduling=='batched':
                transfers=[HostTransfer(config['envs'],config['agents'],config['device'])]
                window=lambda steps:cpu_batched_window(pool,transfers[0],policy,steps,config['agents'])
            else:
                transfers=[HostTransfer(group_envs,config['agents'],config['device']) for _ in pool.conns]
                window=lambda steps:cpu_window(pool,transfers,policy,steps,config['agents'])
            reset=pool.reset
        else:
            env=ResidentLifelong(tasks,config['device'],max(4096,config['measured_steps']+1))
            window=lambda steps:gpu_window(env,policy,steps,config['agents'],env_batch)
            reset=env.reset
        window(config['warmup_steps'])
        torch.cuda.synchronize()
        reset()
        if pool is not None:
            for t in transfers:t.bytes=0
        torch.cuda.synchronize()
        setup_seconds=time.perf_counter()-began
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        window_stats=window(config['measured_steps'])
        torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        if pool is not None:
            for c in pool.conns:c.send(('result',None))
            reports=[pool.receive(c) for c in pool.conns]
            state={k:np.concatenate([r[k] for r in reports]) for k in ['positions','goals','cursors']}
            cpu_memory=sum(r['peak_rss_bytes'] for r in reports)
            cached=sum(r['cached_goals'] for r in reports)
            payload=sum(t.bytes for t in transfers)
            bins=np.sum([r['goal_bins'] for r in reports],axis=0).tolist()
            diagnostics=dict(parent_host_ranges=window_stats['parent_host_ranges'],
                worker_seconds=[dict(build=r['build_seconds'],transition_and_goals=r['step_seconds']) for r in reports],
                interpretation='worker durations overlap each other and GPU work; host policy range includes pending GPU transfers')
        else:
            state=env.result();cpu_memory=None;cached=None;payload=0
            bins=window_stats['goal_bins_device'].cpu().tolist();diagnostics={}
        completed=int(state['cursors'].sum())
        joint=config['envs']*config['measured_steps']
        result=dict(path=path,config=config,setup_seconds=setup_seconds,wall_seconds=seconds,
            joint_transitions=joint,joint_transitions_per_second=joint/seconds,
            batch_decision_ms=1000*seconds/config['measured_steps'],completed_goals=completed,
            goals_per_1000_joint_transitions=1000*completed/joint,goals_per_second=completed/seconds,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),worker_peak_rss_sum_bytes=cpu_memory,
            cached_cpu_goal_maps=cached,expanded_h2d_bytes=payload,map_shape=list(tasks.grid.shape),
            cumulative_goals_every_128_steps=bins,diagnostics=diagnostics,
            goal_protocol='reachable component; unique goals; not own cell; candidate stream with agent-order arbitration',
            cpu_scheduling=scheduling,
            policy_env_microbatch=env_batch,precision='float32; TF32 disabled',
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),extension=ext.__file__,
            timer='current-state inputs through final successor; lazy next-input refresh; boundary GPU sync only')
        out.mkdir(parents=True,exist_ok=True)
        (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        (out/'result.json').write_text(json.dumps(result,indent=2))
        np.savez_compressed(out/'final-state.npz',**state)
        if path=='jointloom' and config.get('profile_steps',0):
            env.reset();gpu_window(env,policy,config['warmup_steps'],config['agents'],env_batch)
            profile=gpu_profile(env,policy,config['agents'],env_batch,config['profile_steps'])
            (out/'profile.json').write_text(json.dumps(profile,indent=2))
        print(json.dumps({k:result[k] for k in ['path','wall_seconds','joint_transitions_per_second','completed_goals','goals_per_second']}),flush=True)
    finally:
        if pool:pool.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/e2.yaml')
    p.add_argument('--paths',nargs='+',default=['cpu_pipeline','jointloom'])
    p.add_argument('--path',choices=['cpu_pipeline','pogema_original','jointloom'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int);p.add_argument('--warmup',type=int)
    p.add_argument('--workers',type=int);p.add_argument('--envs',type=int);p.add_argument('--agents',type=int)
    p.add_argument('--map-name');p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--profile-steps',type=int)
    p.add_argument('--scheduling',choices=['pipelined','batched'])
    p.add_argument('--gpu-env-batch',type=int,help='resident inference group size, independent of CPU worker groups')
    p.add_argument('--native-extent-inputs',action='store_true')
    args=p.parse_args()
    c=yaml.safe_load(args.config.read_text())
    c['checkpoint']=os.environ.get('JOINTLOOM_DECISION_CHECKPOINT',c['checkpoint'])
    for arg,key in [('steps','measured_steps'),('warmup','warmup_steps'),('workers','workers'),('envs','envs'),('agents','agents'),('map_name','map_name'),('profile_steps','profile_steps'),('scheduling','scheduling'),('gpu_env_batch','gpu_env_batch')]:
        if getattr(args,arg) is not None:c[key]=getattr(args,arg)
    if args.native_extent_inputs:c['native_extent_inputs']=True
    if args.path:
        run_child(c,args.path,args.out);return
    args.out.mkdir(parents=True,exist_ok=True)
    run_config=args.out/'run-config.yaml';run_config.write_text(yaml.safe_dump(c,sort_keys=False))
    all_results=[];pairs=[]
    for rep in range(args.repetitions):
        paths=args.paths if rep%2==0 else list(reversed(args.paths))
        current={}
        for path in paths:
            directory=args.out/f'repeat-{rep}'/path
            if (directory/'result.json').exists():raise FileExistsError(directory)
            subprocess.run([sys.executable,__file__,'--config',str(run_config.resolve()),'--path',path,'--out',str(directory.resolve())],check=True)
            result=json.loads((directory/'result.json').read_text());current[path]=result;all_results.append(result)
        cpu_path=next((p for p in ['pogema_original','cpu_pipeline'] if p in current),None)
        if cpu_path is not None and 'jointloom' in current:
            a=np.load(args.out/f'repeat-{rep}'/cpu_path/'final-state.npz')
            b=np.load(args.out/f'repeat-{rep}/jointloom/final-state.npz')
            pairs.append(dict(repetition=rep,cpu_path=cpu_path,speedup=current[cpu_path]['wall_seconds']/current['jointloom']['wall_seconds'],
                final_state_equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files},
                completed_goals_cpu=current[cpu_path]['completed_goals'],completed_goals_gpu=current['jointloom']['completed_goals']))
    summary=dict(config=c,runs=all_results,pairs=pairs,
        median_paired_speedup=float(np.median([r['speedup'] for r in pairs])) if pairs else None)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'summary':str(args.out/'summary.json'),'pairs':pairs}),flush=True)


if __name__=='__main__':main()
