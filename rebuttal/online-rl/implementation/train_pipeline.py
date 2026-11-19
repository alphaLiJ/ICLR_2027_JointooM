"""Compare a fixed CPU-buffer reference with JointLoom-only optimizations."""
import bootstrap
import argparse
import copy
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from model import CostTracer,configure_torch
from configuration import SPEC
from tasks import make_tasks
from environments import CpuPool,CudaEnvironment
from ppo import gae,make_optimizer,update
from reference_pipeline import cpu_gae,reference_update


def run(args):
    configure_torch(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cpu=args.profile=='reference_cpu_buffers';optimized=args.profile=='jointloom_optimized'
    ne=getattr(args,'num_envs',16)
    started=time.perf_counter();n=ne*128;t=32;m=2048
    if getattr(args,'sf2_compatible',False):
        from sf_tasks import load
        tasks=load(args.seed,num_envs=ne,bank_episodes=getattr(args,'task_bank_episodes',64))
    else:tasks=make_tasks(args.seed,envs=ne,bank=getattr(args,'task_bank_episodes',64))
    model=CostTracer().cuda();torch.save(model.state_dict(),out/'initial.pt')
    if optimized:
        from optimized_gpu import StaticLearner,fused_gae
        loss_function=None
        if getattr(args,'sf2_compatible',False):
            from sf2_compat import sf2_loss,sf2_gae
            loss_function=sf2_loss;fused_gae=sf2_gae
        learner=StaticLearner(model,n=n,minibatch=m,channels_last=not args.nchw,
                              benchmark=not args.no_cudnn_tune,graphs=not args.no_graphs,
                              compile_loss=args.compile_loss,compile_model=args.compile_model,
                              loss_function=loss_function)
        optimizer=learner.optimizer;policy=learner.policy
    else:
        optimizer=make_optimizer(model);policy=lambda x:model(x.reshape(n,2,11,11))
    env=CpuPool(tasks,8) if cpu else CudaEnvironment(tasks)
    shapes=dict(obs=(t,n,2,11,11),actions=(t,n),logp=(t,n),values=(t,n),
                next_values=(t,n),rewards=(t,n),reached=(t,n),reset=(t,n))
    dtypes={k:(torch.int64 if k=='actions' else torch.bool if k=='reset' else torch.float32) for k in shapes}
    storage={k:torch.empty(shape,dtype=dtypes[k],device='cpu' if cpu else 'cuda',pin_memory=cpu)
             for k,shape in shapes.items()}
    arrays={k:v.numpy() for k,v in storage.items()} if cpu else None
    gpu_batch={k:torch.empty_like(storage[k],device='cuda') for k in
               ('obs','actions','logp','values')} if cpu else None
    advantages=torch.empty((t,n),device='cpu' if cpu else 'cuda',pin_memory=cpu)
    returns=torch.empty_like(advantages,pin_memory=cpu)
    if cpu:
        gpu_batch['advantages']=torch.empty_like(advantages,device='cuda')
        gpu_batch['returns']=torch.empty_like(returns,device='cuda')
    # Reference retains CPU trajectories; only the current observation is uploaded for acting.
    host_step=torch.empty((n,2,11,11),pin_memory=True);gpu_step=torch.empty_like(host_step,device='cuda')
    copied=torch.cuda.Event();has_copy=False
    def send_obs(x):
        nonlocal has_copy
        if has_copy:copied.synchronize()
        np.copyto(host_step.numpy(),x.reshape(n,2,11,11));gpu_step.copy_(host_step,non_blocking=True)
        copied.record();has_copy=True
        return gpu_step
    obs_cpu=env.obs if cpu else None
    obs=send_obs(obs_cpu) if cpu else env.obs
    sample_rng=torch.Generator(device='cuda').manual_seed(args.seed+20000)
    shuffle_rng=torch.Generator(device='cuda').manual_seed(args.seed+30000)
    reference_rng=np.random.RandomState(args.seed+30000)
    stage_recorder=None
    if os.environ.get('JOINTLOOM_MODEL_STAGE_PROFILE'):
        from neural_timing import recorder
        stage_recorder=recorder()
    # Compile fused auxiliaries outside the measured training interval, without updating weights.
    if optimized:
        dummy_r=torch.zeros((t,n),device='cuda');dummy_reset=torch.zeros_like(dummy_r,dtype=torch.bool)
        fused_gae(dummy_r,dummy_r,dummy_r,dummy_reset,advantages,returns)
        from optimized_gpu import _gather_obs,_gather_scalars
        import triton
        dummy_order=torch.arange(t*n,device='cuda')
        for start in (0,m):
            _gather_obs[(triton.cdiv(m*242,1024),)](storage['obs'],dummy_order,learner.mb_obs,start,m,
                learner.channels_last,1024)
            _gather_scalars[(triton.cdiv(m,256),)](dummy_order,storage['actions'],storage['logp'],storage['values'],
                advantages,returns,learner.mb_actions,learner.mb_scalars,start,m,256)
    torch.cuda.synchronize()
    metadata=dict(profile=args.profile,seed=args.seed,num_envs=ne,agents=128,rollout=32,
        minibatch=2048,adam_updates_per_iteration=t*n//m,iterations=args.iterations,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),map_names=tasks['names'],
        setup_seconds=time.perf_counter()-started,hyperparameters=copy.deepcopy(SPEC),
        channels_last=optimized and not args.nchw,cudnn_benchmark=optimized and not args.no_cudnn_tune,
        cuda_graphs=optimized and not args.no_graphs,amp=False,tf32=False,
        compiled_loss=optimized and args.compile_loss,compiled_model=optimized and args.compile_model,
        minibatch_order=getattr(args,'minibatch_order','shuffle'),
        task_bank_episodes=len(tasks['positions']),
        stop_perf_counter_ns=getattr(args,'stop_perf_ns',0),
        sf2_compatible=getattr(args,'sf2_compatible',False),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        CPU_reference='selected SF-v1 CPU-rollout/GAE/upload placement, synchronous adapter; original author trainer unavailable',
        initial_weights_restored=getattr(learner,'warmup_initial_weights_restored',None) if optimized else None,
        optimizer_state_reset=getattr(learner,'warmup_optimizer_zero',None) if optimized else None,
        worker_cpu_affinity=sorted(os.sched_getaffinity(0))[:16] if cpu else [],
        timing='synchronized wall; complete online sampling + GAE/batch preparation + one update per minibatch; excludes setup/log/save')
    metadata['hyperparameters']['environment']['num_envs']=ne
    metadata['hyperparameters']['environment']['task_bank_episodes']=len(tasks['positions'])
    metadata['hyperparameters']['sampling'].update(agent_transitions_per_rollout=t*n,
        joint_environment_transitions_per_rollout=t*ne)
    metadata['hyperparameters']['ppo']['minibatches_per_epoch']=t*n//m
    if getattr(args,'sf2_compatible',False):
        metadata['hyperparameters']['ppo'].update(loss_reference='sample_factory_v2.1.1',
            advantage_min_std=1e-7,bootstrap_observation='native_SF2_timeout_current_value_approximation')
        metadata['CPU_reference']='actual Sample Factory 2.1.1 via sf_native.py; no local CPU-buffer surrogate'
    metadata['hyperparameters']['ppo']['shuffle_samples']=getattr(args,'minibatch_order','shuffle')=='shuffle'
    metadata['hyperparameters']['numerics']['cudnn_benchmark']=metadata['cudnn_benchmark']
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    torch.cuda.reset_peak_memory_stats();rows=[];joint_step=0
    torch.cuda.synchronize()
    (out/'training-clock.json').write_text(json.dumps(dict(start_perf_counter_ns=time.perf_counter_ns(),
        timing='training sampling loop start; excludes initialization/compile/capture; includes collection/update/control')))
    checkpoint_clock=time.perf_counter()
    stop_reason='iteration_budget'
    try:
        for iteration in range(args.iterations):
            if iteration and getattr(args,'stop_perf_ns',0) and time.perf_counter_ns()>=args.stop_perf_ns:
                stop_reason='wall_deadline';break
            if getattr(args,'stop_file','') and Path(args.stop_file).exists():
                stop_reason='user_stop_request';break
            torch.cuda.synchronize();begin=time.perf_counter()
            begin_ns=time.perf_counter_ns()
            with torch.no_grad():
                logits,values=policy(obs)
                for k in range(t):
                    with stage_recorder.stage(10) if stage_recorder else nullcontext():
                        lp=logits.log_softmax(-1)
                        actions=torch.multinomial(lp.exp(),1,generator=sample_rng).squeeze(-1)
                        selected=lp.gather(-1,actions[:,None]).squeeze(-1)
                    if cpu:
                        meta=torch.stack((values,selected,actions.float()),dim=1).cpu().numpy()
                        arrays['obs'][k]=obs_cpu.reshape(n,2,11,11)
                        arrays['values'][k]=meta[:,0];arrays['logp'][k]=meta[:,1];arrays['actions'][k]=meta[:,2]
                        obs_cpu,r,h,reset=env.step(meta[:,2].astype(np.int64).reshape(ne,128))
                        arrays['rewards'][k]=r.reshape(n);arrays['reached'][k]=h.reshape(n)
                        arrays['reset'][k]=np.broadcast_to(reset[:,None],(ne,128)).reshape(n)
                        obs=send_obs(obs_cpu)
                    else:
                        storage['obs'][k].copy_(obs.reshape(n,2,11,11));storage['values'][k].copy_(values)
                        storage['logp'][k].copy_(selected);storage['actions'][k].copy_(actions)
                        obs,r,h,reset=env.step(actions.reshape(ne,128))
                        storage['rewards'][k].copy_(r.reshape(n));storage['reached'][k].copy_(h.reshape(n))
                        storage['reset'][k].copy_(reset[:,None].expand(-1,128).reshape(n))
                    logits,values=policy(obs)
                    if cpu:
                        # Shift next values after rollout, except the final-observation bootstrap.
                        if k==t-1:arrays['next_values'][k]=values.cpu().numpy()
                    else:storage['next_values'][k].copy_(values)
                    joint_step+=1
                    if joint_step%512==0:
                        reset_obs=env.reset_due()
                        if cpu:obs_cpu=reset_obs;obs=send_obs(obs_cpu)
                        else:obs=reset_obs
                        logits,values=policy(obs)
            torch.cuda.synchronize();sample_end=time.perf_counter()
            sample_completed_ns=time.perf_counter_ns()
            if cpu:
                arrays['next_values'][:-1]=arrays['values'][1:]
                adv,ret=cpu_gae(arrays['rewards'],arrays['values'],arrays['next_values'],arrays['reset'])
                np.copyto(advantages.numpy(),adv);np.copyto(returns.numpy(),ret)
                for key,dest in gpu_batch.items():
                    source=advantages if key=='advantages' else returns if key=='returns' else storage[key]
                    dest.copy_(source,non_blocking=True)
                flat={k:v.flatten(0,1) for k,v in gpu_batch.items()}
            else:
                if optimized:fused_gae(storage['rewards'],storage['values'],storage['next_values'],storage['reset'],advantages,returns)
                else:
                    adv,ret=gae(storage['rewards'],storage['values'],storage['next_values'],storage['reset'])
                    advantages.copy_(adv);returns.copy_(ret)
                flat={k:storage[k].flatten(0,1) for k in ('obs','actions','logp','values')}
                flat.update(advantages=advantages.flatten(),returns=returns.flatten())
            torch.cuda.synchronize();prepared=time.perf_counter()
            if cpu:stats=reference_update(model,optimizer,flat,reference_rng,m)
            elif optimized:stats=learner.update(flat,shuffle_rng,order_mode=getattr(args,'minibatch_order','shuffle'))
            else:stats=update(model,optimizer,flat,shuffle_rng,m)
            torch.cuda.synchronize();end=time.perf_counter()
            end_ns=time.perf_counter_ns()
            numbers=stats.cpu().tolist()
            row=dict(iteration=iteration,agent_samples=(iteration+1)*t*n,
                sample_seconds=sample_end-begin,gae_prepare_seconds=prepared-sample_end,
                update_seconds=end-prepared,iteration_seconds=end-begin,
                agent_samples_per_second=t*n/(end-begin),mean_progress_reward=float(storage['rewards'].mean()),
                goals=float(storage['reached'].sum()),loss=numbers[0],actor_loss=numbers[1],
                value_loss=numbers[2],entropy=numbers[3],approx_kl=numbers[4],clip_fraction=numbers[5],
                gradnorm=numbers[6],mean_return=float(returns.mean()),
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                begin_perf_counter_ns=begin_ns,end_perf_counter_ns=end_ns)
            row.update(sample_completed_perf_counter_ns=sample_completed_ns,
                rollout_agent_samples=t*n,env_joint_steps=joint_step,
                collected_reward_sum=float(storage['rewards'].sum()))
            if not np.isfinite(numbers).all():raise FloatingPointError(row)
            rows.append(row)
            with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            every=getattr(args,'checkpoint_every',0)
            seconds=getattr(args,'checkpoint_seconds',0.)
            due_by_time=bool(seconds and time.perf_counter()-checkpoint_clock>=seconds)
            if due_by_time or (every and (iteration+1)%every==0 and iteration+1<args.iterations):
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    agent_samples=(iteration+1)*t*n,seed=args.seed,completed_iterations=iteration+1,
                    task_bank_episodes=len(tasks['positions'])),out/f'checkpoint-{iteration+1:06d}.pt')
                checkpoint_clock=time.perf_counter()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            agent_samples=len(rows)*t*n,seed=args.seed),out/'final.pt')
        measured=rows[2:] if len(rows)>2 else rows
        adam_steps=sorted(set(float(state['step']) for state in optimizer.state.values()))
        result=dict(profile=args.profile,seed=args.seed,iterations=len(rows),agent_samples=len(rows)*t*n,
            stop_reason=stop_reason,
            adam_step_counters=adam_steps,expected_adam_steps=len(rows)*t*n//m,
            steady_mean={k:float(np.mean([r[k] for r in measured])) for k in
                ('sample_seconds','gae_prepare_seconds','update_seconds','iteration_seconds')},
            first_progress_reward=rows[0]['mean_progress_reward'],last_progress_reward=rows[-1]['mean_progress_reward'],
            total_goals=sum(r['goals'] for r in rows),setup_seconds=metadata['setup_seconds'])
        (out/'summary.json').write_text(json.dumps(result,indent=2))
    finally:
        if cpu:env.close()
        if os.environ.get('JOINTLOOM_NEURAL_PROFILE_DIR'):
            from neural_timing import flush
            flush()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['reference_cpu_buffers','jointloom_optimized','jointloom_eager'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--nchw',dest='nchw',action='store_true')
    p.add_argument('--channels-last',dest='nchw',action='store_false')
    p.add_argument('--no-cudnn-tune',action='store_true')
    p.add_argument('--no-graphs',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--compile-loss',dest='compile_loss',action='store_true')
    p.add_argument('--no-compile-loss',dest='compile_loss',action='store_false')
    p.add_argument('--compile-model',dest='compile_model',action='store_true')
    p.add_argument('--no-compile-model',dest='compile_model',action='store_false')
    p.add_argument('--sf2-compatible',action='store_true')
    p.add_argument('--minibatch-order',choices=['shuffle','sf_env_major'],default='shuffle')
    p.add_argument('--task-bank-episodes',type=int,default=64)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--checkpoint-seconds',type=float,default=0.)
    p.add_argument('--stop-perf-ns',type=int,default=0)
    p.add_argument('--stop-file',type=str,default='')
    p.set_defaults(nchw=True,compile_loss=True,compile_model=True)
    run(p.parse_args())
