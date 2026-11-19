"""Paired CostTracer PPO runner. The timing unit is a complete rollout + update."""
import bootstrap
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import CostTracer,configure_torch
from tasks import make_tasks
from environments import CudaEnvironment,CpuPool
from ppo import gae,make_optimizer,update
from configuration import SPEC


def run(args):
    configure_torch(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    import os
    import copy
    config=copy.deepcopy(SPEC)
    config['environment'].update(num_envs=args.envs,agents_per_env=args.agents,time_limit_steps=args.horizon)
    config['sampling'].update(rollout_steps=args.rollout,
        agent_transitions_per_rollout=args.rollout*args.envs*args.agents,
        joint_environment_transitions_per_rollout=args.rollout*args.envs)
    config['ppo']['minibatch_size']=args.minibatch
    setup_started=time.perf_counter()
    tasks=make_tasks(args.seed,envs=args.envs,agents=args.agents,horizon=args.horizon)
    model=CostTracer().cuda();optimizer=make_optimizer(model)
    torch.save(model.state_dict(),output/'initial.pt')
    env= CudaEnvironment(tasks) if args.backend=='cuda' else CpuPool(tasks,args.workers)
    cpu=args.backend=='cpu';n=args.envs*args.agents;t=args.rollout
    # CPU transfer uses persistent pinned staging; both paths store rollouts on GPU.
    host_obs=torch.empty((args.envs,args.agents,2,11,11),pin_memory=True)
    gpu_obs=torch.empty_like(host_obs,device='cuda')
    host_reward=torch.empty((args.envs,args.agents),pin_memory=True)
    host_hit=torch.empty_like(host_reward,pin_memory=True)
    host_reset=torch.empty((args.envs,),dtype=torch.bool,pin_memory=True)
    copy_done=torch.cuda.Event()
    has_copy=False
    def transfer_obs(array):
        nonlocal has_copy
        if has_copy:copy_done.synchronize()
        np.copyto(host_obs.numpy(),array);gpu_obs.copy_(host_obs,non_blocking=True)
        copy_done.record();has_copy=True
        return gpu_obs
    obs=transfer_obs(env.obs) if cpu else env.obs
    storage=dict(obs=torch.empty((t,n,2,11,11),device='cuda'),
        actions=torch.empty((t,n),dtype=torch.int64,device='cuda'))
    for key in ('logp','values','next_values','rewards','reached'):
        storage[key]=torch.empty((t,n),device='cuda')
    storage['reset']=torch.empty((t,n),dtype=torch.bool,device='cuda')
    sample_rng=torch.Generator(device='cuda').manual_seed(args.seed+20000)
    shuffle_rng=torch.Generator(device='cuda').manual_seed(args.seed+30000)
    joint_steps=0;rows=[]
    torch.cuda.synchronize()
    metadata=dict(backend=args.backend,seed=args.seed,envs=args.envs,agents=args.agents,
        rollout=t,iterations=args.iterations,workers=args.workers if cpu else 0,
        map_names=tasks['names'],native_extents=[list(g.shape) for g in tasks['grids']],
        setup_seconds=time.perf_counter()-setup_started,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        worker_cpu_affinity=sorted(os.sched_getaffinity(0))[:16] if cpu else [],
        hyperparameter_spec=config,goal_source='uniform connected-component compact target stream',
        interpretation='short synchronous adaptation pilot; not original trainer reproduction',
        timing='wall synchronized complete rollout + GAE + PPO; excludes setup/log/checkpoint; first two iterations warmup')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    torch.cuda.reset_peak_memory_stats()
    try:
        for iteration in range(args.iterations):
            torch.cuda.synchronize();start=time.perf_counter()
            # Do not reuse logits across an optimizer update: every rollout starts on current policy.
            with torch.no_grad():
                logits,values=model(obs.reshape(n,2,11,11))
                for k in range(t):
                    storage['obs'][k].copy_(obs.reshape(n,2,11,11))
                    storage['values'][k].copy_(values)
                    logprobs=logits.log_softmax(-1)
                    actions=torch.multinomial(logprobs.exp(),1,generator=sample_rng).squeeze(-1)
                    storage['actions'][k].copy_(actions)
                    storage['logp'][k].copy_(logprobs.gather(-1,actions[:,None]).squeeze(-1))
                    command=actions.reshape(args.envs,args.agents)
                    if cpu:
                        final,rewards,reached,reset=env.step(command.cpu().numpy())
                        obs=transfer_obs(final)
                        np.copyto(host_reward.numpy(),rewards);np.copyto(host_hit.numpy(),reached)
                        np.copyto(host_reset.numpy(),reset)
                        storage['rewards'][k].copy_(host_reward.flatten(),non_blocking=True)
                        storage['reached'][k].copy_(host_hit.flatten(),non_blocking=True)
                        storage['reset'][k].copy_(host_reset[:,None].expand(-1,args.agents).reshape(-1),non_blocking=True)
                    else:
                        obs,rewards,reached,reset=env.step(command)
                        storage['rewards'][k].copy_(rewards.flatten())
                        storage['reached'][k].copy_(reached.flatten())
                        storage['reset'][k].copy_(reset[:,None].expand(-1,args.agents).reshape(-1))
                    logits,values=model(obs.reshape(n,2,11,11))
                    storage['next_values'][k].copy_(values)
                    joint_steps+=1
                    # Both task banks have equal ages. CUDA reset itself supports selected environments.
                    # This controller branch reads no CUDA scalar; it is known from rollout position.
                    if joint_steps%args.horizon==0:
                        reset_obs=env.reset_due()
                        obs=transfer_obs(reset_obs) if cpu else reset_obs
                        logits,values=model(obs.reshape(n,2,11,11))
            adv,returns=gae(storage['rewards'],storage['values'],storage['next_values'],storage['reset'])
            torch.cuda.synchronize();sample_end=time.perf_counter()
            flat={k:storage[k].flatten(0,1) for k in ('obs','actions','logp','values')}
            flat.update(advantages=adv.flatten(),returns=returns.flatten())
            stats=update(model,optimizer,flat,shuffle_rng,args.minibatch)
            torch.cuda.synchronize();end=time.perf_counter()
            loss,actor,value,entropy,kl,clip_fraction,gradnorm=stats.cpu().tolist()
            return_var=returns.var(correction=0)
            explained=1-(returns-storage['values']).var(correction=0)/return_var.clamp_min(1e-12)
            hits=float(storage['reached'].sum());reward=float(storage['rewards'].mean())
            row=dict(iteration=iteration,agent_samples=(iteration+1)*t*n,iteration_seconds=end-start,
                sample_gae_seconds=sample_end-start,update_seconds=end-sample_end,
                agent_samples_per_second=t*n/(end-start),goals=hits,
                goals_per_joint_env_step=hits/(t*args.envs),goals_per_second=hits/(end-start),
                mean_progress_reward=reward,loss=loss,actor_loss=actor,value_loss=value,
                entropy=entropy,approx_kl=kl,clip_fraction=clip_fraction,gradnorm=gradnorm,
                mean_value=float(storage['values'].mean()),mean_return=float(returns.mean()),
                explained_variance=float(explained),
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
            if not np.isfinite([loss,entropy,gradnorm]).all():raise FloatingPointError(row)
            rows.append(row)
            with (output/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        agent_samples=args.iterations*t*n,seed=args.seed),output/'final.pt')
        measured=rows[2:] if len(rows)>2 else rows
        summary=dict(backend=args.backend,seed=args.seed,iterations=len(rows),
            mean_iteration_seconds=float(np.mean([r['iteration_seconds'] for r in measured])),
            mean_sample_gae_seconds=float(np.mean([r['sample_gae_seconds'] for r in measured])),
            mean_update_seconds=float(np.mean([r['update_seconds'] for r in measured])),
            warmup_iterations=min(2,len(rows)-1),agent_samples=args.iterations*t*n,
            first_progress_reward=rows[0]['mean_progress_reward'],last_progress_reward=rows[-1]['mean_progress_reward'],
            total_goals=sum(r['goals'] for r in rows),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
    finally:
        if cpu:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['cpu','cuda'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--envs',type=int,default=SPEC['environment']['num_envs'])
    p.add_argument('--agents',type=int,default=SPEC['environment']['agents_per_env'])
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--horizon',type=int,default=SPEC['environment']['time_limit_steps'])
    p.add_argument('--rollout',type=int,default=SPEC['sampling']['rollout_steps'])
    p.add_argument('--minibatch',type=int,default=SPEC['ppo']['minibatch_size'])
    p.add_argument('--output',required=True)
    run(p.parse_args())
