"""Resolved parameter checks, actual-framework probes, then bounded reward runs."""
import bootstrap
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from model import CostTracer,configure_torch
from sf_native import configuration
from sf_tasks import SF_ROOT,load,prepare


def audit(root,bank_episodes=64,seeds=(0,1,2)):
    args=['--algo=APPO','--env=jointloom_costtracer_pogema','--seed=0',
        '--num_workers=8','--num_envs_per_worker=2','--shuffle_minibatches=False',
        '--costtracer_task_bank_episodes='+str(bank_episodes),
        '--costtracer_initial_weights='+str(SF_ROOT/'initial-seed-0.pt'),
        '--costtracer_observer_dir='+str(root)]
    cfg=configuration(args)
    keys=['num_workers','num_envs_per_worker','worker_num_splits','rollout','batch_size',
        'num_batches_per_epoch','num_epochs','gamma','gae_lambda','learning_rate','lr_schedule',
        'ppo_clip_ratio','ppo_clip_value','value_loss_coeff','exploration_loss_coeff',
        'max_grad_norm','adam_beta1','adam_beta2','adam_eps','shuffle_minibatches',
        'use_rnn','recurrence','with_vtrace','value_bootstrap','normalize_input','normalize_returns']
    actual={k:getattr(cfg,k) for k in keys}
    assert cfg.num_workers*cfg.num_envs_per_worker==16
    assert cfg.batch_size==2048 and cfg.rollout==32 and cfg.num_batches_per_epoch==32 and cfg.num_epochs==1
    assert not cfg.shuffle_minibatches and not cfg.use_rnn and not cfg.with_vtrace
    initial=[];banks=[]
    for seed in seeds:
        configure_torch(seed);model=CostTracer()
        saved=torch.load(SF_ROOT/f'initial-seed-{seed}.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
        assert sum(p.numel() for p in model.parameters())==161734
        initial.append(dict(seed=seed,parameter_count=161734,canonical_initial_matches=True))
        tasks=load(seed,num_envs=16,bank_episodes=bank_episodes)
        assert tasks['agents']==128 and tasks['envs']==16 and tasks['horizon']==512
        assert tasks['positions'].shape==(bank_episodes,16,128,2)
        assert tasks['goals'].shape==(bank_episodes,16,128,2)
        assert tasks['streams'].shape==(bank_episodes,16,128,513,2)
        banks.append(dict(seed=seed,positions=list(tasks['positions'].shape),goals=list(tasks['goals'].shape),
            streams=list(tasks['streams'].shape),maps=tasks['names']))
    # Match the actual native SF minibatch layout with the new GPU gather ordering.
    from optimized_gpu import StaticLearner,_gather_scalars
    from sample_factory.algo.learning.learner import Learner
    from types import SimpleNamespace
    import triton
    n=2048;count=32*n;m=2048
    self=SimpleNamespace(n=n)
    order=StaticLearner.sample_order(self,count,None,mode='sf_env_major')
    identifiers=torch.arange(count,device='cuda')
    output_a=torch.empty(m,dtype=torch.int64,device='cuda');output_s=torch.empty((4,m),device='cuda')
    scalars=identifiers.float()
    fake=SimpleNamespace(cfg=cfg)
    slices=Learner._get_minibatches(fake,m,count)
    native_ids=identifiers.reshape(32,n).T.contiguous().flatten()
    for index,s in enumerate(slices):
        _gather_scalars[(triton.cdiv(m,256),)](order,identifiers,scalars,scalars,scalars,scalars,
            output_a,output_s,index*m,m,256)
        torch.testing.assert_close(output_a,native_ids[s],atol=0,rtol=0)
        torch.testing.assert_close(output_s[0],native_ids[s].float(),atol=0,rtol=0)
    record=dict(resolved_native_parameters=actual,initial_weights=initial,task_banks=banks,
        agents_per_env=128,num_envs=16,concurrent_agents=2048,
        agent_samples_per_training_batch=65536,Adam_steps_per_training_batch=32,
        reward='raw env strict historical-distance progress +0.1; no agent-count division',
        bootstrap='SF2 current-value timeout approximation; advantage std minimum1e-7',
        matched_layout='native [actor,time] flatten, no minibatch shuffling',
        all_native_minibatch_slices_matched=True,cpu_affinity=sorted(os.sched_getaffinity(0)))
    if bank_episodes>64:
        from environments import CpuEnvironment,CudaEnvironment
        tasks=load(seeds[0],env_id=0,num_envs=16,bank_episodes=bank_episodes)
        cpu=CpuEnvironment(tasks);gpu=CudaEnvironment(tasks)
        cpu.episodes[:]=63;cpu.steps[:]=512
        gpu.episodes.fill_(63);gpu.sim.step_counts.fill_(512)
        co=cpu.reset_due();go=gpu.reset_due()
        for key,value in cpu.state().items():assert np.array_equal(value,gpu.state()[key]),key
        assert np.array_equal(cpu.positions(),tasks['positions'][64])
        torch.testing.assert_close(torch.from_numpy(co),go.cpu(),atol=1e-6,rtol=0)
        record['extended_bank_reset_63_to_64']=dict(equal=True,observation_max_error=float(np.max(np.abs(co-go.cpu().numpy()))))
    (root/'parameter-check.json').write_text(json.dumps(record,indent=2))
    lines=['Pretraining parameter checks completed','',
        '16 environments,128agents/env,2048 concurrent agents; rollout32,65536 samples and32 Adam updates per macrobatch.',
        f'CostTracer has161734 parameters; canonical initial weights for seeds{list(seeds)} match tensor by tensor.',
        f'Each task bank has positions/goals shaped[{bank_episodes},16,128,2] and streams shaped[{bank_episodes},16,128,513,2].',
        'This RL workload uses128 agents per environment.',
        'SF keeps its default unshuffled order; JointLoom uses matching actor-major trajectory groups. All32 minibatches were checked for equality.',
        'The SF2-compatible entry point uses advantage std floor1e-7 and current-value timeout approximation in place of the older v1-style1e-3/final-observation bootstrap settings.',
        'Reward statistics use raw environment reward before GAE/bootstrap adjustment. Scale1 and native clipping at1000 leave rewards in[0,.1] unchanged.','']
    lines += [f'{k}: {v}' for k,v in actual.items()]
    (root/'TRAINING_PARAMETERS.txt').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=128)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--profiles',nargs='+',choices=['sf-sync','sf-async','jointloom'],default=['sf-sync','sf-async','jointloom'])
    p.add_argument('--task-bank-episodes',type=int,default=64)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--plan-file',type=Path)
    p.add_argument('--experiment-prefix',type=str)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=False)
    prefix=args.experiment_prefix or root.name
    impl=Path(__file__).resolve().parent;python=bootstrap.PYTHON
    shutil.copytree(impl,root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(args.plan_file or impl.parent/'REWARD_CURVE_PLAN.txt',root/'REWARD_CURVE_PLAN.txt')
    for seed in args.seeds:prepare(seed,num_envs=16,bank_episodes=args.task_bank_episodes)
    audit(root,bank_episodes=args.task_bank_episodes,seeds=args.seeds)
    print('parameters and 128 agents/env checked',flush=True)
    jobs=[];results=[]
    def execute(profile,seed,iterations,phase):
        name=f'{phase}-{profile}-s{seed}';out=root/name
        samples=iterations*65536
        if profile=='jointloom':
            cmd=[python,impl/'train_pipeline.py','--profile','jointloom_optimized','--sf2-compatible',
                '--num-envs','16','--seed',str(seed),'--iterations',str(iterations),
                '--minibatch-order','sf_env_major','--output',out]
            cmd += ['--task-bank-episodes',str(args.task_bank_episodes),'--checkpoint-every',str(args.checkpoint_every)]
        else:
            out.mkdir()
            cmd=[python,impl/'sf_native.py','--algo=APPO','--env=jointloom_costtracer_pogema',
                '--experiment='+prefix+'-'+name,'--train_dir='+str(SF_ROOT/'runs'),
                '--seed='+str(seed),'--async_rl='+str(profile=='sf-async'),
                '--num_workers=8','--num_envs_per_worker=2','--shuffle_minibatches=False',
                '--costtracer_reward_log','--costtracer_iterations='+str(iterations),
                '--train_for_env_steps='+str(samples-1),
                '--costtracer_initial_weights='+str(SF_ROOT/f'initial-seed-{seed}.pt'),
                '--costtracer_observer_dir='+str(out)]
            cmd += ['--costtracer_task_bank_episodes='+str(args.task_bank_episodes),
                '--train_for_seconds=3600','--save_every_sec=60','--keep_checkpoints=20']
        cmd=[str(v) for v in cmd]
        env=dict(os.environ,WANDB_MODE='disabled',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        for key in ('JOINTLOOM_NEURAL_PROFILE_DIR','JOINTLOOM_MODEL_STAGE_PROFILE'):env.pop(key,None)
        entry=dict(run=name,profile=profile,seed=seed,iterations=iterations,status='running',started=time.time(),command=cmd)
        jobs.append(entry);(root/'status.json').write_text(json.dumps(jobs,indent=2));print('starting',name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env).returncode
        entry.update(status='complete' if code==0 else 'failed',finished=time.time(),exit_code=code)
        (root/'status.json').write_text(json.dumps(jobs,indent=2))
        if code:raise RuntimeError('failed: '+str(root/(name+'.log')))
        if profile=='jointloom':
            checkpoint=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
            trained=checkpoint['agent_samples']
            rows=[json.loads(s) for s in (out/'iterations.jsonl').read_text().splitlines()]
            assert len(rows)==iterations and all('sample_completed_perf_counter_ns' in r for r in rows)
            canonical=torch.load(SF_ROOT/f'initial-seed-{seed}.pt',map_location='cpu',weights_only=True)
            initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v,initial[k]) for k,v in canonical.items())
            log_groups=iterations;envs=16;agents=128
        else:
            folders=list((SF_ROOT/'runs'/(prefix+'-'+name)/'checkpoint_p0').glob('checkpoint_*.pth'))
            checkpoint=torch.load(max(folders,key=lambda p:p.stat().st_mtime),map_location='cpu',weights_only=False)
            trained=checkpoint['env_steps']
            records=[json.loads(p.read_text()) for p in out.glob('reward-env-*-pid-*.json')]
            active=[r for r in records if r['records']]
            assert {r['env_id'] for r in active}==set(range(16)) and all(r['agents']==128 for r in active)
            log_groups=sum(sum(v['kind']=='rollout' for v in r['records']) for r in active)//16
            assert log_groups>=iterations
            envs=16;agents=128
        steps=sorted(set(float(v['step']) for v in checkpoint['optimizer']['state'].values()))
        assert steps==[trained/2048] and all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values())
        if profile!='sf-async':assert trained==samples and steps==[iterations*32.]
        result=dict(run=name,phase=phase,profile=profile,seed=seed,iterations=iterations,
            num_envs=envs,agents_per_env=agents,requested_processed_samples=samples,
            checkpoint_processed_samples=trained,checkpoint_adam_steps=steps,
            recorded_full_reward_groups=log_groups,inflight_extra_samples=trained-samples)
        results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2))
        print('complete',json.dumps(result),flush=True)
    for profile in args.profiles:execute(profile,args.seeds[0],2,'probe')
    for seed in args.seeds:
        order=('sf-sync','sf-async','jointloom') if seed%2==0 else ('jointloom','sf-async','sf-sync')
        for profile in order:
            if profile in args.profiles:execute(profile,seed,args.iterations,'formal')
    print('all reward training runs complete',flush=True)


if __name__=='__main__':main()
