"""Direct data-pipeline replay with ALL model evaluation/optimization removed.

Native SF buffers, TensorDict batch copies, Learner._prepare_batch (cached
bootstrap only), and native minibatch selection are retained. Driver is an
explicit synchronous replay, not a full native SF training-throughput run.
"""
import bootstrap
import argparse
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import triton
from environments import CudaEnvironment
from measure_env_delivery import cpu_worker,receive
from model import CostTracer,configure_torch
from sf_tasks import SF_ROOT,load
from sf2_compat import sf2_gae
from optimized_gpu import _gather_obs,_gather_scalars
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing


class CachedBootstrap:
    training=True
    values=None
    def __call__(self,obs,rnn,values_only=True):return {'values':self.values}
    def normalize_obs(self,obs):return obs
    def device_for_input_tensor(self,key):return torch.device('cuda')
    def type_for_input_tensor(self,key):return torch.float32


def make_trace(seed,ne,steps,out):
    configure_torch(seed)
    torch.backends.cudnn.benchmark=True
    env=CudaEnvironment(load(seed,num_envs=ne));n=ne*128
    checkpoint=SF_ROOT/'supply-scaling-8cores-v1'/f'jointloom-e{ne}-s{seed}'/'final.pt'
    model=CostTracer().cuda()
    model.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
    packed=torch.empty((steps,n,8),device='cuda')
    values=torch.empty((steps+1,n),device='cuda')
    rng=torch.Generator(device='cuda').manual_seed(seed+47000)
    with torch.no_grad():
        for i in range(steps+1):
            logits,v=model(env.obs.reshape(n,2,11,11));values[i].copy_(v)
            if i==steps:break
            actions=torch.multinomial(logits.softmax(-1),1,generator=rng)
            lp=logits.log_softmax(-1).gather(-1,actions)
            packed[i].copy_(torch.cat((actions.float(),lp,v[:,None],logits),-1))
            env.step(actions.reshape(ne,128))
            if (i+1)%512==0:env.reset_due()
    torch.cuda.synchronize()
    torch.save(dict(packed=packed.cpu(),values=values.cpu()),out/'cached-model-outputs.pt')
    del model,env
    return packed,values,str(checkpoint)


def state_check(step,state,obs,adv,ret):
    return dict(step=step,state=state,obs=obs,
        advantages=adv.detach().cpu().numpy(),returns=ret.detach().cpu().numpy())


def native_data(seed,ne,steps,packed,values):
    n=ne*128
    cfg=AttrDict(json.loads((SF_ROOT/'runs'/f'neural-residual-8cores-v1-sf-async-e{ne}-s{seed}'/'config.json').read_text()))
    env_info=EnvInfo(obs_space=gym.spaces.Dict(obs=gym.spaces.Box(-1.,1.,(2,11,11),dtype=np.float32)),
        action_space=gym.spaces.Discrete(5),num_agents=128,gpu_actions=False,
        gpu_observations=False,action_splits=None,all_discrete=None,frameskip=1)
    trajectory=alloc_trajectory_tensors(env_info,n,32,1,'cpu',True)
    batch=alloc_trajectory_tensors(env_info,n,32,1,'cuda',False)
    trajectory['rnn_states'].zero_()
    learner=object.__new__(Learner)
    learner.cfg=cfg;learner.policy_id=0;learner.train_step=0;learner.device=torch.device('cuda')
    learner.actor_critic=CachedBootstrap()
    learner.param_server=SimpleNamespace(policy_lock=nullcontext())
    learner.timing=Timing('native SF data replay')
    shared_actions=torch.empty((ne,128),dtype=torch.int32).share_memory_()
    shared_outputs=torch.empty((ne,128,8)).share_memory_()
    shared_obs=torch.empty((ne,128,2,11,11)).share_memory_()
    gpu_obs=torch.empty_like(shared_obs,device='cuda')
    ctx=mp.get_context('spawn');conns=[];processes=[]
    try:
        for worker in range(8):
            parent,child=ctx.Pipe()
            process=ctx.Process(target=cpu_worker,args=(child,seed,ne,worker,shared_actions,shared_obs,
                trajectory,shared_outputs))
            process.start();child.close();conns.append(parent);processes.append(process)
        for conn in conns:assert receive(conn)=='ready'
        gpu_obs.copy_(shared_obs);torch.cuda.synchronize()
        # Warm native GAE/minibatch code on correctly shaped data, outside timing.
        from sample_factory.algo.utils.rl_utils import gae_advantages
        for _ in range(3):
            gae_advantages(torch.zeros(n,32,device='cuda'),torch.zeros(n,32,device='cuda'),
                torch.zeros(n,33,device='cuda'),torch.ones(n,33,device='cuda'),.96,.95)
        torch.cuda.synchronize()
        rows=[];checks=[]
        for block in range(steps//32):
            torch.cuda.synchronize();begin=time.perf_counter()
            for k in range(32):
                i=block*32+k
                # Only model outputs are replayed; CPU environment/views remain online.
                shared_outputs.copy_(packed[i].reshape(ne,128,8).cpu())
                shared_actions.copy_(shared_outputs[:,:,0])
                for conn in conns:conn.send(('step',k))
                for conn in conns:assert receive(conn)=='done'
                gpu_obs.copy_(shared_obs,non_blocking=True)
            torch.cuda.synchronize();sample_end=time.perf_counter()
            # Same TensorDict slice-copy expression as native Batcher, 16 worker splits.
            for group in range(16):
                s=slice(group*n//16,(group+1)*n//16)
                batch[s]=trajectory[s]
            learner.actor_critic.values=values[(block+1)*32]
            prepared,size,invalids=learner._prepare_batch(batch)
            torch.cuda.synchronize();prepare_end=time.perf_counter()
            minibatches=learner._get_minibatches(2048,size)
            for indices in minibatches:
                mb=learner._get_minibatch(prepared,indices)
            torch.cuda.synchronize();end=time.perf_counter()
            rows.append(dict(block=block,sample_seconds=sample_end-begin,
                batch_gae_seconds=prepare_end-sample_end,minibatch_seconds=end-prepare_end,
                total_seconds=end-begin,minibatches=len(minibatches),invalids=invalids))
            if block in (7,14,15):
                for conn in conns:conn.send('state')
                states=[receive(conn) for conn in conns]
                state={k:np.concatenate([s[k] for s in states]) for k in states[0]}
                checks.append(state_check((block+1)*32,state,gpu_obs.cpu().numpy(),
                    prepared['advantages'].reshape(n,32).T,prepared['returns'].reshape(n,32).T))
        return rows,checks,dict(shuffle_minibatches=cfg.shuffle_minibatches,
            native_methods=['alloc_trajectory_tensors','TensorDict.__setitem__',
                'Learner._prepare_batch','Learner._get_minibatches','Learner._get_minibatch'])
    finally:
        for conn in conns:
            try:conn.send('close')
            except (BrokenPipeError,EOFError):pass
        for process in processes:
            process.join(5)
            if process.is_alive():process.terminate();process.join()


def cuda_data(seed,ne,steps,packed,values):
    n=ne*128;m=2048
    env=CudaEnvironment(load(seed,num_envs=ne))
    obs=torch.empty((32,n,2,11,11),device='cuda')
    fields={k:torch.empty((32,n),device='cuda',dtype=torch.bool if k=='reset' else
        torch.int64 if k=='actions' else torch.float32) for k in ('actions','logp','values','next_values','rewards','reset')}
    adv=torch.empty((32,n),device='cuda');ret=torch.empty_like(adv)
    actor_input=torch.empty((n,2,11,11),device='cuda')
    mb_obs=torch.empty((m,2,11,11),device='cuda');mb_actions=torch.empty(m,dtype=torch.int64,device='cuda')
    mb_scalars=torch.empty((4,m),device='cuda')
    sample_actions=packed[:,:,0].long().reshape(steps,ne,128)
    rng=torch.Generator(device='cuda').manual_seed(seed+30000)
    sf2_gae(fields['rewards'],fields['values'],fields['next_values'],fields['reset'],adv,ret)
    order=torch.arange(32*n,device='cuda')
    def gather(start):
        _gather_obs[(triton.cdiv(m*242,1024),)](obs,order,mb_obs,start,m,False,1024)
        _gather_scalars[(triton.cdiv(m,256),)](order,fields['actions'],fields['logp'],fields['values'],adv,ret,
            mb_actions,mb_scalars,start,m,256)
    gather(0);torch.cuda.synchronize()
    rows=[];checks=[]
    for block in range(steps//32):
        torch.cuda.synchronize();begin=time.perf_counter()
        for k in range(32):
            i=block*32+k
            obs[k].copy_(env.obs.reshape(n,2,11,11))
            fields['actions'][k].copy_(sample_actions[i].reshape(n))
            fields['logp'][k].copy_(packed[i,:,1]);fields['values'][k].copy_(values[i])
            _,reward,_,reset=env.step(sample_actions[i])
            fields['rewards'][k].copy_(reward.reshape(n))
            fields['reset'][k].copy_(reset[:,None].expand(-1,128).reshape(n))
            fields['next_values'][k].copy_(values[i+1])
            if (i+1)%512==0:env.reset_due()
            actor_input.copy_(env.obs.reshape(n,2,11,11))
        torch.cuda.synchronize();sample_end=time.perf_counter()
        sf2_gae(fields['rewards'],fields['values'],fields['next_values'],fields['reset'],adv,ret)
        torch.cuda.synchronize();prepare_end=time.perf_counter()
        order=torch.randperm(32*n,device='cuda',generator=rng)
        for start in range(0,32*n,m):gather(start)
        torch.cuda.synchronize();end=time.perf_counter()
        rows.append(dict(block=block,sample_seconds=sample_end-begin,
            batch_gae_seconds=prepare_end-sample_end,minibatch_seconds=end-prepare_end,
            total_seconds=end-begin,minibatches=32*n//m,invalids=0))
        if block in (7,14,15):
            checks.append(state_check((block+1)*32,env.state(),env.obs.cpu().numpy(),adv,ret))
    return rows,checks,dict(shuffle_minibatches=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--envs',type=int,nargs='+',default=[16,32]);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args();root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(__file__).parent,root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (root/'metadata.json').write_text(json.dumps(dict(metric='direct model-free data-pipeline replay',
        cpu_affinity=sorted(os.sched_getaffinity(0)),gpu=torch.cuda.get_device_name(),agents=128,
        steps=512,rollout=32,minibatch=2048,workers=8,splits=2,seeds=args.seeds,envs=args.envs,
        removed='all model evaluation/action distribution sampling, losses/backprop/grad clipping/Adam, model invocation/update scheduling and weight synchronization',
        retained='online environment/reward/goals/distances/observations, observation input delivery, trajectory writes and transfers, native SF batch preparation/GAE, minibatch preparation',
        identity='standalone synchronous replay of native SF data functions, not native SF training throughput',
        replay_outputs='same frozen checkpoint action/logit/logp/value trace for both paths; inputs and environment not precomputed',
        CPU_output_transfer='packed replay outputs, one D2H transfer per joint step (fewer than original SF field-wise copies)',
        warmup='exclude first two of 16 rollout blocks; include final time-limit reset'),indent=2))
    results=[]
    for seed in args.seeds:
        for ne in args.envs:
            out=root/f'e{ne}-s{seed}';out.mkdir()
            print(f'recording e{ne} s{seed}',flush=True)
            packed,values,checkpoint=make_trace(seed,ne,512,out)
            measured={}
            for profile in (('native-data','jointloom') if seed%2==0 else ('jointloom','native-data')):
                print(f'measuring {profile} e{ne} s{seed}',flush=True)
                measured[profile]=(native_data if profile=='native-data' else cuda_data)(seed,ne,512,packed,values)
            validations=[]
            for c,g in zip(measured['native-data'][1],measured['jointloom'][1]):
                equality={k:bool(np.array_equal(c['state'][k],g['state'][k])) for k in c['state']}
                errors={k:float(np.max(np.abs(c[k]-g[k]))) for k in ('obs','advantages','returns')}
                if not all(equality.values()) or not all(np.isfinite(v) for v in errors.values()) or any(v>1e-5 for v in errors.values()):
                    raise AssertionError(dict(step=c['step'],states=equality,errors=errors))
                validations.append(dict(step=c['step'],state_equal=equality,max_errors=errors))
            result=dict(seed=seed,num_envs=ne,checkpoint=checkpoint,validations=validations)
            for profile,(rows,_,config) in measured.items():
                result[profile]=dict(config=config,rows=rows,
                    mean={k:float(np.mean([r[k] for r in rows[2:]])) for k in
                        ('sample_seconds','batch_gae_seconds','minibatch_seconds','total_seconds')})
            result['speedup']=result['native-data']['mean']['total_seconds']/result['jointloom']['mean']['total_seconds']
            (out/'result.json').write_text(json.dumps(result,indent=2));results.append(result)
            (root/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(dict(seed=seed,num_envs=ne,speedup=result['speedup'],
                cpu=result['native-data']['mean'],cuda=result['jointloom']['mean'])),flush=True)
    summary={}
    for ne in args.envs:
        rows=[r for r in results if r['num_envs']==ne];summary[ne]={}
        for profile in ('native-data','jointloom'):
            summary[ne][profile]={k:float(np.mean([r[profile]['mean'][k] for r in rows])) for k in rows[0][profile]['mean']}
        summary[ne]['speedup']=summary[ne]['native-data']['total_seconds']/summary[ne]['jointloom']['total_seconds']
        summary[ne]['seed_speedups']=[r['speedup'] for r in rows]
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
