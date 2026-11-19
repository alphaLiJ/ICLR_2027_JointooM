"""Original E4b workload/configuration with initial and post-DLPack JIT warmup.

Only the one-time compilation exclusion is corrected. The timed input remains
the original second decision, not a resampled task or a long-rollout average.
"""
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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch

from probe_legacy_dispatch import LEGACY,load


def configure():
    os.sched_setaffinity(0,set(range(16)))
    torch.set_num_threads(1)
    # Original E4b does not override these defaults. Explicitly restore the
    # original runner's precision behavior, rather than the later TF32-off test.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False


def tasks(envs):
    from aligned_tasks import PAPER_INPUT
    with np.load(PAPER_INPUT) as source:
        data={k:source[k][:envs].copy() for k in ('grids','positions','goals','arrived')}
    return SimpleNamespace(**data,num_envs=envs,num_agents=256,horizon=2)


class LegacyJAX:
    def __init__(self,batch):
        self.reference=load('paper_aligned_legacy_jax',LEGACY/'baselines/jax_sim_baseline.py')
        self.helper=load('paper_aligned_legacy_complete',LEGACY/'mapf_cuda/evaluation/jax_torch_complete.py')
        c=self.reference.EnvConfig(height=batch.grids.shape[1],width=batch.grids.shape[2],
            num_agents=256,task_mode='standard_mapf',max_episode_steps=2)
        self.config=c
        self.grids=jax.device_put(batch.grids.astype(np.int32))
        positions=jax.device_put(batch.positions.astype(np.int32))
        goals=jax.device_put(batch.goals.astype(np.int32))
        arrived=jax.device_put(batch.arrived.astype(bool))
        keys=jax.random.split(jax.random.PRNGKey(20260723),batch.num_envs)
        self.initial=self.reference.reset_frozen_jit(c,positions,goals,arrived,self.grids,keys)
        cold=jax.jit(self.reference.build_batched_energy_maps,static_argnums=(0,))
        parts=[]
        for start in range(0,batch.num_envs,16):
            part=cold(c,self.grids[start:start+16],goals[start:start+16])
            jax.block_until_ready(part);parts.append(part)
        self.energy=parts[0] if len(parts)==1 else jnp.concatenate(parts,axis=0)
        jax.block_until_ready(self.energy)
        self.obs=jax.jit(self.reference.build_batched_aligned_observations_from_energy_maps,static_argnums=(0,))
        self.obs.lower(c,self.initial,self.grids,self.energy).compile()
        self.transition=jax.jit(lambda s,a,g:self.reference.step_transition_jit(c,s,a,g))
        self.transition.lower(self.initial,jax.ShapeDtypeStruct((batch.num_envs,256),jnp.int32),self.grids).compile()
        self.statics=self.helper.make_eager_batch_statics(batch.num_envs,256,device='cuda:0')
        self.reset()

    def reset(self):self.state=self.initial

    def build(self):
        obs=self.obs(self.config,self.state,self.grids,self.energy)
        return self.helper.build_eager_magat_batch(
            self.helper._torch_from_jax(obs),self.helper._torch_from_jax(self.state.pos),
            self.helper._torch_from_jax(self.state.arrived),statics=self.statics)

    def step(self,actions):
        action=self.helper._jax_from_torch(actions.to(torch.int32).contiguous())
        self.state,*_=self.transition(self.state,action,self.grids)

    def synchronize(self):
        jax.block_until_ready(self.state);torch.cuda.synchronize()

    def result(self):
        return {k:np.asarray(v) for k,v in dict(positions=self.state.pos,goals=self.state.target,
            arrived=self.state.arrived,terminated=self.state.terminated,
            truncated=self.state.truncated,step_counts=self.state.step_count).items()}

    def cache_sizes(self):return dict(observation=self.obs._cache_size(),transition=self.transition._cache_size())


class PaperCUDA:
    def __init__(self,batch):
        from expert.a4_runtime import _build_stateful_simulator
        from expert.fixed_magat_plus_runtime import PyGBatchBuilder
        self.batch=batch
        self.sim=_build_stateful_simulator(batch,device='cuda:0',max_episode_steps=2)
        self.builder=PyGBatchBuilder()
        self.needs_energy=True

    def reset(self):
        b=self.batch
        self.sim.load_state(torch.as_tensor(b.positions.astype(np.int16),device='cuda'),
            torch.as_tensor(b.goals.astype(np.int16),device='cuda'),
            torch.as_tensor(b.arrived.astype(np.uint8),device='cuda'),
            torch.zeros(b.num_envs,dtype=torch.int32,device='cuda'))
        self.needs_energy=True

    def build(self):
        if self.needs_energy:
            self.sim.update_derived_state();self.needs_energy=False
        # Match E4b's fixed-goal path: do not poll dirty goals or apply the
        # later native-extent patch on unchanged full 128x128 maps.
        self.sim.build_magat_plus_inputs()
        return self.builder.view_stateful_outputs(self.sim,materialize_edges=False)

    def step(self,actions):
        self.sim.update_actions(actions.to(torch.uint8));self.sim.step_sim_only()

    def synchronize(self):torch.cuda.synchronize()

    def result(self):
        s=self.sim
        return dict(positions=torch.stack((s.cur_x,s.cur_y),-1).cpu().numpy(),
            goals=torch.stack((s.goal_x,s.goal_y),-1).cpu().numpy(),arrived=s.arrived.cpu().numpy(),
            terminated=s.terminated.cpu().numpy(),truncated=s.truncated.cpu().numpy(),
            step_counts=s.step_counts.cpu().numpy())


def make_env(path,batch):
    if path=='legacy_jax':return LegacyJAX(batch)
    if path=='jointloom':return PaperCUDA(batch)
    from runtime import CamarRuntime
    return CamarRuntime(batch.grids,batch.positions,batch.goals,batch.arrived,horizon=2,paper_cache=True)


def child(args):
    configure()
    from aligned_tasks import PAPER_INPUT,PAPER_CHECKPOINT
    from policies import MagatPolicy
    b=tasks(args.envs);env=make_env(args.path,b)
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda')

    def step():
        data=env.build();logits=policy.model(data.x,data)
        dtype=torch.uint8 if args.path=='jointloom' else torch.int32
        action=logits.argmax(-1).reshape(args.envs,256).to(dtype).contiguous()
        env.step(action)

    with torch.no_grad():
        warm=[]
        for _ in range(3):
            env.synchronize();start=time.perf_counter();step();env.synchronize()
            warm.append(1000*(time.perf_counter()-start))
        # Restore the exact original task and advance once. Compiled variants
        # remain warm; input to the timer is still the original second decision.
        env.reset();step();env.synchronize()
        before=env.cache_sizes() if hasattr(env,'cache_sizes') else {}
        start=time.perf_counter();step();env.synchronize()
        wall_ms=1000*(time.perf_counter()-start)
        after=env.cache_sizes() if hasattr(env,'cache_sizes') else {}
    snapshot=env.result()
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'final-state.npz',**snapshot)
    result=dict(path=args.path,envs=args.envs,agents=256,wall_ms=wall_ms,
        original_frozen_input=str(PAPER_INPUT),checkpoint=str(PAPER_CHECKPOINT),
        task_mode='standard_mapf',max_episode_steps=2,resampled_initial_tasks=False,
        initial_arrived_agents=int(b.arrived.sum()),warmup_decision_ms=warm,
        cache_entries_before=before,cache_entries_after=after,
        precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32),
        energy_cache='uint8; 16-environment chunks for JAX/CAMAR; static fixed goals',
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,jax_version=jax.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
        scope='original second-decision input, after warming initial and post-DLPack JIT variants; no sustained rollout claim')
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def check(args):
    configure()
    from aligned_tasks import PAPER_CHECKPOINT
    from policies import MagatPolicy
    b=tasks(args.envs);a=make_env('camar_discrete',b);c=make_env('jointloom',b)
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda');rows=[]
    with torch.no_grad():
        for step in range(2):
            x=a.build();y=c.build()
            # Materialize CUDA edges outside the model for input comparison.
            y=c.builder.view_stateful_outputs(c.sim,materialize_edges=True)
            torch.testing.assert_close(x.x,y.x,rtol=0,atol=1e-5)
            def ordered(d):
                key=d.edge_index[0]*len(d.x)+d.edge_index[1];order=key.argsort()
                return d.edge_index[:,order],d.edge_attr[order]
            ex,ax=ordered(x);ey,ay=ordered(y)
            torch.testing.assert_close(ex,ey,rtol=0,atol=0);torch.testing.assert_close(ax,ay,rtol=0,atol=0)
            lx=policy.model(x.x,x);ly=policy.model(y.x,y)
            tx=lx.argmax(-1);ty=ly.argmax(-1)
            torch.testing.assert_close(lx,ly,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(tx,ty,rtol=0,atol=0)
            action=tx.reshape(args.envs,256).to(torch.int32)
            a.step(action);c.step(action)
            aa=a.result();cc=c.result()
            for key in cc:np.testing.assert_array_equal(aa[key],cc[key],err_msg=key)
            rows.append(dict(step=step,max_input_error=float((x.x-y.x).abs().max().item()),
                max_logit_error=float((lx-ly).abs().max().item()),actions_and_state_equal=True))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(envs=args.envs,agents=256,original_frozen_input=True,rows=rows),indent=2))
    print(args.out.read_text(),flush=True)


def compare(args):
    args.out.mkdir(parents=True,exist_ok=True);runs=[];pairs=[]
    for repeat in range(args.repetitions):
        paths=['legacy_jax','camar_discrete','jointloom'] if repeat%2==0 else ['jointloom','camar_discrete','legacy_jax']
        current={}
        for path in paths:
            out=args.out/f'repeat-{repeat}'/path
            if not (out/'result.json').exists():
                subprocess.run([sys.executable,__file__,'child','--path',path,'--envs',str(args.envs),'--out',str(out)],check=True)
            current[path]=json.loads((out/'result.json').read_text());runs.append(current[path])
        snapshots={p:np.load(args.out/f'repeat-{repeat}'/p/'final-state.npz') for p in paths}
        for path in ['legacy_jax','camar_discrete']:
            equal={k:bool(np.array_equal(snapshots[path][k],snapshots['jointloom'][k])) for k in snapshots['jointloom'].files}
            row=dict(repeat=repeat,path=path,control_ms=current[path]['wall_ms'],jointloom_ms=current['jointloom']['wall_ms'],
                speedup=current[path]['wall_ms']/current['jointloom']['wall_ms'],final_state_equal=equal)
            pairs.append(row);print(json.dumps(row),flush=True)
        (args.out/'summary.json').write_text(json.dumps(dict(envs=args.envs,runs=runs,pairs=pairs),indent=2))
    text=['# Paper E4b configuration, corrected compilation exclusion','',
        f'A256/N{args.envs}; original frozen tasks; uint8 static cache; JAX BFS chunks16;',
        'same checkpoint; original cuDNN TF32 default enabled; original second decision;',
        'initial and post-DLPack JIT variants warmed before timing.','',
        '| Control | Median control ms | Median JointLoom ms | Median paired speedup |',
        '|---|---:|---:|---:|']
    for path in ['legacy_jax','camar_discrete']:
        selected=[p for p in pairs if p['path']==path]
        text.append(f"| {path} | {statistics.median(p['control_ms'] for p in selected):.6f} | {statistics.median(p['jointloom_ms'] for p in selected):.6f} | {statistics.median(p['speedup'] for p in selected):.6f} |")
    text+=['','All canonical final-state fields are reported in summary.json.',
        'This is a retimed second-decision result, not a 256-step rollout average.']
    (args.out/'RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['compare','child','check'])
    p.add_argument('--path',choices=['legacy_jax','camar_discrete','jointloom'])
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='child':child(args)
    elif args.command=='check':check(args)
    else:compare(args)
