"""Re-measure the identical legacy second-decision input to diagnose JIT cache misses.

This reads the retained E4b source without modifying it. No training or new
task distribution is introduced. Repeated calls all receive exactly the same
post-warm-up state, so episode progress does not confound the cache diagnosis.
"""
import bootstrap
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch


LEGACY = bootstrap.input_path('JOINTLOOM_LEGACY_SOURCE', bootstrap.ROOT/'src')


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    os.sched_setaffinity(0,set(range(16)))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from aligned_tasks import PAPER_INPUT,PAPER_CHECKPOINT
    from policies import MagatPolicy
    old=load('legacy_jax_dispatch',LEGACY/'baselines/jax_sim_baseline.py')
    complete=load('legacy_complete_dispatch',LEGACY/'mapf_cuda/evaluation/jax_torch_complete.py')
    with np.load(PAPER_INPUT) as data:
        arrays={name:data[name][:args.envs].copy() for name in ('grids','positions','goals','arrived')}
    config=old.EnvConfig(height=arrays['grids'].shape[1],width=arrays['grids'].shape[2],
        num_agents=256,task_mode='standard_mapf',max_episode_steps=2)
    # Match the legacy device_put calls, which do not explicitly bind a device.
    grids=jax.device_put(arrays['grids'].astype(np.int32))
    positions=jax.device_put(arrays['positions'].astype(np.int32))
    goals=jax.device_put(arrays['goals'].astype(np.int32))
    arrived=jax.device_put(arrays['arrived'].astype(bool))
    keys=jax.random.split(jax.random.PRNGKey(20260723),args.envs)
    state=old.reset_frozen_jit(config,positions,goals,arrived,grids,keys)
    energy_jit=jax.jit(old.build_batched_energy_maps,static_argnums=(0,))
    energy_parts=[]
    for start in range(0,args.envs,16):
        e=energy_jit(config,grids[start:start+16],goals[start:start+16])
        jax.block_until_ready(e);energy_parts.append(e)
    energy=energy_parts[0] if len(energy_parts)==1 else jnp.concatenate(energy_parts,axis=0)
    jax.block_until_ready(energy)
    obs=jax.jit(old.build_batched_aligned_observations_from_energy_maps,static_argnums=(0,))
    obs.lower(config,state,grids,energy).compile()
    transition=jax.jit(lambda s,a,g:old.step_transition_jit(config,s,a,g))
    transition.lower(state,jax.ShapeDtypeStruct((args.envs,256),jnp.int32),grids).compile()
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda')
    statics=complete.make_eager_batch_statics(args.envs,256,device='cuda:0')
    obs_call=lambda s,g,e:obs(config,s,g,e)

    def step(value):
        return complete._jax_torch_policy_step(policy,value,grids,energy,obs_call,transition,
            statics,device='cuda:0',communication_radius=7.0)

    def commitment(value):
        return {k:bool(v.committed) for k,v in zip(value._fields,value)}

    initial_commitment=commitment(state)
    with torch.inference_mode():
        warm_started=time.perf_counter()
        warm_state,_=step(state)
        jax.block_until_ready(warm_state);torch.cuda.synchronize()
        warm_ms=(time.perf_counter()-warm_started)*1000
        rows=[];previous=None
        for repeat in range(3):
            jax.block_until_ready(warm_state);torch.cuda.synchronize()
            before=dict(observation=obs._cache_size(),transition=transition._cache_size())
            # First repeated call has exactly the original steady-after-one
            # input. Logging exposes whether compilation happens in its timer.
            with jax.log_compiles(repeat==0):
                started=time.perf_counter()
                successor,actions=step(warm_state)
                jax.block_until_ready(successor)
                elapsed=(time.perf_counter()-started)*1000
            after=dict(observation=obs._cache_size(),transition=transition._cache_size())
            snapshot=[np.asarray(v) for v in jax.tree.leaves((successor,actions))]
            identical=True if previous is None else all(np.array_equal(a,b) for a,b in zip(snapshot,previous))
            previous=snapshot
            row=dict(repeat=repeat,wall_ms=elapsed,cache_entries_before=before,
                cache_entries_after=after,identical_output_to_previous=identical)
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(envs=args.envs,agents=256,source=str(LEGACY),frozen_input=str(PAPER_INPUT),
        checkpoint=str(PAPER_CHECKPOINT),initial_state_committed=initial_commitment,
        post_warm_state_committed=commitment(warm_state),warmup_full_step_ms=warm_ms,
        rows=rows,scope='same legacy second-decision state reused; first repeat matches steady-after-one boundary; no modified legacy code')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
