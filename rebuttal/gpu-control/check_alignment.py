"""Meaningful MAPF semantic and policy-integration checks, with ordinary reports."""
import bootstrap
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from baselines import jax_sim_baseline as reference
from discrete_mapf import MAPFConfig, DiscreteMAPF, batch_transition_environment
from observations import CachedObservations


def compare_state(actual, expected):
    for name in ("pos", "target", "arrived", "rng_key", "step_count", "terminated", "truncated"):
        np.testing.assert_array_equal(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)), err_msg=name)


def canonical_goal_channel(config, positions, goals):
    delta=np.asarray(goals,dtype=np.int64)-np.asarray(positions,dtype=np.int64)
    radius=config.fov_radius; mid=config.pyg_obs_diam//2
    near=np.max(np.abs(delta),axis=-1)<=radius
    vertical=np.abs(delta[...,1])>=np.abs(delta[...,0])
    x=np.where(vertical,mid+np.rint(mid*delta[...,0]/np.maximum(1,np.abs(delta[...,1]))),mid*(np.sign(delta[...,0])+1)).astype(int)
    y=np.where(vertical,mid*(np.sign(delta[...,1])+1),mid+np.rint(mid*delta[...,1]/np.maximum(1,np.abs(delta[...,0])))).astype(int)
    x=np.where(near,mid+delta[...,0],x);y=np.where(near,mid+delta[...,1],y)
    result=np.zeros((*delta.shape[:-1],config.pyg_obs_diam,config.pyg_obs_diam),dtype=np.float32)
    env,agent=np.indices(delta.shape[:-1])
    result[env,agent,x,y]=1
    return result


def check_traces(config, grids, positions, goals, arrived, actions):
    env = DiscreteMAPF(config)
    batch = batch_transition_environment(env, len(grids))
    keys = jax.random.split(jax.random.PRNGKey(31), len(grids))
    grids, positions, goals, arrived = map(jnp.asarray, (grids, positions, goals, arrived))
    actual = jax.jit(jax.vmap(env.reset_from_arrays))(keys, grids, positions, goals, arrived)
    kwargs = {k:v for k,v in asdict(config).items() if k in reference.EnvConfig._fields}
    rc = reference.EnvConfig(**kwargs)
    expected = reference.reset_frozen_jit(rc, positions, goals, arrived, grids, keys)
    cache = CachedObservations(config, grids)
    compare_state(actual, expected)
    maximum = 0.0
    legacy_goal_differences=0
    for step, action in enumerate(actions):
        if step in (0, 1, len(actions) - 1):
            a = np.asarray(cache.prepare(actual))
            b = np.asarray(reference.build_batched_aligned_observations(rc, expected, grids))
            maximum = max(maximum, float(np.max(np.abs(a[:,:,[0,1,3]]-b[:,:,[0,1,3]]))))
            np.testing.assert_allclose(a[:,:,[0,1,3]], b[:,:,[0,1,3]], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(a[:,:,2],canonical_goal_channel(config,actual.pos,actual.target))
            legacy_goal_differences+=int(np.count_nonzero(np.any(a[:,:,2]!=b[:,:,2],axis=(-1,-2))))
        _, actual, reward, done, info = batch.step(jax.random.PRNGKey(step), actual, jnp.asarray(action))
        expected, expected_reward, terminated, truncated, expected_info = reference.step_transition_jit(
            rc, expected, jnp.asarray(action), grids)
        compare_state(actual, expected)
        np.testing.assert_array_equal(np.asarray(reward), np.asarray(expected_reward))
        np.testing.assert_array_equal(np.asarray(done), np.asarray(terminated | truncated))
        np.testing.assert_array_equal(np.asarray(info['goal_changed_flags']), np.asarray(expected_info['goal_changed_flags']))
    return dict(config=asdict(config), envs=len(grids), transitions=len(actions)*len(grids),
        non_goal_input_max_error=maximum,canonical_goal_projection_equal=True,
        legacy_jax_float_projection_differences=legacy_goal_differences)


def semantic_checks():
    config = MAPFConfig(height=8, width=9, num_agents=4, max_episode_steps=4)
    # Shared batch contains vertex, swap, blocked chain, allowed rotation,
    # following chain, wall and boundary cases. Different maps stay independent.
    positions = np.array([
        [[2,1],[2,3],[6,6],[6,7]], [[2,1],[2,2],[6,6],[6,7]],
        [[2,1],[2,2],[2,3],[6,7]], [[2,1],[2,2],[3,2],[3,1]],
        [[2,1],[2,2],[2,3],[6,7]], [[2,1],[4,4],[6,6],[6,7]],
        [[0,1],[4,4],[6,6],[6,7]], [[1,1],[4,4],[6,6],[6,7]],
    ],dtype=np.int32)
    goals = np.broadcast_to(np.array([[1,1],[1,2],[1,3],[1,4]]),positions.shape).copy()
    goals[7] = positions[7]
    arrived = np.zeros(positions.shape[:2],dtype=bool)
    arrived[7] = True
    grids = np.zeros((8,8,9),dtype=np.uint8)
    grids[5,2,2] = 1
    initial = np.array([[4,3,0,0],[4,3,0,0],[4,4,0,0],[4,2,3,1],
                        [4,4,4,0],[4,0,0,0],[1,0,0,0],[4,2,3,1]])
    actions = np.stack([initial]*6)
    reports = [check_traces(config,grids,positions,goals,arrived,actions)]
    # Independent explicit expected positions prevent shared-reference errors
    # from hiding incorrect conflict behavior.
    env = DiscreteMAPF(config)
    keys = jax.random.split(jax.random.PRNGKey(31),8)
    state = jax.jit(jax.vmap(env.reset_from_arrays))(keys,jnp.asarray(grids),jnp.asarray(positions),jnp.asarray(goals),jnp.asarray(arrived))
    _, state, *_ = batch_transition_environment(env,8).step(jax.random.PRNGKey(1),state,jnp.asarray(initial))
    expected_pos=positions.copy()
    expected_pos[0,0]=[2,2]
    expected_pos[3]=[[2,2],[3,2],[3,1],[2,1]]
    expected_pos[4,:3]=[[2,2],[2,3],[2,4]]
    np.testing.assert_array_equal(np.asarray(state.pos),expected_pos)
    rng=np.random.default_rng(20261109)
    for mode in ('standard_mapf','lifelong'):
        c=MAPFConfig(height=12,width=13,num_agents=8,task_mode=mode,max_episode_steps=48,max_steps=48)
        g=(rng.random((4,12,13))<.12).astype(np.uint8)
        p=[]; t=[]
        for grid in g:
            cells=np.argwhere(grid==0)
            chosen=rng.choice(len(cells),16,replace=False)
            p.append(cells[chosen[:8]]);t.append(cells[chosen[8:]])
        p=np.array(p,dtype=np.int32);t=np.array(t,dtype=np.int32)
        # Ensure goal refresh is exercised immediately in lifelong mode.
        t[:,0]=p[:,0]
        a=rng.integers(0,5,size=(64,4,8),dtype=np.int32)
        a[0,:,0]=0
        reports.append(check_traces(c,g,p,t,np.zeros((4,8),dtype=bool),a))
    # Arrival on the horizon must terminate, not truncate; arrived agents freeze.
    c=MAPFConfig(height=4,width=5,num_agents=1,max_episode_steps=1)
    reports.append(check_traces(c,np.zeros((2,4,5),dtype=np.uint8),np.array([[[1,1]],[[1,1]]]),
        np.array([[[2,1]],[[3,3]]]),np.zeros((2,1),dtype=bool),np.array([[[2],[0]],[[4],[4]]])))
    # Public CAMAR reset/step and upstream batch wrapper also return real views.
    from discrete_mapf import FrozenGridMap
    from camar.wrappers import BatchEnvWrapper
    public=DiscreteMAPF(c,FrozenGridMap(c,positions=[[1,1]],goals=[[2,1]]))
    wrapped=BatchEnvWrapper(public,2)
    obs,state=wrapped.reset(jax.random.PRNGKey(2))
    assert obs.shape==(2,1,4,13,13)
    obs,state,reward,done,_=wrapped.step(jax.random.PRNGKey(3),state,jnp.array([[2],[2]]))
    assert np.all(np.asarray(done)) and np.all(np.asarray(reward)==1)
    return reports


def policy_check():
    import torch
    from aligned_tasks import load_paper_tasks,AlignedResident,PAPER_CHECKPOINT
    from native_extent_inputs import NativeExtentBatchBuilder
    from policies import MagatPolicy
    from runtime import CamarRuntime
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tasks=load_paper_tasks(2)
    camar=CamarRuntime(tasks.grids,tasks.positions,tasks.goals,tasks.initial_arrived)
    resident=AlignedResident(tasks,'standard_mapf')
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda')
    policy.builder=NativeExtentBatchBuilder([g.shape for g in tasks.grids],256,'cuda')
    maxima=[]
    with torch.inference_mode():
        for step in range(8):
            a=camar.build()
            resident.sim.update_derived_state()
            b=policy.builder.build_from_stateful(resident.sim,materialize_edges=True)
            maximum=float((a.x-b.x).abs().max().item())
            torch.testing.assert_close(a.x,b.x,rtol=0,atol=1e-5)
            def ordered(data):
                k=data.edge_index[0]*len(data.x)+data.edge_index[1]
                order=k.argsort()
                return data.edge_index[:,order],data.edge_attr[order]
            ea,aa=ordered(a);eb,ab=ordered(b)
            torch.testing.assert_close(ea,eb,rtol=0,atol=0)
            torch.testing.assert_close(aa,ab,rtol=0,atol=0)
            la=policy.model(a.x,a);lb=policy.model(b.x,b)
            torch.testing.assert_close(la,lb,rtol=1e-4,atol=1e-5)
            action=la.argmax(-1).reshape(2,256).to(torch.uint8)
            maxima.append(dict(step=step,input_max_error=maximum,
                logit_max_error=float((la-lb).abs().max().item()),
                differing_argmax=int((la.argmax(-1)!=lb.argmax(-1)).sum().item())))
            # Shared actions distinguish transition semantics from numerical
            # near-tie differences between independently computed model inputs.
            camar.step(action)
            resident.sim.update_actions(action);resident.sim.step_sim_only()
            ca=camar.result();cu=resident.result()
            for name in ('positions','goals','arrived'):
                np.testing.assert_array_equal(ca[name],cu[name],err_msg=name)
    return dict(envs=2,agents=256,steps=8,checkpoint=str(PAPER_CHECKPOINT),comparisons=maxima,
                shared_action_final_state_equal=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--skip-policy',action='store_true')
    args=p.parse_args()
    started=time.perf_counter()
    result=dict(semantic=semantic_checks(),jax_version=jax.__version__,devices=list(map(str,jax.devices())))
    if not args.skip_policy:result['policy']=policy_check()
    result['seconds']=time.perf_counter()-started
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
