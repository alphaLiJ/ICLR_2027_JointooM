"""Call installed Sample Factory model/GAE/loss implementations directly."""
import bootstrap
import json
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
from sf_native import register,configuration,make_actor_critic,CostTracerPogema
from sf_tasks import SF_ROOT,load
from model import CostTracer,configure_torch
from environments import CudaEnvironment
from optimized_gpu import StaticLearner
from sf2_compat import sf2_gae,sf2_loss
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.algo.utils.action_distributions import CategoricalActionDistribution
from sample_factory.algo.learning.learner import Learner


def config():
    folder=SF_ROOT/'validation';folder.mkdir(exist_ok=True)
    return configuration(['--algo=APPO','--env=jointloom_costtracer_pogema','--seed=0',
        '--costtracer_initial_weights='+str(SF_ROOT/'initial-seed-0.pt'),
        '--costtracer_observer_dir='+str(folder)])


def native_loss(logits,values,actions,old_logp,old_values,adv,returns):
    distribution=CategoricalActionDistribution(logits)
    ratio=(distribution.log_prob(actions[:,None])-old_logp).exp().clamp(.05,20.)
    std,mean=torch.std_mean(adv);normalized=(adv-mean)/std.clamp_min(1e-7)
    valid=torch.ones_like(adv,dtype=torch.bool)
    fake=SimpleNamespace(cfg=SimpleNamespace(value_loss_coeff=.5,exploration_loss_coeff=.06878))
    policy= Learner._policy_loss(ratio,normalized,1/1.2,1.2,valid,0)
    value= Learner._value_loss(fake,values,old_values,returns,1.,valid,0)
    entropy=Learner._entropy_exploration_loss(fake,distribution,valid,0)
    return policy+value+entropy


def main():
    configure_torch(0);register();cfg=config()
    obs_space=gym.spaces.Dict({'obs':gym.spaces.Box(-1.,1.,shape=(2,11,11),dtype=np.float32)})
    action_space=gym.spaces.Discrete(5)
    native=make_actor_critic(cfg,obs_space,action_space).cuda()
    canonical=CostTracer().cuda();canonical.load_state_dict(native.state_dict(),strict=True)
    x=torch.rand(128,2,11,11,device='cuda')
    with torch.no_grad():
        result=native({'obs':x},torch.zeros(128,1,device='cuda'))
        logits,values=canonical(x)
    forward=dict(parameters=sum(p.numel() for p in native.parameters()),
        logits_max_error=float((logits-result['action_logits']).abs().max()),
        values_max_error=float((values-result['values']).abs().max()))
    assert forward['parameters']==161734 and forward['logits_max_error']==0 and forward['values_max_error']==0
    print('network',forward,flush=True)
    worst_gae=0.
    for t,n in ((1,1),(7,37),(32,2048)):
        v=torch.randn(t+1,n,device='cuda');r=torch.randn(t,n,device='cuda')
        done=torch.rand(t,n,device='cuda')<.12
        adjusted=r+.96*v[:-1]*done
        expected=gae_advantages(adjusted.T,done.T,v.T,torch.ones_like(v.T,dtype=torch.bool),.96,.95).T
        a,ret=sf2_gae(r,v[:-1],v[1:],done)
        worst_gae=max(worst_gae,float((a-expected).abs().max()))
        torch.testing.assert_close(a,expected,atol=3e-6,rtol=3e-6)
    print('GAE',worst_gae,flush=True)
    worst_loss=0.
    for scale in (1.,1e-5,1e-9):
        l=torch.randn(257,5,dtype=torch.float64,requires_grad=True)
        value=torch.randn(257,dtype=torch.float64,requires_grad=True)
        a=torch.randint(5,(257,));old=torch.randn(257,dtype=torch.float64)
        oldv=torch.randn_like(old);adv=torch.randn_like(old)*scale;ret=torch.randn_like(old)
        ours,_=sf2_loss(l,value,a,old,oldv,adv,ret)
        ref=native_loss(l,value,a,old,oldv,adv,ret)
        g1=torch.autograd.grad(ours,(l,value),retain_graph=True);g2=torch.autograd.grad(ref,(l,value))
        worst_loss=max(worst_loss,abs(float(ours.detach()-ref.detach())),
                       *(float((u-w).abs().max()) for u,w in zip(g1,g2)))
    assert worst_loss<1e-10,worst_loss
    print('loss_and_gradient',worst_loss,flush=True)
    # Actual SF multi-agent adapter and the existing resident simulator see identical states/actions.
    env=CostTracerPogema(cfg,{'env_id':0});co,_=env.reset()
    gpu=CudaEnvironment(load(0,0));worst_obs=0.
    rng=np.random.default_rng(7)
    for k in range(12):
        if k==6:
            env.inner.steps[:]=511;gpu.sim.step_counts.fill_(511)
        actions=rng.integers(5,size=128)
        co,cr,ct,ctr,info=env.step(actions)
        go,gr,gh,gt=gpu.step(torch.as_tensor(actions[None],device='cuda'))
        assert np.array_equal(cr,gr.cpu().numpy()[0]) and np.array_equal(ctr,np.repeat(gt.cpu().numpy(),128))
        if ctr.any():go=gpu.reset_due()
        worst_obs=max(worst_obs,float(np.max(np.abs(co-go.cpu().numpy()[0]))))
        torch.testing.assert_close(torch.from_numpy(co),go[0].cpu(),atol=1e-6,rtol=0)
        for key,a in env.inner.state().items():assert np.array_equal(a,gpu.state()[key]),key
    print('multiagent_adapter',worst_obs,flush=True)
    # Optimized updates are checked against the installed SF loss and native SF network.
    own=CostTracer().cuda();own.load_state_dict(native.state_dict())
    engine=StaticLearner(own,n=2048,minibatch=2048,compile_loss=True,compile_model=True,loss_function=sf2_loss)
    optimizer=torch.optim.Adam(native.parameters(),lr=.00013,betas=(.9,.999),eps=1e-6)
    worst_parameter=0.
    for step in range(3):
        x=torch.rand(2048,2,11,11,device='cuda')
        with torch.no_grad():
            prediction=native({'obs':x},torch.zeros(2048,1,device='cuda'))
            actions=torch.randint(5,(2048,),device='cuda')
            oldlp=prediction['action_logits'].log_softmax(-1).gather(-1,actions[:,None]).squeeze(-1)
        data=dict(obs=x,actions=actions,logp=oldlp,values=prediction['values'],
                  advantages=torch.randn(2048,device='cuda'),returns=torch.randn(2048,device='cuda'))
        prediction=native({'obs':x},torch.zeros(2048,1,device='cuda'))
        total=native_loss(prediction['action_logits'],prediction['values'],*(data[k] for k in
            ('actions','logp','values','advantages','returns')))
        optimizer.zero_grad(set_to_none=True);total.backward()
        torch.nn.utils.clip_grad_norm_(native.parameters(),4.);optimizer.step()
        engine.load_minibatch(data);engine.step_loaded()
        for k,v in native.state_dict().items():
            other=own.state_dict()[k]
            worst_parameter=max(worst_parameter,float((v.detach()-other.detach()).abs().max()))
            torch.testing.assert_close(v,other,atol=1e-5,rtol=2e-4)
    counters=[float(s['step']) for s in engine.optimizer.state.values()]
    assert all(s==3 for s in counters),counters
    result=dict(network=forward,max_native_GAE_error=worst_gae,
        max_native_loss_gradient_error=worst_loss,max_adapter_observation_error=worst_obs,
        update_minibatch_size=2048,update_steps=3,max_native_update_parameter_error=worst_parameter,
        bootstrap='unmodified SF2 value_bootstrap current-state approximation',advantage_min_std=1e-7)
    (SF_ROOT/'validation.json').write_text(json.dumps(result,indent=2));print('validated',result,flush=True)


if __name__=='__main__':main()
