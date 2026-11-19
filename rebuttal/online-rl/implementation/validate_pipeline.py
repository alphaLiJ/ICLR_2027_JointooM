"""Numerical checks for fusion, layout, capture replay, and fused Adam."""
import bootstrap
import argparse
import json
import time
import numpy as np
import torch
from model import CostTracer,configure_torch
from ppo import gae,loss,make_optimizer
from reference_pipeline import cpu_gae
from optimized_gpu import StaticLearner,fused_gae,_gather_obs,_gather_scalars
import triton


def check_gae():
    torch.manual_seed(42);worst=0.;cpu_worst=0.
    for t,n in ((1,1),(7,37),(32,2048),(48,65)):
        r=torch.randn(t,n,device='cuda');v=torch.randn_like(r);nv=torch.randn_like(r)
        reset=torch.rand(t,n,device='cuda')<.08
        a,ret=gae(r,v,nv,reset);b,bret=fused_gae(r,v,nv,reset)
        ca,cret=cpu_gae(*(x.cpu().numpy() for x in (r,v,nv,reset)))
        for x,y in ((a,b),(ret,bret)):
            worst=max(worst,float((x-y).abs().max()));torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
        cpu_worst=max(cpu_worst,float(np.max(np.abs(a.cpu().numpy()-ca))))
        np.testing.assert_allclose(a.cpu().numpy(),ca,atol=2e-6,rtol=2e-6)
    return dict(shapes=4,max_fused_gae_error=worst,max_cpu_gae_error=cpu_worst)


def check_gather():
    n=257;m=128;order=torch.randperm(n,device='cuda')
    obs=torch.randn((n,2,11,11),device='cuda');actions=torch.randint(5,(n,),device='cuda')
    vals=[torch.randn(n,device='cuda') for _ in range(4)]
    for cl in (False,True):
        out=torch.empty((m,2,11,11),device='cuda',memory_format=torch.channels_last if cl else torch.contiguous_format)
        oa=torch.empty(m,dtype=torch.int64,device='cuda');scalar=torch.empty((4,m),device='cuda')
        _gather_obs[(triton.cdiv(m*242,1024),)](obs,order,out,32,m,cl,1024)
        _gather_scalars[(1,)](order,actions,*vals,oa,scalar,32,m,256)
        ids=order[32:32+m]
        assert torch.equal(out,obs[ids]) and torch.equal(oa,actions[ids])
        for i,v in enumerate(vals):assert torch.equal(scalar[i],v[ids])
    return dict(NCHW='exact',channels_last='exact',actions_and_scalar_fields='exact')


def check_updates(n=128,steps=5,compile_loss=False,compile_model=False):
    configure_torch(0)
    plain=CostTracer().cuda();fast=CostTracer().cuda();fast.load_state_dict(plain.state_dict())
    baseline=make_optimizer(plain)
    engine=StaticLearner(fast,n=n,minibatch=n,compile_loss=compile_loss,compile_model=compile_model)
    assert engine.warmup_initial_weights_restored and engine.warmup_optimizer_zero
    maximum=dict(logits=0.,values=0.,loss=0.,gradient=0.,parameter=0.)
    for step in range(steps):
        x=torch.rand((n,2,11,11),device='cuda')
        x[:,0]=(x[:,0]>.7).float();x[:,1,5,5]=-.015625
        with torch.no_grad():
            logits,values=plain(x);fl,fv=engine.policy(x)
            maximum['logits']=max(maximum['logits'],float((logits-fl).abs().max()))
            maximum['values']=max(maximum['values'],float((values-fv).abs().max()))
            torch.testing.assert_close(logits,fl,atol=2e-5,rtol=2e-4)
            torch.testing.assert_close(values,fv,atol=2e-5,rtol=2e-4)
            actions=torch.randint(5,(n,),device='cuda')
            oldlp=logits.log_softmax(-1).gather(-1,actions[:,None]).squeeze(-1)
        data=dict(obs=x,actions=actions,logp=oldlp,values=values,
            advantages=torch.randn(n,device='cuda'),returns=torch.randn(n,device='cuda'))
        logit,value=plain(x)
        total,_=loss(logit,value,*(data[k] for k in ('actions','logp','values','advantages','returns')))
        baseline.zero_grad(set_to_none=True);total.backward()
        torch.nn.utils.clip_grad_norm_(plain.parameters(),4.,foreach=True)
        baseline.step()
        engine.load_minibatch(data);stats=engine.step_loaded()
        maximum['loss']=max(maximum['loss'],abs(float(total.detach())-float(stats[0])))
        for (name,p),(other,q) in zip(plain.named_parameters(),fast.named_parameters()):
            assert name==other
            maximum['gradient']=max(maximum['gradient'],float((p.grad-q.grad).abs().max()))
            maximum['parameter']=max(maximum['parameter'],float((p-q).abs().max()))
            torch.testing.assert_close(p,q,atol=1e-5,rtol=2e-4)
    assert maximum['loss']<2e-5 and maximum['gradient']<2e-4,maximum
    counters=[float(s['step']) for s in engine.optimizer.state.values()]
    assert all(s==steps for s in counters),counters
    return dict(minibatch_updates=steps,minibatch_size=n,max_errors=maximum,adam_steps=steps,
                compiled_model=compile_model,compiled_loss=compile_loss,
                warmup_initial_weights_restored=True,warmup_optimizer_reset=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,default=128);p.add_argument('--steps',type=int,default=5)
    p.add_argument('--compile-loss',action='store_true');p.add_argument('--compile-model',action='store_true')
    p.add_argument('--output',default=str(bootstrap.ARTIFACTS/'pipeline-v2-validation.json'))
    args=p.parse_args()
    configure_torch(0);result={}
    for key,fn in [('GAE',check_gae),('gather',check_gather),('network_graph_adam',
                  lambda:check_updates(args.size,args.steps,args.compile_loss,args.compile_model))]:
        result[key]=fn();print(key,json.dumps(result[key]),flush=True)
    from pathlib import Path
    path=Path(args.output)
    path.write_text(json.dumps(result,indent=2));print('saved',path,flush=True)


if __name__=='__main__':main()
