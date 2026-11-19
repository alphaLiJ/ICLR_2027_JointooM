"""JointLoom mathematical profile matching unmodified SF 2.1.1 PPO/GAE.

This adapts our implementation. No Sample Factory code is changed.
"""
import bootstrap
import torch
from optimized_gpu import triton,tl


def sf2_loss(logits,values,actions,old_logp,old_values,advantages,returns):
    logprobs=logits.log_softmax(-1)
    logp=logprobs.gather(-1,actions[:,None]).squeeze(-1)
    ratio=(logp-old_logp).exp().clamp(.05,20.)
    std,mean=torch.std_mean(advantages)
    adv=(advantages-mean)/std.clamp_min(1e-7)
    clipped_ratio=ratio.clamp(1/1.2,1.2)
    actor=-torch.minimum(ratio*adv,clipped_ratio*adv).mean()
    clipped_values=old_values+(values-old_values).clamp(-1.,1.)
    critic=torch.maximum((values-returns).square(),(clipped_values-returns).square()).mean()
    entropy=-(logprobs.exp()*logprobs).sum(-1).mean()
    total=actor+.5*critic-.06878*entropy
    kl=((ratio-1)-(logp-old_logp)).mean()
    fraction=(ratio!=clipped_ratio).float().mean()
    return total,torch.stack((actor.detach(),critic.detach(),entropy.detach(),kl.detach(),fraction.detach()))


@triton.jit
def _sf2_gae(R,V,NV,D,A,RET,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    carry=tl.full((B,),0,tl.float32)
    for t in range(T-1,-1,-1):
        at=t*N+i
        r=tl.load(R+at,i<N,0);v=tl.load(V+at,i<N,0);nv=tl.load(NV+at,i<N,0)
        done=tl.load(D+at,i<N,0).to(tl.float32)
        # SF value_bootstrap=True approximates the timeout value using current V(s_t).
        reward=r+(.96*v)*done
        delta=(reward-v)+(1-done)*(.96*nv)
        carry=delta+(.96*.95)*carry*(1-done)
        tl.store(A+at,carry,i<N);tl.store(RET+at,carry+v,i<N)


def sf2_gae(rewards,values,next_values,reset,advantages=None,returns=None):
    if advantages is None:advantages=torch.empty_like(rewards)
    if returns is None:returns=torch.empty_like(rewards)
    t,n=rewards.shape
    _sf2_gae[(triton.cdiv(n,128),)](rewards,values,next_values,reset,advantages,returns,n,t,128,
                                   enable_fp_fusion=False)
    return advantages,returns
