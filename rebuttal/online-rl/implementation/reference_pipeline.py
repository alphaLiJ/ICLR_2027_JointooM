"""CPU rollout/GAE and batch upload, following the selected SF-v1 placement.

This is a synchronous adapter, not the unpublished original CostTracer trainer.
The shared time-limit bootstrap and PPO math remain the audited definition.
"""
import bootstrap
import numpy as np
import torch
from ppo import loss
from configuration import SPEC


def cpu_gae(rewards,values,next_values,reset):
    advantages=np.empty_like(rewards);carry=np.zeros_like(rewards[0])
    gamma=np.float32(SPEC['ppo']['gamma'])
    coefficient=np.float32(SPEC['ppo']['gamma']*SPEC['ppo']['gae_lambda'])
    for t in reversed(range(len(rewards))):
        delta=(rewards[t]+gamma*next_values[t])-values[t]
        carry=delta+coefficient*(~reset[t]).astype(np.float32)*carry
        advantages[t]=carry
    return advantages,advantages+values


def reference_update(model,optimizer,data,rng,minibatch=2048):
    # SF-v1 generates permutations on CPU and indexes an already-uploaded batch.
    order=rng.permutation(len(data['actions']))
    recorded=[]
    for start in range(0,len(order),minibatch):
        indices=order[start:start+minibatch]
        logits,value=model(data['obs'][indices])
        total,stats=loss(logits,value,*(data[k][indices] for k in
            ('actions','logp','values','advantages','returns')))
        optimizer.zero_grad(set_to_none=True);total.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),SPEC['optimizer']['max_global_grad_norm'],foreach=True)
        optimizer.step()
        recorded.append(torch.cat((total.detach()[None],stats,norm.detach()[None])))
    return torch.stack(recorded).mean(0)
