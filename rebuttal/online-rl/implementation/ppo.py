"""Synchronous GAE/PPO using the audited Sample Factory v1 loss conventions."""
import bootstrap
import torch
from configuration import SPEC

PPO=SPEC['ppo']
OPT=SPEC['optimizer']


def gae(rewards, values, next_values, reset, gamma=PPO['gamma'], lam=PPO['gae_lambda']):
    # next_values is evaluated on the final observation BEFORE any time reset.
    # Continuing-task time limits bootstrap, but do not carry GAE into a reset.
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        carry = delta + gamma * lam * (~reset[t]).to(rewards.dtype) * carry
        advantages[t] = carry
    return advantages, advantages + values


def loss(logits, values, actions, old_logp, old_values, advantages, returns):
    logprobs = logits.log_softmax(-1)
    logp = logprobs.gather(-1,actions[:,None]).squeeze(-1)
    ratio = (logp-old_logp).exp().clamp(*PPO['ratio_numerical_clamp'])
    adv = (advantages-advantages.mean()) / advantages.std(correction=PPO['advantage_std_correction']).clamp_min(PPO['advantage_min_std'])
    clipped_ratio=ratio.clamp(PPO['clip_ratio_lower'],PPO['clip_ratio_upper'])
    actor = -torch.minimum(ratio*adv, clipped_ratio*adv).mean()
    clipped = old_values + (values-old_values).clamp(-PPO['value_clip'],PPO['value_clip'])
    critic = torch.maximum((values-returns).square(), (clipped-returns).square()).mean()
    entropy = -(logprobs.exp()*logprobs).sum(-1).mean()
    total = actor + PPO['value_loss_coefficient']*critic - PPO['entropy_coefficient']*entropy
    approx_kl = ((ratio-1)-(logp-old_logp)).mean()
    fraction_clipped=(ratio!=clipped_ratio).float().mean()
    return total, torch.stack((actor.detach(),critic.detach(),entropy.detach(),approx_kl.detach(),fraction_clipped.detach()))


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=OPT['learning_rate'], betas=OPT['betas'],
                            eps=OPT['epsilon'],weight_decay=OPT['weight_decay'])


def update(model, optimizer, data, generator, minibatch=2048):
    n = len(data['actions'])
    order = torch.randperm(n, device=data['obs'].device, generator=generator)
    metrics = []
    for _ in range(PPO['epochs']):
        for indices in order.split(minibatch):
            logits, value = model(data['obs'][indices])
            total, stats = loss(logits, value, *(data[k][indices] for k in
                               ('actions','logp','values','advantages','returns')))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),OPT['max_global_grad_norm'], foreach=True)
            optimizer.step()
            metrics.append(torch.cat((total.detach()[None], stats,norm.detach()[None])))
    return torch.stack(metrics).mean(0)
