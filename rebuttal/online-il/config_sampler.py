"""Balanced full-configuration orders; each configuration is used once."""
import numpy as np


ORDERS = ('ordered','window_shuffle_128','global_shuffle')


def configuration_order(stages, envs, order, seed, window=128):
    if order not in ORDERS:
        raise ValueError(order)
    plan = np.broadcast_to(np.arange(stages, dtype=np.int64)[:,None], (stages,envs)).copy()
    for e in range(envs):
        rng = np.random.default_rng(np.random.SeedSequence([seed,e]))
        if order == 'global_shuffle':
            rng.shuffle(plan[:,e])
        elif order == 'window_shuffle_128':
            for begin in range(0,stages,window):
                rng.shuffle(plan[begin:begin+window,e])
    return plan
