"""GPU-only CAMAR discrete environment / PyTorch MAGAT integration."""
import bootstrap
import jax
import jax.numpy as jnp
import numpy as np
import torch

from discrete_mapf import MAPFConfig, DiscreteMAPF, batch_transition_environment
from observations import CachedObservations, PaperCachedObservations
from expert.fixed_magat_plus_runtime import RuntimePyGBatch


class CamarRuntime:
    def __init__(self, grids, positions, goals, arrived, *, mode="standard_mapf", horizon=4096, paper_cache=False):
        n, h, w = grids.shape
        self.agents = positions.shape[1]
        self.config = MAPFConfig(height=h, width=w, num_agents=self.agents,
                                 task_mode=mode, max_episode_steps=horizon, max_steps=horizon)
        self.env = DiscreteMAPF(self.config)
        self.batch = batch_transition_environment(self.env, n)
        self.grids = jax.device_put(np.asarray(grids, dtype=np.int32 if paper_cache else np.uint8))
        positions = jax.device_put(np.asarray(positions, dtype=np.int32))
        goals = jax.device_put(np.asarray(goals, dtype=np.int32))
        arrived = jax.device_put(np.asarray(arrived, dtype=bool))
        keys = jax.random.split(jax.random.PRNGKey(20260723 if paper_cache else 20261109), n)
        self.initial_state = jax.jit(jax.vmap(self.env.reset_from_arrays))(
            keys, self.grids, positions, goals, arrived)
        self.inputs = (PaperCachedObservations if paper_cache else CachedObservations)(self.config, self.grids)
        self.batch_ids = torch.arange(n, device="cuda").repeat_interleave(self.agents)
        self.ptr = torch.arange(0, (n + 1) * self.agents, self.agents, device="cuda")
        self.reset()

    def reset(self):
        self.state = self.initial_state
        self.key = jax.random.PRNGKey(20261109)
        self.inputs.invalidate()

    def build(self):
        """Node views in JAX; sparse MAGAT graph in PyTorch, all on GPU.

        DLPack is used for the framework boundary. Graph compaction, its dynamic
        output size and stream synchronization are included in decision timing.
        This adapter does not invoke JointLoom's CUDA input builders.
        """
        obs = self.inputs.prepare(self.state)
        x = torch.from_dlpack(obs).reshape(-1, 4, self.config.pyg_obs_diam, self.config.pyg_obs_diam)
        pos = torch.from_dlpack(self.state.pos)
        delta = pos[:, :, None, :] - pos[:, None, :, :]
        squared = (delta * delta).sum(dim=-1)
        env, source, target = torch.nonzero((squared > 0) & (squared <= 49), as_tuple=True)
        relative = delta[env, source, target].to(torch.float32)
        edge_index = torch.stack((env * self.agents + source, env * self.agents + target))
        edge_attr = torch.cat((relative, relative.abs().sum(dim=-1, keepdim=True)), dim=-1)
        arrived = torch.from_dlpack(self.state.arrived).reshape(-1)
        return RuntimePyGBatch(x=x, edge_index=edge_index, edge_attr=edge_attr,
                               batch=self.batch_ids, ptr=self.ptr, arrived=arrived,
                               terminated=torch.zeros_like(arrived))

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = jax.dlpack.from_dlpack(actions.contiguous())
        self.key, key = jax.random.split(self.key)
        _, self.state, _, _, _ = self.batch.step(key, self.state, actions)

    def synchronize(self):
        jax.block_until_ready(self.state)
        if self.inputs.energy is not None:
            jax.block_until_ready(self.inputs.energy)
        torch.cuda.synchronize()

    def result(self):
        return {name:np.asarray(jax.device_get(value)) for name,value in {
            "positions":self.state.pos, "goals":self.state.goal_pos,
            "arrived":self.state.arrived, "terminated":self.state.terminated,
            "truncated":self.state.truncated, "step_counts":self.state.step}.items()}
