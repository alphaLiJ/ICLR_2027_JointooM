"""
Tests that CUDA-generated MAGAT observations stay consistent with Pogema-side observations.
"""

import os
import sys

import numpy as np
import pytest
import torch

TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(TESTS_ROOT)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for path in (PROJECT_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

try:
    import grid_world_cpp as ext
    EXT_AVAILABLE = True
except ImportError:
    EXT_AVAILABLE = False

from expert.expert_running import put_maps_into_registry
from mapf_cuda.simulation.grids import extract_single_env_grid
from mapf_cuda.simulation.pogema_envs import (
    build_random_standard_env,
    build_topology_standard_env,
)
from mapf_cuda.training.topology_async import _build_step_data
from expert.expert_running import RESET_FLAG_COL, initial_refresh_flags, no_refresh_flags
from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter, OBS_RADIUS
from magat_plus.additional_data.cost_to_go_calculator import CostToGoCalculator


MAGAT_INPUT_OBS_DIAM = 2 * OBS_RADIUS + 3


def _build_reference_obs_tensor(observations, env_unwrapped):
    node_features = []
    for observation in observations:
        obs_obstacle = np.pad(observation["obstacles"], 1)
        obs_agent = np.pad(observation["agents"], 1)
        obs_goal = np.zeros_like(obs_obstacle)
        centre = (obs_goal.shape[0] // 2, obs_goal.shape[1] // 2)
        goal = tuple(
            tpos - apos
            for tpos, apos in zip(
                observation["global_target_xy"], observation["global_xy"]
            )
        )

        if np.all(np.abs(goal) <= OBS_RADIUS):
            obs_goal[centre[0] + goal[0], centre[1] + goal[1]] = 1.0
        else:
            angle = np.arctan2(goal[1], goal[0])
            goal_sign = np.sign(goal)
            dist = obs_goal.shape[0] // 2
            if (angle >= np.pi / 4 and angle <= np.pi * 3 / 4) or (
                angle >= -np.pi * (3 / 4) and angle <= -np.pi / 4
            ):
                goal_y_fov = int(dist * (goal_sign[1] + 1))
                goal_x_fov = int(
                    centre[0] + np.round(dist * goal[0] / np.abs(goal[1]))
                )
            else:
                goal_x_fov = int(dist * (goal_sign[0] + 1))
                goal_y_fov = int(
                    centre[1] + np.round(dist * goal[1] / np.abs(goal[0]))
                )
            obs_goal[goal_x_fov, goal_y_fov] = 1.0

        node_features.append(np.stack([obs_obstacle, obs_agent, obs_goal]))

    node_features = torch.from_numpy(np.stack(node_features)).to(torch.float32)
    if not hasattr(env_unwrapped, "num_agents"):
        env_unwrapped.num_agents = env_unwrapped.get_num_agents()
    ctg_calc = CostToGoCalculator(
        env=env_unwrapped,
        obs_radius=OBS_RADIUS,
        dtype="float32",
        pad_cost_to_go=True,
        clamp_value=1.0,
        clamp_values_doubled=False,
    )
    cost_to_go = torch.from_numpy(
        ctg_calc.generate_cost_to_go(env_unwrapped, normalized=True)
    ).unsqueeze(1)
    return torch.cat([node_features, cost_to_go], dim=1)


def _build_cuda_batch(runtime, simulator, observations, actions, refresh_flags, device: str):
    raw_batch = torch.from_numpy(
        _build_step_data(
            observations,
            actions.astype(np.uint16, copy=False),
            env_id=0,
            reset_flag=refresh_flags,
        ).astype(np.int16, copy=False)
    ).to(device)
    agents_to_update = raw_batch[raw_batch[:, RESET_FLAG_COL] != 0]
    if agents_to_update.shape[0] > 0:
        simulator.update_energy_maps(agents_to_update, agents_to_update.shape[0])
    simulator.refresh_compact_state_from_raw_batch(raw_batch)
    return runtime.build_batch(simulator, raw_batch)


def _assert_obs_consistency_for_env(env, *, device: str):
    num_agents = env.get_num_agents()
    simulator = ext.StatelessGridWorldSimulator(
        extract_single_env_grid(env, device=device),
        num_agents,
        3,
    )
    runtime = MAGATRuntimeAdapter(device=device)

    observations = env.env.unwrapped._obs()
    actions = np.zeros(num_agents, dtype=np.uint16)
    reference_batch = _build_reference_obs_tensor(observations, env.env.unwrapped)
    cuda_batch = _build_cuda_batch(
        runtime,
        simulator,
        observations,
        actions,
        initial_refresh_flags(len(observations)),
        device,
    )

    assert cuda_batch.x.shape == (num_agents, 4, MAGAT_INPUT_OBS_DIAM, MAGAT_INPUT_OBS_DIAM)
    assert cuda_batch.x.shape == reference_batch.shape
    assert torch.allclose(cuda_batch.x.cpu(), reference_batch, atol=1e-6)

    _, _, terminated, truncated, _ = env.step(actions.astype(np.int64, copy=False))
    assert not all(terminated)
    assert not all(truncated)

    next_observations = env.env.unwrapped._obs()
    next_actions = np.zeros(num_agents, dtype=np.uint16)
    next_reference_batch = _build_reference_obs_tensor(next_observations, env.env.unwrapped)
    next_cuda_batch = _build_cuda_batch(
        runtime,
        simulator,
        next_observations,
        next_actions,
        no_refresh_flags(len(next_observations)),
        device,
    )

    assert next_cuda_batch.x.shape == next_reference_batch.shape
    assert torch.allclose(next_cuda_batch.x.cpu(), next_reference_batch, atol=1e-6)


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateless_obs_matches_pogema_reference_for_reset_and_non_reset_steps():
    env = build_random_standard_env(
        num_agents=8,
        map_size=32,
        density=0.0,
        seed=42,
        max_episode_steps=64,
    )
    _assert_obs_consistency_for_env(env, device="cuda:0")


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateless_obs_matches_pogema_reference_on_topology_map():
    put_maps_into_registry(os.path.join(PROJECT_ROOT, "maps", "maps.yaml"))
    env = build_topology_standard_env(
        num_agents=8,
        map_name="mazes-s0_wc8_od55",
        seed=42,
        max_episode_steps=64,
    )
    _assert_obs_consistency_for_env(env, device="cuda:0")
