"""Official MAGAT+ preprocessing oracle for frozen B1 rows.

This module intentionally mirrors the model-input construction used by the
existing MAGAT parity tests.  It is a correctness oracle, not a performance
baseline: the performance baseline remains ``original_magat_inmemory_baseline``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expert.fixed_magat_plus_runtime import OBS_RADIUS


MAGAT_COMM_RADIUS = 7


@dataclass(frozen=True)
class MagatReferenceBatch:
    """One environment's official MAGAT model-ready inputs."""

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    labels: np.ndarray
    positions: np.ndarray


def build_magat_reference_batch(
    grid: np.ndarray,
    compact_rows: np.ndarray,
    *,
    max_episode_steps: int = 1,
) -> MagatReferenceBatch:
    """Rebuild one B1 MAGAT stage from the original POGEMA/MAGAT semantics."""

    grid = np.asarray(grid, dtype=np.uint8)
    rows = np.asarray(compact_rows, dtype=np.uint16)
    if grid.ndim != 2:
        raise ValueError("MAGAT reference grid must have shape [height, width]")
    if rows.ndim != 2 or rows.shape[1] != 8 or rows.shape[0] == 0:
        raise ValueError("MAGAT compact rows must have shape [agents, 8]")
    if np.any(rows[:, 0] != rows[0, 0]):
        raise ValueError("MAGAT reference input must contain exactly one environment")

    from magat_plus.additional_data.cost_to_go_calculator import CostToGoCalculator
    from pogema.envs import PogemaCoopFinish
    from pogema.grid_config import GridConfig

    num_agents = int(rows.shape[0])
    config = GridConfig(
        map=grid.tolist(),
        agents_xy=rows[:, 2:4].tolist(),
        targets_xy=rows[:, 4:6].tolist(),
        num_agents=num_agents,
        obs_radius=OBS_RADIUS,
        on_target="nothing",
        collision_system="soft",
        observation_type="MAPF",
        max_episode_steps=int(max_episode_steps),
        empty_outside=True,
    )
    env = PogemaCoopFinish(grid_config=config)
    env._initialize_grid()
    env.update_was_on_goal()
    return build_magat_reference_batch_from_env(env, rows)


def build_magat_reference_batch_from_env(
    env,
    compact_rows: np.ndarray,
    *,
    cost_to_go_calculator=None,
) -> MagatReferenceBatch:
    """Build the same oracle batch from an initialized live POGEMA episode.

    A4 reuses the optional calculator across steps so its CPU baseline preserves
    the official lazy distance tables instead of reconstructing them each step.
    """

    from magat_plus.additional_data.cost_to_go_calculator import CostToGoCalculator

    rows = np.asarray(compact_rows, dtype=np.uint16)
    if rows.ndim != 2 or rows.shape[1] != 8 or rows.shape[0] == 0:
        raise ValueError("MAGAT compact rows must have shape [agents, 8]")
    if rows.shape[0] != int(env.num_agents):
        raise ValueError("MAGAT compact rows must match the live environment")
    observations = env._obs()

    global_xys = np.asarray(
        [observation["global_xy"] for observation in observations], dtype=np.int64
    )
    node_features: list[np.ndarray] = []
    for observation in observations:
        obstacle = np.pad(np.asarray(observation["obstacles"], dtype=np.float32), 1)
        agents = np.pad(np.asarray(observation["agents"], dtype=np.float32), 1)
        goals = np.zeros_like(obstacle)
        centre = (goals.shape[0] // 2, goals.shape[1] // 2)
        goal = np.asarray(observation["global_target_xy"], dtype=np.int64) - np.asarray(
            observation["global_xy"], dtype=np.int64
        )
        if np.all(np.abs(goal) <= OBS_RADIUS):
            goals[centre[0] + goal[0], centre[1] + goal[1]] = 1.0
        else:
            angle = np.arctan2(goal[1], goal[0])
            sign = np.sign(goal)
            distance = goals.shape[0] // 2
            if (np.pi / 4 <= angle <= 3 * np.pi / 4) or (
                -3 * np.pi / 4 <= angle <= -np.pi / 4
            ):
                goal_y = int(distance * (sign[1] + 1))
                goal_x = int(centre[0] + np.round(distance * goal[0] / abs(goal[1])))
            else:
                goal_x = int(distance * (sign[0] + 1))
                goal_y = int(centre[1] + np.round(distance * goal[1] / abs(goal[0])))
            goals[goal_x, goal_y] = 1.0
        node_features.append(np.stack((obstacle, agents, goals), axis=0))

    calculator = cost_to_go_calculator
    if calculator is None:
        calculator = CostToGoCalculator(
            env=env,
            obs_radius=OBS_RADIUS,
            dtype="float32",
            pad_cost_to_go=True,
            clamp_value=1.0,
            clamp_values_doubled=False,
        )
    cost_to_go = calculator.generate_cost_to_go(env, normalized=True)
    x = np.concatenate(
        (np.asarray(node_features, dtype=np.float32), cost_to_go[:, None]), axis=1
    ).astype(np.float32, copy=False)

    position_deltas = global_xys[:, None, :] - global_xys[None, :, :]
    distances = np.linalg.norm(position_deltas, axis=-1)
    # Match the original pdist/squareform construction exactly: zero-distance
    # pairs are absent because its weighted adjacency is later passed through
    # nonzero.  Valid standard-MAPF streams never co-locate agents, but keeping
    # this rule makes the oracle faithful even for malformed diagnostics.
    adjacency = (distances <= MAGAT_COMM_RADIUS) & (distances > 0.0)
    edge_index = np.argwhere(adjacency).T.astype(np.int64, copy=False)
    edge_delta = global_xys[edge_index[0]] - global_xys[edge_index[1]]
    edge_attr = np.concatenate(
        (
            edge_delta.astype(np.float32),
            np.abs(edge_delta).sum(axis=1, keepdims=True, dtype=np.int64).astype(
                np.float32
            ),
        ),
        axis=1,
    )
    return MagatReferenceBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=rows[:, 6].astype(np.int64, copy=True),
        positions=global_xys,
    )


__all__ = [
    "MAGAT_COMM_RADIUS",
    "MagatReferenceBatch",
    "build_magat_reference_batch",
    "build_magat_reference_batch_from_env",
]
