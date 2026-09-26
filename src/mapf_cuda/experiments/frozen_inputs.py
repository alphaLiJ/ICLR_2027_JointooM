"""Deterministic standard-MAPF workloads for scaling experiments."""

from __future__ import annotations

import hashlib

import numpy as np

from expert.benchmark_contract import FrozenTransitionBatch, build_frozen_pool


def _topology_seed(pool_identity: str) -> int:
    digest = hashlib.sha256(pool_identity.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _generate_obstacles(
    *, topology: str, density: float, map_size: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.zeros((map_size, map_size), dtype=np.uint8)
    if topology == "random":
        grid = (rng.random((map_size, map_size)) < density).astype(np.uint8)
    elif topology == "maze":
        for row in range(5, map_size - 1, 6):
            grid[row, 1:-1] = 1
            gate = 1 + int(rng.integers(0, map_size - 3))
            grid[row, gate : gate + 2] = 0
    elif topology == "warehouse":
        for column in range(4, map_size - 1, 6):
            grid[1:-1, column : column + 2] = 1
            gate = 1 + int(rng.integers(0, map_size - 5))
            grid[gate : gate + 4, column : column + 2] = 0
    else:
        raise ValueError(f"unsupported topology {topology!r}")
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid


def _make_instance(
    *,
    seed: int,
    horizon: int,
    topology: str,
    density: float,
    num_agents: int,
    map_size: int,
    topology_seed: int,
) -> dict[str, np.ndarray]:
    if horizon <= 0 or map_size < 3 or num_agents <= 0:
        raise ValueError("horizon, map_size, and num_agents must be positive")
    grid = _generate_obstacles(
        topology=topology,
        density=density,
        map_size=map_size,
        seed=topology_seed ^ int(seed),
    )
    free_cells = np.argwhere(grid == 0)
    if free_cells.shape[0] < 2 * num_agents:
        raise ValueError(
            f"{topology} map has {free_cells.shape[0]} free cells, "
            f"need at least {2 * num_agents}"
        )
    rng = np.random.default_rng(seed)
    selected = free_cells[rng.permutation(free_cells.shape[0])[: 2 * num_agents]]
    positions = selected[:num_agents].astype(np.uint16, copy=False)
    arrived_count = min(
        int(rng.integers(0, max(2, num_agents // 8 + 1))), num_agents
    )
    goals = np.empty_like(positions)
    goals[:arrived_count] = positions[:arrived_count]
    goals[arrived_count:] = selected[
        num_agents : 2 * num_agents - arrived_count
    ].astype(np.uint16, copy=False)
    arrived = np.zeros(num_agents, dtype=np.bool_)
    arrived[:arrived_count] = True
    return {
        "grid": grid,
        "positions": positions,
        "goals": goals,
        "arrived": arrived,
        "active": np.logical_not(arrived),
        "actions": rng.integers(
            0, 5, size=(horizon, num_agents), dtype=np.uint8
        ),
    }


def build_frozen_standard_mapf_pool(
    *,
    topology: str,
    density: float,
    num_agents: int,
    seeds: tuple[int, ...],
    horizon: int,
    map_size: int,
    pool_identity: str,
) -> FrozenTransitionBatch:
    """Build a deterministic pool without the historical A1 registry stack."""

    topology_seed = _topology_seed(pool_identity)
    return build_frozen_pool(
        seeds=seeds,
        horizon=horizon,
        instance_factory=lambda *, seed, horizon: _make_instance(
            seed=seed,
            horizon=horizon,
            topology=topology,
            density=density,
            num_agents=num_agents,
            map_size=map_size,
            topology_seed=topology_seed,
        ),
    )

