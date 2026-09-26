"""Construct CUDA simulators from runtime-sized topology maps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import grid_world_cpp as ext
from expert.profiled_topology_loader import (
    build_real_map_extent_tensor,
    embed_obstacles_in_compiled_profile,
    stack_embedded_obstacle_grids,
)


def stack_grids_for_compiled_simulator(
    grids: Sequence[np.ndarray],
    *,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad runtime maps to the compiled profile and upload them once."""

    embedded, real_extents = stack_embedded_obstacle_grids(
        grids,
        compiled_map_w=ext.COMPILED_MAP_W,
        compiled_map_h=ext.COMPILED_MAP_H,
    )
    grids_cuda = torch.from_numpy(embedded).to(
        device=device, dtype=torch.int32
    ).contiguous()
    extents_cuda = build_real_map_extent_tensor(real_extents, device=device)
    return grids_cuda, extents_cuda


def build_stateless_simulator(
    grids: Sequence[np.ndarray],
    *,
    num_agents: int,
    device: str = "cuda:0",
):
    grids_cuda, _ = stack_grids_for_compiled_simulator(grids, device=device)
    return ext.StatelessGridWorldSimulator(grids_cuda, int(num_agents), 3)


def extract_single_env_grid(env, *, device: str = "cuda:0") -> torch.Tensor:
    """Upload one POGEMA obstacle grid using the compiled map profile."""

    obstacles = env.env.unwrapped.grid.get_obstacles(
        ignore_borders=True
    ).astype(np.int32)
    padded, _ = embed_obstacles_in_compiled_profile(
        obstacles,
        compiled_map_w=ext.COMPILED_MAP_W,
        compiled_map_h=ext.COMPILED_MAP_H,
    )
    return torch.from_numpy(padded).to(device).unsqueeze(0).contiguous()


def configure_magat_builder(
    simulator,
    *,
    mode: str | None = None,
    local_gather_impl: str | None = None,
) -> str | None:
    if mode is not None:
        simulator.pyg_builder_mode = mode
    if local_gather_impl is not None:
        simulator.pyg_local_gather_impl = local_gather_impl
    return getattr(simulator, "resolved_pyg_builder_impl", None)
