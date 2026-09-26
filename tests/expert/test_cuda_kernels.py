"""Focused integration contracts for the compiled CUDA simulator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

try:
    import grid_world_cpp as ext
except ImportError:
    ext = None


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    pytest.mark.skipif(ext is None, reason="CUDA extension is not built"),
]


def _states(num_agents: int, *, reset: int) -> torch.Tensor:
    data = torch.zeros((num_agents, 8), dtype=torch.int16, device="cuda:0")
    data[:, 1] = torch.arange(num_agents, dtype=torch.int16, device="cuda:0")
    data[:, 2] = 16 + torch.arange(num_agents, dtype=torch.int16, device="cuda:0")
    data[:, 3] = 16
    data[:, 4] = 48
    data[:, 5] = 48
    data[:, 7] = reset
    return data


def _simulator(num_envs: int, num_agents: int):
    grids = torch.zeros(
        (num_envs, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda:0",
    )
    return ext.StatelessGridWorldSimulator(grids, num_agents, 3)


def test_extension_exposes_compiled_profile():
    assert ext.COMPILED_MAP_W > 0
    assert ext.COMPILED_MAP_H > 0
    assert ext.COMPILED_MAP_H % 32 == 0


@pytest.mark.parametrize("num_agents", [1, 4, 32])
def test_energy_and_observation_shapes_follow_runtime_agent_count(num_agents):
    sim = _simulator(1, num_agents)
    sim.update_energy_maps(_states(num_agents, reset=1), num_agents)
    sim.setup_imitation_obs(_states(num_agents, reset=0))
    torch.cuda.synchronize()

    assert sim.pyg_x.shape[0] == num_agents
    assert sim.pyg_pos.shape == (num_agents, 2)
    assert sim.pyg_ptr.tolist() == [0, num_agents]


def test_energy_map_marks_the_goal_cell():
    sim = _simulator(1, 1)
    states = _states(1, reset=1)
    sim.update_energy_maps(states, 1)
    torch.cuda.synchronize()

    assert int(sim.energy_maps[0, 0, 48, 48].item()) == 255


def test_pyg_edges_are_bounded_by_runtime_agent_count():
    num_agents = 4
    sim = _simulator(1, num_agents)
    states = _states(num_agents, reset=1)
    sim.update_energy_maps(states, num_agents)
    states[:, 7] = 0
    sim.setup_imitation_obs(states)
    torch.cuda.synchronize()

    edges = sim.pyg_edge_index
    assert edges.shape[0] == 2
    if edges.numel():
        assert int(edges.min().item()) >= 0
        assert int(edges.max().item()) < num_agents


def test_compiled_simulator_rejects_wrong_grid_shape():
    grids = torch.zeros(
        (1, ext.COMPILED_MAP_W + 1, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda:0",
    )
    with pytest.raises(RuntimeError):
        ext.StatelessGridWorldSimulator(grids, 1, 3)


def test_stateless_memory_inventory_exposes_backing_storage_without_allocation():
    sim = _simulator(1, 4)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    inventory = sim.memory_tensors
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    assert after == before
    assert {
        "grid_compressed",
        "energy_maps",
        "edge_counts",
        "pyg_edge_index_storage",
        "pyg_edge_attr_storage",
        "pyg_edge_prefix",
        "scan_temp_storage",
    }.issubset(inventory)
    assert inventory["pyg_edge_index"].untyped_storage().data_ptr() == inventory[
        "pyg_edge_index_storage"
    ].untyped_storage().data_ptr()
