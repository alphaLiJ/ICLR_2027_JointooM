import numpy as np
import pytest
import torch

from mapf_cuda.simulation.grids import build_stateless_simulator
from expert.profiled_topology_loader import (
    build_real_map_extent_tensor,
    embed_obstacles_in_compiled_profile,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

try:
    import grid_world_cpp as ext
    EXT_AVAILABLE = True
except ImportError:
    EXT_AVAILABLE = False


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_compiled_map_profile_constants_are_exposed():
    assert ext.COMPILED_MAP_W > 0
    assert ext.COMPILED_MAP_H > 0
    assert ext.COMPILED_MAP_H % 32 == 0


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateless_simulator_accepts_only_compiled_shape():
    grids = torch.zeros(
        (1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )

    sim = ext.StatelessGridWorldSimulator(grids, 8, 3)

    assert sim is not None


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateless_simulator_rejects_non_compiled_shape():
    wrong_w = ext.COMPILED_MAP_W + 1
    wrong_h = ext.COMPILED_MAP_H
    grids = torch.zeros((1, wrong_w, wrong_h), dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError):
        ext.StatelessGridWorldSimulator(grids, 8, 3)


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_energy_map_update_works_for_compiled_profile():
    grids = torch.zeros(
        (1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )
    sim = ext.StatelessGridWorldSimulator(grids, 1, 3)

    center_x = ext.COMPILED_MAP_W // 2
    center_y = ext.COMPILED_MAP_H // 2
    states = torch.zeros((1, 8), dtype=torch.int16, device="cuda")
    states[0, 0] = 0
    states[0, 1] = 0
    states[0, 2] = center_x
    states[0, 3] = center_y
    states[0, 4] = center_x
    states[0, 5] = center_y
    states[0, 7] = 1

    sim.update_energy_maps(states, 1)
    torch.cuda.synchronize()

    assert int(sim.energy_maps[0, 0, center_x, center_y].item()) == 255


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateful_free_cell_cache_respects_real_map_extents():
    real_obstacles = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=torch.int32,
    ).cpu().numpy()
    embedded, real_extent = embed_obstacles_in_compiled_profile(
        real_obstacles,
        compiled_map_w=ext.COMPILED_MAP_W,
        compiled_map_h=ext.COMPILED_MAP_H,
    )
    grid = torch.from_numpy(embedded).to(device="cuda", dtype=torch.int32).unsqueeze(0).contiguous()
    extents = build_real_map_extent_tensor([real_extent], device="cuda")

    sim = ext.GridWorldSimulator(
        grid,
        4,
        2,
        ext.COMPILED_MAP_W * ext.COMPILED_MAP_H,
        123,
    )
    sim.set_real_map_extents(extents)
    sim.run_initialization()
    torch.cuda.synchronize()

    expected_free = int((real_obstacles == 0).sum())
    assert int(sim.counts[0].item()) == expected_free
    assert int(sim.cur_x.max().item()) < real_extent[0]
    assert int(sim.cur_y.max().item()) < real_extent[1]


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_benchmark_stateless_builder_accepts_smaller_raw_grids():
    raw_grids = [
        np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                [0, 0],
                [1, 0],
                [0, 0],
            ],
            dtype=np.int32,
        ),
    ]

    sim = build_stateless_simulator(
        raw_grids, num_agents=4, device="cuda"
    )

    assert sim is not None
