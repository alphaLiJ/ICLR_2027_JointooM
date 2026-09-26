import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

try:
    import grid_world_cpp as ext
    EXT_AVAILABLE = True
except ImportError:
    EXT_AVAILABLE = False


MAP_W = getattr(ext, "COMPILED_MAP_W", 128) if EXT_AVAILABLE else 128
MAP_H = getattr(ext, "COMPILED_MAP_H", 128) if EXT_AVAILABLE else 128
COL_OFFSET = MAP_H // 32
FEATURE_DIM = 8
PYG_OBS_DIAM = 2 * 5 + 3
LEGACY_FULLMAP_SUPPORTED = MAP_W == 128 and MAP_H == 128
ASYNC_LOCAL_GATHER_HW_SUPPORTED = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
)


@pytest.fixture
def small_empty_grid():
    return torch.zeros((1, MAP_W, MAP_H), dtype=torch.int32, device="cuda")


def _bitmap_cell_is_set(bitmap_cpu: torch.Tensor, x: int, y: int) -> bool:
    word_idx = x * COL_OFFSET + (y >> 5)
    mask = 1 << (y & 31)
    return (int(bitmap_cpu[0, word_idx].item()) & mask) != 0


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_builder_resolution_metadata_is_exposed(small_empty_grid):
    sim = ext.StatelessGridWorldSimulator(small_empty_grid.clone(), 32, 3)

    assert sim.pyg_builder_mode == "auto"
    assert sim.resolved_pyg_builder_impl in {
        "legacy_fullmap",
        "local_gather_scalar",
        "local_gather_async_sm89plus",
    }


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_local_gather_impl_override_accepts_auto_scalar_async(small_empty_grid):
    sim = ext.StatelessGridWorldSimulator(small_empty_grid.clone(), 32, 3)

    sim.pyg_builder_mode = "local_gather"
    sim.pyg_local_gather_impl = "scalar"
    assert sim.pyg_local_gather_impl == "scalar"

    sim.pyg_local_gather_impl = "auto"
    assert sim.pyg_local_gather_impl == "auto"

    sim.pyg_local_gather_impl = "async_sm89plus"
    assert sim.pyg_local_gather_impl == "async_sm89plus"


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateful_and_stateless_compact_state_methods_exist(small_empty_grid):
    num_agents = 4
    stateful = ext.GridWorldSimulator(small_empty_grid, num_agents, 2, MAP_W * MAP_H, 123)
    stateless = ext.StatelessGridWorldSimulator(small_empty_grid, num_agents, 3)

    assert hasattr(stateful, "step_sim_only")
    assert hasattr(stateful, "build_magat_plus_inputs")
    assert hasattr(stateful, "step_compact_only")
    assert hasattr(stateful, "materialize_pyg_inputs")
    assert hasattr(stateful, "state_packed")
    assert hasattr(stateful, "grid_ocp")

    assert hasattr(stateless, "setup_imitation_obs")
    assert hasattr(stateless, "build_magat_plus_inputs")
    assert hasattr(stateless, "refresh_compact_state_from_raw_batch")
    assert hasattr(stateless, "materialize_pyg_inputs")
    assert hasattr(stateless, "state_packed")
    assert hasattr(stateless, "grid_ocp")


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateless_compact_state_wrappers_match_legacy_setup(small_empty_grid):
    num_agents = 2
    legacy = ext.StatelessGridWorldSimulator(small_empty_grid, num_agents, 3)
    wrapped = ext.StatelessGridWorldSimulator(small_empty_grid, num_agents, 3)

    states = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device="cuda")
    states[:, 0] = 0
    states[:, 1] = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    states[:, 2] = torch.tensor([40, 41], dtype=torch.int16, device="cuda")
    states[:, 3] = torch.tensor([50, 51], dtype=torch.int16, device="cuda")
    states[:, 4] = torch.tensor([60, 61], dtype=torch.int16, device="cuda")
    states[:, 5] = torch.tensor([62, 63], dtype=torch.int16, device="cuda")
    states[:, 7] = 1

    legacy.update_energy_maps(states, num_agents)
    wrapped.update_energy_maps(states, num_agents)

    states[:, 7] = 0
    legacy.setup_imitation_obs(states)
    wrapped.refresh_compact_state_from_raw_batch(states)
    wrapped.materialize_pyg_inputs()
    torch.cuda.synchronize()

    assert torch.equal(wrapped.state_packed[:num_agents].cpu(), states.cpu())
    assert torch.equal(legacy.pyg_pos.cpu(), wrapped.pyg_pos.cpu())
    assert torch.allclose(legacy.pyg_x.cpu(), wrapped.pyg_x.cpu())
    assert torch.equal(legacy.pyg_edge_index.cpu(), wrapped.pyg_edge_index.cpu())
    assert torch.allclose(legacy.pyg_edge_attr.cpu(), wrapped.pyg_edge_attr.cpu())


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_refresh_compact_state_populates_stateless_occupancy_bitmap(small_empty_grid):
    num_agents = 2
    sim = ext.StatelessGridWorldSimulator(small_empty_grid, num_agents, 3)

    states = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device="cuda")
    states[:, 0] = 0
    states[:, 1] = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    states[:, 2] = torch.tensor([40, 41], dtype=torch.int16, device="cuda")
    states[:, 3] = torch.tensor([50, 51], dtype=torch.int16, device="cuda")
    states[:, 4] = 64
    states[:, 5] = 64

    sim.refresh_compact_state_from_raw_batch(states)
    torch.cuda.synchronize()

    bitmap = sim.grid_ocp.cpu()
    assert _bitmap_cell_is_set(bitmap, 40, 50)
    assert _bitmap_cell_is_set(bitmap, 41, 51)
    assert not _bitmap_cell_is_set(bitmap, 42, 52)


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_step_compact_only_keeps_state_packed_in_sync(small_empty_grid):
    num_agents = 4
    sim = ext.GridWorldSimulator(small_empty_grid, num_agents, 2, MAP_W * MAP_H, 123)
    sim.run_initialization()

    actions = torch.zeros((1, num_agents), dtype=torch.uint8, device="cuda")
    sim.update_actions(actions)
    sim.step_compact_only()
    torch.cuda.synchronize()

    packed = sim.state_packed.cpu().to(torch.int64)
    cur_x = sim.cur_x.reshape(-1).cpu().to(torch.int64)
    cur_y = sim.cur_y.reshape(-1).cpu().to(torch.int64)
    goal_x = sim.goal_x.reshape(-1).cpu().to(torch.int64)
    goal_y = sim.goal_y.reshape(-1).cpu().to(torch.int64)
    goal_changed = sim.goal_changed_flags.reshape(-1).cpu().to(torch.int64)

    assert torch.equal(packed[:, 0], torch.zeros(num_agents, dtype=torch.int64))
    assert torch.equal(packed[:, 1], torch.arange(num_agents, dtype=torch.int64))
    assert torch.equal(packed[:, 2], cur_x)
    assert torch.equal(packed[:, 3], cur_y)
    assert torch.equal(packed[:, 4], goal_x)
    assert torch.equal(packed[:, 5], goal_y)
    assert torch.equal(packed[:, 6], torch.zeros(num_agents, dtype=torch.int64))
    assert torch.equal(packed[:, 7], goal_changed)


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateful_split_compact_pipeline_matches_full_step(small_empty_grid):
    num_agents = 4
    pool_capacity = MAP_W * MAP_H
    full = ext.GridWorldSimulator(small_empty_grid, num_agents, 2, pool_capacity, 321)
    split = ext.GridWorldSimulator(small_empty_grid, num_agents, 2, pool_capacity, 321)

    full.run_initialization()
    split.run_initialization()

    actions = torch.zeros((1, num_agents), dtype=torch.uint8, device="cuda")
    full.update_actions(actions)
    split.update_actions(actions)

    full.step()
    split.step_compact_only()
    split.update_derived_state()
    split.materialize_pyg_inputs()
    torch.cuda.synchronize()

    assert torch.equal(full.state_packed.cpu(), split.state_packed.cpu())
    assert torch.equal(full.pyg_pos.cpu(), split.pyg_pos.cpu())
    assert torch.allclose(full.pyg_x.cpu(), split.pyg_x.cpu())
    assert torch.equal(full.pyg_edge_index.cpu(), split.pyg_edge_index.cpu())
    assert torch.allclose(full.pyg_edge_attr.cpu(), split.pyg_edge_attr.cpu())


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_incremental_energy_refresh_tracks_changed_agents_and_packs_them(small_empty_grid):
    num_agents = 4
    sim = ext.GridWorldSimulator(small_empty_grid, num_agents, 2, MAP_W * MAP_H, 999)
    sim.run_initialization()

    sim.cur_x[0, :] = torch.tensor([10, 20, 30, 40], dtype=torch.int16, device="cuda")
    sim.cur_y[0, :] = torch.tensor([10, 20, 30, 40], dtype=torch.int16, device="cuda")
    sim.goal_x[0, :] = torch.tensor([11, 21, 31, 41], dtype=torch.int16, device="cuda")
    sim.goal_y[0, :] = torch.tensor([11, 21, 31, 41], dtype=torch.int16, device="cuda")
    sim.update_actions(torch.zeros((1, num_agents), dtype=torch.uint8, device="cuda"))

    sim.step_compact_only()
    sim.update_derived_state()
    torch.cuda.synchronize()

    assert sim.goal_changed_count.item() == 0

    sim.cur_x[0, 0] = sim.goal_x[0, 0]
    sim.cur_y[0, 0] = sim.goal_y[0, 0]
    old_goal = (
        int(sim.goal_x[0, 0].item()),
        int(sim.goal_y[0, 0].item()),
    )

    sim.step_compact_only()
    sim.update_derived_state()
    torch.cuda.synchronize()

    new_goal = (
        int(sim.goal_x[0, 0].item()),
        int(sim.goal_y[0, 0].item()),
    )
    changed = sim.changed_state_packed.cpu().to(torch.int64)

    assert new_goal != old_goal
    assert sim.goal_changed_flags.reshape(-1).sum().item() == 1
    assert sim.goal_changed_count.item() == 1
    assert torch.equal(
        changed[0],
        sim.state_packed[0].cpu().to(torch.int64),
    )


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_incremental_energy_refresh_matches_full_recompute_for_changed_agents(small_empty_grid):
    num_agents = 4
    sim = ext.GridWorldSimulator(small_empty_grid, num_agents, 2, MAP_W * MAP_H, 1001)
    ref = ext.StatelessGridWorldSimulator(small_empty_grid, num_agents, 3)
    sim.run_initialization()

    sim.cur_x[0, :] = torch.tensor([10, 20, 30, 40], dtype=torch.int16, device="cuda")
    sim.cur_y[0, :] = torch.tensor([10, 20, 30, 40], dtype=torch.int16, device="cuda")
    sim.goal_x[0, :] = torch.tensor([11, 21, 31, 41], dtype=torch.int16, device="cuda")
    sim.goal_y[0, :] = torch.tensor([11, 21, 31, 41], dtype=torch.int16, device="cuda")
    sim.update_actions(torch.zeros((1, num_agents), dtype=torch.uint8, device="cuda"))

    sim.step_compact_only()
    sim.update_derived_state()
    torch.cuda.synchronize()

    sim.cur_x[0, 0] = sim.goal_x[0, 0]
    sim.cur_y[0, 0] = sim.goal_y[0, 0]
    sim.step_compact_only()

    packed_after_step = sim.state_packed.clone()
    ref.update_energy_maps(packed_after_step, num_agents)
    sim.update_derived_state()
    torch.cuda.synchronize()

    assert sim.goal_changed_count.item() == 1
    assert torch.equal(sim.energy_maps.cpu(), ref.energy_maps.cpu())


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_stateless_materialization_handles_more_than_256_agents(small_empty_grid):
    num_agents = 300
    sim = ext.StatelessGridWorldSimulator(small_empty_grid, num_agents, 3)

    agent_ids = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    states = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device="cuda")
    states[:, 0] = 0
    states[:, 1] = agent_ids
    states[:, 2] = torch.div(agent_ids, 20, rounding_mode="floor") + 10
    states[:, 3] = torch.remainder(agent_ids, 20) + 10
    states[:, 4] = 64
    states[:, 5] = 64
    states[:, 7] = 1

    sim.update_energy_maps(states, num_agents)
    states[:, 7] = 0
    sim.refresh_compact_state_from_raw_batch(states)
    sim.materialize_pyg_inputs()
    torch.cuda.synchronize()

    expected_pos = states[:, 2:4].to(torch.float32).cpu()
    actual_pos = sim.pyg_pos.cpu()
    assert torch.equal(actual_pos, expected_pos)
    assert sim.pyg_num_edges.item() > 0


@pytest.mark.skipif(not EXT_AVAILABLE, reason="Extension not compiled")
def test_large_agent_local_gather_matches_legacy_prefix_features(small_empty_grid):
    legacy_agents = 256
    large_agents = 300
    grid = small_empty_grid.clone()
    grid[0, 20, 21] = 1

    legacy = ext.StatelessGridWorldSimulator(grid, legacy_agents, 3)
    large = ext.StatelessGridWorldSimulator(grid, large_agents, 3)

    def build_states(num_agents: int) -> torch.Tensor:
        agent_ids = torch.arange(num_agents, dtype=torch.int16, device="cuda")
        states = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device="cuda")
        states[:, 0] = 0
        states[:, 1] = agent_ids
        states[:, 2] = torch.div(agent_ids, 16, rounding_mode="floor") + 40
        states[:, 3] = torch.remainder(agent_ids, 16) + 40
        states[:, 4] = 64
        states[:, 5] = 64
        states[:, 7] = 1
        return states

    legacy_states = build_states(legacy_agents)
    large_states = build_states(large_agents)
    extra_agent_ids = torch.arange(large_agents - legacy_agents, dtype=torch.int16, device="cuda")
    large_states[legacy_agents:, 2] = torch.div(extra_agent_ids, 16, rounding_mode="floor") + 90
    large_states[legacy_agents:, 3] = torch.remainder(extra_agent_ids, 16) + 90

    for states in (legacy_states, large_states):
        states[0, 2:6] = torch.tensor([20, 20, 20, 24], dtype=torch.int16, device="cuda")
        states[1, 2:6] = torch.tensor([20, 22, 30, 30], dtype=torch.int16, device="cuda")
        states[2, 2:6] = torch.tensor([0, 0, 4, 4], dtype=torch.int16, device="cuda")

    legacy.update_energy_maps(legacy_states, legacy_agents)
    large.update_energy_maps(large_states, large_agents)
    legacy_states[:, 7] = 0
    large_states[:, 7] = 0

    legacy.refresh_compact_state_from_raw_batch(legacy_states)
    large.refresh_compact_state_from_raw_batch(large_states)
    legacy.materialize_pyg_inputs()
    large.materialize_pyg_inputs()
    torch.cuda.synchronize()

    assert torch.equal(large.pyg_pos[:legacy_agents].cpu(), legacy.pyg_pos.cpu())
    assert torch.allclose(large.pyg_x[:legacy_agents].cpu(), legacy.pyg_x.cpu())

    large_x = large.pyg_x.view(large_agents, 4, PYG_OBS_DIAM, PYG_OBS_DIAM).cpu()

    agent0 = large_x[0]
    assert agent0[1, 6, 6].item() == pytest.approx(1.0)
    assert agent0[0, 6, 7].item() == pytest.approx(1.0)
    assert agent0[1, 6, 8].item() == pytest.approx(1.0)
    assert agent0[2, 6, 10].item() == pytest.approx(1.0)
    assert agent0[3, 6, 6].item() == pytest.approx(0.0)

    agent2 = large_x[2]
    assert agent2[0, 6, 5].item() == pytest.approx(1.0)
    assert agent2[0, 6, 4].item() == pytest.approx(0.0)
    assert agent2[0, 5, 6].item() == pytest.approx(1.0)
    assert agent2[0, 4, 6].item() == pytest.approx(0.0)
    assert agent2[3, 6, 5].item() == pytest.approx(1.0)
    assert agent2[3, 6, 4].item() == pytest.approx(1.0)


@pytest.mark.skipif(
    not EXT_AVAILABLE or not LEGACY_FULLMAP_SUPPORTED,
    reason="legacy_fullmap equivalence check requires a legacy-supported compiled profile",
)
def test_builder_mode_property_can_force_equivalent_legacy_and_local_gather_paths(small_empty_grid):
    num_agents = 64
    grid = small_empty_grid.clone()
    grid[0, 39, 41] = 1

    legacy = ext.StatelessGridWorldSimulator(grid, num_agents, 3)
    local = ext.StatelessGridWorldSimulator(grid, num_agents, 3)

    legacy.pyg_builder_mode = "legacy_fullmap"
    local.pyg_builder_mode = "local_gather"

    assert legacy.pyg_builder_mode == "legacy_fullmap"
    assert local.pyg_builder_mode == "local_gather"

    agent_ids = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    states = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device="cuda")
    states[:, 0] = 0
    states[:, 1] = agent_ids
    states[:, 2] = torch.div(agent_ids, 8, rounding_mode="floor") + 40
    states[:, 3] = torch.remainder(agent_ids, 8) + 40
    states[:, 4] = 64
    states[:, 5] = 64
    states[:, 7] = 1

    legacy.update_energy_maps(states, num_agents)
    local.update_energy_maps(states, num_agents)
    states[:, 7] = 0

    legacy.refresh_compact_state_from_raw_batch(states)
    local.refresh_compact_state_from_raw_batch(states)
    legacy.materialize_pyg_inputs()
    local.materialize_pyg_inputs()
    torch.cuda.synchronize()

    assert torch.equal(legacy.pyg_pos.cpu(), local.pyg_pos.cpu())
    assert torch.allclose(legacy.pyg_x.cpu(), local.pyg_x.cpu())
    assert torch.equal(legacy.pyg_edge_index.cpu(), local.pyg_edge_index.cpu())
    assert torch.allclose(legacy.pyg_edge_attr.cpu(), local.pyg_edge_attr.cpu())


@pytest.mark.skipif(
    not EXT_AVAILABLE or LEGACY_FULLMAP_SUPPORTED,
    reason="unsupported legacy_fullmap path only applies to larger compiled profiles",
)
def test_builder_mode_property_rejects_legacy_fullmap_when_profile_exceeds_budget(small_empty_grid):
    sim = ext.StatelessGridWorldSimulator(small_empty_grid.clone(), 64, 3)

    with pytest.raises(RuntimeError, match="legacy_fullmap builder mode requires"):
        sim.pyg_builder_mode = "legacy_fullmap"


@pytest.mark.skipif(
    not EXT_AVAILABLE or not ASYNC_LOCAL_GATHER_HW_SUPPORTED,
    reason="async local gather requires a compiled extension on sm89+ hardware",
)
def test_async_local_gather_matches_scalar_on_supported_gpu(small_empty_grid):
    num_agents = 64
    grid = small_empty_grid.clone()
    grid[0, 39, 41] = 1

    scalar = ext.StatelessGridWorldSimulator(grid, num_agents, 3)
    async_sim = ext.StatelessGridWorldSimulator(grid, num_agents, 3)

    scalar.pyg_builder_mode = "local_gather"
    async_sim.pyg_builder_mode = "local_gather"
    scalar.pyg_local_gather_impl = "scalar"
    async_sim.pyg_local_gather_impl = "async_sm89plus"

    assert scalar.resolved_pyg_builder_impl == "local_gather_scalar"
    assert async_sim.resolved_pyg_builder_impl == "local_gather_async_sm89plus"

    agent_ids = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    states = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device="cuda")
    states[:, 0] = 0
    states[:, 1] = agent_ids
    states[:, 2] = torch.div(agent_ids, 8, rounding_mode="floor") + 40
    states[:, 3] = torch.remainder(agent_ids, 8) + 40
    states[:, 4] = 64
    states[:, 5] = 64
    states[:, 7] = 1

    scalar.update_energy_maps(states, num_agents)
    async_sim.update_energy_maps(states, num_agents)
    states[:, 7] = 0

    scalar.refresh_compact_state_from_raw_batch(states)
    async_sim.refresh_compact_state_from_raw_batch(states)
    scalar.materialize_pyg_inputs()
    async_sim.materialize_pyg_inputs()
    torch.cuda.synchronize()

    assert torch.equal(scalar.pyg_pos.cpu(), async_sim.pyg_pos.cpu())
    assert torch.allclose(scalar.pyg_x.cpu(), async_sim.pyg_x.cpu())
    assert torch.equal(scalar.pyg_edge_index.cpu(), async_sim.pyg_edge_index.cpu())
    assert torch.allclose(scalar.pyg_edge_attr.cpu(), async_sim.pyg_edge_attr.cpu())
