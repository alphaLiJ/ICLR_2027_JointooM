from __future__ import annotations

from experiments.runners.run_s3_memory_decomposition import (
    ROW_SPECS,
    STATEFUL_CLASSIFICATIONS,
    capacity_preflight_result,
    get_row_spec,
)


def test_s3_row_matrix_has_matched_full_and_microbatch_pair():
    full = get_row_spec("S3-M2F")
    micro = get_row_spec("S3-M2B")

    assert (full.num_agents, full.num_envs, full.scan_cell) == (
        micro.num_agents,
        micro.num_envs,
        micro.scan_cell,
    )
    assert full.model_microbatch_envs is None
    assert micro.model_microbatch_envs == 16


def test_s3_capacity_row_is_preflight_only_and_rejects_16_gib_fixture():
    result = capacity_preflight_result(
        get_row_spec("S3-C1"),
        device="cuda:0",
        memory_info=(15 * 1024**3, 16 * 1024**3),
    )

    assert result["estimate"]["mandatory_known_bytes"] > 15 * 1024**3
    assert result["preflight"]["allowed"] is False


def test_stateful_inventory_table_covers_current_native_surface():
    expected = {
        "grid_compressed",
        "free_cell_list",
        "pool_ptr",
        "offsets",
        "counts",
        "grid_ocp",
        "real_map_extents",
        "cur_x",
        "cur_y",
        "goal_x",
        "goal_y",
        "agents_obs",
        "rewards",
        "actions",
        "rng_states",
        "max_free_cell_count",
        "goal_changed_flags",
        "arrived",
        "terminated",
        "truncated",
        "step_counts",
        "goal_changed_prefix",
        "changed_state_packed",
        "goal_changed_count",
        "energy_maps",
        "state_packed",
        "pyg_x",
        "pyg_pos",
        "pyg_edge_index",
        "pyg_edge_attr",
        "pyg_batch",
        "pyg_ptr",
        "pyg_num_edges",
        "edge_counts",
        "pyg_edge_index_storage",
        "pyg_edge_attr_storage",
        "pyg_edge_prefix",
        "scan_temp_storage",
    }
    assert set(STATEFUL_CLASSIFICATIONS) == expected
    assert set(ROW_SPECS) == {
        "S3-M1",
        "S3-M2F",
        "S3-M2B",
        "S3-M3",
        "S3-G1",
        "S3-T1",
        "S3-C1",
    }
