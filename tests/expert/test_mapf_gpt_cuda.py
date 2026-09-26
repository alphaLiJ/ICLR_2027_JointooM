import pytest
import torch
import numpy as np


ext = pytest.importorskip("grid_world_cpp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _grids(num_envs=2, *, device="cuda", dtype=torch.int32):
    return torch.zeros(
        (num_envs, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=dtype,
        device=device,
    )


def _valid_rows(num_envs=2, num_agents=4):
    rows = torch.zeros((num_envs * num_agents, 13), dtype=torch.int16, device="cuda")
    blocks = rows.view(num_envs, num_agents, 13)
    blocks[:, :, 0] = torch.arange(num_envs, dtype=torch.int16, device="cuda")[:, None]
    blocks[:, :, 1] = torch.arange(num_agents, dtype=torch.int16, device="cuda")[None, :]
    blocks[:, :, 2] = 10
    blocks[:, :, 3] = torch.arange(num_agents, dtype=torch.int16, device="cuda")[None, :] + 10
    blocks[:, :, 4] = 20
    blocks[:, :, 5] = torch.arange(num_agents, dtype=torch.int16, device="cuda")[None, :] + 10
    blocks[:, :, 6] = 0
    blocks[:, :, 7] = 1
    blocks[:, :, 8:13] = 5
    return rows


def test_mapf_gpt_builder_api_allocates_only_linear_outputs():
    builder = ext.MapfGPTObservationBuilder(_grids(), 4)

    assert builder.tokens.shape == (8, 256)
    assert builder.tokens.dtype == torch.int32
    assert builder.labels.shape == (8,)
    assert builder.labels.dtype == torch.int64
    assert builder.cost_to_go.shape == (
        2,
        4,
        ext.COMPILED_MAP_W,
        ext.COMPILED_MAP_H,
    )
    assert builder.cost_to_go.dtype == torch.int16
    assert builder.state_packed.shape == (8, 8)
    assert builder.state_packed.dtype == torch.int16
    assert builder.agent_at_cell.shape == (
        2,
        ext.COMPILED_MAP_W,
        ext.COMPILED_MAP_H,
    )
    assert builder.agent_at_cell.dtype == torch.int32
    assert builder.pyg_ptr.tolist() == [0, 4, 8]

    assert not hasattr(builder, "pyg_edge_index_storage")
    assert not hasattr(builder, "pyg_edge_attr_storage")
    assert not hasattr(builder, "pyg_x")


def test_mapf_gpt_memory_inventory_includes_goal_cache_without_allocation():
    builder = ext.MapfGPTObservationBuilder(_grids(), 4)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    inventory = builder.memory_tensors
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    assert after == before
    assert inventory["goal_cache"].shape == (8, 2)
    assert inventory["tokens"] is builder.tokens


def test_token_kernel_materializes_on_target_active_mask():
    builder = ext.MapfGPTObservationBuilder(_grids(num_envs=1), 4)
    rows = _valid_rows(num_envs=1, num_agents=4)
    rows[0, 4:6] = rows[0, 2:4]
    rows[2, 4:6] = rows[2, 2:4]

    builder.build_tokens(rows)

    assert builder.active_mask.is_cuda
    assert builder.active_mask.dtype == torch.bool
    assert builder.active_mask.shape == (4,)
    assert builder.active_mask.cpu().tolist() == [False, True, False, True]


def test_mapf_gpt_builder_work_is_owned_by_current_stream():
    num_agents = 512
    builder = ext.MapfGPTObservationBuilder(_grids(num_envs=1), num_agents)
    source = _valid_rows(num_envs=1, num_agents=num_agents)
    agent_ids = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    source[:, 2] = agent_ids // 64 + 20
    source[:, 3] = agent_ids % 64 + 20
    source[:, 4] = 100
    source[:, 5] = source[:, 3]
    source[:, 6] = agent_ids % 5
    delayed_rows = torch.zeros_like(source)
    stream = torch.cuda.Stream(priority=-1)
    current_done = torch.cuda.Event()
    default_done = torch.cuda.Event()
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        delayed_rows.copy_(source)
        builder.build_tokens(delayed_rows)
        current_done.record()
    default_done.record(torch.cuda.default_stream())

    current_done.synchronize()
    assert default_done.query(), "builder work escaped to the CUDA default stream"
    assert builder.labels.cpu().tolist() == (agent_ids % 5).cpu().tolist()
    assert builder.diagnostics.cpu().tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "grids, message",
    [
        (lambda: _grids(device="cpu"), "CUDA"),
        (lambda: _grids(dtype=torch.float32), "int32"),
        (
            lambda: torch.zeros((1, 8, 8), dtype=torch.int32, device="cuda"),
            "shape",
        ),
    ],
)
def test_mapf_gpt_builder_rejects_invalid_grids(grids, message):
    with pytest.raises(RuntimeError, match=message):
        ext.MapfGPTObservationBuilder(grids(), 4)


def test_mapf_gpt_builder_rejects_invalid_raw_stage_layouts():
    builder = ext.MapfGPTObservationBuilder(_grids(), 4)
    rows = _valid_rows()

    with pytest.raises(RuntimeError, match="int16"):
        builder.build_tokens(rows.to(torch.int32))
    with pytest.raises(RuntimeError, match="13"):
        builder.build_tokens(rows[:, :8].contiguous())
    with pytest.raises(RuntimeError, match="row count"):
        builder.build_tokens(rows[:-1].contiguous())
    backing = torch.empty((8, 26), dtype=torch.int16, device="cuda")
    non_contiguous = backing[:, ::2]
    non_contiguous.copy_(rows)
    assert not non_contiguous.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        builder.build_tokens(non_contiguous)


def test_invalid_position_is_diagnosed_without_cuda_out_of_bounds():
    builder = ext.MapfGPTObservationBuilder(_grids(num_envs=1), 1)
    rows = _valid_rows(num_envs=1, num_agents=1)
    rows[0, 2] = ext.COMPILED_MAP_W

    builder.build_tokens(rows)
    diagnostics = builder.diagnostics.detach().cpu().tolist()

    assert diagnostics[3] > 0
    assert torch.all(builder.tokens == 66)


def _unsigned_cost(builder):
    return builder.cost_to_go.to(torch.int32).bitwise_and(0xFFFF)


def test_cost_to_go_is_exact_uint16_and_exceeds_254():
    grid = torch.ones(
        (1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )
    # Two full horizontal corridors connected only at the far right:
    # distance (0,0) -> (2,0) is 128 + 128 = 256.
    grid[0, 0, :] = 0
    grid[0, 1, -1] = 0
    grid[0, 2, :] = 0
    builder = ext.MapfGPTObservationBuilder(grid, 1)
    rows = _valid_rows(num_envs=1, num_agents=1)
    rows[0, 2:6] = torch.tensor([2, 0, 0, 0], dtype=torch.int16, device="cuda")
    rows[0, 7] = 0  # cache-invalid must still force the first refresh

    builder.build_tokens(rows)
    cost = _unsigned_cost(builder)[0, 0]
    assert cost[0, 0].item() == 0
    assert cost[0, 1].item() == 1
    assert cost[2, 0].item() == 256
    assert cost[3, 0].item() == 65535
    assert builder.diagnostics[0].item() == 0


def test_cost_to_go_refreshes_on_goal_change_and_replay_without_flag():
    builder = ext.MapfGPTObservationBuilder(_grids(num_envs=1), 1)
    rows_a = _valid_rows(num_envs=1, num_agents=1)
    rows_a[0, 2:6] = torch.tensor([10, 10, 10, 20], dtype=torch.int16, device="cuda")
    rows_a[0, 7] = 0
    rows_b = rows_a.clone()
    rows_b[0, 4:6] = torch.tensor([30, 30], dtype=torch.int16, device="cuda")

    builder.build_tokens(rows_a)
    cost_a = _unsigned_cost(builder).clone()
    builder.build_tokens(rows_b)
    cost_b = _unsigned_cost(builder).clone()
    builder.build_tokens(rows_a)
    replay_a = _unsigned_cost(builder).clone()

    assert cost_a[0, 0, 10, 20].item() == 0
    assert cost_b[0, 0, 30, 30].item() == 0
    assert not torch.equal(cost_a, cost_b)
    assert torch.equal(cost_a, replay_a)


def _parity_fixture(num_agents=16):
    grid = np.zeros((ext.COMPILED_MAP_W, ext.COMPILED_MAP_H), dtype=np.int32)
    # Obstacles exercise unreachable tokens and multi-direction greedy masks.
    grid[7:25, 17] = 1
    grid[15, 17] = 0
    positions = [(15, 15)]
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if len(positions) >= num_agents:
                break
            pos = (15 + dx, 15 + dy)
            if pos != positions[0] and grid[pos] == 0:
                positions.append(pos)
        if len(positions) >= num_agents:
            break
    rows = np.zeros((num_agents, 13), dtype=np.uint16)
    rows[:, 0] = 0
    rows[:, 1] = np.arange(num_agents, dtype=np.uint16)
    rows[:, 2:4] = positions
    rows[:, 4] = 25
    rows[:, 5] = np.arange(num_agents, dtype=np.uint16) % 10 + 20
    rows[:, 6] = np.arange(num_agents, dtype=np.uint16) % 5
    rows[:, 7] = 1
    for agent_id in range(num_agents):
        rows[agent_id, 8:13] = [
            5 if slot < 2 else (agent_id + slot) % 5 for slot in range(5)
        ]
    return grid[None, ...], rows


def test_cuda_tokens_match_reference_for_top13_boundaries_and_histories():
    from expert.mapf_gpt_schema import tokenize_stage_reference

    grids_np, rows_np = _parity_fixture()
    expected_tokens, expected_labels = tokenize_stage_reference(
        grids_np, rows_np, num_agents=rows_np.shape[0]
    )
    builder = ext.MapfGPTObservationBuilder(
        torch.from_numpy(grids_np).to(device="cuda", dtype=torch.int32),
        rows_np.shape[0],
    )
    builder.build_tokens(
        torch.from_numpy(rows_np.astype(np.int16)).to(device="cuda")
    )

    assert torch.equal(
        builder.tokens.cpu(), torch.from_numpy(expected_tokens)
    )
    assert torch.equal(
        builder.labels.cpu(), torch.from_numpy(expected_labels)
    )
    assert builder.diagnostics.cpu().tolist() == [0, 0, 0, 0]


def test_cuda_tokens_are_replay_deterministic():
    grids_np, rows_a_np = _parity_fixture()
    rows_b_np = rows_a_np.copy()
    rows_b_np[:, 4] = 30
    builder = ext.MapfGPTObservationBuilder(
        torch.from_numpy(grids_np).to(device="cuda", dtype=torch.int32),
        rows_a_np.shape[0],
    )
    rows_a = torch.from_numpy(rows_a_np.astype(np.int16)).to("cuda")
    rows_b = torch.from_numpy(rows_b_np.astype(np.int16)).to("cuda")

    builder.build_tokens(rows_a)
    tokens_a = builder.tokens.clone()
    builder.build_tokens(rows_b)
    builder.build_tokens(rows_a)
    assert torch.equal(tokens_a, builder.tokens)


def test_resident_builder_matches_raw_rows_without_current_action_leakage():
    grids_np, rows_np = _parity_fixture(num_agents=8)
    grids = torch.from_numpy(grids_np).to(device="cuda", dtype=torch.int32)
    rows = torch.from_numpy(rows_np.astype(np.int16)).to("cuda")
    raw_builder = ext.MapfGPTObservationBuilder(grids, 8)
    resident_builder = ext.MapfGPTObservationBuilder(grids, 8)
    resident_builder.histories.copy_(rows[:, 8:13])

    raw_builder.build_tokens(rows)
    blocks = rows.view(1, 8, 13)
    resident_builder.build_tokens_from_state(
        blocks[:, :, 2].contiguous(),
        blocks[:, :, 3].contiguous(),
        blocks[:, :, 4].contiguous(),
        blocks[:, :, 5].contiguous(),
    )

    assert torch.equal(resident_builder.tokens, raw_builder.tokens)
    assert torch.equal(resident_builder.active_mask, raw_builder.active_mask)
    assert resident_builder.diagnostics.cpu().tolist() == [0, 0, 0, 0]


def test_resident_history_rolls_only_after_action_selection_and_masks_on_target():
    builder = ext.MapfGPTObservationBuilder(_grids(num_envs=1), 3)
    cur_x = torch.tensor([[10, 11, 12]], dtype=torch.int16, device="cuda")
    cur_y = torch.tensor([[10, 11, 12]], dtype=torch.int16, device="cuda")
    goal_x = torch.tensor([[20, 11, 22]], dtype=torch.int16, device="cuda")
    goal_y = torch.tensor([[20, 11, 22]], dtype=torch.int16, device="cuda")

    builder.reset_histories()
    builder.build_tokens_from_state(cur_x, cur_y, goal_x, goal_y)
    assert builder.histories.cpu().tolist() == [[5] * 5, [5] * 5, [5] * 5]
    assert builder.active_mask.cpu().tolist() == [True, False, True]

    builder.append_actions(
        torch.tensor([[1, 4, 3]], dtype=torch.uint8, device="cuda")
    )
    assert builder.histories.cpu().tolist() == [
        [5, 5, 5, 5, 1],
        [5, 5, 5, 5, 0],
        [5, 5, 5, 5, 3],
    ]

    builder.build_tokens_from_state(cur_x, cur_y, goal_x, goal_y)
    rows = torch.zeros((3, 13), dtype=torch.int16, device="cuda")
    rows[:, 1] = torch.arange(3, dtype=torch.int16, device="cuda")
    rows[:, 2] = cur_x[0]
    rows[:, 3] = cur_y[0]
    rows[:, 4] = goal_x[0]
    rows[:, 5] = goal_y[0]
    rows[:, 8:13] = builder.histories
    reference = ext.MapfGPTObservationBuilder(_grids(num_envs=1), 3)
    reference.build_tokens(rows)
    assert torch.equal(builder.tokens, reference.tokens)


def test_resident_builder_rejects_invalid_state_and_action_contracts():
    builder = ext.MapfGPTObservationBuilder(_grids(num_envs=1), 2)
    valid = torch.tensor([[10, 11]], dtype=torch.int16, device="cuda")

    with pytest.raises(RuntimeError, match="int16"):
        builder.build_tokens_from_state(
            valid.to(torch.int32), valid, valid, valid
        )
    with pytest.raises(RuntimeError, match="shape"):
        builder.build_tokens_from_state(
            valid[:, :1].contiguous(), valid, valid, valid
        )
    with pytest.raises(RuntimeError, match="uint8"):
        builder.append_actions(valid)
    with pytest.raises(RuntimeError, match="shape"):
        builder.append_actions(
            torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
        )


@pytest.mark.parametrize("num_agents", [256, 512])
def test_cuda_token_builder_scales_total_agents_independently_of_visible_cap(num_agents):
    grids = _grids(num_envs=1)
    rows = _valid_rows(num_envs=1, num_agents=num_agents)
    blocks = rows.view(1, num_agents, 13)
    agent_ids = torch.arange(num_agents, dtype=torch.int16, device="cuda")
    blocks[0, :, 2] = agent_ids // 64 + 20
    blocks[0, :, 3] = agent_ids % 64 + 20
    blocks[0, :, 4] = 100
    blocks[0, :, 5] = agent_ids % 64 + 20
    builder = ext.MapfGPTObservationBuilder(grids, num_agents)

    builder.build_tokens(rows)
    assert builder.tokens.shape == (num_agents, 256)
    assert torch.all((builder.tokens >= 0) & (builder.tokens <= 66))
    assert builder.diagnostics[0].item() == 0
