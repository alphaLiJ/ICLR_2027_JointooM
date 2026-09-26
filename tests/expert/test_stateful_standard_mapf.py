"""Focused CUDA tests for explicit finite-horizon standard MAPF semantics."""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the stateful simulator"
)

try:
    import grid_world_cpp as ext
except ImportError:
    ext = None


MAP_W = 128
MAP_H = 128
MAP_OFFSET = MAP_W * MAP_H // 32


def _sim(
    *,
    num_envs: int = 1,
    num_agents: int = 1,
    seed: int = 7,
    task_mode: str | None = None,
    max_episode_steps: int = 256,
):
    if ext is None:
        pytest.skip("grid_world_cpp extension is not built")
    grid = torch.zeros(
        (num_envs, MAP_W, MAP_H), dtype=torch.int32, device="cuda"
    )
    args = [grid, num_agents, 2, num_envs * MAP_W * MAP_H, seed]
    if task_mode is not None:
        args.extend([task_mode, max_episode_steps])
    return ext.GridWorldSimulator(*args)


def _load(
    sim,
    positions,
    goals,
    *,
    arrived=None,
    step_counts=None,
):
    positions = torch.tensor(positions, dtype=torch.int16, device="cuda").contiguous()
    goals = torch.tensor(goals, dtype=torch.int16, device="cuda").contiguous()
    num_envs, num_agents, _ = positions.shape
    if arrived is None:
        arrived = [[False] * num_agents for _ in range(num_envs)]
    if step_counts is None:
        step_counts = [0] * num_envs
    arrived = torch.tensor(arrived, dtype=torch.uint8, device="cuda").contiguous()
    step_counts = torch.tensor(
        step_counts, dtype=torch.int32, device="cuda"
    ).contiguous()
    sim.load_state(positions, goals, arrived, step_counts)
    torch.cuda.synchronize()


def _set_actions(sim, actions):
    sim.update_actions(
        torch.tensor(actions, dtype=torch.uint8, device="cuda").contiguous()
    )


def _valid_state_tensors(*, num_envs=2, num_agents=2, device="cuda"):
    positions = torch.zeros(
        (num_envs, num_agents, 2), dtype=torch.int16, device=device
    )
    goals = torch.zeros_like(positions)
    for env_id in range(num_envs):
        for agent_id in range(num_agents):
            cell = env_id * num_agents + agent_id
            positions[env_id, agent_id] = torch.tensor(
                [2 + cell, 3 + cell], dtype=torch.int16, device=device
            )
            goals[env_id, agent_id] = torch.tensor(
                [20 + cell, 21 + cell], dtype=torch.int16, device=device
            )
    arrived = torch.zeros(
        (num_envs, num_agents), dtype=torch.uint8, device=device
    )
    step_counts = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    return positions, goals, arrived, step_counts


def test_default_mode_is_lifelong_and_mode_is_read_only():
    sim = _sim()
    assert sim.task_mode == "lifelong"
    assert sim.max_episode_steps == 256
    with pytest.raises(AttributeError):
        sim.task_mode = "standard_mapf"
    with pytest.raises(AttributeError):
        sim.max_episode_steps = 7


def test_lifecycle_tensor_properties_are_read_only_and_readable():
    sim = _sim(task_mode="standard_mapf")
    properties = ("arrived", "terminated", "truncated", "step_counts")
    for name in properties:
        value = getattr(sim, name)
        assert isinstance(value, torch.Tensor)
        with pytest.raises(AttributeError):
            setattr(sim, name, value.clone())
        assert getattr(sim, name) is value


def test_standard_mode_accepts_known_mode_and_rejects_unknown_mode():
    sim = _sim(task_mode="standard_mapf", max_episode_steps=17)
    assert sim.task_mode == "standard_mapf"
    assert sim.max_episode_steps == 17
    with pytest.raises((RuntimeError, ValueError), match="Unknown task mode"):
        _sim(task_mode="not-a-mapf-mode")
    with pytest.raises((RuntimeError, ValueError), match="positive"):
        _sim(task_mode="standard_mapf", max_episode_steps=0)


def test_load_state_rebuilds_split_state_occupancy_and_packed_state():
    sim = _sim(num_envs=2, num_agents=2, task_mode="standard_mapf")
    positions = [[[2, 3], [4, 35]], [[7, 8], [9, 10]]]
    goals = [[[2, 3], [6, 7]], [[11, 12], [13, 14]]]
    arrived = [[True, False], [False, False]]
    _load(sim, positions, goals, arrived=arrived, step_counts=[3, 4])

    assert torch.equal(
        sim.cur_x.cpu(), torch.tensor([[2, 4], [7, 9]], dtype=torch.int16)
    )
    assert torch.equal(
        sim.cur_y.cpu(), torch.tensor([[3, 35], [8, 10]], dtype=torch.int16)
    )
    assert torch.equal(
        sim.goal_x.cpu(), torch.tensor([[2, 6], [11, 13]], dtype=torch.int16)
    )
    assert torch.equal(
        sim.goal_y.cpu(), torch.tensor([[3, 7], [12, 14]], dtype=torch.int16)
    )
    assert torch.equal(sim.arrived.cpu(), torch.tensor(arrived, dtype=torch.uint8))
    assert torch.equal(sim.step_counts.cpu(), torch.tensor([3, 4], dtype=torch.int32))
    assert not bool(sim.terminated.any().item())
    assert not bool(sim.truncated.any().item())
    assert bool(sim.goal_changed_flags.all().item())

    occupancy = sim.grid_ocp.cpu()
    expected = torch.zeros((2, MAP_OFFSET), dtype=torch.int32)
    for env_id, env_positions in enumerate(positions):
        for x, y in env_positions:
            expected[env_id, x * (MAP_H // 32) + (y >> 5)] |= 1 << (y & 31)
    assert torch.equal(occupancy, expected)

    assert sim.state_packed.cpu().tolist() == [
        [0, 0, 2, 3, 2, 3, 0, 1],
        [0, 1, 4, 35, 6, 7, 0, 1],
        [1, 0, 7, 8, 11, 12, 0, 1],
        [1, 1, 9, 10, 13, 14, 0, 1],
    ]


def test_load_state_marks_goals_for_first_derived_state_rebuild():
    grid = torch.zeros((1, MAP_W, MAP_H), dtype=torch.int32, device="cuda")
    stateful = ext.GridWorldSimulator(
        grid, 1, 0, MAP_W * MAP_H, 7, "standard_mapf", 256
    )
    positions = torch.tensor([[[10, 10]]], dtype=torch.int16, device="cuda")
    goals = torch.tensor([[[20, 20]]], dtype=torch.int16, device="cuda")
    arrived = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    counts = torch.zeros((1,), dtype=torch.int32, device="cuda")
    stateful.load_state(positions, goals, arrived, counts)
    stateful.update_derived_state()
    stateful.build_magat_plus_inputs()

    stateless = ext.StatelessGridWorldSimulator(grid, 1, 3)
    raw = torch.tensor(
        [[0, 0, 10, 10, 20, 20, 0, 1]], dtype=torch.int16, device="cuda"
    )
    stateless.update_energy_maps(raw, 1)
    stateless.build_magat_plus_inputs(raw)
    torch.cuda.synchronize()

    assert torch.equal(stateful.energy_maps, stateless.energy_maps)
    assert torch.equal(stateful.pyg_x, stateless.pyg_x)


@pytest.mark.parametrize(
    ("positions", "goals", "arrived", "step_counts", "message"),
    [
        ([[[-1, 2]]], [[[3, 4]]], [[0]], [0], "bounds"),
        ([[[2, 2]]], [[[128, 4]]], [[0]], [0], "bounds"),
        ([[[2, 2]]], [[[3, 4]]], [[1]], [0], "arrived agents"),
        ([[[2, 2]]], [[[3, 4]]], [[0]], [-1], "step_counts"),
        ([[[2, 2]]], [[[3, 4]]], [[0]], [257], "step_counts"),
    ],
)
def test_load_state_validates_bounds_arrival_and_step_counts(
    positions, goals, arrived, step_counts, message
):
    sim = _sim(task_mode="standard_mapf")
    with pytest.raises(RuntimeError, match=message):
        _load(
            sim,
            positions,
            goals,
            arrived=arrived,
            step_counts=step_counts,
        )


def test_load_state_requires_exact_cuda_contiguous_tensor_contract():
    sim = _sim(task_mode="standard_mapf")
    pos = torch.tensor([[[2, 2]]], dtype=torch.int16, device="cuda")
    goal = torch.tensor([[[3, 3]]], dtype=torch.int16, device="cuda")
    arrived = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    counts = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="positions must be int16"):
        sim.load_state(pos.to(torch.int32), goal, arrived, counts)
    with pytest.raises(RuntimeError, match="positions must have shape"):
        sim.load_state(pos.reshape(1, 2), goal, arrived, counts)
    noncontiguous = torch.zeros((1, 1, 4), dtype=torch.int16, device="cuda")[:, :, ::2]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(RuntimeError, match="positions must be contiguous"):
        sim.load_state(noncontiguous, goal, arrived, counts)


@pytest.mark.parametrize("field", ["positions", "goals", "arrived", "step_counts"])
@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("cpu", "must be CUDA"),
        ("dtype", "must be (int16|uint8|int32)"),
        ("shape", "must have shape"),
        ("noncontiguous", "must be contiguous"),
    ],
)
def test_load_state_validates_each_tensor_contract(field, violation, message):
    sim = _sim(num_envs=2, num_agents=2, task_mode="standard_mapf")
    tensors = list(_valid_state_tensors())
    field_index = {
        "positions": 0,
        "goals": 1,
        "arrived": 2,
        "step_counts": 3,
    }[field]
    original = tensors[field_index]

    if violation == "cpu":
        invalid = original.cpu()
    elif violation == "dtype":
        invalid = original.to(torch.int64)
    elif violation == "shape":
        invalid = original.unsqueeze(-1)
    elif field in {"positions", "goals"}:
        backing = torch.zeros((2, 2, 4), dtype=torch.int16, device="cuda")
        invalid = backing[:, :, ::2]
        invalid.copy_(original)
    elif field == "arrived":
        backing = torch.zeros((2, 4), dtype=torch.uint8, device="cuda")
        invalid = backing[:, ::2]
        invalid.copy_(original)
    else:
        backing = torch.zeros((4,), dtype=torch.int32, device="cuda")
        invalid = backing[::2]
        invalid.copy_(original)

    if violation == "noncontiguous":
        assert invalid.shape == original.shape
        assert not invalid.is_contiguous()
    tensors[field_index] = invalid
    with pytest.raises(RuntimeError, match=message):
        sim.load_state(*tensors)


def test_load_state_rejects_arrived_values_above_one():
    sim = _sim(num_envs=2, num_agents=2, task_mode="standard_mapf")
    positions, goals, arrived, step_counts = _valid_state_tensors()
    arrived[1, 1] = 2
    with pytest.raises(RuntimeError, match="arrived values must be 0 or 1"):
        sim.load_state(positions, goals, arrived, step_counts)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("field_index", [0, 1, 2, 3])
def test_load_state_rejects_tensor_on_different_cuda_device(field_index):
    grid = torch.zeros((2, MAP_W, MAP_H), dtype=torch.int32, device="cuda:0")
    sim = ext.GridWorldSimulator(
        grid,
        2,
        2,
        2 * MAP_W * MAP_H,
        7,
        "standard_mapf",
        256,
    )
    tensors = list(_valid_state_tensors(device="cuda:0"))
    tensors[field_index] = tensors[field_index].to("cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        sim.load_state(*tensors)


def test_update_actions_copies_from_caller_owned_tensor():
    sim = _sim(task_mode="standard_mapf")
    _load(sim, [[[10, 10]]], [[[20, 20]]])
    caller_actions = torch.tensor([[2]], dtype=torch.uint8, device="cuda")
    sim.update_actions(caller_actions)
    caller_actions.zero_()
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[11]]


def test_load_state_resets_previous_actions_to_stay():
    sim = _sim(task_mode="standard_mapf")
    _load(sim, [[[10, 10]]], [[[20, 20]]])
    _set_actions(sim, [[2]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[11]]

    _load(sim, [[[30, 30]]], [[[40, 40]]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[30]]
    assert sim.cur_y.cpu().tolist() == [[30]]
    assert sim.state_packed[:, 6].cpu().tolist() == [0]


def test_run_initialization_resets_actions_to_stay():
    sim = _sim(num_agents=2)
    _set_actions(sim, [[2, 4]])
    assert sim.actions.cpu().tolist() == [[2, 4]]
    sim.run_initialization()
    torch.cuda.synchronize()
    assert sim.actions.cpu().tolist() == [[0, 0]]


def test_arrived_agent_packs_actual_stay_action():
    # Pair the arrived agent with an unfinished one so the environment still
    # executes a transition.
    sim = _sim(num_agents=2, task_mode="standard_mapf")
    _load(
        sim,
        [[[10, 10], [20, 20]]],
        [[[10, 10], [30, 30]]],
        arrived=[[True, False]],
    )
    _set_actions(sim, [[2, 0]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[10, 20]]
    assert sim.state_packed[:, 6].cpu().tolist() == [0, 0]


def test_update_actions_validates_device_and_contiguous_layout():
    sim = _sim(num_agents=2, task_mode="standard_mapf")
    with pytest.raises(RuntimeError, match="CUDA"):
        sim.update_actions(torch.zeros((1, 2), dtype=torch.uint8))
    backing = torch.zeros((1, 4), dtype=torch.uint8, device="cuda")
    noncontiguous = backing[:, ::2]
    assert noncontiguous.shape == (1, 2)
    assert not noncontiguous.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        sim.update_actions(noncontiguous)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_update_actions_rejects_different_cuda_device():
    grid = torch.zeros((1, MAP_W, MAP_H), dtype=torch.int32, device="cuda:0")
    sim = ext.GridWorldSimulator(
        grid, 2, 2, MAP_W * MAP_H, 7, "standard_mapf", 256
    )
    actions = torch.zeros((1, 2), dtype=torch.uint8, device="cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        sim.update_actions(actions)


def test_nondefault_stream_orders_load_actions_and_step():
    sim = _sim(task_mode="standard_mapf")
    stream = torch.cuda.Stream()
    positions = torch.tensor([[[10, 10]]], dtype=torch.int16, device="cuda")
    goals = torch.tensor([[[20, 20]]], dtype=torch.int16, device="cuda")
    arrived = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    step_counts = torch.zeros((1,), dtype=torch.int32, device="cuda")
    actions = torch.tensor([[2]], dtype=torch.uint8, device="cuda")

    with torch.cuda.stream(stream):
        sim.load_state(positions, goals, arrived, step_counts)
        sim.update_actions(actions)
        sim.step_sim_only()
    stream.synchronize()

    assert sim.cur_x.cpu().tolist() == [[11]]
    assert sim.cur_y.cpu().tolist() == [[10]]
    assert sim.step_counts.cpu().tolist() == [1]


def test_standard_mode_keeps_fixed_goals():
    sim = _sim(num_agents=2, task_mode="standard_mapf")
    goals = [[[20, 20], [30, 30]]]
    _load(sim, [[[10, 10], [12, 12]]], goals)
    _set_actions(sim, [[0, 0]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.goal_x.cpu().tolist() == [[20, 30]]
    assert sim.goal_y.cpu().tolist() == [[20, 30]]
    assert not bool(sim.goal_changed_flags.any().item())
    assert sim.state_packed[:, 7].cpu().tolist() == [0, 0]


def test_arrived_agent_stays_and_occupies_goal():
    sim = _sim(num_agents=2, task_mode="standard_mapf")
    _load(
        sim,
        [[[10, 10], [11, 10]]],
        [[[10, 10], [20, 20]]],
        arrived=[[True, False]],
    )
    _set_actions(sim, [[2, 1]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[10, 11]]
    assert sim.cur_y.cpu().tolist() == [[10, 10]]
    word = 10 * (MAP_H // 32)
    assert int(sim.grid_ocp[0, word].item()) & (1 << 10)
    assert sim.arrived.cpu().tolist() == [[1, 0]]


def test_final_arrival_terminates_immediately():
    sim = _sim(task_mode="standard_mapf")
    _load(sim, [[[10, 10]]], [[[11, 10]]])
    _set_actions(sim, [[2]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[11]]
    assert sim.arrived.cpu().tolist() == [[1]]
    assert sim.step_counts.cpu().tolist() == [1]
    assert sim.terminated.cpu().tolist() == [1]
    assert sim.truncated.cpu().tolist() == [0]


def test_horizon_truncates_unfinished_env():
    sim = _sim(task_mode="standard_mapf", max_episode_steps=3)
    _load(sim, [[[10, 10]]], [[[20, 20]]], step_counts=[2])
    _set_actions(sim, [[0]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.step_counts.cpu().tolist() == [3]
    assert sim.terminated.cpu().tolist() == [0]
    assert sim.truncated.cpu().tolist() == [1]


def test_final_arrival_at_horizon_prefers_termination_over_truncation():
    sim = _sim(task_mode="standard_mapf", max_episode_steps=3)
    _load(sim, [[[10, 10]]], [[[11, 10]]], step_counts=[2])
    _set_actions(sim, [[2]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[11]]
    assert sim.step_counts.cpu().tolist() == [3]
    assert sim.terminated.cpu().tolist() == [1]
    assert sim.truncated.cpu().tolist() == [0]


def test_same_step_isolates_terminated_and_truncated_environments():
    sim = _sim(
        num_envs=2,
        num_agents=1,
        task_mode="standard_mapf",
        max_episode_steps=3,
    )
    _load(
        sim,
        [[[10, 10]], [[20, 20]]],
        [[[11, 10]], [[30, 30]]],
        step_counts=[2, 2],
    )
    _set_actions(sim, [[2], [0]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert sim.cur_x.cpu().tolist() == [[11], [20]]
    assert sim.cur_y.cpu().tolist() == [[10], [20]]
    assert sim.arrived.cpu().tolist() == [[1], [0]]
    assert sim.step_counts.cpu().tolist() == [3, 3]
    assert sim.terminated.cpu().tolist() == [1, 0]
    assert sim.truncated.cpu().tolist() == [0, 1]


@pytest.mark.parametrize("done_kind", ["terminated", "truncated"])
def test_done_environment_no_longer_advances(done_kind):
    if done_kind == "terminated":
        sim = _sim(task_mode="standard_mapf")
        _load(sim, [[[10, 10]]], [[[11, 10]]])
        _set_actions(sim, [[2]])
    else:
        sim = _sim(task_mode="standard_mapf", max_episode_steps=1)
        _load(sim, [[[10, 10]]], [[[20, 20]]])
        _set_actions(sim, [[0]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    before = (
        sim.cur_x.clone(),
        sim.cur_y.clone(),
        sim.step_counts.clone(),
        sim.state_packed.clone(),
        sim.grid_ocp.clone(),
    )
    _set_actions(sim, [[2]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    after = (sim.cur_x, sim.cur_y, sim.step_counts, sim.state_packed, sim.grid_ocp)
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_lifelong_fixture_still_reassigns_goal():
    sim = _sim(seed=7)
    sim.run_initialization()
    sim.cur_x[0, 0] = 10
    sim.cur_y[0, 0] = 10
    sim.goal_x[0, 0] = 10
    sim.goal_y[0, 0] = 10
    _set_actions(sim, [[0]])
    sim.step_sim_only()
    torch.cuda.synchronize()
    assert (int(sim.goal_x[0, 0]), int(sim.goal_y[0, 0])) == (46, 39)
    assert sim.goal_changed_flags.cpu().tolist() == [[1]]
    assert sim.state_packed.cpu().tolist() == [[0, 0, 10, 10, 46, 39, 0, 1]]
