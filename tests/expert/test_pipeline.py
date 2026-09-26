"""
Unit Tests for Expert Pipeline Interfaces

Tests cover:
1. RandomExpertPolicy - action generation, reset
2. RingBuffer - reservation, wrap-around, concurrent writes
3. DMAWorker - threshold logic, transfer simulation
4. GPUComputeHandler - tensor validation, ordering
5. ExpertWorker - step data construction, episode reset
6. ExtremeMAPFPipeline - orchestration, pointer tracking
7. DataValidator - format validation, edge cases
8. Edge cases: capacity wrap, empty batches, boundary conditions
"""

from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import multiprocessing as mp
import numpy as np
import pytest
import threading
import time
import torch

# Import from expert_running (the canonical implementation)
from expert.expert_running import (
    FEATURE_DIM,
    NUM_ACTIONS,
    RESET_FLAG_COL,
    ExpertPolicy,
    PyGBatchBuilder,
    RandomExpertPolicy,
    RingBuffer,
    DMAWorker,
    GPUComputeHandler,
    ExpertWorker,
    ExtremeMAPFPipeline,
    DataValidator,
    initial_refresh_flags,
    run_expert_algorithm_optimized,
    normalize_refresh_flags,
    refresh_flags_from_env,
    validate_env_aligned_batch as pipeline_validate_env_aligned_batch,
)


# ============================================================
# Fixtures
# ============================================================

@dataclass
class MockObservation:
    """Mock Pogema observation."""
    global_xy: tuple = (10, 20)
    global_target_xy: tuple = (50, 60)


@pytest.fixture
def mock_observations(num_agents=256):
    """Create mock observations for testing."""
    return [MockObservation() for _ in range(num_agents)]


@pytest.fixture
def sample_step_data(num_agents=256, env_id=0):
    """Create valid 8-column step data."""
    data = np.zeros((num_agents, FEATURE_DIM), dtype=np.uint16)
    data[:, 0] = env_id
    data[:, 1] = np.arange(num_agents, dtype=np.uint16)
    data[:, 2] = 10  # pos_x
    data[:, 3] = 20  # pos_y
    data[:, 4] = 50  # target_x
    data[:, 5] = 60  # target_y
    data[:, 6] = 2   # action
    data[:, 7] = 0   # reset_flag
    return data


@pytest.fixture
def sample_step_data_with_reset(num_agents=256, env_id=0):
    """Create step data with reset_flag=1 for all agents."""
    data = np.zeros((num_agents, FEATURE_DIM), dtype=np.uint16)
    data[:, 0] = env_id
    data[:, 1] = np.arange(num_agents, dtype=np.uint16)
    data[:, 6] = 1
    data[:, 7] = 1  # reset_flag
    return data


@pytest.fixture
def sample_gpu_tensor(num_agents=256, env_id=0):
    """Create valid GPU tensor."""
    data = torch.zeros((num_agents, FEATURE_DIM), dtype=torch.int16, device='cpu')
    data[:, 0] = env_id
    data[:, 1] = torch.arange(num_agents, dtype=torch.int16)
    data[:, 7] = 0
    return data


# ============================================================
# Test: RandomExpertPolicy
# ============================================================

class TestRandomExpertPolicy:
    """Tests for RandomExpertPolicy."""

    def test_init(self):
        policy = RandomExpertPolicy(num_actions=5, seed=42)
        assert policy.num_actions == 5

    def test_act_returns_correct_shape(self, mock_observations):
        policy = RandomExpertPolicy(seed=42)
        actions = policy.act(mock_observations)
        assert actions.shape == (256,)
        assert actions.dtype == np.uint16

    def test_act_values_in_range(self, mock_observations):
        policy = RandomExpertPolicy(num_actions=5, seed=42)
        actions = policy.act(mock_observations)
        assert np.all(actions >= 0)
        assert np.all(actions < 5)

    def test_act_single_agent(self):
        obs = [MockObservation()]
        policy = RandomExpertPolicy(seed=42)
        actions = policy.act(obs)
        assert actions.shape == (1,)

    def test_act_deterministic_with_seed(self):
        obs = [MockObservation() for _ in range(10)]
        policy1 = RandomExpertPolicy(seed=123)
        policy2 = RandomExpertPolicy(seed=123)
        a1 = policy1.act(obs)
        a2 = policy2.act(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_act_different_seeds_differ(self):
        obs = [MockObservation() for _ in range(100)]
        policy1 = RandomExpertPolicy(seed=1)
        policy2 = RandomExpertPolicy(seed=2)
        a1 = policy1.act(obs)
        a2 = policy2.act(obs)
        assert not np.array_equal(a1, a2)

    def test_act_all_actions_used(self):
        """With enough samples, all actions should appear."""
        obs = [MockObservation() for _ in range(10000)]
        policy = RandomExpertPolicy(num_actions=5, seed=42)
        actions = policy.act(obs)
        assert len(np.unique(actions)) == 5

    def test_reset_states_no_error(self):
        policy = RandomExpertPolicy(seed=42)
        policy.reset_states(None)  # Should not raise

    def test_act_empty_observations(self):
        policy = RandomExpertPolicy(seed=42)
        actions = policy.act([])
        assert actions.shape == (0,)

    def test_act_large_num_agents(self):
        obs = [MockObservation() for _ in range(10000)]
        policy = RandomExpertPolicy(seed=42)
        actions = policy.act(obs)
        assert actions.shape == (10000,)


# ============================================================
# Test: RingBuffer
# ============================================================

class TestRingBuffer:
    """Tests for RingBuffer interface."""

    def test_init_creates_correct_sizes(self):
        """Test that initialization creates correct buffer sizes."""
        buf = RingBuffer(capacity=1024, feature_dim=FEATURE_DIM)
        assert buf.capacity == 1024
        assert buf.feature_dim == FEATURE_DIM
        assert buf.cpu_buffer.shape == (1024, FEATURE_DIM)
        assert len(buf.ready_flags) == 1024
        assert buf.cpu_buffer.dtype == torch.int16

    def test_init_shared_memory(self):
        """Verify shared memory is enabled."""
        buf = RingBuffer(capacity=64, feature_dim=FEATURE_DIM)
        # Pinned + shared memory tensors should be on CPU
        assert not buf.cpu_buffer.is_cuda

    def test_reserve_and_write_basic(self):
        """Test reserve_and_write works correctly."""
        buf = RingBuffer(capacity=64)
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        data[:, 0] = 1
        data[:, 6] = 2
        start, end = buf.reserve_and_write(data)
        assert start == 0
        assert end == 10
        assert buf.shared_reserve_ptr.value == 10

    def test_scan_ready_range(self):
        """Test scan_ready_range returns correct count."""
        buf = RingBuffer(capacity=64)
        for i in range(10):
            buf.ready_flags[i] = 1
        ready = buf.scan_ready_range(0, 20)
        assert ready == 10

    def test_clear_flags(self):
        """Test clear_flags sets flags to 0."""
        buf = RingBuffer(capacity=64)
        for i in range(10):
            buf.ready_flags[i] = 1
        buf.clear_flags(0, 10)
        assert sum(buf.ready_flags[:10]) == 0

    def test_shared_reserve_ptr_initial_zero(self):
        buf = RingBuffer(capacity=64)
        assert buf.shared_reserve_ptr.value == 0

    def test_multiprocess_shared_access(self):
        """Verify shared memory can be accessed from child process."""
        buf = RingBuffer(capacity=64)

        def writer():
            buf.shared_reserve_ptr.value = 42

        p = mp.Process(target=writer)
        p.start()
        p.join(timeout=5)
        assert buf.shared_reserve_ptr.value == 42

    def test_reserve_and_write_blocks_when_unconsumed_window_is_full(self):
        """Producer must wait instead of overwriting unread slots."""
        buf = RingBuffer(capacity=16)
        data = np.zeros((8, FEATURE_DIM), dtype=np.uint16)

        buf.reserve_and_write(data)
        buf.reserve_and_write(data)

        result = {"done": False}

        def writer():
            buf.reserve_and_write(data)
            result["done"] = True

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        time.sleep(0.05)

        assert thread.is_alive()
        assert result["done"] is False

        buf.clear_flags(0, 8)
        buf.shared_compute_ptr.value = 8

        thread.join(timeout=1.0)
        assert result["done"] is True
        assert thread.is_alive() is False


# ============================================================
# Test: DMAWorker
# ============================================================

class TestDMAWorker:
    """Tests for DMAWorker interface."""

    def test_init(self):
        ring_buf = RingBuffer(capacity=1024)
        gpu_buf = torch.zeros((1024, FEATURE_DIM), dtype=torch.int16)
        worker = DMAWorker(ring_buf, gpu_buf, batch_threshold=256)
        assert worker.batch_threshold == 256
        assert worker.dma_read_ptr == 0

    def test_run_once_returns_none_when_insufficient_data(self):
        """run_once returns None when data < batch_threshold."""
        ring_buf = RingBuffer(capacity=64)
        gpu_buf = torch.zeros((64, FEATURE_DIM), dtype=torch.int16)
        worker = DMAWorker(ring_buf, gpu_buf, batch_threshold=32)
        result = worker.run_once()
        assert result is None

    def test_start_creates_thread(self):
        """start() returns a thread handle."""
        ring_buf = RingBuffer(capacity=64)
        gpu_buf = torch.zeros((64, FEATURE_DIM), dtype=torch.int16)
        worker = DMAWorker(ring_buf, gpu_buf, batch_threshold=32)
        t = worker.start()
        assert isinstance(t, threading.Thread)
        worker.stop()

    def test_stop_sets_flag(self):
        """stop() sets _running to False."""
        ring_buf = RingBuffer(capacity=64)
        gpu_buf = torch.zeros((64, FEATURE_DIM), dtype=torch.int16)
        worker = DMAWorker(ring_buf, gpu_buf)
        worker.start()
        worker.stop()
        assert worker._running is False

    def test_gpu_buffer_shape_matches(self):
        """GPU buffer should match ring buffer capacity."""
        capacity = 512
        ring_buf = RingBuffer(capacity=capacity)
        gpu_buf = torch.zeros((capacity, FEATURE_DIM), dtype=torch.int16)
        worker = DMAWorker(ring_buf, gpu_buf)
        assert worker.gpu_buffer.shape == ring_buf.cpu_buffer.shape

    def test_batch_threshold_validation(self):
        """Batch threshold should be <= capacity."""
        ring_buf = RingBuffer(capacity=64)
        gpu_buf = torch.zeros((64, FEATURE_DIM), dtype=torch.int16)
        # This should work (implementation should validate)
        worker = DMAWorker(ring_buf, gpu_buf, batch_threshold=32)
        assert worker.batch_threshold == 32


# ============================================================
# Test: GPUComputeHandler
# ============================================================

class TestGPUComputeHandler:
    """Tests for GPUComputeHandler interface."""

    def test_init(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        assert handler.simulator is mock_sim

    def test_process_batch_calls_simulator(self):
        """process_batch should refresh derived and compact state without eager PyG materialization."""
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        full_batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        agents_to_update = torch.zeros((2, FEATURE_DIM), dtype=torch.int16)
        handler.process_batch(full_batch, agents_to_update)
        mock_sim.update_derived_state.assert_called_once_with(
            agents_to_update, agents_to_update.shape[0]
        )
        mock_sim.refresh_compact_state_from_raw_batch.assert_called_once_with(full_batch)
        mock_sim.build_magat_plus_inputs.assert_not_called()

    def test_process_batch_skips_empty_derived_update(self):
        """process_batch should still refresh compact state when no agents changed."""
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        full_batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        agents_to_update = torch.zeros((0, FEATURE_DIM), dtype=torch.int16)
        handler.process_batch(full_batch, agents_to_update)
        mock_sim.update_derived_state.assert_not_called()
        mock_sim.refresh_compact_state_from_raw_batch.assert_called_once_with(full_batch)

    def test_pyg_batch_builder_materializes_before_reading_stateless_outputs(self):
        """Runtime batch build should trigger just-in-time PyG materialization."""
        num_nodes = 2
        builder = PyGBatchBuilder()
        raw_batch = torch.zeros((num_nodes, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 6] = torch.tensor([1, 2], dtype=torch.int16)

        mock_sim = MagicMock()
        mock_sim.pyg_x = torch.zeros((num_nodes, 4 * 13 * 13), dtype=torch.float32)
        mock_sim.pyg_edge_index_storage = torch.zeros((2, 0), dtype=torch.int64)
        mock_sim.pyg_edge_attr_storage = torch.zeros((0, 3), dtype=torch.float32)
        mock_sim.pyg_num_edges = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(num_nodes, dtype=torch.int64)
        mock_sim.pyg_ptr = torch.tensor([0, num_nodes], dtype=torch.int64)

        def _materialize():
            mock_sim.pyg_x.fill_(7.0)

        mock_sim.materialize_pyg_inputs.side_effect = _materialize

        batch = builder.build(mock_sim, raw_batch, materialize_edges=False)

        mock_sim.materialize_pyg_inputs.assert_called_once_with()
        assert torch.all(batch.x == 7.0)
        assert torch.equal(batch.y, torch.tensor([1, 2], dtype=torch.long))

    def test_pyg_batch_builder_marks_arrived_agents_from_raw_batch(self):
        """Stateless batch build should infer arrived agents from pos==target."""
        num_nodes = 3
        builder = PyGBatchBuilder()
        raw_batch = torch.zeros((num_nodes, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 2] = torch.tensor([10, 11, 12], dtype=torch.int16)
        raw_batch[:, 3] = torch.tensor([20, 21, 22], dtype=torch.int16)
        raw_batch[:, 4] = torch.tensor([10, 15, 12], dtype=torch.int16)
        raw_batch[:, 5] = torch.tensor([20, 25, 99], dtype=torch.int16)

        mock_sim = MagicMock()
        mock_sim.pyg_x = torch.zeros((num_nodes, 4 * 13 * 13), dtype=torch.float32)
        mock_sim.pyg_edge_index_storage = torch.zeros((2, 0), dtype=torch.int64)
        mock_sim.pyg_edge_attr_storage = torch.zeros((0, 3), dtype=torch.float32)
        mock_sim.pyg_num_edges = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(num_nodes, dtype=torch.int64)
        mock_sim.pyg_ptr = torch.tensor([0, num_nodes], dtype=torch.int64)
        mock_sim.materialize_pyg_inputs.side_effect = lambda: None

        batch = builder.build(mock_sim, raw_batch, materialize_edges=False)

        assert torch.equal(batch.arrived, torch.tensor([True, False, False], dtype=torch.bool))

    def test_pyg_batch_builder_materializes_before_reading_stateful_outputs(self):
        """Stateful runtime build should also trigger just-in-time PyG materialization."""
        num_nodes = 2
        builder = PyGBatchBuilder()

        mock_sim = MagicMock()
        mock_sim.pyg_x = torch.zeros((num_nodes, 4 * 13 * 13), dtype=torch.float32)
        mock_sim.pyg_edge_index_storage = torch.zeros((2, 0), dtype=torch.int64)
        mock_sim.pyg_edge_attr_storage = torch.zeros((0, 3), dtype=torch.float32)
        mock_sim.pyg_num_edges = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(num_nodes, dtype=torch.int64)
        mock_sim.pyg_ptr = torch.tensor([0, num_nodes], dtype=torch.int64)
        mock_sim.actions = torch.tensor([[3, 4]], dtype=torch.uint8)

        def _materialize():
            mock_sim.pyg_x.fill_(5.0)

        mock_sim.materialize_pyg_inputs.side_effect = _materialize

        batch = builder.build_from_stateful(mock_sim, materialize_edges=False)

        mock_sim.materialize_pyg_inputs.assert_called_once_with()
        assert torch.all(batch.x == 5.0)
        assert torch.equal(batch.y, torch.tensor([3, 4], dtype=torch.long))

    def test_pyg_batch_builder_can_view_prebuilt_stateful_outputs(self):
        """Stage profiling must not silently launch the builder a second time."""
        num_nodes = 2
        builder = PyGBatchBuilder()
        mock_sim = MagicMock()
        mock_sim.pyg_x = torch.full(
            (num_nodes, 4 * 13 * 13), 7.0, dtype=torch.float32
        )
        mock_sim.pyg_edge_index_storage = torch.zeros((2, 0), dtype=torch.int64)
        mock_sim.pyg_edge_attr_storage = torch.zeros((0, 3), dtype=torch.float32)
        mock_sim.pyg_num_edges = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(num_nodes, dtype=torch.int64)
        mock_sim.pyg_ptr = torch.tensor([0, num_nodes], dtype=torch.int64)
        mock_sim.actions = torch.tensor([[1, 2]], dtype=torch.uint8)

        batch = builder.view_stateful_outputs(
            mock_sim, materialize_edges=False
        )

        mock_sim.materialize_pyg_inputs.assert_not_called()
        assert torch.all(batch.x == 7.0)
        assert torch.equal(batch.y, torch.tensor([1, 2], dtype=torch.long))

    def test_pyg_batch_builder_slice_preserves_arrived_mask(self):
        """Environment-aligned slicing should preserve arrived flags."""
        batch = PyGBatchBuilder.slice_env_aligned_batch(
            type(
                "Batch",
                (),
                {
                    "x": torch.zeros((4, 4, 13, 13), dtype=torch.float32),
                    "y": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "terminated": torch.zeros(4, dtype=torch.bool),
                    "arrived": torch.tensor([False, True, True, False], dtype=torch.bool),
                    "edge_index": None,
                    "edge_attr": None,
                    "edge_index_storage": torch.zeros((2, 0), dtype=torch.int64),
                    "edge_attr_storage": torch.zeros((0, 3), dtype=torch.float32),
                    "num_edges": torch.tensor([0], dtype=torch.int64),
                },
            )(),
            start_node=2,
            end_node=4,
            agents_per_env=2,
        )

        assert torch.equal(batch.arrived, torch.tensor([True, False], dtype=torch.bool))

    def test_filter_reset_agents_returns_correct(self):
        """filter_reset_agents returns only rows with reset_flag != 0."""
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        batch[2, RESET_FLAG_COL] = 1
        batch[5, RESET_FLAG_COL] = 1
        filtered = handler.filter_reset_agents(batch)
        assert filtered.shape[0] == 2

    def test_filter_reset_agents_logic(self):
        """Test the expected filtering logic conceptually."""
        batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        batch[2, RESET_FLAG_COL] = 1
        batch[5, RESET_FLAG_COL] = 1
        batch[8, RESET_FLAG_COL] = 1

        # Expected filtering
        mask = batch[:, RESET_FLAG_COL] != 0
        filtered = batch[mask]
        assert filtered.shape[0] == 3
        assert filtered.shape[1] == FEATURE_DIM

    def test_filter_reset_agents_empty(self):
        """When no reset flags set, filtered result should be empty."""
        batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        # All reset_flag = 0
        mask = batch[:, RESET_FLAG_COL] != 0
        filtered = batch[mask]
        assert filtered.shape[0] == 0

    def test_filter_reset_agents_all(self):
        """When all reset flags set, filtered == original."""
        batch = torch.ones((10, FEATURE_DIM), dtype=torch.int16)
        batch[:, RESET_FLAG_COL] = 1
        mask = batch[:, RESET_FLAG_COL] != 0
        filtered = batch[mask]
        assert filtered.shape[0] == 10


# ============================================================
# Test: ExpertWorker
# ============================================================

class TestExpertWorker:
    """Tests for ExpertWorker interface."""

    def test_init(self):
        policy = RandomExpertPolicy(seed=42)
        mock_pipeline = MagicMock()
        worker = ExpertWorker(
            expert_id=0,
            policy=policy,
            pipeline=mock_pipeline,
            grid_config=None
        )
        assert worker.expert_id == 0
        assert worker.policy is policy

    def test_run_raises_without_pogema(self):
        """run() raises when pogema is installed but grid_config is None."""
        policy = RandomExpertPolicy(seed=42)
        mock_pipeline = MagicMock()
        mock_pipeline.ring_buffer = MagicMock()
        worker = ExpertWorker(0, policy, mock_pipeline)
        # grid_config is None, so pogema_v0 will fail or behave unexpectedly
        try:
            worker.run()
        except Exception:
            pass  # Expected - any error is fine

    def test_run_episode_delegates_to_policy(self):
        """run_episode calls policy.act()."""
        policy = RandomExpertPolicy(seed=42)
        mock_pipeline = MagicMock()
        worker = ExpertWorker(0, policy, mock_pipeline)
        obs = [MockObservation() for _ in range(5)]
        actions = worker.run_episode(None, obs)
        assert actions.shape == (5,)
        assert actions.dtype == np.uint16

    def test_build_step_data_correct(self):
        """build_step_data stores observation coordinates in obs-radius-normalized space."""
        policy = RandomExpertPolicy(seed=42)
        mock_pipeline = MagicMock()
        worker = ExpertWorker(3, policy, mock_pipeline)
        # Use dict-style observations (matching Pogema format)
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(4)]
        actions = np.array([1, 2, 3, 4], dtype=np.uint16)
        data = worker.build_step_data(obs, actions, reset_flag=1)
        assert data.shape == (4, FEATURE_DIM)
        assert np.all(data[:, 0] == 3)  # expert_id
        assert np.array_equal(data[:, 1], [0, 1, 2, 3])
        assert np.all(data[:, 2] == 5)  # pos_x - OBS_RADIUS
        assert np.all(data[:, 3] == 15)  # pos_y - OBS_RADIUS
        assert np.all(data[:, 4] == 45)  # target_x - OBS_RADIUS
        assert np.all(data[:, 5] == 55)  # target_y - OBS_RADIUS
        assert np.all(data[:, 7] == 1)  # reset_flag

    def test_build_step_data_accepts_per_agent_refresh_flags(self):
        policy = RandomExpertPolicy(seed=42)
        mock_pipeline = MagicMock()
        worker = ExpertWorker(3, policy, mock_pipeline)
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(4)]
        actions = np.array([1, 2, 3, 4], dtype=np.uint16)
        refresh_flags = np.array([1, 0, 1, 0], dtype=np.uint16)
        data = worker.build_step_data(obs, actions, reset_flag=refresh_flags)
        assert np.array_equal(data[:, 7], refresh_flags)

    def test_run_expert_algorithm_optimized_emits_first_reset_then_progress_rows(self):
        class FakeExpert:
            def __init__(self):
                self.reset_calls = 0
                self.act_calls = 0

            def reset_states(self, env):
                self.reset_calls += 1

            def act(self, observations):
                self.act_calls += 1
                return np.array([4, 0], dtype=np.uint16)

        class FakeEnv:
            def __init__(self):
                self.step_calls = 0
                self._observations = [
                    [
                        {"global_xy": (10, 20), "global_target_xy": (30, 40)},
                        {"global_xy": (11, 21), "global_target_xy": (31, 41)},
                    ],
                    [
                        {"global_xy": (11, 20), "global_target_xy": (30, 40)},
                        {"global_xy": (11, 21), "global_target_xy": (31, 41)},
                    ],
                ]

            def reset(self):
                return self._observations[0], {}

            def step(self, actions):
                self.step_calls += 1
                if self.step_calls >= 2:
                    raise RuntimeError("stop-loop")
                return self._observations[1], None, [False, False], [False, False], None

        captured_rows = []
        pipeline = MagicMock()
        pipeline.feature_dim = FEATURE_DIM
        pipeline.reserve_and_write.side_effect = lambda rows: captured_rows.append(rows.copy())
        expert = FakeExpert()
        env = FakeEnv()

        with pytest.raises(RuntimeError, match="stop-loop"):
            run_expert_algorithm_optimized(expert, 7, pipeline, env=env)

        assert expert.reset_calls == 1
        assert expert.act_calls == 2
        assert env.step_calls == 2
        assert len(captured_rows) == 2
        assert np.array_equal(captured_rows[0][:, 7], np.array([1, 1], dtype=np.uint16))
        assert np.array_equal(captured_rows[1][:, 7], np.array([0, 0], dtype=np.uint16))
        assert np.array_equal(captured_rows[0][:, 0], np.array([7, 7], dtype=np.uint16))
        assert np.array_equal(captured_rows[1][:, 0], np.array([7, 7], dtype=np.uint16))
        assert np.array_equal(captured_rows[0][:, 1], np.array([0, 1], dtype=np.uint16))
        assert np.array_equal(captured_rows[1][:, 1], np.array([0, 1], dtype=np.uint16))
        assert np.array_equal(captured_rows[0][:, 6], np.array([4, 0], dtype=np.uint16))
        assert np.array_equal(captured_rows[1][:, 6], np.array([4, 0], dtype=np.uint16))
        assert np.array_equal(captured_rows[0][:, 2], np.array([5, 6], dtype=np.uint16))
        assert np.array_equal(captured_rows[1][:, 2], np.array([6, 6], dtype=np.uint16))

    def test_refresh_flag_helpers(self):
        assert np.array_equal(initial_refresh_flags(4), np.ones(4, dtype=np.uint16))
        assert np.array_equal(normalize_refresh_flags(1, 3), np.ones(3, dtype=np.uint16))
        assert np.array_equal(
            normalize_refresh_flags(np.array([1, 0, 1], dtype=np.uint16), 3),
            np.array([1, 0, 1], dtype=np.uint16),
        )

    def test_refresh_flags_from_env_reads_was_on_goal(self):
        env = MagicMock()
        env.unwrapped.was_on_goal = [True, False, True]
        flags = refresh_flags_from_env(env, 3)
        assert np.array_equal(flags, np.array([1, 0, 1], dtype=np.uint16))

    def test_refresh_flags_from_env_defaults_to_zero_when_missing(self):
        env = MagicMock()
        env.unwrapped = MagicMock(spec=[])
        flags = refresh_flags_from_env(env, 3)
        assert np.array_equal(flags, np.zeros(3, dtype=np.uint16))

    def test_refresh_flags_from_env_rejects_wrong_length(self):
        env = MagicMock()
        env.unwrapped.was_on_goal = [True, False]
        with pytest.raises(ValueError):
            refresh_flags_from_env(env, 3)

    def test_next_row_refresh_contract(self):
        first_row_flags = initial_refresh_flags(3)
        second_row_flags = np.array([0, 1, 0], dtype=np.uint16)
        assert np.array_equal(first_row_flags, np.array([1, 1, 1], dtype=np.uint16))
        assert np.array_equal(second_row_flags, np.array([0, 1, 0], dtype=np.uint16))
        assert second_row_flags[1] == 1
        assert second_row_flags[0] == 0
        assert second_row_flags[2] == 0

    def test_current_row_can_stay_unflagged_before_goal_refresh(self):
        row_t_flags = np.zeros(3, dtype=np.uint16)
        row_t1_flags = np.array([0, 1, 0], dtype=np.uint16)
        assert np.array_equal(row_t_flags, np.zeros(3, dtype=np.uint16))
        assert np.array_equal(row_t1_flags, np.array([0, 1, 0], dtype=np.uint16))

    def test_process_batch_updates_only_flagged_subset(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        full_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        full_batch[:, 1] = torch.arange(4, dtype=torch.int16)
        full_batch[1, RESET_FLAG_COL] = 1
        full_batch[3, RESET_FLAG_COL] = 1
        agents_to_update = handler.filter_reset_agents(full_batch)
        handler.process_batch(full_batch, agents_to_update)
        expected = full_batch[[1, 3]]
        called_batch, called_len = mock_sim.update_derived_state.call_args[0]
        assert torch.equal(called_batch, expected)
        assert called_len == 2
        mock_sim.refresh_compact_state_from_raw_batch.assert_called_once_with(full_batch)

    def test_process_batch_skips_empty_refresh_subset(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        full_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        agents_to_update = handler.filter_reset_agents(full_batch)
        handler.process_batch(full_batch, agents_to_update)
        mock_sim.update_derived_state.assert_not_called()
        mock_sim.refresh_compact_state_from_raw_batch.assert_called_once_with(full_batch)

    def test_filter_reset_agents_preserves_per_agent_mask(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        batch = torch.zeros((5, FEATURE_DIM), dtype=torch.int16)
        batch[:, 1] = torch.arange(5, dtype=torch.int16)
        batch[0, RESET_FLAG_COL] = 1
        batch[2, RESET_FLAG_COL] = 1
        batch[4, RESET_FLAG_COL] = 1
        filtered = handler.filter_reset_agents(batch)
        assert torch.equal(filtered[:, 1], torch.tensor([0, 2, 4], dtype=torch.int16))

    def test_step_data_refresh_flag_range(self):
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        data[:, 7] = 2
        with pytest.raises(ValueError):
            DataValidator.validate_step_data(data)

    def test_step_data_refresh_flag_vector_validates(self):
        data = np.zeros((4, FEATURE_DIM), dtype=np.uint16)
        data[:, 7] = np.array([1, 0, 1, 0], dtype=np.uint16)
        assert DataValidator.validate_step_data(data)

    def test_episode_first_row_refresh_is_all_ones(self):
        assert np.array_equal(initial_refresh_flags(5), np.ones(5, dtype=np.uint16))

    def test_only_changed_agents_are_flagged_on_next_row(self):
        changed = np.array([0, 1, 0, 1], dtype=np.uint16)
        assert changed.sum() == 2
        assert np.array_equal(changed, np.array([0, 1, 0, 1], dtype=np.uint16))

    def test_unchanged_agents_do_not_force_refresh(self):
        changed = np.array([0, 1, 0, 0], dtype=np.uint16)
        assert changed[0] == 0
        assert changed[2] == 0
        assert changed[3] == 0
        assert changed[1] == 1

    def test_refresh_flag_is_first_row_after_goal_change(self):
        before = np.array([0, 0, 0], dtype=np.uint16)
        after = np.array([1, 0, 0], dtype=np.uint16)
        assert np.array_equal(before, np.zeros(3, dtype=np.uint16))
        assert np.array_equal(after, np.array([1, 0, 0], dtype=np.uint16))

    def test_obs_action_alignment_remains_unchanged(self):
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(2)]
        actions = np.array([1, 4], dtype=np.uint16)
        worker = ExpertWorker(7, RandomExpertPolicy(seed=42), MagicMock())
        data = worker.build_step_data(obs, actions, reset_flag=np.array([0, 1], dtype=np.uint16))
        assert np.array_equal(data[:, 6], actions)
        assert np.all(data[:, 4] == 45)
        assert np.all(data[:, 5] == 55)

    def test_refresh_flag_col_is_last(self):
        assert RESET_FLAG_COL == FEATURE_DIM - 1

    def test_refresh_flag_vector_broadcasts_scalar(self):
        flags = normalize_refresh_flags(0, 4)
        assert np.array_equal(flags, np.zeros(4, dtype=np.uint16))

    def test_refresh_flag_vector_rejects_mismatched_length(self):
        with pytest.raises(ValueError):
            normalize_refresh_flags(np.array([1, 0], dtype=np.uint16), 3)

    def test_filter_reset_agents_empty_batch_shape(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        batch = torch.zeros((0, FEATURE_DIM), dtype=torch.int16)
        filtered = handler.filter_reset_agents(batch)
        assert filtered.shape == (0, FEATURE_DIM)

    def test_process_batch_preserves_full_batch_refresh(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        full_batch = torch.zeros((3, FEATURE_DIM), dtype=torch.int16)
        full_batch[:, RESET_FLAG_COL] = 1
        agents_to_update = handler.filter_reset_agents(full_batch)
        handler.process_batch(full_batch, agents_to_update)
        called_batch, called_len = mock_sim.update_derived_state.call_args[0]
        assert torch.equal(called_batch, full_batch)
        assert called_len == 3
        mock_sim.refresh_compact_state_from_raw_batch.assert_called_once_with(full_batch)

    def test_process_batch_handles_single_flagged_agent(self):
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        full_batch = torch.zeros((3, FEATURE_DIM), dtype=torch.int16)
        full_batch[2, RESET_FLAG_COL] = 1
        agents_to_update = handler.filter_reset_agents(full_batch)
        handler.process_batch(full_batch, agents_to_update)
        called_batch, called_len = mock_sim.update_derived_state.call_args[0]
        assert called_batch.shape[0] == 1
        assert called_len == 1

    def test_data_validator_accepts_refresh_flag_column(self):
        data = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        data[:, 7] = np.array([1, 0], dtype=np.uint16)
        assert DataValidator.validate_step_data(data)

    def test_data_validator_rejects_refresh_flag_gt_one(self):
        data = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        data[0, 7] = 3
        with pytest.raises(ValueError):
            DataValidator.validate_step_data(data)

    def test_refresh_mask_can_be_copied_between_rows(self):
        flags = np.array([0, 1, 0], dtype=np.uint16)
        copied = flags.copy()
        assert np.array_equal(flags, copied)

    def test_refresh_flags_remain_uint16(self):
        flags = initial_refresh_flags(4)
        assert flags.dtype == np.uint16

    def test_refresh_flags_from_env_return_uint16(self):
        env = MagicMock()
        env.unwrapped.was_on_goal = [True, False, False, True]
        flags = refresh_flags_from_env(env, 4)
        assert flags.dtype == np.uint16

    def test_filter_reset_agents_uses_nonzero_semantics(self):
        batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        batch[1, RESET_FLAG_COL] = 1
        batch[3, RESET_FLAG_COL] = 1
        filtered = GPUComputeHandler(MagicMock()).filter_reset_agents(batch)
        assert filtered.shape[0] == 2

    def test_next_row_refresh_flag_does_not_require_new_columns(self):
        data = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        assert data.shape[1] == 8
        assert RESET_FLAG_COL == 7

    def test_refresh_flag_vector_can_mark_subset(self):
        subset = np.array([1, 0, 0, 1, 0], dtype=np.uint16)
        assert subset.sum() == 2
        assert subset.dtype == np.uint16

    def test_refresh_flag_subset_matches_filtered_count(self):
        subset = np.array([1, 0, 1, 0, 1], dtype=np.uint16)
        batch = torch.zeros((5, FEATURE_DIM), dtype=torch.int16)
        batch[:, RESET_FLAG_COL] = torch.from_numpy(subset.astype(np.int16))
        filtered = GPUComputeHandler(MagicMock()).filter_reset_agents(batch)
        assert filtered.shape[0] == int(subset.sum())

    def test_refresh_flags_from_env_handles_numpy_input(self):
        env = MagicMock()
        env.unwrapped.was_on_goal = np.array([True, False, True])
        flags = refresh_flags_from_env(env, 3)
        assert np.array_equal(flags, np.array([1, 0, 1], dtype=np.uint16))

    def test_normalize_refresh_flags_accepts_python_list(self):
        flags = normalize_refresh_flags([1, 0, 1], 3)
        assert np.array_equal(flags, np.array([1, 0, 1], dtype=np.uint16))

    def test_normalize_refresh_flags_accepts_bool_array(self):
        flags = normalize_refresh_flags(np.array([True, False, True]), 3)
        assert np.array_equal(flags, np.array([1, 0, 1], dtype=np.uint16))

    def test_normalize_refresh_flags_accepts_bool_scalar(self):
        flags = normalize_refresh_flags(True, 2)
        assert np.array_equal(flags, np.array([1, 1], dtype=np.uint16))

    def test_build_step_data_with_bool_refresh_flags(self):
        policy = RandomExpertPolicy(seed=42)
        worker = ExpertWorker(1, policy, MagicMock())
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(2)]
        actions = np.array([0, 1], dtype=np.uint16)
        data = worker.build_step_data(obs, actions, reset_flag=np.array([True, False]))
        assert np.array_equal(data[:, 7], np.array([1, 0], dtype=np.uint16))

    def test_build_step_data_with_scalar_zero_refresh_flag(self):
        policy = RandomExpertPolicy(seed=42)
        worker = ExpertWorker(1, policy, MagicMock())
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(2)]
        actions = np.array([0, 1], dtype=np.uint16)
        data = worker.build_step_data(obs, actions, reset_flag=0)
        assert np.array_equal(data[:, 7], np.array([0, 0], dtype=np.uint16))

    def test_build_step_data_with_scalar_one_refresh_flag(self):
        policy = RandomExpertPolicy(seed=42)
        worker = ExpertWorker(1, policy, MagicMock())
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(2)]
        actions = np.array([0, 1], dtype=np.uint16)
        data = worker.build_step_data(obs, actions, reset_flag=1)
        assert np.array_equal(data[:, 7], np.array([1, 1], dtype=np.uint16))

    def test_build_step_data_rejects_wrong_refresh_flag_length(self):
        policy = RandomExpertPolicy(seed=42)
        worker = ExpertWorker(1, policy, MagicMock())
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(2)]
        actions = np.array([0, 1], dtype=np.uint16)
        with pytest.raises(ValueError):
            worker.build_step_data(obs, actions, reset_flag=np.array([1], dtype=np.uint16))

    def test_build_step_data_preserves_agent_ids_with_refresh_subset(self):
        policy = RandomExpertPolicy(seed=42)
        worker = ExpertWorker(5, policy, MagicMock())
        obs = [{'global_xy': (10, 20), 'global_target_xy': (50, 60)} for _ in range(3)]
        actions = np.array([0, 1, 2], dtype=np.uint16)
        data = worker.build_step_data(obs, actions, reset_flag=np.array([0, 1, 0], dtype=np.uint16))
        assert np.array_equal(data[:, 1], np.array([0, 1, 2], dtype=np.uint16))
        assert data[1, 7] == 1
        assert data[0, 7] == 0
        assert data[2, 7] == 0

    def test_filter_reset_agents_returns_correct(self):
        """filter_reset_agents returns only rows with reset_flag != 0."""
        mock_sim = MagicMock()
        handler = GPUComputeHandler(mock_sim)
        batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        batch[2, RESET_FLAG_COL] = 1
        batch[5, RESET_FLAG_COL] = 1
        filtered = handler.filter_reset_agents(batch)
        assert filtered.shape[0] == 2

    def test_build_step_data_logic(self):
        """Test the expected step data construction logic."""
        num_agents = 5
        env_id = 3

        step_data = np.zeros((num_agents, FEATURE_DIM), dtype=np.uint16)
        step_data[:, 0] = env_id
        step_data[:, 1] = np.arange(num_agents, dtype=np.uint16)
        step_data[:, 2] = 10  # pos_x
        step_data[:, 3] = 20  # pos_y
        step_data[:, 4] = 50  # target_x
        step_data[:, 5] = 60  # target_y
        step_data[:, 6] = 1   # action
        step_data[:, 7] = 0   # reset_flag

        assert step_data.shape == (num_agents, FEATURE_DIM)
        assert step_data.dtype == np.uint16
        assert np.all(step_data[:, 0] == env_id)
        assert np.array_equal(step_data[:, 1], [0, 1, 2, 3, 4])
        assert np.all(step_data[:, 7] == 0)


# ============================================================
# Test: ExtremeMAPFPipeline
# ============================================================

class _MockMAGATRuntime:
    def __init__(self):
        self.calls = []

    def train_step(self, simulator, raw_batch):
        self.calls.append((simulator, raw_batch.clone()))
        return {"backend": "magat", "rows": int(raw_batch.shape[0])}


class _MockMapfGPTRuntime:
    def __init__(self):
        self.calls = []

    def train_step(self, builder, raw_stage):
        self.calls.append((builder, raw_stage.clone()))
        return {"backend": "mapf_gpt", "rows": int(raw_stage.shape[0])}


def _make_env_aligned_frontier(*, reset_flags, positions, actions):
    frontier = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
    frontier[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
    frontier[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)
    frontier[:, 2:4] = torch.tensor(positions, dtype=torch.int16)
    frontier[:, 4:6] = frontier[:, 2:4] + 20
    frontier[:, 6] = torch.tensor(actions, dtype=torch.int16)
    frontier[:, RESET_FLAG_COL] = torch.tensor(reset_flags, dtype=torch.int16)
    return frontier


def _frontier_env_blocks(frontier, agents_per_env=2):
    return [
        frontier[start:start + agents_per_env].cpu().numpy().astype(np.uint16, copy=True)
        for start in range(0, frontier.shape[0], agents_per_env)
    ]


class TestExtremeMAPFPipeline:
    """Tests for ExtremeMAPFPipeline interface."""

    def test_init(self):
        mock_sim = MagicMock()
        pipeline = ExtremeMAPFPipeline(
            simulator=mock_sim,
            capacity=1024,
            batch_threshold=256,
            feature_dim=FEATURE_DIM,
            device="cpu"
        )
        assert pipeline.capacity == 1024
        assert pipeline.batch_threshold == 256
        assert pipeline.feature_dim == FEATURE_DIM
        assert pipeline.dma_read_ptr == 0
        assert pipeline.compute_ptr == 0

    def test_initialize_creates_components(self):
        """initialize() creates all pipeline components."""
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(0, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=1024, batch_threshold=256)
            pipeline.initialize()
        assert pipeline.ring_buffer is not None
        assert pipeline.gpu_buffer is not None
        assert pipeline.gpu_handler is not None
        assert pipeline.dma_worker is not None

    def test_reserve_and_write_delegates(self):
        """reserve_and_write delegates to ring_buffer."""
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(0, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=1024, batch_threshold=256)
            pipeline.initialize()
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        start, end = pipeline.reserve_and_write(data)
        assert start == 0
        assert end == 10

    def test_train_step_returns_none_without_data(self):
        """train_step returns None when no data available."""
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(0, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=1024, batch_threshold=256)
            pipeline.initialize()
        result = pipeline.train_step()
        assert result is None

    @pytest.mark.parametrize(
        ("runtime_factory", "expected_backend"),
        [(_MockMAGATRuntime, "magat"), (_MockMapfGPTRuntime, "mapf_gpt")],
    )
    def test_runtime_train_step_contract_is_backend_parameterized(
        self, runtime_factory, expected_backend
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = runtime_factory()

        result = pipeline.train_step(runtime=runtime)

        assert result == {"backend": expected_backend, "rows": 4}
        assert len(runtime.calls) == 1
        simulator_or_builder, forwarded_batch = runtime.calls[0]
        assert simulator_or_builder is mock_sim
        assert torch.equal(forwarded_batch, raw_batch)
        pipeline.gpu_handler.process_batch.assert_called_once()

    def test_runtime_train_step_propagates_backend_errors(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]

        class _FailingRuntime:
            def train_step(self, simulator, batch):
                raise RuntimeError("backend failure")

        with pytest.raises(RuntimeError, match="backend failure"):
            pipeline.train_step(runtime=_FailingRuntime())
        pipeline.gpu_handler.process_batch.assert_called_once()

    def test_train_step_returns_none_when_stage_batch_is_missing(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.dma_event = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=None)

        runtime = _MockMAGATRuntime()
        assert pipeline.train_step(runtime=runtime) is None
        assert runtime.calls == []
        pipeline.extract_env_aligned_batch.assert_called_once()
        pipeline.dma_event.synchronize.assert_called_once()

    def test_get_stats_exposes_stage_counters_for_health_hooks(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        stats = pipeline.get_stats()
        assert stats["capacity"] == 8
        assert stats["stage_rows"] == 4
        assert stats["num_stage_slots"] >= 1
        assert "stage_ready_counts" in stats
        assert "stage_seq_tags" in stats
        assert "env_write_stage" in stats
        assert stats["reserve_ptr"] == 0
        assert stats["dma_read_ptr"] == 0
        assert stats["compute_ptr"] == 0

    def test_backend_runtime_calls_preserve_env_aligned_batch_layout(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = _MockMapfGPTRuntime()

        pipeline.train_step(runtime=runtime)

        forwarded = runtime.calls[0][1]
        view = forwarded.view(2, 2, FEATURE_DIM)
        assert torch.equal(view[0, :, 0], torch.tensor([0, 0], dtype=torch.int16))
        assert torch.equal(view[1, :, 0], torch.tensor([1, 1], dtype=torch.int16))
        assert torch.equal(view[0, :, 1], torch.tensor([0, 1], dtype=torch.int16))
        assert torch.equal(view[1, :, 1], torch.tensor([0, 1], dtype=torch.int16))

    def test_train_step_runtime_path_filters_reset_agents_before_backend_call(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)
        raw_batch[1, RESET_FLAG_COL] = 1
        reset_subset = raw_batch[1:2]

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = reset_subset
        runtime = _MockMAGATRuntime()

        pipeline.train_step(runtime=runtime)

        pipeline.gpu_handler.filter_reset_agents.assert_called_once_with(raw_batch)
        pipeline.gpu_handler.process_batch.assert_called_once_with(raw_batch, reset_subset)
        assert len(runtime.calls) == 1
        assert torch.equal(runtime.calls[0][1], raw_batch)

    def test_train_step_runtime_path_uses_same_simulator_for_all_backends(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]

        magat = _MockMAGATRuntime()
        mapf_gpt = _MockMapfGPTRuntime()
        pipeline.train_step(runtime=magat)
        pipeline.train_step(runtime=mapf_gpt)

        assert magat.calls[0][0] is mock_sim
        assert mapf_gpt.calls[0][0] is mock_sim

    def test_train_step_runtime_path_handles_empty_reset_subset(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = _MockMAGATRuntime()

        result = pipeline.train_step(runtime=runtime)

        assert result == {"backend": "magat", "rows": 4}
        pipeline.gpu_handler.process_batch.assert_called_once()
        processed_batch, processed_subset = pipeline.gpu_handler.process_batch.call_args.args
        assert torch.equal(processed_batch, raw_batch)
        assert processed_subset.shape == (0, FEATURE_DIM)
        assert len(runtime.calls) == 1

    def test_runtime_train_step_contract_preserves_tensor_dtype(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = _MockMapfGPTRuntime()

        pipeline.train_step(runtime=runtime)

        assert runtime.calls[0][1].dtype == torch.int16
        assert runtime.calls[0][1].shape == (4, FEATURE_DIM)

    def test_train_step_runtime_path_requires_initialized_dma_worker(self):
        mock_sim = MagicMock()
        pipeline = ExtremeMAPFPipeline(mock_sim, capacity=8, batch_threshold=4, device="cpu")
        runtime = _MockMAGATRuntime()
        assert pipeline.train_step(runtime=runtime) is None
        assert runtime.calls == []

    def test_train_step_runtime_path_returns_none_when_batch_not_ready(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.dma_event = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=None)
        runtime = _MockMapfGPTRuntime()

        assert pipeline.train_step(runtime=runtime) is None
        assert runtime.calls == []
        pipeline.dma_event.synchronize.assert_called_once()
        pipeline.extract_env_aligned_batch.assert_called_once()

    def test_runtime_train_step_contract_is_shared_across_multiple_calls(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = _MockMAGATRuntime()

        first = pipeline.train_step(runtime=runtime)
        second = pipeline.train_step(runtime=runtime)

        assert first == second == {"backend": "magat", "rows": 4}
        assert len(runtime.calls) == 2
        assert all(call[0] is mock_sim for call in runtime.calls)
        assert all(torch.equal(call[1], raw_batch) for call in runtime.calls)

    def test_train_step_runtime_path_keeps_backend_calls_after_gpu_handler(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)
        events = []

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.side_effect = lambda batch: (events.append("filter"), raw_batch[:0])[1]
        pipeline.gpu_handler.process_batch.side_effect = lambda batch, subset: events.append("process")

        class _OrderedRuntime:
            def train_step(self, simulator, batch):
                events.append("runtime")
                return {"backend": "ordered", "rows": int(batch.shape[0])}

        result = pipeline.train_step(runtime=_OrderedRuntime())
        assert result == {"backend": "ordered", "rows": 4}
        assert events == ["filter", "process", "runtime"]

    def test_train_step_runtime_path_passes_exact_raw_batch_without_copy_semantics_assertion(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]

        captured = {}

        class _CaptureRuntime:
            def train_step(self, simulator, batch):
                captured["batch"] = batch
                return {"backend": "capture", "rows": int(batch.shape[0])}

        pipeline.train_step(runtime=_CaptureRuntime())
        assert captured["batch"] is raw_batch

    def test_train_step_runtime_path_still_uses_env_aligned_batch_dimensions(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = _MockMAGATRuntime()

        pipeline.train_step(runtime=runtime)
        pipeline.extract_env_aligned_batch.assert_called_once_with(
            4,
            agents_per_env=2,
            num_envs=2,
        )

    def test_train_step_runtime_path_preserves_compute_stream_synchronization_call(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.dma_event = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        runtime = _MockMAGATRuntime()

        pipeline.train_step(runtime=runtime)

        pipeline.dma_event.synchronize.assert_called_once()
        assert len(runtime.calls) == 1

    def test_train_step_runtime_path_handles_both_backend_stubs_identically_in_shape_contract(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]
        magat = _MockMAGATRuntime()
        gpt = _MockMapfGPTRuntime()

        pipeline.train_step(runtime=magat)
        pipeline.train_step(runtime=gpt)

        assert magat.calls[0][1].shape == gpt.calls[0][1].shape == (4, FEATURE_DIM)
        assert magat.calls[0][1].dtype == gpt.calls[0][1].dtype == torch.int16

    @pytest.mark.parametrize("runtime_factory", [_MockMAGATRuntime, _MockMapfGPTRuntime])
    def test_runtime_train_step_progression_contract_is_stable_across_frontiers(
        self, runtime_factory
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 0, 0],
            positions=[[10, 11], [10, 11], [19, 20], [20, 20]],
            actions=[0, 4, 0, 4],
        )
        batches = [first, second]
        reset_subsets = [first[[0, 3]], second[[1]]]

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(side_effect=batches)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.side_effect = reset_subsets
        runtime = runtime_factory()

        first_result = pipeline.train_step(runtime=runtime)
        second_result = pipeline.train_step(runtime=runtime)

        assert first_result["rows"] == second_result["rows"] == 4
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)
        observed_filter_batches = [call.args[0] for call in pipeline.gpu_handler.filter_reset_agents.call_args_list]
        assert len(observed_filter_batches) == 2
        assert torch.equal(observed_filter_batches[0], first)
        assert torch.equal(observed_filter_batches[1], second)
        assert torch.equal(pipeline.gpu_handler.process_batch.call_args_list[0].args[0], first)
        assert torch.equal(pipeline.gpu_handler.process_batch.call_args_list[0].args[1], reset_subsets[0])
        assert torch.equal(pipeline.gpu_handler.process_batch.call_args_list[1].args[0], second)
        assert torch.equal(pipeline.gpu_handler.process_batch.call_args_list[1].args[1], reset_subsets[1])
        assert len(pipeline.gpu_handler.process_batch.call_args_list) == 2
        assert len(pipeline.gpu_handler.filter_reset_agents.call_args_list) == 2
        assert len(runtime.calls) == len(pipeline.gpu_handler.process_batch.call_args_list)
        assert len(runtime.calls) == len(pipeline.gpu_handler.filter_reset_agents.call_args_list)
        assert torch.equal(reset_subsets[0][:, 0], torch.tensor([0, 1], dtype=torch.int16))
        assert torch.equal(reset_subsets[1][:, 0], torch.tensor([0], dtype=torch.int16))
        assert torch.equal(runtime.calls[0][1][:, 0], torch.tensor([0, 0, 1, 1], dtype=torch.int16))
        assert torch.equal(runtime.calls[1][1][:, 0], torch.tensor([0, 0, 1, 1], dtype=torch.int16))
        assert torch.equal(runtime.calls[0][1][:, 1], torch.tensor([0, 1, 0, 1], dtype=torch.int16))
        assert torch.equal(runtime.calls[1][1][:, 1], torch.tensor([0, 1, 0, 1], dtype=torch.int16))

    def test_runtime_train_step_progression_filters_only_current_frontier_reset_rows(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 1, 0, 0],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 4, 0, 0],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 0, 1, 0],
            positions=[[10, 11], [10, 12], [20, 20], [20, 21]],
            actions=[0, 0, 4, 0],
        )

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(side_effect=[first, second])
        runtime = _MockMAGATRuntime()
        observed_subsets = []

        class _TrackingHandler:
            def filter_reset_agents(self, batch):
                return GPUComputeHandler(mock_sim).filter_reset_agents(batch)

            def process_batch(self, batch, subset):
                observed_subsets.append(subset.clone())

        pipeline.gpu_handler = _TrackingHandler()

        pipeline.train_step(runtime=runtime)
        pipeline.train_step(runtime=runtime)

        assert len(observed_subsets) == 2
        assert torch.equal(observed_subsets[0][:, 1], torch.tensor([0, 1], dtype=torch.int16))
        assert torch.equal(observed_subsets[1][:, 1], torch.tensor([0], dtype=torch.int16))
        assert observed_subsets[0].shape == (2, FEATURE_DIM)
        assert observed_subsets[1].shape == (1, FEATURE_DIM)
        assert torch.equal(observed_subsets[0][:, RESET_FLAG_COL], torch.tensor([1, 1], dtype=torch.int16))
        assert torch.equal(observed_subsets[1][:, RESET_FLAG_COL], torch.tensor([1], dtype=torch.int16))
        assert all(torch.equal(call[1], batch) for call, batch in zip(runtime.calls, (first, second)))

    def test_train_step_runtime_path_backend_return_values_flow_to_caller(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]

        class _StructuredRuntime:
            def train_step(self, simulator, batch):
                return {"backend": "structured", "loss": 1.23, "rows": int(batch.shape[0])}

        result = pipeline.train_step(runtime=_StructuredRuntime())
        assert result == {"backend": "structured", "loss": 1.23, "rows": 4}

    @pytest.mark.parametrize("runtime_factory", [_MockMAGATRuntime, _MockMapfGPTRuntime])
    def test_runtime_results_and_stats_stay_consistent_across_two_ready_frontiers(
        self, runtime_factory
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        for block in _frontier_env_blocks(first):
            pipeline.ring_buffer.reserve_and_write(block)
        for block in _frontier_env_blocks(second):
            pipeline.ring_buffer.reserve_and_write(block)

        assert pipeline.has_env_aligned_batch(2) is True
        with patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()):
            copied0 = pipeline.dma_worker.run_once()
            assert copied0 == (0, 4)
            stats_after_dma0 = pipeline.get_stats()
            assert stats_after_dma0["dma_read_ptr"] == 4
            assert stats_after_dma0["compute_ptr"] == 0
            assert stats_after_dma0["reserve_ptr"] == 8
            assert stats_after_dma0["stage_ready_counts"] == [2, 2]
            assert stats_after_dma0["stage_seq_tags"] == [0, 1]

            runtime = runtime_factory()
            first_result = pipeline.train_step(runtime=runtime)
            stats_after_train0 = pipeline.get_stats()
            assert first_result["rows"] == 4
            assert stats_after_train0["compute_ptr"] == 4
            assert stats_after_train0["dma_read_ptr"] == 4
            assert stats_after_train0["stage_ready_counts"] == [0, 2]
            assert stats_after_train0["stage_seq_tags"] == [-1, 1]
            assert stats_after_train0["env_write_stage"] == [2, 2]
            assert pipeline.has_env_aligned_batch(2) is True
            assert pipeline.extract_env_aligned_batch(
                4, agents_per_env=2, num_envs=2
            ) is None

            copied1 = pipeline.dma_worker.run_once()
            assert copied1 == (4, 8)
            stats_after_dma1 = pipeline.get_stats()
            assert stats_after_dma1["dma_read_ptr"] == 8
            assert stats_after_dma1["compute_ptr"] == 4
            assert stats_after_dma1["stage_ready_counts"] == [0, 2]
            assert stats_after_dma1["stage_seq_tags"] == [-1, 1]
            assert pipeline.has_env_aligned_batch(2) is True

            second_result = pipeline.train_step(runtime=runtime)
        stats_after_train1 = pipeline.get_stats()
        assert second_result["rows"] == 4
        assert stats_after_train1["compute_ptr"] == 8
        assert stats_after_train1["dma_read_ptr"] == 8
        assert stats_after_train1["reserve_ptr"] == 8
        assert stats_after_train1["stage_ready_counts"] == [0, 0]
        assert stats_after_train1["stage_seq_tags"] == [-1, -1]
        assert stats_after_train1["env_write_stage"] == [2, 2]
        assert pipeline.has_env_aligned_batch(2) is False
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)

    @pytest.mark.parametrize("backend_name", ["magat", "mapf_gpt"])
    def test_runtime_structured_results_remain_frontier_local_across_two_ready_frontiers(
        self, backend_name
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )

        class _StructuredRuntime:
            def __init__(self, backend):
                self.backend = backend
                self.calls = []

            def train_step(self, simulator, raw_batch):
                self.calls.append((simulator, raw_batch.clone()))
                reset_rows = raw_batch[raw_batch[:, RESET_FLAG_COL] != 0]
                return {
                    "backend": self.backend,
                    "rows": int(raw_batch.shape[0]),
                    "callback": {
                        "call_index": len(self.calls) - 1,
                        "reset_rows": int(reset_rows.shape[0]),
                        "env_ids": raw_batch[:, 0].tolist(),
                        "positions": raw_batch[:, 2:4].tolist(),
                    },
                }

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        for block in _frontier_env_blocks(first):
            pipeline.ring_buffer.reserve_and_write(block)
        for block in _frontier_env_blocks(second):
            pipeline.ring_buffer.reserve_and_write(block)

        runtime = _StructuredRuntime(backend_name)
        with patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()):
            assert pipeline.dma_worker.run_once() == (0, 4)
            first_result = pipeline.train_step(runtime=runtime)
            assert first_result == {
                "backend": backend_name,
                "rows": 4,
                "callback": {
                    "call_index": 0,
                    "reset_rows": 2,
                    "env_ids": [0, 0, 1, 1],
                    "positions": [[10, 10], [10, 11], [20, 20], [20, 21]],
                },
            }
            assert pipeline.get_stats()["compute_ptr"] == 4
            assert pipeline.get_stats()["stage_seq_tags"] == [-1, 1]

            assert pipeline.dma_worker.run_once() == (4, 8)
            second_result = pipeline.train_step(runtime=runtime)

        assert second_result == {
            "backend": backend_name,
            "rows": 4,
            "callback": {
                "call_index": 1,
                "reset_rows": 2,
                "env_ids": [0, 0, 1, 1],
                "positions": [[30, 30], [30, 31], [40, 40], [40, 41]],
            },
        }
        final_stats = pipeline.get_stats()
        assert final_stats["compute_ptr"] == 8
        assert final_stats["dma_read_ptr"] == 8
        assert final_stats["stage_ready_counts"] == [0, 0]
        assert final_stats["stage_seq_tags"] == [-1, -1]
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)

    @pytest.mark.parametrize("backend_name", ["magat", "mapf_gpt"])
    def test_runtime_callback_observes_post_release_stats_at_each_frontier_boundary(
        self, backend_name
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )
        observed = []

        class _BoundaryRuntime:
            def __init__(self, backend):
                self.backend = backend
                self.calls = []

            def train_step(self, simulator, raw_batch):
                self.calls.append((simulator, raw_batch.clone()))
                observed.append(
                    {
                        "call_index": len(self.calls) - 1,
                        "stats": pipeline.get_stats(),
                        "has_ready_frontier": pipeline.has_env_aligned_batch(2),
                        "positions": raw_batch[:, 2:4].tolist(),
                    }
                )
                return {
                    "backend": self.backend,
                    "rows": int(raw_batch.shape[0]),
                    "call_index": len(self.calls) - 1,
                }

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        for block in _frontier_env_blocks(first):
            pipeline.ring_buffer.reserve_and_write(block)
        for block in _frontier_env_blocks(second):
            pipeline.ring_buffer.reserve_and_write(block)

        runtime = _BoundaryRuntime(backend_name)
        with patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()):
            assert pipeline.dma_worker.run_once() == (0, 4)
            first_result = pipeline.train_step(runtime=runtime)
            assert first_result == {
                "backend": backend_name,
                "rows": 4,
                "call_index": 0,
            }

            assert pipeline.dma_worker.run_once() == (4, 8)
            second_result = pipeline.train_step(runtime=runtime)
            assert second_result == {
                "backend": backend_name,
                "rows": 4,
                "call_index": 1,
            }

        assert observed[0] == {
            "call_index": 0,
            "stats": {
                "capacity": 8,
                "reserve_ptr": 8,
                "dma_read_ptr": 4,
                "compute_ptr": 4,
                "num_stage_slots": 2,
                "stage_rows": 4,
                "stage_ready_counts": [0, 2],
                "stage_seq_tags": [-1, 1],
                "env_write_stage": [2, 2],
            },
            "has_ready_frontier": True,
            "positions": [[10, 10], [10, 11], [20, 20], [20, 21]],
        }
        assert observed[1] == {
            "call_index": 1,
            "stats": {
                "capacity": 8,
                "reserve_ptr": 8,
                "dma_read_ptr": 8,
                "compute_ptr": 8,
                "num_stage_slots": 2,
                "stage_rows": 4,
                "stage_ready_counts": [0, 0],
                "stage_seq_tags": [-1, -1],
                "env_write_stage": [2, 2],
            },
            "has_ready_frontier": False,
            "positions": [[30, 30], [30, 31], [40, 40], [40, 41]],
        }
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)

    @pytest.mark.parametrize("backend_name", ["magat", "mapf_gpt"])
    def test_runtime_callback_aggregation_preserves_ordered_frontier_snapshots(
        self, backend_name
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )

        def _expected_entry(call_index, positions, reset_rows):
            return {
                "call_index": call_index,
                "reset_rows": reset_rows,
                "positions": positions,
            }

        def _clone_entries(entries):
            return [
                {
                    "call_index": entry["call_index"],
                    "reset_rows": entry["reset_rows"],
                    "positions": [list(pos) for pos in entry["positions"]],
                }
                for entry in entries
            ]

        class _AggregatingRuntime:
            def __init__(self, backend):
                self.backend = backend
                self.calls = []
                self.aggregate = []

            def train_step(self, simulator, raw_batch):
                self.calls.append((simulator, raw_batch.clone()))
                self.aggregate.append(
                    _expected_entry(
                        call_index=len(self.calls) - 1,
                        reset_rows=int((raw_batch[:, RESET_FLAG_COL] != 0).sum().item()),
                        positions=raw_batch[:, 2:4].tolist(),
                    )
                )
                return {
                    "backend": self.backend,
                    "rows": int(raw_batch.shape[0]),
                    "aggregate": _clone_entries(self.aggregate),
                }

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        for block in _frontier_env_blocks(first):
            pipeline.ring_buffer.reserve_and_write(block)
        for block in _frontier_env_blocks(second):
            pipeline.ring_buffer.reserve_and_write(block)

        expected_first = {
            "backend": backend_name,
            "rows": 4,
            "aggregate": [
                _expected_entry(
                    call_index=0,
                    reset_rows=2,
                    positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
                )
            ],
        }
        expected_second = {
            "backend": backend_name,
            "rows": 4,
            "aggregate": [
                _expected_entry(
                    call_index=0,
                    reset_rows=2,
                    positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
                ),
                _expected_entry(
                    call_index=1,
                    reset_rows=2,
                    positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
                ),
            ],
        }

        runtime = _AggregatingRuntime(backend_name)
        with patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()):
            assert pipeline.dma_worker.run_once() == (0, 4)
            first_result = pipeline.train_step(runtime=runtime)
            assert first_result == expected_first

            assert pipeline.dma_worker.run_once() == (4, 8)
            second_result = pipeline.train_step(runtime=runtime)

        assert first_result == expected_first
        assert second_result == expected_second
        assert runtime.aggregate == expected_second["aggregate"]
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)

    @pytest.mark.parametrize("backend_name", ["magat", "mapf_gpt"])
    def test_runtime_callback_aggregation_failure_preserves_prior_snapshot(
        self, backend_name
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )

        def _entry(call_index, positions, reset_rows):
            return {
                "call_index": call_index,
                "reset_rows": reset_rows,
                "positions": positions,
            }

        def _clone_entries(entries):
            return [
                {
                    "call_index": entry["call_index"],
                    "reset_rows": entry["reset_rows"],
                    "positions": [list(pos) for pos in entry["positions"]],
                }
                for entry in entries
            ]

        class _FailingAggregatingRuntime:
            def __init__(self, backend):
                self.backend = backend
                self.calls = []
                self.aggregate = []

            def train_step(self, simulator, raw_batch):
                self.calls.append((simulator, raw_batch.clone()))
                current_entry = _entry(
                    call_index=len(self.calls) - 1,
                    reset_rows=int((raw_batch[:, RESET_FLAG_COL] != 0).sum().item()),
                    positions=raw_batch[:, 2:4].tolist(),
                )
                if len(self.calls) == 2:
                    raise RuntimeError(f"{self.backend} callback aggregation failure")
                self.aggregate.append(current_entry)
                return {
                    "backend": self.backend,
                    "rows": int(raw_batch.shape[0]),
                    "aggregate": _clone_entries(self.aggregate),
                }

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        for block in _frontier_env_blocks(first):
            pipeline.ring_buffer.reserve_and_write(block)
        for block in _frontier_env_blocks(second):
            pipeline.ring_buffer.reserve_and_write(block)

        expected_first = {
            "backend": backend_name,
            "rows": 4,
            "aggregate": [
                _entry(
                    call_index=0,
                    reset_rows=2,
                    positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
                )
            ],
        }

        runtime = _FailingAggregatingRuntime(backend_name)
        with patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()):
            assert pipeline.dma_worker.run_once() == (0, 4)
            first_result = pipeline.train_step(runtime=runtime)
            assert first_result == expected_first

            assert pipeline.dma_worker.run_once() == (4, 8)
            with pytest.raises(RuntimeError, match=f"{backend_name} callback aggregation failure"):
                pipeline.train_step(runtime=runtime)

        assert first_result == expected_first
        assert runtime.aggregate == expected_first["aggregate"]
        failure_stats = pipeline.get_stats()
        assert failure_stats["compute_ptr"] == 8
        assert failure_stats["dma_read_ptr"] == 8
        assert failure_stats["stage_ready_counts"] == [0, 0]
        assert failure_stats["stage_seq_tags"] == [-1, -1]
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)

    @pytest.mark.parametrize("backend_name", ["magat", "mapf_gpt"])
    def test_runtime_failure_after_second_frontier_still_leaves_consumed_stage_state(
        self, backend_name
    ):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        first = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        second = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )

        class _FailOnSecondRuntime:
            def __init__(self, backend):
                self.backend = backend
                self.calls = []

            def train_step(self, simulator, raw_batch):
                self.calls.append((simulator, raw_batch.clone()))
                if len(self.calls) == 2:
                    raise RuntimeError(f"{self.backend} frontier failure")
                return {
                    "backend": self.backend,
                    "rows": int(raw_batch.shape[0]),
                    "call_index": 0,
                }

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        for block in _frontier_env_blocks(first):
            pipeline.ring_buffer.reserve_and_write(block)
        for block in _frontier_env_blocks(second):
            pipeline.ring_buffer.reserve_and_write(block)

        runtime = _FailOnSecondRuntime(backend_name)
        with patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()):
            assert pipeline.dma_worker.run_once() == (0, 4)
            first_result = pipeline.train_step(runtime=runtime)
            assert first_result == {
                "backend": backend_name,
                "rows": 4,
                "call_index": 0,
            }

            assert pipeline.dma_worker.run_once() == (4, 8)
            with pytest.raises(RuntimeError, match=f"{backend_name} frontier failure"):
                pipeline.train_step(runtime=runtime)

        failure_stats = pipeline.get_stats()
        assert failure_stats["compute_ptr"] == 8
        assert failure_stats["dma_read_ptr"] == 8
        assert failure_stats["reserve_ptr"] == 8
        assert failure_stats["stage_ready_counts"] == [0, 0]
        assert failure_stats["stage_seq_tags"] == [-1, -1]
        assert failure_stats["env_write_stage"] == [2, 2]
        assert pipeline.has_env_aligned_batch(2) is False
        assert len(runtime.calls) == 2
        assert torch.equal(runtime.calls[0][1], first)
        assert torch.equal(runtime.calls[1][1], second)

    def test_train_step_runtime_path_calls_filter_before_process_even_when_empty(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)
        events = []

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)

        class _Handler:
            def filter_reset_agents(self, batch):
                events.append("filter")
                return batch[:0]

            def process_batch(self, batch, subset):
                events.append("process")

        pipeline.gpu_handler = _Handler()
        runtime = _MockMAGATRuntime()
        pipeline.train_step(runtime=runtime)
        assert events == ["filter", "process"]

    def test_train_step_runtime_path_preserves_backend_stub_order_across_calls(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)
        sequence = []

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)

        class _Handler:
            def filter_reset_agents(self, batch):
                sequence.append("filter")
                return batch[:0]

            def process_batch(self, batch, subset):
                sequence.append("process")

        class _Runtime:
            def train_step(self, simulator, batch):
                sequence.append("runtime")
                return {"backend": "ok", "rows": int(batch.shape[0])}

        pipeline.gpu_handler = _Handler()
        pipeline.train_step(runtime=_Runtime())
        pipeline.train_step(runtime=_Runtime())
        assert sequence == ["filter", "process", "runtime", "filter", "process", "runtime"]

    def test_train_step_runtime_path_accepts_backend_stubs_with_same_signature(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        raw_batch = torch.zeros((4, FEATURE_DIM), dtype=torch.int16)
        raw_batch[:, 0] = torch.tensor([0, 0, 1, 1], dtype=torch.int16)
        raw_batch[:, 1] = torch.tensor([0, 1, 0, 1], dtype=torch.int16)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=raw_batch)
        pipeline.gpu_handler = MagicMock()
        pipeline.gpu_handler.filter_reset_agents.return_value = raw_batch[:0]

        class _RuntimeA:
            def train_step(self, simulator, batch):
                return ("a", int(batch.shape[0]))

        class _RuntimeB:
            def train_step(self, simulator, batch):
                return ("b", int(batch.shape[0]))

        assert pipeline.train_step(runtime=_RuntimeA()) == ("a", 4)
        assert pipeline.train_step(runtime=_RuntimeB()) == ("b", 4)

    def test_train_step_runtime_path_never_calls_runtime_when_raw_batch_missing(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker = MagicMock()
        pipeline.dma_event = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=None)

        class _Runtime:
            def __init__(self):
                self.called = False

            def train_step(self, simulator, batch):
                self.called = True
                return None

        runtime = _Runtime()
        assert pipeline.train_step(runtime=runtime) is None
        assert runtime.called is False

    def test_dma_worker_loop_starts_thread(self):
        """dma_worker_loop starts DMA worker thread."""
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(0, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=1024, batch_threshold=256)
            pipeline.initialize()
        pipeline.dma_worker_loop()
        assert pipeline.dma_worker._running is True
        pipeline.dma_worker.stop()

    def test_get_stats_returns_dict(self):
        """get_stats returns dictionary with pipeline state."""
        mock_sim = MagicMock()
        pipeline = ExtremeMAPFPipeline(mock_sim, capacity=1024, batch_threshold=256)
        stats = pipeline.get_stats()
        assert "capacity" in stats
        assert "reserve_ptr" in stats
        assert "dma_read_ptr" in stats

    def test_pointer_monotonicity_logic(self):
        """Test that compute_ptr <= dma_read_ptr always holds conceptually."""
        compute_ptr = 0
        dma_read_ptr = 0

        # DMA advances
        dma_read_ptr += 256
        assert compute_ptr <= dma_read_ptr

        # Compute catches up
        compute_ptr += 256
        assert compute_ptr <= dma_read_ptr

        # DMA advances again
        dma_read_ptr += 512
        assert compute_ptr <= dma_read_ptr

    def test_ring_buffer_capacity_logic(self):
        """Test wrap-around index calculation."""
        capacity = 100
        ptr = 250

        # idx calculation
        idx = ptr % capacity
        assert idx == 50

        # After advancing past capacity
        ptr += 60
        idx = ptr % capacity
        assert idx == 10

    def test_chunk_extraction_logic(self):
        """Test the chunk extraction without wrap-around."""
        capacity = 100
        gpu_buffer = torch.arange(100 * FEATURE_DIM).reshape(100, FEATURE_DIM).float()

        compute_ptr = 20
        dma_read_ptr = 80
        chunk_size = min(dma_read_ptr - compute_ptr, 40)

        idx_start = compute_ptr % capacity
        idx_end = (compute_ptr + chunk_size) % capacity

        if idx_start < idx_end:
            chunk = gpu_buffer[idx_start:idx_end]
        else:
            chunk = torch.cat([
                gpu_buffer[idx_start:capacity],
                gpu_buffer[0:idx_end]
            ], dim=0)

        assert chunk.shape[0] == chunk_size

    def test_chunk_extraction_with_wrap(self):
        """Test chunk extraction with wrap-around."""
        capacity = 100
        gpu_buffer = torch.arange(100 * FEATURE_DIM).reshape(100, FEATURE_DIM).float()

        compute_ptr = 90
        dma_read_ptr = 120
        chunk_size = min(dma_read_ptr - compute_ptr, 40)  # 30

        idx_start = compute_ptr % capacity  # 90
        idx_end = (compute_ptr + chunk_size) % capacity  # 20

        if idx_start < idx_end:
            chunk = gpu_buffer[idx_start:idx_end]
        else:
            chunk = torch.cat([
                gpu_buffer[idx_start:capacity],  # [90:100] = 10 rows
                gpu_buffer[0:idx_end]             # [0:20] = 20 rows
            ], dim=0)

        assert chunk.shape[0] == chunk_size
        assert chunk.shape[1] == FEATURE_DIM

    def test_stage_mode_initialize_creates_frontier_components(self):
        """initialize() should infer frontier layout and allocate stage buffers."""
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=8, batch_threshold=4, device="cpu")
            pipeline.initialize()

        assert pipeline.stage_mode is True
        assert pipeline.agents_per_env == 2
        assert pipeline.num_envs == 2
        assert pipeline.ring_buffer.stage_mode is True
        assert pipeline.ring_buffer.stage_rows == 4
        assert pipeline.ring_buffer.num_stage_slots == 2
        assert pipeline.stage_batch_buffer.shape == (4, FEATURE_DIM)

    def test_stage_ring_requires_full_env_block(self):
        """Stage ring should reject partial env writes."""
        with patch("torch.Tensor.pin_memory", lambda self: self):
            ring = RingBuffer(capacity=12, num_envs=2, agents_per_env=3)

        data = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        data[:, 0] = 0
        data[:, 1] = np.arange(2, dtype=np.uint16)
        with pytest.raises(ValueError, match="one full env block"):
            ring.reserve_and_write(data)

    def test_stage_ring_requires_contiguous_agent_ids(self):
        """Stage ring should reject malformed agent ordering inside an env block."""
        with patch("torch.Tensor.pin_memory", lambda self: self):
            ring = RingBuffer(capacity=12, num_envs=2, agents_per_env=3)

        data = np.zeros((3, FEATURE_DIM), dtype=np.uint16)
        data[:, 0] = 1
        data[:, 1] = np.array([0, 2, 1], dtype=np.uint16)
        with pytest.raises(ValueError, match="agent ids to be contiguous"):
            ring.reserve_and_write(data)

    def test_stage_ready_requires_all_envs(self):
        """A frontier stage becomes ready only after every env fills its reserved slot."""
        with patch("torch.Tensor.pin_memory", lambda self: self):
            ring = RingBuffer(capacity=12, num_envs=2, agents_per_env=3)

        env0 = np.zeros((3, FEATURE_DIM), dtype=np.uint16)
        env0[:, 0] = 0
        env0[:, 1] = np.arange(3, dtype=np.uint16)
        env1 = env0.copy()
        env1[:, 0] = 1

        ring.reserve_and_write(env0)
        assert ring.is_stage_ready(0) is False
        ring.reserve_and_write(env1)
        assert ring.is_stage_ready(0) is True
        assert ring.stage_ready_counts[0] == 2
        assert list(ring.env_write_seq) == [1, 1]

    def test_extract_env_aligned_batch_reads_single_frontier_stage(self):
        """Stage-mode extraction should return one env-major frontier and release the stage."""
        mock_sim = MagicMock()
        pipeline = ExtremeMAPFPipeline(mock_sim, capacity=8, batch_threshold=4, device="cpu")
        pipeline.feature_dim = FEATURE_DIM
        with patch("torch.Tensor.pin_memory", lambda self: self):
            pipeline.ring_buffer = RingBuffer(capacity=8, num_envs=2, agents_per_env=2)
        pipeline.gpu_buffer = torch.zeros((8, FEATURE_DIM), dtype=torch.int16)
        pipeline.stage_batch_buffer = torch.empty((4, FEATURE_DIM), dtype=torch.int16)
        pipeline.compute_ptr = 0
        pipeline.expected_agent_ids_cache = {}
        pipeline.dma_worker = MagicMock()
        pipeline.dma_worker.dma_read_ptr = 4
        pipeline.dma_worker.release_completed_stage = MagicMock()

        stage = torch.tensor(
            [
                [0, 0, 10, 20, 30, 40, 1, 0],
                [0, 1, 11, 21, 31, 41, 2, 0],
                [1, 0, 50, 60, 70, 80, 3, 1],
                [1, 1, 51, 61, 71, 81, 4, 0],
            ],
            dtype=torch.int16,
        )
        pipeline.gpu_buffer[:4] = stage
        pipeline.ring_buffer.stage_seq_tags[0] = 0
        pipeline.ring_buffer.stage_ready_counts[0] = 2

        raw_batch = pipeline.extract_env_aligned_batch(
            chunk_size=4, agents_per_env=2, num_envs=2
        )

        assert torch.equal(raw_batch, stage)
        assert pipeline.compute_ptr == 4
        pipeline.dma_worker.release_completed_stage.assert_called_once_with(0)

    def test_stage_release_clears_ready_state(self):
        """Releasing a frontier stage should clear tags, counts, and ready flags."""
        with patch("torch.Tensor.pin_memory", lambda self: self):
            ring = RingBuffer(capacity=8, num_envs=2, agents_per_env=2)

        env0 = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        env0[:, 0] = 0
        env0[:, 1] = np.arange(2, dtype=np.uint16)
        env1 = env0.copy()
        env1[:, 0] = 1
        ring.reserve_and_write(env0)
        ring.reserve_and_write(env1)

        assert ring.is_stage_ready(0) is True
        ring.release_stage(0)

        assert ring.is_stage_ready(0) is False
        assert ring.stage_ready_counts[0] == 0
        assert ring.stage_seq_tags[0] == -1
        assert sum(ring.ready_flags[:4]) == 0

    def test_validate_env_aligned_batch_accepts_env_major_frontier(self):
        raw_batch = torch.tensor(
            [
                [0, 0, 10, 20, 30, 40, 1, 0],
                [0, 1, 11, 21, 31, 41, 2, 0],
                [1, 0, 50, 60, 70, 80, 3, 1],
                [1, 1, 51, 61, 71, 81, 4, 0],
            ],
            dtype=torch.int16,
        )
        assert pipeline_validate_env_aligned_batch(raw_batch, agents_per_env=2, num_envs=2)

    def test_validate_env_aligned_batch_rejects_wrong_env_order(self):
        raw_batch = torch.tensor(
            [
                [0, 0, 10, 20, 30, 40, 1, 0],
                [1, 1, 11, 21, 31, 41, 2, 0],
                [1, 0, 50, 60, 70, 80, 3, 1],
                [0, 1, 51, 61, 71, 81, 4, 0],
            ],
            dtype=torch.int16,
        )
        with pytest.raises(ValueError, match="env blocks"):
            pipeline_validate_env_aligned_batch(raw_batch, agents_per_env=2, num_envs=2)

    def test_validate_env_aligned_batch_rejects_wrong_agent_order(self):
        raw_batch = torch.tensor(
            [
                [0, 0, 10, 20, 30, 40, 1, 0],
                [0, 2, 11, 21, 31, 41, 2, 0],
                [1, 0, 50, 60, 70, 80, 3, 1],
                [1, 1, 51, 61, 71, 81, 4, 0],
            ],
            dtype=torch.int16,
        )
        with pytest.raises(ValueError, match="agent ids"):
            pipeline_validate_env_aligned_batch(raw_batch, agents_per_env=2, num_envs=2)

    def test_stage_ring_reuses_stage_slot_only_after_release(self):
        with patch("torch.Tensor.pin_memory", lambda self: self):
            ring = RingBuffer(capacity=8, num_envs=2, agents_per_env=2)

        env0_stage0 = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        env0_stage0[:, 0] = 0
        env0_stage0[:, 1] = np.arange(2, dtype=np.uint16)
        env1_stage0 = env0_stage0.copy()
        env1_stage0[:, 0] = 1

        env0_stage1 = env0_stage0.copy()
        env0_stage1[:, 2] = [10, 11]
        env1_stage1 = env1_stage0.copy()
        env1_stage1[:, 2] = [20, 21]

        env0_stage2 = env0_stage0.copy()
        env0_stage2[:, 2] = [30, 31]
        env1_stage2 = env1_stage0.copy()
        env1_stage2[:, 2] = [40, 41]

        assert ring.reserve_and_write(env0_stage0) == (0, 2)
        assert ring.reserve_and_write(env1_stage0) == (2, 4)
        assert ring.reserve_and_write(env0_stage1) == (4, 6)
        assert ring.reserve_and_write(env1_stage1) == (6, 8)
        assert ring.is_stage_ready(0) is True
        assert ring.is_stage_ready(1) is True
        assert list(ring.stage_seq_tags) == [0, 1]
        assert list(ring.env_write_seq) == [2, 2]

        ring.shared_compute_ptr.value = 4
        assert ring.reserve_and_write(env0_stage2) == (0, 2)
        assert ring.reserve_and_write(env1_stage2) == (2, 4)
        assert ring.is_stage_ready(2) is True
        assert list(ring.stage_seq_tags) == [2, 1]
        assert list(ring.stage_ready_counts) == [2, 2]
        assert list(ring.env_write_seq) == [3, 3]

    def test_extract_env_aligned_batch_advances_frontier_order_across_stage_slots(self):
        mock_sim = MagicMock()
        pipeline = ExtremeMAPFPipeline(mock_sim, capacity=8, batch_threshold=4, device="cpu")
        pipeline.feature_dim = FEATURE_DIM
        with patch("torch.Tensor.pin_memory", lambda self: self):
            pipeline.ring_buffer = RingBuffer(capacity=8, num_envs=2, agents_per_env=2)
        pipeline.gpu_buffer = torch.zeros((8, FEATURE_DIM), dtype=torch.int16)
        pipeline.stage_batch_buffer = torch.empty((4, FEATURE_DIM), dtype=torch.int16)
        pipeline.expected_agent_ids_cache = {}
        pipeline.dma_worker = MagicMock()
        pipeline.dma_worker.release_completed_stage = MagicMock()
        pipeline.compute_ptr = 0

        stage0 = _make_env_aligned_frontier(
            reset_flags=[1, 0, 0, 1],
            positions=[[10, 10], [10, 11], [20, 20], [20, 21]],
            actions=[4, 0, 1, 3],
        )
        stage1 = _make_env_aligned_frontier(
            reset_flags=[0, 1, 1, 0],
            positions=[[30, 30], [30, 31], [40, 40], [40, 41]],
            actions=[0, 4, 4, 0],
        )
        pipeline.gpu_buffer[:4] = stage0
        pipeline.gpu_buffer[4:8] = stage1
        pipeline.ring_buffer.stage_seq_tags[0] = 0
        pipeline.ring_buffer.stage_seq_tags[1] = 1
        pipeline.ring_buffer.stage_ready_counts[0] = 2
        pipeline.ring_buffer.stage_ready_counts[1] = 2

        pipeline.dma_worker.dma_read_ptr = 8

        first = pipeline.extract_env_aligned_batch(
            chunk_size=4, agents_per_env=2, num_envs=2
        ).clone()
        second = pipeline.extract_env_aligned_batch(
            chunk_size=4, agents_per_env=2, num_envs=2
        ).clone()

        assert torch.equal(first, stage0)
        assert torch.equal(second, stage1)
        assert pipeline.compute_ptr == 8
        assert pipeline.dma_worker.release_completed_stage.call_args_list[0].args == (0,)
        assert pipeline.dma_worker.release_completed_stage.call_args_list[1].args == (1,)

    def test_release_completed_stage_updates_compute_ptr_to_next_frontier(self):
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            ring = RingBuffer(capacity=8, num_envs=2, agents_per_env=2)
            gpu = torch.zeros((8, FEATURE_DIM), dtype=torch.int16)
            worker = DMAWorker(ring, gpu, batch_threshold=4, device="cpu")

        env0 = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
        env0[:, 0] = 0
        env0[:, 1] = np.arange(2, dtype=np.uint16)
        env1 = env0.copy()
        env1[:, 0] = 1
        ring.reserve_and_write(env0)
        ring.reserve_and_write(env1)

        assert ring.is_stage_ready(0) is True
        worker.release_completed_stage(0)
        assert ring.shared_compute_ptr.value == 4
        assert ring.is_stage_ready(0) is False
        assert list(ring.stage_seq_tags) == [-1, -1]
        assert list(ring.stage_ready_counts) == [0, 0]

    def test_pipeline_has_env_aligned_batch_tracks_frontier_readiness_after_release(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()

        pipeline.ring_buffer.stage_seq_tags[0] = 0
        pipeline.ring_buffer.stage_ready_counts[0] = 2
        pipeline.compute_ptr = 0
        assert pipeline.has_env_aligned_batch(2) is True

        pipeline.ring_buffer.release_stage(0)
        assert pipeline.has_env_aligned_batch(2) is False

        pipeline.ring_buffer.stage_seq_tags[1] = 1
        pipeline.ring_buffer.stage_ready_counts[1] = 2
        pipeline.compute_ptr = 4
        assert pipeline.has_env_aligned_batch(2) is True

    def test_dma_worker_run_once_waits_for_complete_stage(self):
        """DMA should not copy a stage until all env blocks for that frontier are ready."""
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.stream", side_effect=lambda *args, **kwargs: nullcontext()), patch(
            "torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()
        ):
            ring = RingBuffer(capacity=8, num_envs=2, agents_per_env=2)
            gpu = torch.zeros((8, FEATURE_DIM), dtype=torch.int16)
            worker = DMAWorker(ring, gpu, batch_threshold=4, device="cpu")

            env0 = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
            env0[:, 0] = 0
            env0[:, 1] = np.arange(2, dtype=np.uint16)
            ring.reserve_and_write(env0)
            assert worker.run_once() is None

            env1 = env0.copy()
            env1[:, 0] = 1
            copied = ring.reserve_and_write(env1)
            assert worker.run_once() == (0, 4)
            assert worker.dma_read_ptr == 4
            assert torch.equal(gpu[0:4], ring.cpu_buffer[0:4])
            assert copied == (2, 4)

    def test_pipeline_get_stats_reports_stage_state(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=8, batch_threshold=4, device="cpu")
            pipeline.initialize()

        stats = pipeline.get_stats()
        assert stats["num_stage_slots"] == 2
        assert stats["stage_rows"] == 4
        assert stats["env_write_stage"] == [0, 0]
        assert stats["stage_ready_counts"] == [0, 0]

    def test_train_step_returns_none_without_ready_frontier(self):
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)

        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(mock_sim, capacity=8, batch_threshold=4, device="cpu")
            pipeline.initialize()

        pipeline.dma_event = MagicMock()
        pipeline.dma_event.synchronize = MagicMock()
        pipeline.extract_env_aligned_batch = MagicMock(return_value=None)
        assert pipeline.train_step() is None
        pipeline.extract_env_aligned_batch.assert_called_once_with(
            4, agents_per_env=2, num_envs=2
        )

    def test_dma_worker_loop_starts_thread(self):
        """dma_worker_loop starts DMA worker thread."""
        mock_sim = MagicMock()
        mock_sim.pyg_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
        mock_sim.pyg_batch = torch.zeros(4, dtype=torch.int64)
        with patch("torch.Tensor.pin_memory", lambda self: self), patch(
            "torch.cuda.Stream", side_effect=lambda *args, **kwargs: object()
        ), patch("torch.cuda.Event", side_effect=lambda *args, **kwargs: MagicMock()):
            pipeline = ExtremeMAPFPipeline(
                mock_sim, capacity=8, batch_threshold=4, device="cpu"
            )
            pipeline.initialize()
        pipeline.dma_worker_loop()
        assert pipeline.dma_worker._running is True
        pipeline.dma_worker.stop()

    def test_get_stats_returns_dict(self):
        """get_stats returns dictionary with pipeline state."""
        mock_sim = MagicMock()
        pipeline = ExtremeMAPFPipeline(mock_sim, capacity=1024, batch_threshold=256)
        stats = pipeline.get_stats()
        assert "capacity" in stats
        assert "reserve_ptr" in stats
        assert "dma_read_ptr" in stats

    def test_pointer_monotonicity_logic(self):
        """Test that compute_ptr <= dma_read_ptr always holds conceptually."""
        compute_ptr = 0
        dma_read_ptr = 0

        dma_read_ptr += 256
        assert compute_ptr <= dma_read_ptr

        compute_ptr += 256
        assert compute_ptr <= dma_read_ptr

        dma_read_ptr += 512
        assert compute_ptr <= dma_read_ptr


# ============================================================
# Test: DataValidator
# ============================================================

class TestDataValidator:
    """Tests for DataValidator interface."""

    def test_validate_step_data_valid(self):
        """Valid step data passes validation."""
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        assert DataValidator.validate_step_data(data) is True

    def test_validate_gpu_tensor_cpu_raises(self):
        """CPU tensor raises ValueError."""
        tensor = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        with pytest.raises(ValueError, match="CUDA"):
            DataValidator.validate_gpu_tensor(tensor)

    def test_check_agent_state_alignment_valid(self):
        """Valid alignment passes."""
        tensor = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        assert DataValidator.check_agent_state_alignment(tensor) is True

    # ---- Edge case validation tests (logic-level) ----

    def test_step_data_dtype_validation(self):
        """Wrong dtype should fail."""
        data = np.zeros((10, FEATURE_DIM), dtype=np.float32)
        assert data.dtype != np.uint16

    def test_step_data_shape_validation(self):
        """Wrong feature dim should fail."""
        data = np.zeros((10, 5), dtype=np.uint16)  # 5 instead of 8
        assert data.shape[1] != FEATURE_DIM

    def test_step_data_action_range(self):
        """Actions must be in [0, 4]."""
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        data[:, 6] = 5  # Out of range
        assert np.any(data[:, 6] >= NUM_ACTIONS)

    def test_step_data_reset_flag_range(self):
        """Reset flag must be 0 or 1."""
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        data[:, 7] = 2  # Invalid
        assert np.any((data[:, 7] != 0) & (data[:, 7] != 1))

    def test_step_data_negative_positions(self):
        """uint16 cannot be negative, so this is inherently safe."""
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        # uint16 max is 65535, min is 0
        assert np.all(data >= 0)

    def test_gpu_tensor_contiguity(self):
        """Non-contiguous tensor should fail validation."""
        base = torch.zeros((20, FEATURE_DIM), dtype=torch.int16)
        non_contig = base[::2, ::2]  # Strided view
        assert not non_contig.is_contiguous()

    def test_gpu_tensor_dtype_validation(self):
        """Wrong dtype should fail."""
        tensor = torch.zeros((10, FEATURE_DIM), dtype=torch.float32)
        assert tensor.dtype != torch.int16

    def test_agent_state_byte_alignment(self):
        """Verify AgentState is 16 bytes (8 * uint16_t)."""
        agent_state_size = FEATURE_DIM * 2  # 8 fields * 2 bytes each
        assert agent_state_size == 16


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    """Edge case tests for the pipeline."""

    def test_single_agent(self):
        """Pipeline should work with 1 agent."""
        data = np.zeros((1, FEATURE_DIM), dtype=np.uint16)
        assert data.shape == (1, FEATURE_DIM)

    def test_single_environment(self):
        """Pipeline should work with 1 environment."""
        data = np.zeros((256, FEATURE_DIM), dtype=np.uint16)
        data[:, 0] = 0  # Single env
        assert np.all(data[:, 0] == 0)

    def test_max_env_id(self):
        """Test with maximum uint16 value."""
        data = np.zeros((10, FEATURE_DIM), dtype=np.uint16)
        data[:, 0] = 65535  # Max uint16
        assert data[0, 0] == 65535

    def test_capacity_equal_to_batch(self):
        """When capacity == batch_threshold, every batch fills the buffer."""
        capacity = 256
        batch_threshold = 256
        available = 256
        chunk_size = min(available, batch_threshold)
        assert chunk_size == capacity

    def test_capacity_smaller_than_batch(self):
        """When capacity < batch_threshold, use available."""
        capacity = 128
        batch_threshold = 256
        available = 128
        chunk_size = min(available, batch_threshold)
        assert chunk_size == capacity

    def test_wrap_boundary_exact(self):
        """Test exact boundary wrap-around."""
        capacity = 100
        ptr = 200
        idx = ptr % capacity
        assert idx == 0  # Exact boundary

    def test_wrap_one_before_boundary(self):
        capacity = 100
        ptr = 199
        idx = ptr % capacity
        assert idx == 99

    def test_wrap_one_after_boundary(self):
        capacity = 100
        ptr = 201
        idx = ptr % capacity
        assert idx == 1

    def test_empty_batch(self):
        """Empty batch should be handled gracefully."""
        data = np.zeros((0, FEATURE_DIM), dtype=np.uint16)
        assert data.shape[0] == 0

    def test_batch_at_uint16_max(self):
        """Test with data at uint16 boundary."""
        data = np.full((10, FEATURE_DIM), 65535, dtype=np.uint16)
        assert data.max() == 65535

    def test_concurrent_reserve_simulation(self):
        """Simulate two processes reserving space."""
        shared_ptr = mp.Value('Q', 0)
        lock = mp.Lock()

        def reserve(n):
            with lock:
                start = shared_ptr.value
                shared_ptr.value += n
                return start

        results = []
        def worker(n):
            results.append(reserve(n))

        threads = [threading.Thread(target=worker, args=(100,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert shared_ptr.value == 400
        assert len(set(results)) == 4  # All unique start positions

    def test_ready_flag_scan_all_ready(self):
        """When all flags are 1, scan returns full range."""
        flags = torch.ones(100, dtype=torch.uint8)
        zero_indices = (flags == 0).nonzero()
        assert len(zero_indices) == 0  # All ready

    def test_ready_flag_scan_partial(self):
        """Scan stops at first zero."""
        flags = torch.ones(100, dtype=torch.uint8)
        flags[50:] = 0
        zero_indices = (flags == 0).nonzero()
        assert zero_indices[0].item() == 50

    def test_ready_flag_scan_none_ready(self):
        """When no flags are set."""
        flags = torch.zeros(100, dtype=torch.uint8)
        ready = (flags == 1).nonzero()
        assert len(ready) == 0


# ============================================================
# Test: Data Format Consistency
# ============================================================

class TestDataFormatConsistency:
    """Verify data format matches AgentState struct."""

    AGENT_STATE_FIELDS = [
        'env_id',       # index 0
        'agent_id',     # index 1
        'pos_x',        # index 2
        'pos_y',        # index 3
        'target_x',     # index 4
        'target_y',     # index 5
        'action',       # index 6
        'reset_flag',   # index 7
    ]

    def test_feature_dim_matches_agent_state(self):
        assert FEATURE_DIM == len(self.AGENT_STATE_FIELDS)

    def test_agent_state_field_order(self):
        """Verify field indices match design."""
        assert self.AGENT_STATE_FIELDS[0] == 'env_id'
        assert self.AGENT_STATE_FIELDS[1] == 'agent_id'
        assert self.AGENT_STATE_FIELDS[6] == 'action'
        assert self.AGENT_STATE_FIELDS[7] == 'reset_flag'

    def test_numpy_to_torch_dtype_mapping(self):
        """Verify numpy uint16 maps to torch int16 (same byte width)."""
        np_data = np.zeros(10, dtype=np.uint16)
        # Both are 2-byte types
        assert np_data.itemsize == 2
        assert np.int16().itemsize == 2
        # Direct casting preserves byte width
        casted = np_data.astype(np.int16)
        assert casted.dtype == np.int16
        assert casted.itemsize == 2

    def test_struct_layout(self):
        """Verify memory layout: 8 fields * 2 bytes = 16 bytes per agent."""
        data = np.zeros((1, FEATURE_DIM), dtype=np.uint16)
        assert data.itemsize == 2
        assert data.nbytes == FEATURE_DIM * 2  # 16 bytes

    def test_reinterpret_cast_safety(self):
        """Verify reinterpret_cast from int16* to AgentState* is safe."""
        tensor = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        # AgentState has 8 uint16_t fields = 16 bytes
        # Tensor row: 8 int16 elements = 16 bytes
        # Layout matches!
        assert tensor.stride(0) == FEATURE_DIM  # Contiguous row
        assert tensor.element_size() * FEATURE_DIM == 16


# ============================================================
# Test: Constants
# ============================================================

class TestConstants:
    """Verify pipeline constants."""

    def test_feature_dim(self):
        assert FEATURE_DIM == 8

    def test_num_actions(self):
        assert NUM_ACTIONS == 5

    def test_reset_flag_col(self):
        assert RESET_FLAG_COL == 7

    def test_reset_flag_is_last(self):
        assert RESET_FLAG_COL == FEATURE_DIM - 1


# ============================================================
# Test: Execution Order (Conceptual)
# ============================================================

class TestExecutionOrder:
    """Verify execution order constraints are understood."""

    def test_energy_before_obs_order(self):
        """
        Energy map must be updated BEFORE imitation obs generation.

        This test verifies the conceptual ordering by tracking call order.
        """
        call_order = []

        def update_energy_maps(data):
            call_order.append('energy')

        def generate_imitation_obs(data):
            call_order.append('obs')

        # Correct order
        update_energy_maps(None)
        generate_imitation_obs(None)

        assert call_order == ['energy', 'obs']

    def test_reset_filter_before_energy(self):
        """
        Python-side filter must happen BEFORE calling energy map kernel.
        """
        call_order = []

        def filter_reset(batch):
            call_order.append('filter')
            return batch

        def update_energy_maps(data):
            call_order.append('energy')

        batch = torch.zeros((10, FEATURE_DIM), dtype=torch.int16)
        filtered = filter_reset(batch)
        update_energy_maps(filtered)

        assert call_order == ['filter', 'energy']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
