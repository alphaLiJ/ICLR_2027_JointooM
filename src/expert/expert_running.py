import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_DEFAULT_CPU_THREADS_PER_PROCESS = os.environ.get("MAPF_CPU_THREADS_PER_PROCESS", "1")
for _env_var in CPU_THREAD_ENV_VARS:
    os.environ.setdefault(_env_var, _DEFAULT_CPU_THREADS_PER_PROCESS)
os.environ.setdefault("LACAM_NO_MULTI_THREAD", "1")

import multiprocessing as mp
import sys
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
import torch

from expert._lagat_imports import ensure_lagat_on_path, repo_root_from_module_file

PROJECT_ROOT = repo_root_from_module_file(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MAGAT_PLUS_PARENT = ensure_lagat_on_path(PROJECT_ROOT)

from expert.fixed_magat_plus_runtime import (
    OBS_RADIUS,
    OBS_DIAM,
    PYG_NUM_CHANNELS,
    MAGAT_EMBEDDING_SIZE,
    MAGAT_NUM_GNN_LAYERS,
    MAGAT_NUM_ATTENTION_HEADS,
    MAGAT_LR,
    RuntimePyGBatch,
    PyGBatchBuilder,
    build_fixed_magat_args,
    MAGATRuntimeAdapter,
)

# ============================================================
# Constants
# ============================================================
FEATURE_DIM = 8
NUM_ACTIONS = 5
RESET_FLAG_COL = 7
DEFAULT_EXPERT_TIMEOUTS = (1.0, 5.0, 10.0, 30.0)


def resolve_cpu_threads_per_process(default: int = 1) -> int:
    raw_value = os.environ.get("MAPF_CPU_THREADS_PER_PROCESS")
    if raw_value is None:
        return max(1, int(default))
    try:
        return max(1, int(raw_value))
    except ValueError:
        return max(1, int(default))


def configure_cpu_thread_limits(num_threads: int | None = None) -> int:
    resolved_num_threads = resolve_cpu_threads_per_process(
        default=1 if num_threads is None else num_threads
    )
    resolved_str = str(resolved_num_threads)
    for env_var in CPU_THREAD_ENV_VARS:
        os.environ.setdefault(env_var, resolved_str)
    os.environ.setdefault("LACAM_NO_MULTI_THREAD", "1")
    try:
        torch.set_num_threads(resolved_num_threads)
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.set_num_interop_threads(resolved_num_threads)
    except (AttributeError, RuntimeError):
        pass
    return resolved_num_threads


configure_cpu_thread_limits()


# ============================================================
# Expert Policy Protocol
# ============================================================
class ExpertPolicy(ABC):
    @abstractmethod
    def act(self, observations) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset_states(self, env) -> None:
        raise NotImplementedError


class RandomExpertPolicy(ExpertPolicy):
    def __init__(self, num_actions: int = NUM_ACTIONS, seed: int = 42):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def act(self, observations) -> np.ndarray:
        return self.rng.integers(0, self.num_actions, size=len(observations), dtype=np.uint16)

    def reset_states(self, env) -> None:
        pass


class LacamExpertPolicy(ExpertPolicy):
    """LaCAM expert algorithm adapter."""

    def __init__(self, time_limit=60.0, timeouts=None, lib_path=None):
        from real_expert_alg.lacam.inference import LacamInference, LacamInferenceConfig

        if lib_path is None:
            from real_expert_alg.lacam.inference import lacam_library_path

            lib_path = str(lacam_library_path())
        cfg = LacamInferenceConfig(
            time_limit=time_limit,
            timeouts=resolve_expert_timeouts(timeouts),
            lacam_lib_path=lib_path,
        )
        self.base = LacamInference(cfg)
        self._max_episode_steps = None

    def act(self, observations) -> np.ndarray:
        actions_list = self.base.act(observations)
        return np.array(actions_list, dtype=np.uint16)

    def reset_states(self, env) -> None:
        self.base.reset_states()
        try:
            self._max_episode_steps = env.grid.config.max_episode_steps
        except (AttributeError, TypeError):
            self._max_episode_steps = None


def resolve_expert_timeouts(expert_timeouts=None) -> list[float]:
    return list(expert_timeouts or DEFAULT_EXPERT_TIMEOUTS)


# ============================================================
# Helpers
# ============================================================
def _normalize_obs_coords_array(observations):
    pos = np.array([obs["global_xy"] for obs in observations], dtype=np.int32)
    tgt = np.array([obs["global_target_xy"] for obs in observations], dtype=np.int32)
    pos -= OBS_RADIUS
    tgt -= OBS_RADIUS
    return pos, tgt


def initial_refresh_flags(num_agents: int) -> np.ndarray:
    return np.ones(num_agents, dtype=np.uint16)


def no_refresh_flags(num_agents: int) -> np.ndarray:
    return np.zeros(num_agents, dtype=np.uint16)


def refresh_flags_from_env(env, num_agents: int) -> np.ndarray:
    flags = np.asarray(
        getattr(env.unwrapped, "was_on_goal", np.zeros(num_agents, dtype=bool)),
        dtype=np.uint16,
    )
    flags = flags.reshape(-1)
    if flags.shape[0] != num_agents:
        raise ValueError(
            f"refresh flag shape mismatch: expected {num_agents}, got {flags.shape[0]}"
        )
    return flags


def normalize_refresh_flags(refresh_flag, num_agents: int) -> np.ndarray:
    flags = np.asarray(refresh_flag, dtype=np.uint16)
    if flags.ndim == 0:
        return np.full(num_agents, int(flags.item()), dtype=np.uint16)
    flags = flags.reshape(-1)
    if flags.shape[0] != num_agents:
        raise ValueError(
            f"refresh flag shape mismatch: expected {num_agents}, got {flags.shape[0]}"
        )
    return flags


# ============================================================
# DataValidator
# ============================================================
class DataValidator:
    @staticmethod
    def validate_step_data(data: np.ndarray) -> bool:
        if data.dtype != np.uint16:
            raise ValueError(f"dtype must be np.uint16, got {data.dtype}")
        if data.ndim != 2 or data.shape[1] != FEATURE_DIM:
            raise ValueError(f"shape must be [N, {FEATURE_DIM}], got {data.shape}")
        if np.any(data[:, 6] >= NUM_ACTIONS):
            raise ValueError(f"actions out of range [0,{NUM_ACTIONS - 1}]")
        if np.any((data[:, RESET_FLAG_COL] != 0) & (data[:, RESET_FLAG_COL] != 1)):
            raise ValueError("reset_flag must be 0 or 1")
        return True

    @staticmethod
    def validate_gpu_tensor(tensor: torch.Tensor) -> bool:
        if tensor.dtype != torch.int16:
            raise ValueError(f"dtype must be torch.int16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")
        if not tensor.is_cuda:
            raise ValueError("tensor must be on CUDA device")
        if tensor.dim() != 2 or tensor.shape[1] != FEATURE_DIM:
            raise ValueError(f"shape must be [N, {FEATURE_DIM}], got {tensor.shape}")
        return True

    @staticmethod
    def check_agent_state_alignment(tensor: torch.Tensor) -> bool:
        expected_bytes = FEATURE_DIM * 2
        if tensor.element_size() * tensor.shape[1] != expected_bytes:
            raise ValueError(f"Row size mismatch: expected {expected_bytes} bytes")
        return True


def load_maps_from_yaml(file_path):
    from expert.profiled_topology_loader import load_named_topology_strings_from_yaml

    return load_named_topology_strings_from_yaml(file_path)


def put_maps_into_registry(maps_path, max_maps=None):
    from pogema.grid_registry import GRID_STR_REGISTRY, RegisteredGrid, in_registry
    from expert.profiled_topology_loader import parse_topology_catalog_grid_string

    maps_dict = load_maps_from_yaml(maps_path)
    loaded = 0
    for name, map_str in maps_dict.items():
        if max_maps and loaded >= max_maps:
            break
        if in_registry(name):
            loaded += 1
            continue
        try:
            parsed = parse_topology_catalog_grid_string(map_str)
            RegisteredGrid(
                name=name,
                obstacles=parsed.obstacles,
                semantic_positions=parsed.semantic_positions,
            )
        except (ValueError, KeyError, IndexError):
            pass  # Skip maps with unsupported symbols
        else:
            loaded += 1


# ============================================================
# RingBuffer - CPU pinned shared memory ring buffer
# ============================================================
class RingBuffer:
    def __init__(
        self,
        capacity: int,
        feature_dim: int = FEATURE_DIM,
        *,
        num_envs: int | None = None,
        agents_per_env: int | None = None,
    ):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_envs = int(num_envs) if num_envs is not None else None
        self.agents_per_env = int(agents_per_env) if agents_per_env is not None else None
        self.stage_mode = self.num_envs is not None and self.agents_per_env is not None
        self.stage_rows = None
        self.num_stage_slots = None
        if self.stage_mode:
            if self.num_envs <= 0:
                raise ValueError(f"num_envs must be positive, got {self.num_envs}")
            if self.agents_per_env <= 0:
                raise ValueError(
                    f"agents_per_env must be positive, got {self.agents_per_env}"
                )
            self.stage_rows = self.num_envs * self.agents_per_env
            if capacity % self.stage_rows != 0:
                raise ValueError(
                    "stage-based ring capacity must be divisible by one frontier batch: "
                    f"capacity={capacity}, stage_rows={self.stage_rows}"
                )
            self.num_stage_slots = capacity // self.stage_rows
            if self.num_stage_slots <= 0:
                raise ValueError(
                    f"stage-based ring requires at least one stage slot, got capacity={capacity}"
                )
        self.cpu_buffer = torch.zeros(
            (capacity, feature_dim), dtype=torch.int16
        ).pin_memory().share_memory_()
        self.ready_flags = mp.Array("B", capacity)  # uint8, cross-process safe
        self.shared_reserve_ptr = mp.Value("Q", 0)
        self.shared_compute_ptr = mp.Value("Q", 0)
        self.reserve_lock = mp.Lock()
        if self.stage_mode:
            self.env_write_seq = mp.Array("Q", self.num_envs)
            self.stage_ready_counts = mp.Array("I", self.num_stage_slots)
            self.stage_seq_tags = mp.Array("q", [-1] * self.num_stage_slots)
        else:
            self.env_write_seq = None
            self.stage_ready_counts = None
            self.stage_seq_tags = None

    def _stage_row_bounds(self, stage_seq: int, env_id: int | None = None) -> tuple[int, int]:
        if not self.stage_mode:
            raise RuntimeError("stage row bounds are only available in stage mode")
        stage_idx = int(stage_seq) % self.num_stage_slots
        row_start = stage_idx * self.stage_rows
        if env_id is not None:
            row_start += int(env_id) * self.agents_per_env
            return row_start, row_start + self.agents_per_env
        return row_start, row_start + self.stage_rows

    def _validate_stage_block(self, data: np.ndarray) -> tuple[int, torch.Tensor]:
        if data.ndim != 2 or data.shape[1] != self.feature_dim:
            raise ValueError(
                f"shape must be [N, {self.feature_dim}] in stage mode, got {data.shape}"
            )
        if data.shape[0] != self.agents_per_env:
            raise ValueError(
                "stage-based ring expects one full env block per write: "
                f"expected {self.agents_per_env} rows, got {data.shape[0]}"
            )
        env_ids = data[:, 0]
        env_id = int(env_ids[0])
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(
                f"env_id out of range for stage-based ring: env_id={env_id}, num_envs={self.num_envs}"
            )
        if not np.all(env_ids == env_id):
            raise ValueError("stage-based ring requires every written block to contain a single env_id")
        expected_agent_ids = np.arange(self.agents_per_env, dtype=np.uint16)
        if not np.array_equal(data[:, 1], expected_agent_ids):
            raise ValueError(
                "stage-based ring expects agent ids to be contiguous 0..agents_per_env-1"
            )
        return env_id, torch.from_numpy(data.astype(np.int16, copy=False))

    def _reserve_stage_block(self, env_id: int) -> tuple[int, int, int]:
        """Reserve the next fixed stage slot for one logical environment.

        Reservation installs only the sequence tag.  A stage remains invisible
        to the consumer until ``_publish_stage_block`` has been called for all
        logical environments.  Keeping this boundary explicit also makes the
        reserve-before-publish crash window directly testable.
        """
        while True:
            with self.reserve_lock:
                write_seq = int(self.env_write_seq[env_id])
                consumed_seq = self.shared_compute_ptr.value // self.stage_rows
                if write_seq - consumed_seq >= self.num_stage_slots:
                    row_start = row_end = None
                else:
                    stage_idx = write_seq % self.num_stage_slots
                    stage_tag = int(self.stage_seq_tags[stage_idx])
                    if stage_tag not in (-1, write_seq) and stage_tag >= consumed_seq:
                        row_start = row_end = None
                    else:
                        if stage_tag != write_seq:
                            self.stage_seq_tags[stage_idx] = write_seq
                            self.stage_ready_counts[stage_idx] = 0
                        row_start, row_end = self._stage_row_bounds(write_seq, env_id=env_id)
                        break
            time.sleep(0.0005)
        return write_seq, row_start, row_end

    def _publish_stage_block(self, env_id: int, write_seq: int) -> None:
        """Publish one fully written environment block into its stage."""
        with self.reserve_lock:
            stage_idx = write_seq % self.num_stage_slots
            self.stage_ready_counts[stage_idx] += 1
            self.env_write_seq[env_id] = write_seq + 1
            self.shared_reserve_ptr.value += self.agents_per_env

    def _stage_reserve_and_write(self, data: np.ndarray):
        env_id, trans_tensor = self._validate_stage_block(data)
        write_seq, row_start, row_end = self._reserve_stage_block(env_id)

        self.cpu_buffer[row_start:row_end].copy_(trans_tensor)
        self.ready_flags[row_start:row_end] = [1] * self.agents_per_env
        self._publish_stage_block(env_id, write_seq)
        return (row_start, row_end)

    def reserve_and_write(self, data: np.ndarray):
        if self.stage_mode:
            return self._stage_reserve_and_write(data)

        N = data.shape[0]
        if N == 0:
            return (0, 0)
        if N > self.capacity:
            raise ValueError(
                f"cannot write chunk of size {N} into ring buffer capacity {self.capacity}"
            )

        trans_tensor = torch.from_numpy(data.astype(np.int16, copy=False))
        while True:
            with self.reserve_lock:
                start_ptr = self.shared_reserve_ptr.value
                compute_ptr = self.shared_compute_ptr.value
                unread = start_ptr - compute_ptr
                if unread + N <= self.capacity:
                    self.shared_reserve_ptr.value += N
                    break
            time.sleep(0.0005)

        idx_start = start_ptr % self.capacity
        idx_end = (idx_start + N) % self.capacity
        if idx_start < idx_end or idx_end == 0:
            logical_end = self.capacity if idx_end == 0 else idx_end
            self.cpu_buffer[idx_start:logical_end].copy_(trans_tensor)
            self.ready_flags[idx_start:logical_end] = [1] * N
            return (idx_start, logical_end)

        first_len = self.capacity - idx_start
        self.cpu_buffer[idx_start:self.capacity].copy_(trans_tensor[:first_len])
        self.cpu_buffer[0:idx_end].copy_(trans_tensor[first_len:])
        self.ready_flags[idx_start:self.capacity] = [1] * first_len
        self.ready_flags[0:idx_end] = [1] * idx_end
        return (idx_start, idx_end)

    def is_stage_ready(self, stage_seq: int) -> bool:
        if not self.stage_mode:
            return False
        stage_idx = int(stage_seq) % self.num_stage_slots
        with self.reserve_lock:
            return (
                int(self.stage_seq_tags[stage_idx]) == int(stage_seq)
                and int(self.stage_ready_counts[stage_idx]) >= self.num_envs
            )

    def release_stage(self, stage_seq: int) -> None:
        if not self.stage_mode:
            return
        row_start, row_end = self._stage_row_bounds(stage_seq)
        stage_idx = int(stage_seq) % self.num_stage_slots
        with self.reserve_lock:
            self.stage_ready_counts[stage_idx] = 0
            self.stage_seq_tags[stage_idx] = -1
        self.ready_flags[row_start:row_end] = [0] * self.stage_rows

    def scan_ready_range(self, start_ptr: int, max_scan: int) -> int:
        idx_start = start_ptr % self.capacity
        for i in range(max_scan):
            if self.ready_flags[idx_start + i] == 0:
                return i
        return max_scan

    def clear_flags(self, start: int, end: int) -> None:
        self.ready_flags[start:end] = [0] * (end - start)


# ============================================================
# DMAWorker - Async CPU→GPU transfer
# ============================================================
class DMAWorker:
    def __init__(self, ring_buffer: "RingBuffer", gpu_buffer: torch.Tensor, batch_threshold=4096, device="cuda:0"):
        self.ring_buffer = ring_buffer
        self.gpu_buffer = gpu_buffer
        self.batch_threshold = batch_threshold
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.dma_event = torch.cuda.Event(enable_timing=False)
        self.dma_read_ptr = 0
        self._running = False

    def run_once(self):
        if self.ring_buffer.stage_mode:
            stage_seq = self.dma_read_ptr // self.ring_buffer.stage_rows
            if not self.ring_buffer.is_stage_ready(stage_seq):
                return None
            idx_start, idx_end = self.ring_buffer._stage_row_bounds(stage_seq)
            with torch.cuda.stream(self.copy_stream):
                self.gpu_buffer[idx_start:idx_end].copy_(
                    self.ring_buffer.cpu_buffer[idx_start:idx_end],
                    non_blocking=True,
                )
                self.dma_event.record(self.copy_stream)
            self.dma_read_ptr += self.ring_buffer.stage_rows
            return (idx_start, idx_end)

        current_reserve = self.ring_buffer.shared_reserve_ptr.value
        available = current_reserve - self.dma_read_ptr
        if available <= 0:
            return None
        idx_start = self.dma_read_ptr % self.ring_buffer.capacity
        max_scan = min(
            available,
            self.ring_buffer.capacity - idx_start,
            self.ring_buffer.capacity // 4,
        )
        ready = self.ring_buffer.scan_ready_range(self.dma_read_ptr, max_scan)
        if ready <= 0:
            return None
        idx_end = idx_start + ready
        with torch.cuda.stream(self.copy_stream):
            self.gpu_buffer[idx_start:idx_end].copy_(
                self.ring_buffer.cpu_buffer[idx_start:idx_end],
                non_blocking=True,
            )
            self.dma_event.record(self.copy_stream)
        self.ring_buffer.clear_flags(idx_start, idx_end)
        self.dma_read_ptr += ready
        return (idx_start, idx_end)

    def release_completed_stage(self, stage_seq: int) -> None:
        if not self.ring_buffer.stage_mode:
            return
        self.ring_buffer.release_stage(stage_seq)
        self.ring_buffer.shared_compute_ptr.value = (int(stage_seq) + 1) * self.ring_buffer.stage_rows

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return t

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            if self.run_once() is None:
                time.sleep(0.0005)


# ============================================================
# GPUComputeHandler - Kernel execution orchestrator
# ============================================================
class GPUComputeHandler:
    def __init__(self, simulator):
        self.simulator = simulator

    def process_batch(self, full_batch: torch.Tensor, agents_to_update: torch.Tensor):
        if agents_to_update.shape[0] > 0:
            self.simulator.update_derived_state(agents_to_update, agents_to_update.shape[0])
        self.simulator.refresh_compact_state_from_raw_batch(full_batch)

    def filter_reset_agents(self, full_batch: torch.Tensor) -> torch.Tensor:
        return full_batch[full_batch[:, RESET_FLAG_COL] != 0]


# ============================================================
# ExpertWorker - Single expert subprocess worker
# ============================================================
class ExpertWorker:
    def __init__(self, expert_id: int, policy: ExpertPolicy, pipeline: "ExtremeMAPFPipeline", grid_config=None):
        self.expert_id = expert_id
        self.policy = policy
        self.pipeline = pipeline
        self.grid_config = grid_config

    def run(self):
        from pogema import pogema_v0

        env = pogema_v0(grid_config=self.grid_config)
        observations, _ = env.reset()
        self.policy.reset_states(env)
        refresh_flags = initial_refresh_flags(len(observations))

        while True:
            actions = self.policy.act(observations)
            step_data = self.build_step_data(observations, actions, reset_flag=refresh_flags)
            self.pipeline.reserve_and_write(step_data.copy())

            observations, _, terminated, truncated, _ = env.step(actions)
            if all(terminated) or all(truncated):
                observations, _ = env.reset()
                self.policy.reset_states(env)
                refresh_flags = initial_refresh_flags(len(observations))
            else:
                refresh_flags = no_refresh_flags(len(observations))

    def run_episode(self, env, observations):
        return self.policy.act(observations)

    def build_step_data(self, observations, actions: np.ndarray, reset_flag=0) -> np.ndarray:
        n = len(observations)
        data = np.zeros((n, FEATURE_DIM), dtype=np.uint16)
        data[:, 0] = self.expert_id
        data[:, 1] = np.arange(n, dtype=np.uint16)
        pos, tgt = _normalize_obs_coords_array(observations)
        data[:, 2] = pos[:, 0]
        data[:, 3] = pos[:, 1]
        data[:, 4] = tgt[:, 0]
        data[:, 5] = tgt[:, 1]
        data[:, 6] = actions
        data[:, 7] = normalize_refresh_flags(reset_flag, n)
        return data


# ============================================================
# ExtremeMAPFPipeline - Main orchestrator
# ============================================================
class ExtremeMAPFPipeline:
    def __init__(
        self,
        simulator,
        capacity=16384 * 128,
        batch_threshold=16384,
        feature_dim=FEATURE_DIM,
        device="cuda:0",
    ):
        self.device = torch.device(device)
        self.capacity = capacity
        self.batch_threshold = batch_threshold
        self.feature_dim = feature_dim
        self.cuda_simulator = simulator

        self.ring_buffer = None
        self.gpu_buffer = None
        self.gpu_handler = None
        self.dma_worker = None

        self.dma_read_ptr = 0
        self.compute_ptr = 0
        self.dma_event = None
        self.compute_stream = None
        self.graph_rows = None
        self.stage_batch_buffer = None
        self.expected_agent_ids_cache = {}
        self.stage_mode = False
        self.agents_per_env = None
        self.num_envs = None

    def _infer_stage_layout(self) -> tuple[int | None, int | None]:
        pyg_ptr = getattr(self.cuda_simulator, "pyg_ptr", None)
        if pyg_ptr is None or pyg_ptr.shape[0] <= 1:
            return None, None
        agents_per_env = int((pyg_ptr[1] - pyg_ptr[0]).item())
        num_envs = int(pyg_ptr.shape[0] - 1)
        if agents_per_env <= 0 or num_envs <= 0:
            return None, None
        return agents_per_env, num_envs

    def initialize(self):
        self.agents_per_env, self.num_envs = self._infer_stage_layout()
        self.stage_mode = self.agents_per_env is not None and self.num_envs is not None
        self.ring_buffer = RingBuffer(
            self.capacity,
            self.feature_dim,
            num_envs=self.num_envs,
            agents_per_env=self.agents_per_env,
        )
        self.gpu_buffer = torch.zeros(
            (self.capacity, self.feature_dim), dtype=torch.int16, device=self.device
        )
        stage_rows = self.ring_buffer.stage_rows if self.ring_buffer.stage_mode else self.batch_threshold
        self.stage_batch_buffer = torch.empty(
            (stage_rows, self.feature_dim),
            dtype=torch.int16,
            device=self.device,
        )
        self.gpu_handler = GPUComputeHandler(self.cuda_simulator)
        self.dma_worker = DMAWorker(
            self.ring_buffer,
            self.gpu_buffer,
            self.batch_threshold,
            str(self.device),
        )
        self.dma_event = self.dma_worker.dma_event
        self.compute_stream = torch.cuda.Stream(device=self.device, priority=-1)
        self.graph_rows = int(self.cuda_simulator.pyg_batch.shape[0])

    def reserve_and_write(self, transitions_numpy):
        if self.ring_buffer is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.ring_buffer.reserve_and_write(transitions_numpy)

    def dma_worker_loop(self):
        if self.dma_worker is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        self.dma_worker.start()

    def _expected_agent_ids(self, agents_per_env: int, device: torch.device) -> torch.Tensor:
        cache_key = (agents_per_env, device.type, device.index)
        expected = self.expected_agent_ids_cache.get(cache_key)
        if expected is None:
            expected = torch.arange(agents_per_env, dtype=torch.int16, device=device)
            self.expected_agent_ids_cache[cache_key] = expected
        return expected

    def _validate_env_aligned_batch(self, raw_batch: torch.Tensor, agents_per_env: int, num_envs: int) -> None:
        if raw_batch.shape[0] != agents_per_env * num_envs:
            raise ValueError(
                "env-aligned batch size mismatch: "
                f"expected {agents_per_env * num_envs}, got {raw_batch.shape[0]}"
            )
        block_view = raw_batch.view(num_envs, agents_per_env, self.feature_dim)
        block_env_ids = block_view[:, :, 0].to(torch.int64)
        expected_env_ids = torch.arange(num_envs, dtype=torch.int64, device=raw_batch.device)
        if not torch.all(block_env_ids == expected_env_ids.view(num_envs, 1)):
            raise ValueError(
                "stage-based frontier batch must be strict env-major with one block per env"
            )
        expected_agent_ids = self._expected_agent_ids(agents_per_env, raw_batch.device)
        block_agent_ids = block_view[:, :, 1]
        if not torch.all(block_agent_ids == expected_agent_ids.unsqueeze(0)):
            raise ValueError(
                "stage-based frontier batch expects agent ids to be contiguous 0..agents_per_env-1"
            )

    def has_env_aligned_batch(self, num_envs: int) -> bool:
        if self.ring_buffer is None:
            return False
        if self.ring_buffer.stage_mode:
            stage_seq = self.compute_ptr // self.ring_buffer.stage_rows
            return self.ring_buffer.is_stage_ready(stage_seq)
        return False

    def extract_env_aligned_batch(self, chunk_size: int, agents_per_env: int, num_envs: int) -> torch.Tensor:
        if self.ring_buffer is None:
            return None
        if self.ring_buffer.stage_mode:
            stage_seq = self.compute_ptr // self.ring_buffer.stage_rows
            if not self.ring_buffer.is_stage_ready(stage_seq):
                return None
            chunk_size = agents_per_env * num_envs
            dma_available = (
                0
                if self.dma_worker is None
                else int(self.dma_worker.dma_read_ptr - self.compute_ptr)
            )
            if dma_available < chunk_size:
                return None
            row_start, row_end = self.ring_buffer._stage_row_bounds(stage_seq)
            if self.stage_batch_buffer is None or self.stage_batch_buffer.shape[0] < chunk_size:
                self.stage_batch_buffer = torch.empty(
                    (chunk_size, self.feature_dim),
                    dtype=self.gpu_buffer.dtype,
                    device=self.gpu_buffer.device,
                )
            raw_batch = self.stage_batch_buffer[:chunk_size]
            raw_batch.copy_(self.gpu_buffer[row_start:row_end], non_blocking=False)
            self._validate_env_aligned_batch(raw_batch, agents_per_env, num_envs)
            self.compute_ptr += chunk_size
            self.dma_worker.release_completed_stage(stage_seq)
            return raw_batch
        return None

    def train_step(self, model=None, optimizer=None, runtime: MAGATRuntimeAdapter = None):
        if self.dma_worker is None:
            return None
        agents_per_env = (
            self.cuda_simulator.pyg_ptr.shape[0] > 1
            and int((self.cuda_simulator.pyg_ptr[1] - self.cuda_simulator.pyg_ptr[0]).item())
            or 0
        )
        num_envs = int(self.cuda_simulator.pyg_ptr.shape[0] - 1)

        self.dma_event.synchronize()
        raw_batch = self.extract_env_aligned_batch(
            agents_per_env * num_envs,
            agents_per_env=agents_per_env,
            num_envs=num_envs,
        )
        if raw_batch is None:
            return None

        agents_to_update = self.gpu_handler.filter_reset_agents(raw_batch)
        self.gpu_handler.process_batch(raw_batch, agents_to_update)

        if runtime is not None:
            return runtime.train_step(self.cuda_simulator, raw_batch)
        if model is not None:
            loss = model(self.cuda_simulator.pyg_x)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            return loss
        return self.cuda_simulator.pyg_x

    def get_stats(self):
        reserve_ptr = self.ring_buffer.shared_reserve_ptr.value if self.ring_buffer else 0
        dma_read_ptr = self.dma_worker.dma_read_ptr if self.dma_worker is not None else self.dma_read_ptr
        stats = {
            "capacity": self.capacity,
            "reserve_ptr": reserve_ptr,
            "dma_read_ptr": dma_read_ptr,
            "compute_ptr": self.compute_ptr,
        }
        if self.ring_buffer is not None and self.ring_buffer.stage_mode:
            stats.update(
                {
                    "num_stage_slots": self.ring_buffer.num_stage_slots,
                    "stage_rows": self.ring_buffer.stage_rows,
                    "stage_ready_counts": list(self.ring_buffer.stage_ready_counts),
                    "stage_seq_tags": list(self.ring_buffer.stage_seq_tags),
                    "env_write_stage": list(self.ring_buffer.env_write_seq),
                }
            )
        return stats

    def shutdown(self):
        if self.dma_worker is not None:
            self.dma_worker.stop()

    def __del__(self):
        self.shutdown()


# =================================================================
# Env Worker process helper
# =================================================================
def run_expert_algorithm_optimized(
    expert,
    env_id,
    pipeline_instance: ExtremeMAPFPipeline,
    env=None,
    grid_config=None,
):
    if env is None:
        from pogema import pogema_v0

        env = pogema_v0(grid_config=grid_config)

    observations, _ = env.reset()
    expert.reset_states(env)

    num_agents = len(observations)
    feature_dim = pipeline_instance.feature_dim
    step_data_view = np.zeros((num_agents, feature_dim), dtype=np.uint16)
    step_data_view[:, 0] = env_id
    step_data_view[:, 1] = np.arange(num_agents)
    refresh_flags = initial_refresh_flags(num_agents)

    while True:
        actions = expert.act(observations)
        positions, target_xy = _normalize_obs_coords_array(observations)
        step_data_view[:, 2] = positions[:, 0]
        step_data_view[:, 3] = positions[:, 1]
        step_data_view[:, 4] = target_xy[:, 0]
        step_data_view[:, 5] = target_xy[:, 1]
        step_data_view[:, 6] = actions
        step_data_view[:, 7] = refresh_flags

        pipeline_instance.reserve_and_write(step_data_view)
        observations, _, terminated, truncated, _ = env.step(actions)

        if all(terminated) or all(truncated):
            observations, _ = env.reset()
            expert.reset_states(env)
            refresh_flags = initial_refresh_flags(len(observations))
        else:
            refresh_flags = no_refresh_flags(len(observations))


# ============================================================
# Utility / diagnostic helpers
# ============================================================
def validate_env_aligned_batch(raw_batch: torch.Tensor, agents_per_env: int, num_envs: int) -> bool:
    if raw_batch.dim() != 2 or raw_batch.shape[1] != FEATURE_DIM:
        raise ValueError(f"raw_batch must have shape [N, {FEATURE_DIM}], got {tuple(raw_batch.shape)}")
    if agents_per_env <= 0 or num_envs <= 0:
        raise ValueError(
            f"agents_per_env and num_envs must be positive, got {agents_per_env}, {num_envs}"
        )
    expected_rows = agents_per_env * num_envs
    if raw_batch.shape[0] != expected_rows:
        raise ValueError(
            f"raw_batch row count mismatch: expected {expected_rows}, got {raw_batch.shape[0]}"
        )
    block_view = raw_batch.view(num_envs, agents_per_env, FEATURE_DIM)
    expected_env_ids = torch.arange(num_envs, dtype=torch.int16, device=raw_batch.device)
    if not torch.all(block_view[:, :, 0] == expected_env_ids.view(num_envs, 1)):
        raise ValueError("raw_batch env blocks are not strict env-major order")
    expected_agent_ids = torch.arange(agents_per_env, dtype=torch.int16, device=raw_batch.device)
    if not torch.all(block_view[:, :, 1] == expected_agent_ids.unsqueeze(0)):
        raise ValueError("raw_batch agent ids are not contiguous 0..agents_per_env-1 per env block")
    return True


if __name__ == "__main__":
    import grid_world_cpp as ext

    grid = torch.zeros((1, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H), dtype=torch.int32, device="cuda")
    sim = ext.StatelessGridWorldSimulator(grid, 512, 3)
    policy = RandomExpertPolicy(seed=123)
    pipeline = ExtremeMAPFPipeline(sim)
    pipeline.initialize()
    worker = ExpertWorker(0, policy, pipeline)
    sample_obs = [
        {"global_xy": (64, 64), "global_target_xy": (70, 70)}
        for _ in range(512)
    ]
    actions = worker.run_episode(None, sample_obs)
    step_data = worker.build_step_data(sample_obs, actions, reset_flag=1)
    pipeline.reserve_and_write(step_data)
    pipeline.dma_worker.run_once()
    pipeline.dma_event.synchronize()
    pipeline.dma_read_ptr = pipeline.dma_worker.dma_read_ptr
    out = pipeline.train_step()
    print("Output shape:", tuple(out.shape))
    print("Reserve ptr:", pipeline.ring_buffer.shared_reserve_ptr.value)
    print("DMA read ptr:", pipeline.dma_worker.dma_read_ptr)
    print("Compute ptr:", pipeline.compute_ptr)
    pipeline.shutdown()
    print("Pipeline smoke test passed")

    # Topology-map smoke test
    topo_dir = os.path.join(PROJECT_ROOT, "maps", "topology-benchmarks")
    if os.path.isdir(topo_dir):
        maps = [f for f in os.listdir(topo_dir) if f.endswith(".map")]
        if maps:
            print("Found topology maps:", maps[:3])
            cfg = {"num_agents": 16, "map_name": maps[0]}
            print("Topology smoke test config:", cfg)
    else:
        print("No topology-benchmarks directory found; skipping topology smoke test")

    print("All done ✅")
