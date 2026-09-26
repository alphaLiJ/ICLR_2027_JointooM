"""Run one isolated row of the S3 allocator-level memory experiment.

One invocation executes exactly one repetition.  Campaign orchestration must
launch each invocation in a fresh process so CUDA, PyTorch, and model state are
not retained between rows.  This runner never invokes NVML or ``nvidia-smi``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MANIFEST = Path("artifacts/e2_matrix_manifest.json")
DEFAULT_MAGAT_CHECKPOINT = Path("artifacts/checkpoints/magat.pt")
DEFAULT_MAPF_GPT_CHECKPOINT = Path("artifacts/checkpoints/mapf_gpt.pt")

EXPECTED_MANIFEST_SHA256 = (
    "90cd556eed66806c96cb854e2926907921cc5340f557e2ff435c90effa27e0ed"
)
EXPECTED_MAGAT_SHA256 = (
    "ce8342ed384c63c05d76ca0bcdcb4cd503e3bae0bbfde51d364f4e00cc8939b2"
)
EXPECTED_MAPF_GPT_SHA256 = (
    "147e21df43a260cca3981b7ea3c8b6e395aee12b18f9ff6aa92bf8e25cf0913b"
)
EXPECTED_INPUT_SHA256 = {
    "matrix-a128-e0064": (
        "bddeeaeec11fc33bc4253dc870861baf3ef8ffa218fff44cfbd8fb01e24c2e55"
    ),
    "matrix-a256-e0256": (
        "8b5e7b80d05dbb91cc252ce6f40094e9a6f95c4c51f4836bd475b11fa28c057e"
    ),
    "matrix-a512-e0256": (
        "ea5a2147c8d1b4c44278560cd8d2ed759d3b5a70d641dcfcd4db80d165f9114f"
    ),
    "matrix-a512-e1024": (
        "a54c84db080e62f7aefe27d9518f3cdfae67241c45d8014cb5593375553bc905"
    ),
}


@dataclass(frozen=True)
class S3RowSpec:
    row_id: str
    backend: str
    operation: str
    num_agents: int
    num_envs: int
    scan_cell: str | None
    model_microbatch_envs: int | None = None
    model_microbatch_agents: int | None = None
    ring_depth: int | None = None
    batch_rows: int | None = None
    safety_reserve_bytes: int = 1024**3


ROW_SPECS = {
    "S3-M1": S3RowSpec(
        "S3-M1", "magat", "inference", 128, 64, "matrix-a128-e0064"
    ),
    "S3-M2F": S3RowSpec(
        "S3-M2F", "magat", "inference", 256, 256, "matrix-a256-e0256"
    ),
    "S3-M2B": S3RowSpec(
        "S3-M2B",
        "magat",
        "inference",
        256,
        256,
        "matrix-a256-e0256",
        model_microbatch_envs=16,
    ),
    "S3-M3": S3RowSpec(
        "S3-M3",
        "magat",
        "inference",
        512,
        256,
        "matrix-a512-e0256",
        model_microbatch_envs=8,
    ),
    "S3-G1": S3RowSpec(
        "S3-G1",
        "mapf_gpt",
        "inference",
        128,
        64,
        "matrix-a128-e0064",
        model_microbatch_agents=256,
    ),
    "S3-T1": S3RowSpec(
        "S3-T1",
        "magat",
        "training",
        256,
        4,
        "matrix-a256-e0256",
        ring_depth=128,
        batch_rows=1024,
    ),
    "S3-C1": S3RowSpec(
        "S3-C1", "capacity", "preflight", 512, 1024, "matrix-a512-e1024"
    ),
}


STATEFUL_CLASSIFICATIONS: dict[str, tuple[str, str]] = {
    "grid_compressed": ("static_map", "instance"),
    "free_cell_list": ("static_map", "instance"),
    "pool_ptr": ("static_map", "instance"),
    "offsets": ("static_map", "instance"),
    "counts": ("static_map", "instance"),
    "real_map_extents": ("static_map", "instance"),
    "grid_ocp": ("dynamic_simulator", "episode"),
    "cur_x": ("dynamic_simulator", "episode"),
    "cur_y": ("dynamic_simulator", "episode"),
    "goal_x": ("dynamic_simulator", "goal_epoch"),
    "goal_y": ("dynamic_simulator", "goal_epoch"),
    "agents_obs": ("dynamic_simulator", "episode"),
    "rewards": ("dynamic_simulator", "episode"),
    "actions": ("dynamic_simulator", "frontier"),
    "rng_states": ("dynamic_simulator", "run"),
    "max_free_cell_count": ("dynamic_simulator", "instance"),
    "goal_changed_flags": ("dynamic_simulator", "frontier"),
    "arrived": ("dynamic_simulator", "episode"),
    "terminated": ("dynamic_simulator", "episode"),
    "truncated": ("dynamic_simulator", "episode"),
    "step_counts": ("dynamic_simulator", "episode"),
    "goal_changed_prefix": ("compact_state", "frontier"),
    "changed_state_packed": ("compact_state", "frontier"),
    "goal_changed_count": ("compact_state", "frontier"),
    "state_packed": ("compact_state", "frontier"),
    "energy_maps": ("derived_state", "goal_epoch"),
    "pyg_x": ("magat_builder", "run"),
    "pyg_pos": ("magat_builder", "run"),
    "pyg_edge_index": ("magat_builder", "model_call"),
    "pyg_edge_attr": ("magat_builder", "model_call"),
    "pyg_batch": ("magat_builder", "run"),
    "pyg_ptr": ("magat_builder", "run"),
    "pyg_num_edges": ("magat_builder", "frontier"),
    "edge_counts": ("magat_builder", "frontier"),
    "pyg_edge_index_storage": ("magat_builder", "run"),
    "pyg_edge_attr_storage": ("magat_builder", "run"),
    "pyg_edge_prefix": ("magat_builder", "frontier"),
    "scan_temp_storage": ("magat_builder", "run"),
}

MAPF_GPT_CLASSIFICATIONS: dict[str, tuple[str, str]] = {
    "grid_compressed": ("static_map", "instance"),
    "state_packed": ("compact_state", "frontier"),
    "histories": ("mapf_gpt_builder", "run"),
    "labels": ("mapf_gpt_builder", "frontier"),
    "tokens": ("mapf_gpt_builder", "model_call"),
    "active_mask": ("mapf_gpt_builder", "frontier"),
    "cost_to_go": ("derived_state", "goal_epoch"),
    "agent_at_cell": ("mapf_gpt_builder", "frontier"),
    "goal_cache": ("mapf_gpt_builder", "goal_epoch"),
    "diagnostics": ("mapf_gpt_builder", "run"),
    "pyg_ptr": ("mapf_gpt_builder", "run"),
}


def get_row_spec(row_id: str) -> S3RowSpec:
    try:
        return ROW_SPECS[row_id]
    except KeyError as error:
        raise ValueError(f"unknown S3 row {row_id!r}") from error


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_digest(path: Path, expected: str, label: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected:
        raise RuntimeError(
            f"{label} digest mismatch for {resolved}: expected {expected}, got {observed}"
        )
    return observed


def resolve_scan_cell_input(manifest_path: Path, scan_cell: str) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = [cell for cell in manifest["a2_cells"] if cell["cell_id"] == scan_cell]
    if len(cells) != 1:
        raise ValueError(f"scan cell must resolve exactly once: {scan_cell}")
    pool_name = cells[0]["pool_name"]
    entries = [
        entry
        for entry in manifest["entries"]
        if entry.get("pool_name") == pool_name and entry.get("role") == "a2_pool"
    ]
    if len(entries) != 1:
        raise ValueError(f"pool must resolve exactly once: {pool_name}")
    path = Path(entries[0]["frozen_batch_path"]).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_dirty(repo_root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _write_inventory_csv(path: Path, reports: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "checkpoint",
        "name",
        "owner",
        "lifetime",
        "kind",
        "device",
        "dtype",
        "shape",
        "logical_bytes",
        "storage_id",
        "storage_bytes",
        "storage_offset",
        "is_view",
        "is_pinned",
        "counted_unique_storage",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            checkpoint = str(report["checkpoint"])
            for item in report["census"]["rows"]:
                row = dict(item)
                row["checkpoint"] = checkpoint
                row["shape"] = json.dumps(row["shape"], separators=(",", ":"))
                writer.writerow({key: row.get(key) for key in fieldnames})
    temporary.replace(path)


def _runtime_environment(torch, device: str) -> dict[str, Any]:
    resolved = torch.device(device)
    properties = torch.cuda.get_device_properties(resolved)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(resolved),
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "external_nvml_collector_used": False,
    }


class RunRecorder:
    def __init__(self, output_dir: Path, *, torch, device: str):
        self.output_dir = output_dir
        self.torch = torch
        self.device = device
        self.allocator_rows: list[dict[str, Any]] = []
        self.inventory_reports: list[dict[str, Any]] = []
        self.log_path = output_dir / "stdout.log"

    def log(self, message: str) -> None:
        text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(text, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def record(self, checkpoint: str, references) -> dict[str, Any]:
        from mapf_cuda.observability.memory_census import (
            allocator_snapshot,
            census_tensors,
        )

        self.torch.cuda.synchronize(self.device)
        census = census_tensors(references)
        allocator = allocator_snapshot(
            checkpoint, device=self.device, census=census
        )
        self.allocator_rows.append(allocator)
        self.inventory_reports.append(
            {"checkpoint": checkpoint, "census": census}
        )
        self.log(
            f"{checkpoint}: allocated={allocator['allocated_bytes']} "
            f"tensor={allocator['tensor_storage_bytes']} "
            f"residual={allocator['residual_bytes']}"
        )
        return allocator

    def record_allocator_only(self, checkpoint: str) -> dict[str, Any]:
        from mapf_cuda.observability.memory_census import allocator_snapshot

        self.torch.cuda.synchronize(self.device)
        allocator = allocator_snapshot(checkpoint, device=self.device)
        self.allocator_rows.append(allocator)
        self.log(
            f"{checkpoint}: allocated={allocator['allocated_bytes']} "
            f"peak={allocator['peak_allocated_bytes']}"
        )
        return allocator

    def write(self) -> None:
        _write_jsonl(
            self.output_dir / "allocator-checkpoints.jsonl", self.allocator_rows
        )
        _write_inventory_csv(
            self.output_dir / "tensor-inventory.csv", self.inventory_reports
        )


def _module_references(runtime, *, training: bool):
    from mapf_cuda.observability.memory_census import (
        module_tensor_references,
        optimizer_tensor_references,
    )

    references = module_tensor_references(
        runtime.model, prefix="model", include_gradients=training
    )
    if training:
        references.extend(
            optimizer_tensor_references(runtime.optimizer, prefix="optimizer")
        )
    return references


def _stateful_references(simulator):
    from mapf_cuda.observability.memory_census import inventory_tensor_references

    return inventory_tensor_references(
        dict(simulator.memory_tensors),
        classifications=STATEFUL_CLASSIFICATIONS,
        prefix="simulator",
    )


def _mapf_gpt_references(builder):
    from mapf_cuda.observability.memory_census import inventory_tensor_references

    return inventory_tensor_references(
        dict(builder.memory_tensors),
        classifications=MAPF_GPT_CLASSIFICATIONS,
        prefix="mapf_gpt_builder",
    )


def _object_tensor_references(value: Any, *, prefix: str):
    """Enumerate public tensor fields on a lightweight runtime batch."""

    import torch

    from mapf_cuda.observability.memory_census import TensorReference

    references = []
    for name, tensor in sorted(vars(value).items()):
        if isinstance(tensor, torch.Tensor):
            references.append(
                TensorReference(
                    f"{prefix}.{name}",
                    tensor,
                    "model_ephemeral",
                    "model_call",
                    "runtime_batch",
                )
            )
    return references


def _load_magat_runtime(checkpoint: Path, *, device: str):
    from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter
    from mapf_cuda.models.checkpoints import load_magat_checkpoint

    runtime = MAGATRuntimeAdapter(device=device)
    load_magat_checkpoint(runtime, str(checkpoint), device)
    runtime.model.eval()
    return runtime


def _build_magat_data(runtime, simulator):
    """Create zero-copy dense valid-edge views over the resident graph."""

    data = runtime.batch_builder.view_stateful_outputs(
        simulator, materialize_edges=False
    )
    num_edges = int(data.num_edges.item())
    data.edge_index = data.edge_index_storage[:, :num_edges]
    data.edge_attr = data.edge_attr_storage[:num_edges]
    return data


def _slice_magat_envs(data, simulator, *, start_env: int, end_env: int, agents: int):
    """Build one environment-aligned model microbatch without global copies."""

    import torch

    from expert.fixed_magat_plus_runtime import RuntimePyGBatch

    start_node = int(start_env) * int(agents)
    end_node = int(end_env) * int(agents)
    prefix = dict(simulator.memory_tensors)["pyg_edge_prefix"]
    start_edge = int(prefix[start_node].item())
    end_edge = int(prefix[end_node].item())
    num_envs = int(end_env) - int(start_env)
    device = data.x.device
    edge_index = data.edge_index_storage[:, start_edge:end_edge] - start_node
    edge_attr = data.edge_attr_storage[start_edge:end_edge]
    return RuntimePyGBatch(
        x=data.x[start_node:end_node],
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_index_storage=edge_index,
        edge_attr_storage=edge_attr,
        num_edges=torch.tensor([end_edge - start_edge], device=device, dtype=torch.int64),
        batch=torch.arange(num_envs, device=device, dtype=torch.int64).repeat_interleave(agents),
        ptr=torch.arange(
            0,
            (num_envs + 1) * agents,
            step=agents,
            device=device,
            dtype=torch.int64,
        ),
        y=data.y[start_node:end_node],
        terminated=data.terminated[start_node:end_node],
        arrived=data.arrived[start_node:end_node],
    )


def _run_magat_inference(
    spec: S3RowSpec,
    *,
    batch,
    checkpoint: Path,
    device: str,
    seed: int,
    recorder: RunRecorder,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from expert.a3_runner import _build_frozen_stateful_simulator

    runtime = _load_magat_runtime(checkpoint, device=device)
    recorder.record("model_loaded", _module_references(runtime, training=False))

    simulator = _build_frozen_stateful_simulator(batch, device=device, seed=seed)
    staged_actions = torch.as_tensor(
        np.array(batch.actions[0], copy=True), dtype=torch.uint8, device=device
    ).contiguous()
    simulator.update_actions(staged_actions)
    del staged_actions
    gc.collect()
    persistent_refs = _module_references(runtime, training=False) + _stateful_references(
        simulator
    )
    persistent = recorder.record("persistent_allocated", persistent_refs)

    simulator.update_derived_state()
    simulator.build_magat_plus_inputs()
    data = _build_magat_data(runtime, simulator)
    ready_refs = persistent_refs + _object_tensor_references(data, prefix="magat_batch")
    ready = recorder.record("builder_ready", ready_refs)

    outputs = []

    def infer_once():
        outputs.clear()
        with torch.inference_mode():
            if spec.model_microbatch_envs is None:
                outputs.append(runtime.model(data.x, data))
            else:
                for start_env in range(0, spec.num_envs, spec.model_microbatch_envs):
                    end_env = min(start_env + spec.model_microbatch_envs, spec.num_envs)
                    part = _slice_magat_envs(
                        data,
                        simulator,
                        start_env=start_env,
                        end_env=end_env,
                        agents=spec.num_agents,
                    )
                    outputs.append(runtime.model(part.x, part))

    infer_once()
    torch.cuda.synchronize(device)
    outputs.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    infer_once()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - started
    output_refs = []
    from mapf_cuda.observability.memory_census import TensorReference

    for index, output in enumerate(outputs):
        output_refs.append(
            TensorReference(
                f"model.output.{index}",
                output,
                "model_ephemeral",
                "model_call",
                "output",
            )
        )
    peak = recorder.record("inference_peak", ready_refs + output_refs)
    result = {
        "persistent_allocated_bytes": int(persistent["allocated_bytes"]),
        "builder_ready_allocated_bytes": int(ready["allocated_bytes"]),
        "measured_peak_allocated_bytes": int(peak["peak_allocated_bytes"]),
        "measured_peak_reserved_bytes": int(peak["peak_reserved_bytes"]),
        "ephemeral_peak_over_builder_bytes": int(
            peak["peak_allocated_bytes"] - ready["allocated_bytes"]
        ),
        "measured_operation_wall_s_diagnostic": wall_s,
        "model_microbatch_envs": spec.model_microbatch_envs,
        "output_chunks": len(outputs),
        "valid_edges": int(data.num_edges.item()),
        "persistent_reconciliation_residual_ratio": persistent["residual_ratio"],
    }

    del output_refs, outputs, data, ready_refs, persistent_refs
    del simulator, runtime
    gc.collect()
    torch.cuda.empty_cache()
    recorder.record_allocator_only("cleanup_diagnostic")
    return result


def _run_mapf_gpt_inference(
    spec: S3RowSpec,
    *,
    batch,
    checkpoint: Path,
    device: str,
    recorder: RunRecorder,
) -> dict[str, Any]:
    import torch

    from expert.minimal_p0_mapf_gpt_resident_runner import (
        _build_resident_components,
        _infer_actions,
        _load_inference_model,
    )
    from mapf_cuda.observability.memory_census import TensorReference

    model, payload, model_size = _load_inference_model(checkpoint, device=device)
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    from mapf_cuda.observability.memory_census import module_tensor_references

    model_refs = module_tensor_references(model, prefix="model")
    recorder.record("model_loaded", model_refs)

    simulator, builder = _build_resident_components(
        batch, device=device, horizon=batch.horizon
    )
    persistent_refs = model_refs + _stateful_references(simulator) + _mapf_gpt_references(
        builder
    )
    persistent = recorder.record("persistent_allocated", persistent_refs)

    builder.build_tokens_from_state(
        simulator.cur_x, simulator.cur_y, simulator.goal_x, simulator.goal_y
    )
    total_agents = spec.num_envs * spec.num_agents
    actions = torch.empty(total_agents, dtype=torch.uint8, device=device)
    ready_refs = persistent_refs + [
        TensorReference(
            "model.action_output", actions, "model_ephemeral", "model_call", "output"
        )
    ]
    ready = recorder.record("builder_ready", ready_refs)

    microbatch = int(spec.model_microbatch_agents or total_agents)
    _infer_actions(model, builder.tokens, actions, microbatch_size=microbatch)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    microbatches = _infer_actions(
        model, builder.tokens, actions, microbatch_size=microbatch
    )
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - started
    peak = recorder.record("inference_peak", ready_refs)
    result = {
        "model_size": model_size,
        "persistent_allocated_bytes": int(persistent["allocated_bytes"]),
        "builder_ready_allocated_bytes": int(ready["allocated_bytes"]),
        "measured_peak_allocated_bytes": int(peak["peak_allocated_bytes"]),
        "measured_peak_reserved_bytes": int(peak["peak_reserved_bytes"]),
        "ephemeral_peak_over_builder_bytes": int(
            peak["peak_allocated_bytes"] - ready["allocated_bytes"]
        ),
        "measured_operation_wall_s_diagnostic": wall_s,
        "model_microbatch_agents": microbatch,
        "microbatches": int(microbatches),
        "persistent_reconciliation_residual_ratio": persistent["residual_ratio"],
    }

    del actions, builder, simulator, model, ready_refs, persistent_refs, model_refs
    gc.collect()
    torch.cuda.empty_cache()
    recorder.record_allocator_only("cleanup_diagnostic")
    return result


def _run_training_memory(
    spec: S3RowSpec,
    *,
    batch,
    checkpoint: Path,
    device: str,
    seed: int,
    recorder: RunRecorder,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from expert.a3_runner import _build_frozen_stateful_simulator
    from expert.expert_running import RingBuffer
    from mapf_cuda.observability.memory_census import TensorReference

    runtime = _load_magat_runtime(checkpoint, device=device)
    runtime.model.train()
    simulator = _build_frozen_stateful_simulator(batch, device=device, seed=seed)
    actions = torch.as_tensor(
        np.array(batch.actions[0], copy=True), dtype=torch.uint8, device=device
    ).contiguous()
    simulator.update_actions(actions)
    del actions
    gc.collect()
    simulator.update_derived_state()
    simulator.build_magat_plus_inputs()

    ring_capacity = int(spec.ring_depth) * spec.num_envs * spec.num_agents
    ring = RingBuffer(
        ring_capacity,
        feature_dim=8,
        num_envs=spec.num_envs,
        agents_per_env=spec.num_agents,
    )
    stage = torch.empty((int(spec.batch_rows), 8), dtype=torch.int16, device=device)
    transport_refs = [
        TensorReference(
            "transport.ring_payload",
            ring.cpu_buffer,
            "transport_host",
            "run",
            "pinned_ring",
        ),
        TensorReference(
            "transport.gpu_stage",
            stage,
            "transport_gpu",
            "frontier",
            "compact_stage",
        ),
    ]

    base_refs = _module_references(runtime, training=False) + _stateful_references(
        simulator
    ) + transport_refs
    recorder.record("builder_ready", base_refs)

    # The first optimizer step allocates capturable Adam state and gradients.
    runtime.train_step_from_stateful(simulator)
    torch.cuda.synchronize(device)
    training_refs = _module_references(runtime, training=True) + _stateful_references(
        simulator
    ) + transport_refs
    persistent = recorder.record("training_persistent", training_refs)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss = runtime.train_step_from_stateful(simulator)
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - started
    training_refs = _module_references(runtime, training=True) + _stateful_references(
        simulator
    ) + transport_refs
    peak = recorder.record("optimizer_step_peak", training_refs)
    result = {
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": 1,
        "measured_loss_diagnostic": float(loss.item()),
        "training_persistent_allocated_bytes": int(persistent["allocated_bytes"]),
        "measured_peak_allocated_bytes": int(peak["peak_allocated_bytes"]),
        "measured_peak_reserved_bytes": int(peak["peak_reserved_bytes"]),
        "ephemeral_peak_over_training_persistent_bytes": int(
            peak["peak_allocated_bytes"] - persistent["allocated_bytes"]
        ),
        "measured_operation_wall_s_diagnostic": wall_s,
        "ring_raw_payload_bytes": int(ring.cpu_buffer.numel() * ring.cpu_buffer.element_size()),
        "gpu_stage_bytes": int(stage.numel() * stage.element_size()),
        "persistent_reconciliation_residual_ratio": persistent["residual_ratio"],
    }

    del loss, training_refs, base_refs, transport_refs, stage, ring
    del simulator, runtime
    gc.collect()
    torch.cuda.empty_cache()
    cleanup = recorder.record_allocator_only("cleanup_diagnostic")
    from mapf_cuda.observability.memory_census import reconcile_after_cleanup

    reconciliation = reconcile_after_cleanup(
        allocated_bytes=persistent["allocated_bytes"],
        tensor_storage_bytes=persistent["tensor_storage_bytes"],
        cleanup_allocated_bytes=cleanup["allocated_bytes"],
        context_allocated_bytes=recorder.allocator_rows[0]["allocated_bytes"],
    )
    result["allocator_reconciliation_after_cleanup"] = reconciliation
    result["persistent_reconciliation_adjusted_residual_ratio"] = reconciliation[
        "adjusted_residual_ratio"
    ]
    return result


def capacity_preflight_result(
    spec: S3RowSpec,
    *,
    device: str,
    memory_info: tuple[int, int] | None = None,
) -> dict[str, Any]:
    from mapf_cuda.observability.memory_census import (
        build_capacity_estimate,
        preflight_capacity,
    )

    estimate = build_capacity_estimate(
        num_envs=spec.num_envs, num_agents=spec.num_agents
    )
    decision = preflight_capacity(
        required_bytes=estimate["mandatory_known_bytes"],
        safety_reserve_bytes=spec.safety_reserve_bytes,
        device=device,
        memory_info=memory_info,
    )
    return {"estimate": estimate, "preflight": decision}


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_row_spec(args.row_id)
    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    commit = _git_commit(repo_root)
    git_dirty = _git_is_dirty(repo_root)
    if git_dirty and not args.allow_dirty:
        raise RuntimeError(
            "formal S3 runs require a clean worktree; commit the instrumentation "
            "or pass --allow-dirty only for a development smoke"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = Path(args.manifest).expanduser().resolve()
    manifest_sha = _verify_digest(
        manifest, EXPECTED_MANIFEST_SHA256, "input manifest"
    )
    if spec.scan_cell is None:
        raise RuntimeError(f"row {spec.row_id} has no frozen scan cell")
    input_path = resolve_scan_cell_input(manifest, spec.scan_cell)
    input_sha = _verify_digest(
        input_path, EXPECTED_INPUT_SHA256[spec.scan_cell], "frozen input"
    )

    config = {
        **asdict(spec),
        "repetition": int(args.repetition),
        "seed": int(args.seed),
        "device": args.device,
        "manifest": str(manifest),
        "input_path": str(input_path),
        "code_commit": commit,
        "git_dirty": git_dirty,
        "development_dirty_run_allowed": bool(args.allow_dirty),
        "wall_time_is_diagnostic_only": True,
        "nvml_enabled": False,
    }
    _write_json(output_dir / "config.json", config)
    (output_dir / "code-commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output_dir / "input.sha256").write_text(
        f"{input_sha}  {input_path}\n", encoding="utf-8"
    )
    (output_dir / "manifest.sha256").write_text(
        f"{manifest_sha}  {manifest}\n", encoding="utf-8"
    )

    import torch

    torch.manual_seed(int(args.seed))
    try:
        torch.cuda.set_device(args.device)
    except Exception as error:
        raise RuntimeError("S3 requires a working CUDA runtime") from error
    torch.cuda.synchronize(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    _write_json(
        output_dir / "environment.json", _runtime_environment(torch, args.device)
    )

    recorder = RunRecorder(output_dir, torch=torch, device=args.device)
    recorder.log(f"starting {spec.row_id} repetition={args.repetition}")
    context = recorder.record_allocator_only("context")
    capacity = capacity_preflight_result(spec, device=args.device)
    _write_json(output_dir / "capacity-model.json", capacity)

    checkpoint_path = None
    checkpoint_sha = None
    if spec.backend == "capacity":
        operation = {
            "preflight_only": True,
            **capacity,
        }
    else:
        from expert.minimal_scaling_runner import load_scan_cell

        _, batch = load_scan_cell(manifest, spec.scan_cell)
        if batch.num_agents != spec.num_agents or batch.num_envs < spec.num_envs:
            raise RuntimeError(
                f"frozen batch shape mismatch for {spec.row_id}: "
                f"A={batch.num_agents}, E={batch.num_envs}"
            )
        if batch.num_envs != spec.num_envs:
            batch = batch.take_envs(spec.num_envs)
        if spec.backend == "magat":
            checkpoint_path = Path(args.magat_checkpoint).expanduser().resolve()
            checkpoint_sha = _verify_digest(
                checkpoint_path, EXPECTED_MAGAT_SHA256, "MAGAT checkpoint"
            )
            if spec.operation == "training":
                operation = _run_training_memory(
                    spec,
                    batch=batch,
                    checkpoint=checkpoint_path,
                    device=args.device,
                    seed=int(args.seed),
                    recorder=recorder,
                )
            else:
                operation = _run_magat_inference(
                    spec,
                    batch=batch,
                    checkpoint=checkpoint_path,
                    device=args.device,
                    seed=int(args.seed),
                    recorder=recorder,
                )
        elif spec.backend == "mapf_gpt":
            checkpoint_path = Path(args.mapf_gpt_checkpoint).expanduser().resolve()
            checkpoint_sha = _verify_digest(
                checkpoint_path, EXPECTED_MAPF_GPT_SHA256, "MAPF-GPT checkpoint"
            )
            operation = _run_mapf_gpt_inference(
                spec,
                batch=batch,
                checkpoint=checkpoint_path,
                device=args.device,
                recorder=recorder,
            )
        else:  # pragma: no cover - row table construction prevents this
            raise RuntimeError(f"unsupported backend {spec.backend}")

    if checkpoint_path is not None:
        (output_dir / "checkpoint.sha256").write_text(
            f"{checkpoint_sha}  {checkpoint_path}\n", encoding="utf-8"
        )
    recorder.write()
    persistent_residual_ratio = operation.get(
        "persistent_reconciliation_adjusted_residual_ratio",
        operation.get("persistent_reconciliation_residual_ratio"),
    )
    if spec.backend == "capacity":
        acceptance = {
            "capacity_preflight_refused_destructive_allocation": not bool(
                capacity["preflight"]["allowed"]
            ),
            "persistent_reconciliation_within_5pct": None,
        }
    else:
        acceptance = {
            "capacity_preflight_refused_destructive_allocation": None,
            "persistent_reconciliation_within_5pct": (
                persistent_residual_ratio is not None
                and float(persistent_residual_ratio) <= 0.05
            ),
        }
    acceptance["complete"] = all(
        value for value in acceptance.values() if value is not None
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "row_id": spec.row_id,
        "repetition": int(args.repetition),
        "context_allocated_bytes": int(context["allocated_bytes"]),
        "operation": operation,
        "acceptance": acceptance,
        "capacity": capacity,
        "provenance": {
            "code_commit": commit,
            "manifest_sha256": manifest_sha,
            "input_sha256": input_sha,
            "checkpoint_sha256": checkpoint_sha,
        },
    }
    _write_json(output_dir / "result.json", result)
    recorder.log(f"completed {spec.row_id}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-id", required=True, choices=sorted(ROW_SPECS))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--magat-checkpoint", type=Path, default=DEFAULT_MAGAT_CHECKPOINT
    )
    parser.add_argument(
        "--mapf-gpt-checkpoint", type=Path, default=DEFAULT_MAPF_GPT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit an explicitly non-formal development smoke from a dirty worktree",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
