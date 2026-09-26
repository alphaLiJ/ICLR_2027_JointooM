"""Manifest-bound cumulative A3 stateful builder stage scans."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from expert.minimal_scaling_runner import load_scan_cell_with_source


A3_STAGE_NAMES = (
    "transition",
    "derived_state_energy_map",
    "node_or_token_construction",
    "graph_or_token_finalization",
    "model_ready_batch",
)

_A3_REQUIRED_METRICS = (
    "wall_ms_per_step",
    "gpu_ms_per_step",
    "env_steps_s",
    "agent_steps_s",
    "peak_gpu_memory_bytes",
    "graph_edges",
    "average_graph_degree",
    "token_count",
    "padding_fraction",
    "cuda_launch_count",
    "cuda_launch_count_per_step",
    "gpu_utilization_pct",
)


def _summarize_cuda_profile_events(
    events, *, cuda_device_type, profiled_steps: int
) -> dict[str, Any]:
    """Summarize an isolated CUDA profiler pass without touching main timing."""

    if int(profiled_steps) <= 0:
        raise ValueError("profiled_steps must be positive")
    cuda_events = [event for event in events if event.device_type == cuda_device_type]
    if not cuda_events:
        raise RuntimeError("A3 profiler pass reported no CUDA kernel events")
    intervals = sorted(
        (float(event.time_range.start), float(event.time_range.end))
        for event in cuda_events
        if float(event.time_range.end) >= float(event.time_range.start)
    )
    if not intervals:
        raise RuntimeError("A3 profiler pass reported no valid CUDA time ranges")
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    active_us = sum(end - start for start, end in merged)
    span_us = intervals[-1][1] - intervals[0][0]
    launch_count = sum(
        int(getattr(event, "count", 1))
        for event in events
        if str(getattr(event, "name", "")).startswith("cudaLaunch")
    )
    if launch_count <= 0:
        raise RuntimeError("A3 profiler pass reported no CUDA kernel launches")
    return {
        "cuda_launch_count": int(launch_count),
        "cuda_launch_count_per_step": launch_count / int(profiled_steps),
        "cuda_launch_count_profiled_steps": int(profiled_steps),
        "cuda_launch_count_source": "torch_profiler_separate_pass",
        "profiled_gpu_active_time_us": active_us,
        "profiled_gpu_timeline_span_us": span_us,
        "gpu_utilization_pct": 0.0 if span_us <= 0.0 else active_us / span_us * 100.0,
        "gpu_utilization_source": "cuda_event_timeline_separate_pass",
        "profiling_contaminates_main_timing": False,
    }


def _build_frozen_stateful_simulator(batch, *, device: str, seed: int):
    import torch
    import grid_world_cpp as ext

    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator

    grids, extents = stack_grids_for_compiled_simulator(
        list(batch.grids), device=device
    )
    simulator = ext.GridWorldSimulator(
        grids,
        batch.num_agents,
        0,
        int(grids.shape[0] * grids.shape[1] * grids.shape[2]),
        int(seed),
        "standard_mapf",
        int(batch.horizon),
    )
    simulator.set_real_map_extents(extents)
    simulator.load_state(
        torch.as_tensor(batch.positions.astype(np.int16), device=device).contiguous(),
        torch.as_tensor(batch.goals.astype(np.int16), device=device).contiguous(),
        torch.as_tensor(batch.arrived.astype(np.uint8), device=device).contiguous(),
        torch.zeros(batch.num_envs, dtype=torch.int32, device=device),
    )
    simulator.pyg_builder_mode = "local_gather"
    simulator.pyg_local_gather_impl = "auto"
    return simulator


def _measure_stage(
    simulator,
    batch,
    stage_name: str,
    *,
    warmup_steps: int,
    num_steps: int,
    device: str,
) -> dict[str, Any]:
    import torch

    staged_actions = [
        torch.as_tensor(
            np.array(batch.actions[step], copy=True),
            dtype=torch.uint8,
            device=device,
        ).contiguous()
        for step in range(batch.horizon)
    ]

    def run_once(step_index: int) -> None:
        actions = staged_actions[step_index % batch.horizon]
        simulator.update_actions(actions)
        simulator.step_sim_only()
        if stage_name == "transition":
            return
        simulator.update_derived_state()
        if stage_name == "derived_state_energy_map":
            return
        simulator.build_magat_plus_nodes()
        if stage_name == "node_or_token_construction":
            return
        simulator.finalize_magat_plus_graph()
        if stage_name == "graph_or_token_finalization":
            return
        # Materializing the variable-length graph view is the final host-visible
        # operation required before a PyG model can consume the resident tensors.
        _ = simulator.pyg_edge_index
        _ = simulator.pyg_edge_attr

    for step in range(int(warmup_steps)):
        run_once(step)
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    for step in range(int(num_steps)):
        run_once(step + int(warmup_steps))
    end_event.record()
    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - started
    gpu_ms = float(start_event.elapsed_time(end_event))
    profile_steps = min(max(1, int(num_steps)), 5)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        acc_events=True,
    ) as profiler:
        for step in range(profile_steps):
            run_once(step + int(warmup_steps) + int(num_steps))
        torch.cuda.synchronize(torch.device(device))
    profile_metrics = _summarize_cuda_profile_events(
        profiler.events(),
        cuda_device_type=torch.autograd.DeviceType.CUDA,
        profiled_steps=profile_steps,
    )
    num_edges = int(simulator.pyg_num_edges.detach().cpu().item()) if stage_name in {
        "graph_or_token_finalization",
        "model_ready_batch",
    } else 0
    total_nodes = batch.num_envs * batch.num_agents
    return {
        "stage": stage_name,
        "backend": "magat",
        "timed_region": "cumulative_stateful_stage",
        "num_envs": batch.num_envs,
        "num_agents": batch.num_agents,
        "num_steps": int(num_steps),
        "wall_ms_per_step": wall_s / int(num_steps) * 1000.0,
        "gpu_ms_per_step": gpu_ms / int(num_steps),
        "env_steps_s": batch.num_envs * int(num_steps) / wall_s,
        "agent_steps_s": total_nodes * int(num_steps) / wall_s,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
        "graph_edges": num_edges,
        "average_graph_degree": 0.0 if total_nodes == 0 else num_edges / total_nodes,
        "token_count": None,
        "padding_fraction": None,
        "resolved_builder_impl": simulator.resolved_pyg_builder_impl,
        "consumed_input_sha256": batch.semantic_sha256,
        **profile_metrics,
    }


def _default_stage_executor(
    *, batch, backend: str, warmup_steps: int, num_steps: int,
    device: str, seed: int,
) -> list[dict[str, Any]]:
    if backend != "magat":
        raise NotImplementedError(
            "the core A3 stateful stage executor currently targets MAGAT"
        )
    rows = []
    for stage_index, stage_name in enumerate(A3_STAGE_NAMES):
        simulator = _build_frozen_stateful_simulator(
            batch, device=device, seed=int(seed) + stage_index
        )
        rows.append(
            _measure_stage(
                simulator,
                batch,
                stage_name,
                warmup_steps=warmup_steps,
                num_steps=num_steps,
                device=device,
            )
        )
    return rows


def run_builder_stage_scan(
    *,
    manifest_path: str | pathlib.Path,
    scan_cell: str,
    backend: str,
    warmup_steps: int = 5,
    repetitions: int = 5,
    num_steps: int = 20,
    device: str = "cuda:0",
    seed: int = 20260720,
    stage_executor: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if int(repetitions) <= 0 or int(num_steps) <= 0 or int(warmup_steps) < 0:
        raise ValueError("A3 repetitions/num_steps must be positive and warmup non-negative")
    cell, full_batch, batch = load_scan_cell_with_source(
        pathlib.Path(manifest_path), scan_cell
    )
    executor = stage_executor or _default_stage_executor
    rows = []
    for repetition_id in range(int(repetitions)):
        repetition_rows = list(
            executor(
                batch=batch,
                backend=backend,
                warmup_steps=int(warmup_steps),
                num_steps=int(num_steps),
                device=device,
                seed=int(seed) + repetition_id,
            )
        )
        if [row.get("stage") for row in repetition_rows] != list(A3_STAGE_NAMES):
            raise RuntimeError("A3 executor must report the exact five cumulative stages")
        for row in repetition_rows:
            missing_metrics = [key for key in _A3_REQUIRED_METRICS if key not in row]
            if missing_metrics:
                raise RuntimeError(
                    f"A3 stage {row.get('stage')} omitted required metrics: {missing_metrics}"
                )
            if int(row["cuda_launch_count"]) <= 0:
                raise RuntimeError("A3 stage reported no CUDA kernel launches")
            consumed = row.get("consumed_input_sha256", batch.semantic_sha256)
            if consumed != batch.semantic_sha256:
                raise RuntimeError("A3 stage consumed input hash differs from selected pool")
            row.update(
                repetition_id=repetition_id,
                scan_cell=scan_cell,
                source_pool_sha256=full_batch.semantic_sha256,
                selected_input_sha256=batch.semantic_sha256,
                consumed_input_sha256=consumed,
                instance_ids=batch.instance_ids.tolist(),
            )
            rows.append(row)
    return {
        "schema_version": 1,
        "accepted": True,
        "backend": backend,
        "scan_cell": scan_cell,
        "cell": dict(cell),
        "source_pool_sha256": full_batch.semantic_sha256,
        "consumed_input_sha256": batch.semantic_sha256,
        "instance_ids": batch.instance_ids.tolist(),
        "stage_names": list(A3_STAGE_NAMES),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one manifest-bound A3 stage scan.")
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument("--backend", default="magat", choices=("magat", "mapf_gpt"))
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args(argv)
    report = run_builder_stage_scan(
        manifest_path=args.manifest,
        scan_cell=args.scan_cell,
        backend=args.backend,
        warmup_steps=args.warmup_steps,
        repetitions=args.repetitions,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["A3_STAGE_NAMES", "main", "run_builder_stage_scan"]
