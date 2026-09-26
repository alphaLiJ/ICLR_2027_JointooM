"""Lightweight E3 stage decomposition for the resident MAGAT deployment path."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np

from expert.minimal_scaling_runner import GpuMemoryTracker, load_scan_cell


STEADY_STAGES = (
    "transition",
    "derived_state_steady",
    "node_construction",
    "graph_finalization",
    "edge_materialization",
    "model_forward",
    "argmax_action_update",
    "closed_loop",
)


def _build_simulator(batch, *, device: str, seed: int):
    from expert.a3_runner import _build_frozen_stateful_simulator

    simulator = _build_frozen_stateful_simulator(
        batch,
        device=device,
        seed=seed,
    )
    simulator.update_actions(
        __import__("torch").as_tensor(
            np.array(batch.actions[0], copy=True),
            dtype=__import__("torch").uint8,
            device=device,
        ).contiguous()
    )
    return simulator


def _load_runtime(checkpoint: Path, *, device: str):
    from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter
    from mapf_cuda.models.checkpoints import load_magat_checkpoint

    runtime = MAGATRuntimeAdapter(device=device)
    load_magat_checkpoint(runtime, str(checkpoint), device)
    runtime.model.eval()
    return runtime


def _profile_cuda_callable(fn: Callable[[], Any]) -> dict[str, Any]:
    """Count launches in a separate one-step pass, never in the timed samples."""

    import torch

    try:
        from expert.a3_runner import _summarize_cuda_profile_events

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            acc_events=True,
        ) as profiler:
            with torch.inference_mode():
                fn()
            torch.cuda.synchronize()
        summary = _summarize_cuda_profile_events(
            profiler.events(),
            cuda_device_type=torch.autograd.DeviceType.CUDA,
            profiled_steps=1,
        )
        return {
            "cuda_launch_count_per_step": summary["cuda_launch_count_per_step"],
            "profiled_gpu_utilization_pct": summary["gpu_utilization_pct"],
            "profile_status": "ok",
            "profile_error": None,
        }
    except Exception as error:
        return {
            "cuda_launch_count_per_step": None,
            "profiled_gpu_utilization_pct": None,
            "profile_status": "unavailable",
            "profile_error": f"{type(error).__name__}: {error}",
        }


def _measure_callable(
    *,
    stage: str,
    fn: Callable[[], Any],
    num_envs: int,
    num_agents: int,
    warmup_steps: int,
    num_steps: int,
    repetitions: int,
    memory_tracker: GpuMemoryTracker,
    profile: bool = True,
) -> list[dict[str, Any]]:
    import torch

    try:
        with torch.inference_mode():
            for _ in range(int(warmup_steps)):
                fn()
        torch.cuda.synchronize()
        profile_metrics = _profile_cuda_callable(fn) if profile else {
            "cuda_launch_count_per_step": None,
            "profiled_gpu_utilization_pct": None,
            "profile_status": "not_requested",
            "profile_error": None,
        }
        rows: list[dict[str, Any]] = []
        for repetition in range(int(repetitions)):
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(int(num_steps)):
                    fn()
            end_event.record()
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - started
            gpu_ms = float(start_event.elapsed_time(end_event))
            memory_sample = memory_tracker.sample(
                f"{stage}_repetition_{repetition}"
            )
            env_steps = int(num_envs) * int(num_steps)
            agent_steps = env_steps * int(num_agents)
            rows.append(
                {
                    "stage": stage,
                    "repetition": repetition,
                    "num_steps": int(num_steps),
                    "wall_ms_per_step": wall_s / int(num_steps) * 1000.0,
                    "gpu_ms_per_step": gpu_ms / int(num_steps),
                    "env_steps_s": env_steps / wall_s,
                    "agent_steps_s": agent_steps / wall_s,
                    "torch_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "torch_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved()
                    ),
                    "nvml_process_memory_bytes": memory_sample[
                        "nvml_process_memory_bytes"
                    ],
                    "timed_h2d_bytes": 0,
                    "timed_d2h_bytes": 0,
                    **profile_metrics,
                }
            )
        return rows
    except Exception as error:
        raise RuntimeError(f"E3 stage {stage!r} failed: {error}") from error


def _measure_initial_energy_map(
    *,
    batch,
    device: str,
    seed: int,
    repetitions: int,
    memory_tracker: GpuMemoryTracker,
) -> list[dict[str, Any]]:
    """Measure the reset-time full energy-map construction on fresh simulators."""

    rows: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        simulator = _build_simulator(
            batch,
            device=device,
            seed=int(seed) + repetition,
        )
        rows.extend(
            _measure_callable(
                stage="derived_state_initial_energy_map",
                fn=simulator.update_derived_state,
                num_envs=batch.num_envs,
                num_agents=batch.num_agents,
                warmup_steps=0,
                num_steps=1,
                repetitions=1,
                memory_tracker=memory_tracker,
                profile=False,
            )
        )
        rows[-1]["repetition"] = repetition
        del simulator
        gc.collect()
        __import__("torch").cuda.empty_cache()
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for stage in dict.fromkeys(row["stage"] for row in rows):
        selected = [row for row in rows if row["stage"] == stage]
        summary: dict[str, Any] = {
            "stage": stage,
            "repetitions": len(selected),
        }
        for key in (
            "wall_ms_per_step",
            "gpu_ms_per_step",
            "env_steps_s",
            "agent_steps_s",
        ):
            values = [float(row[key]) for row in selected]
            summary[f"median_{key}"] = statistics.median(values)
            summary[f"min_{key}"] = min(values)
            summary[f"max_{key}"] = max(values)
        for key in (
            "torch_peak_allocated_bytes",
            "torch_peak_reserved_bytes",
            "nvml_process_memory_bytes",
        ):
            values = [int(row[key]) for row in selected if row[key] is not None]
            summary[f"max_{key}"] = max(values) if values else None
        summary["cuda_launch_count_per_step"] = selected[0][
            "cuda_launch_count_per_step"
        ]
        summary["profiled_gpu_utilization_pct"] = selected[0][
            "profiled_gpu_utilization_pct"
        ]
        summary["profile_status"] = selected[0]["profile_status"]
        summary["profile_error"] = selected[0]["profile_error"]
        summaries.append(summary)
    return summaries


def run_e3_stage_scan(
    *,
    manifest: Path,
    scan_cell: str,
    checkpoint: Path,
    warmup_steps: int,
    num_steps: int,
    repetitions: int,
    model_microbatch_envs: int = 0,
    device: str = "cuda:0",
    seed: int = 20260723,
) -> dict[str, Any]:
    import torch

    if repetitions <= 0 or num_steps <= 0 or warmup_steps < 0:
        raise ValueError("repetitions/num_steps must be positive and warmup non-negative")
    cell, batch = load_scan_cell(manifest, scan_cell)
    if batch.num_agents != 256:
        raise ValueError("E3 currently requires exactly 256 agents")
    if model_microbatch_envs < 0:
        raise ValueError("model_microbatch_envs must be non-negative")
    memory_tracker = GpuMemoryTracker("cuda")
    memory_tracker.start()

    rows = _measure_initial_energy_map(
        batch=batch,
        device=device,
        seed=seed,
        repetitions=repetitions,
        memory_tracker=memory_tracker,
    )
    runtime = _load_runtime(checkpoint, device=device)
    simulator = _build_simulator(batch, device=device, seed=seed + 100)
    holders: dict[str, Any] = {}

    # Establish a valid resident energy map before steady-state measurement.
    simulator.update_derived_state()
    torch.cuda.synchronize()

    stage_functions: list[tuple[str, Callable[[], Any]]] = [
        ("transition", simulator.step_sim_only),
        ("derived_state_steady", simulator.update_derived_state),
        ("node_construction", simulator.build_magat_plus_nodes),
        ("graph_finalization", simulator.finalize_magat_plus_graph),
    ]
    for stage, fn in stage_functions:
        rows.extend(
            _measure_callable(
                stage=stage,
                fn=fn,
                num_envs=batch.num_envs,
                num_agents=batch.num_agents,
                warmup_steps=warmup_steps,
                num_steps=num_steps,
                repetitions=repetitions,
                memory_tracker=memory_tracker,
            )
        )

    effective_microbatch_envs = (
        batch.num_envs
        if model_microbatch_envs == 0
        else min(int(model_microbatch_envs), batch.num_envs)
    )

    def partition_model_batches(data):
        if effective_microbatch_envs >= batch.num_envs:
            return [data]
        from expert.fixed_magat_plus_runtime import PyGBatchBuilder

        parts = []
        for start_env in range(0, batch.num_envs, effective_microbatch_envs):
            end_env = min(start_env + effective_microbatch_envs, batch.num_envs)
            parts.append(
                PyGBatchBuilder.slice_env_aligned_batch(
                    data,
                    start_node=start_env * batch.num_agents,
                    end_node=end_env * batch.num_agents,
                    agents_per_env=batch.num_agents,
                )
            )
        return parts

    def materialize_edges():
        holders.pop("model_batches", None)
        holders.pop("data", None)
        data_now = runtime.batch_builder.view_stateful_outputs(
            simulator,
            materialize_edges=True,
        )
        batches_now = partition_model_batches(data_now)
        holders["data"] = data_now
        holders["model_batches"] = batches_now
        return data_now

    rows.extend(
        _measure_callable(
            stage="edge_materialization",
            fn=materialize_edges,
            num_envs=batch.num_envs,
            num_agents=batch.num_agents,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            repetitions=repetitions,
            memory_tracker=memory_tracker,
        )
    )
    data = holders["data"]
    graph_edges = int(data.num_edges.detach().cpu().item())
    total_nodes = batch.num_envs * batch.num_agents

    def model_forward():
        outputs = [
            runtime.model(model_batch.x, model_batch)
            for model_batch in holders["model_batches"]
        ]
        holders["logits"] = (
            outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        )
        return holders["logits"]

    rows.extend(
        _measure_callable(
            stage="model_forward",
            fn=model_forward,
            num_envs=batch.num_envs,
            num_agents=batch.num_agents,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            repetitions=repetitions,
            memory_tracker=memory_tracker,
        )
    )
    logits = holders["logits"]

    def argmax_action_update():
        actions = logits.argmax(dim=-1).reshape(
            batch.num_envs, batch.num_agents
        ).to(torch.uint8)
        simulator.update_actions(actions)
        holders["actions"] = actions
        return actions

    rows.extend(
        _measure_callable(
            stage="argmax_action_update",
            fn=argmax_action_update,
            num_envs=batch.num_envs,
            num_agents=batch.num_agents,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            repetitions=repetitions,
            memory_tracker=memory_tracker,
        )
    )
    # The isolated materialization/model measurements intentionally retain
    # their outputs.  They are not part of the production closed loop, so
    # release them before measuring that loop to avoid double-accounting.
    del data, logits
    holders.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    def closed_loop():
        simulator.update_derived_state()
        simulator.build_magat_plus_nodes()
        simulator.finalize_magat_plus_graph()
        resident_data = runtime.batch_builder.view_stateful_outputs(
            simulator,
            materialize_edges=True,
        )
        resident_batches = partition_model_batches(resident_data)
        output_parts = [
            runtime.model(model_batch.x, model_batch)
            for model_batch in resident_batches
        ]
        logits_now = (
            output_parts[0]
            if len(output_parts) == 1
            else torch.cat(output_parts, dim=0)
        )
        actions_now = logits_now.argmax(dim=-1).reshape(
            batch.num_envs, batch.num_agents
        ).to(torch.uint8)
        simulator.update_actions(actions_now)
        simulator.step_sim_only()
        return actions_now

    rows.extend(
        _measure_callable(
            stage="closed_loop",
            fn=closed_loop,
            num_envs=batch.num_envs,
            num_agents=batch.num_agents,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            repetitions=repetitions,
            memory_tracker=memory_tracker,
        )
    )
    return {
        "schema_version": 1,
        "status": "ok",
        "scan_cell": scan_cell,
        "cell": cell,
        "num_agents": batch.num_agents,
        "num_envs": batch.num_envs,
        "horizon": batch.horizon,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "selected_input_sha256": batch.semantic_sha256,
        "warmup_steps": warmup_steps,
        "num_steps": num_steps,
        "repetitions": repetitions,
        "model_microbatch_envs": effective_microbatch_envs,
        "model_microbatch_agents": effective_microbatch_envs * batch.num_agents,
        "model_microbatch_count": (
            batch.num_envs + effective_microbatch_envs - 1
        )
        // effective_microbatch_envs,
        "builder_mode": simulator.pyg_builder_mode,
        "resolved_builder_impl": simulator.resolved_pyg_builder_impl,
        "graph_edges": graph_edges,
        "average_graph_degree": graph_edges / total_nodes,
        "timed_h2d_bytes_per_step": 0,
        "timed_d2h_bytes_per_step": 0,
        "memory": memory_tracker.summary(),
        "rows": rows,
        "stages": _aggregate(rows),
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report.get("rows", [])
    if rows:
        with (output_dir / "measurements.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "stdout.log").write_text(
        json.dumps(
            {
                "status": report["status"],
                "num_agents": report.get("num_agents"),
                "num_envs": report.get("num_envs"),
                "stages": report.get("stages"),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one lightweight E3 stage scan.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--model-microbatch-envs",
        type=int,
        default=0,
        help="0 uses one full model batch; otherwise split into env-aligned batches.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_e3_stage_scan(
            manifest=args.manifest,
            scan_cell=args.scan_cell,
            checkpoint=args.checkpoint,
            warmup_steps=args.warmup_steps,
            num_steps=args.num_steps,
            repetitions=args.repetitions,
            model_microbatch_envs=args.model_microbatch_envs,
            device=args.device,
            seed=args.seed,
        )
        _write_report(args.output_dir, report)
        print(json.dumps({"status": "ok", "stages": report["stages"]}, sort_keys=True))
        return 0
    except Exception as error:
        message = str(error)
        capacity = "out of memory" in message.lower()
        report = {
            "schema_version": 1,
            "status": "capacity_failure" if capacity else "failed",
            "scan_cell": args.scan_cell,
            "exception_type": type(error).__name__,
            "exception_message": message,
            "pid": os.getpid(),
        }
        _write_report(args.output_dir, report)
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STEADY_STAGES",
    "_aggregate",
    "_measure_callable",
    "run_e3_stage_scan",
]
