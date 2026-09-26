"""GPU-resident MAPF-GPT closed-loop inference benchmark.

The measured region keeps simulator state, token construction, model inputs,
action history, and transition updates on one CUDA device.  Checkpoint I/O,
initialization, validation, and result readback are deliberately outside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from expert.minimal_scaling_runner import _slice_batch, load_scan_cell


def _digest_arrays(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_inference_model(checkpoint: Path, *, device: str):
    import torch

    from expert.mapf_gpt_runtime import GPT, GPTConfig, mapf_gpt_config

    resolved = checkpoint.expanduser().resolve()
    payload = torch.load(resolved, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"invalid MAPF-GPT checkpoint: {resolved}")
    runtime_state = payload.get("runtime_state") or {}
    model_args = payload.get("model_args")
    if isinstance(model_args, dict):
        allowed = set(GPTConfig.__dataclass_fields__)
        config = GPTConfig(
            **{key: value for key, value in model_args.items() if key in allowed}
        )
        model_size = str(runtime_state.get("model_size", resolved.stem)).upper()
    else:
        model_size = str(runtime_state.get("model_size", "2M")).upper()
        config = mapf_gpt_config(model_size)
    state = {
        key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key: value
        for key, value in payload["model"].items()
    }
    model = GPT(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload, model_size


def _build_resident_components(batch, *, device: str, horizon: int):
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
        20260728,
        "standard_mapf",
        int(horizon),
    )
    simulator.set_real_map_extents(extents)
    simulator.load_state(
        torch.as_tensor(batch.positions.astype(np.int16), device=device).contiguous(),
        torch.as_tensor(batch.goals.astype(np.int16), device=device).contiguous(),
        torch.as_tensor(batch.arrived.astype(np.uint8), device=device).contiguous(),
        torch.zeros(batch.num_envs, dtype=torch.int32, device=device),
    )
    builder = ext.MapfGPTObservationBuilder(grids, batch.num_agents)
    builder.reset_histories()
    return simulator, builder


def _infer_actions(model, tokens, output, *, microbatch_size: int) -> int:
    import torch

    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    microbatches = 0
    with torch.no_grad():
        for start in range(0, int(tokens.shape[0]), int(microbatch_size)):
            end = min(start + int(microbatch_size), int(tokens.shape[0]))
            output[start:end].copy_(
                model.act(tokens[start:end], do_sample=False).to(torch.uint8)
            )
            microbatches += 1
    return microbatches


def _resident_temporal_parity(
    model,
    batch,
    *,
    steps: int,
    microbatch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    from expert.mapf_gpt_schema import tokenize_stage_reference

    if steps <= 0:
        return {"steps": 0, "token_mismatches": 0, "history_mismatches": 0}
    parity_batch = _slice_batch(batch, np.asarray([0], dtype=np.int64))
    simulator, builder = _build_resident_components(
        parity_batch, device=device, horizon=max(steps, 1)
    )
    total_agents = parity_batch.num_agents
    actions_flat = torch.empty(total_agents, dtype=torch.uint8, device=device)
    actions = actions_flat.view(1, total_agents)
    history = np.full((total_agents, 5), 5, dtype=np.uint16)
    token_hashes: list[str] = []
    action_hashes: list[str] = []
    for _step in range(steps):
        builder.build_tokens_from_state(
            simulator.cur_x,
            simulator.cur_y,
            simulator.goal_x,
            simulator.goal_y,
        )
        positions = torch.stack(
            (simulator.cur_x[0], simulator.cur_y[0]), dim=-1
        ).cpu().numpy().astype(np.uint16, copy=False)
        goals = torch.stack(
            (simulator.goal_x[0], simulator.goal_y[0]), dim=-1
        ).cpu().numpy().astype(np.uint16, copy=False)
        rows = np.zeros((total_agents, 13), dtype=np.uint16)
        rows[:, 1] = np.arange(total_agents, dtype=np.uint16)
        rows[:, 2:4] = positions
        rows[:, 4:6] = goals
        rows[:, 8:13] = history
        reference_tokens, _ = tokenize_stage_reference(
            np.asarray(parity_batch.grids),
            rows,
            num_agents=total_agents,
        )
        actual_tokens = builder.tokens.cpu()
        expected_tokens = torch.from_numpy(reference_tokens)
        if not torch.equal(actual_tokens, expected_tokens):
            mismatch = int((actual_tokens != expected_tokens).sum().item())
            raise RuntimeError(
                f"MAPF-GPT resident temporal token parity failed: {mismatch}"
            )
        _infer_actions(
            model,
            builder.tokens,
            actions_flat,
            microbatch_size=microbatch_size,
        )
        active = builder.active_mask.view(1, total_agents)
        actions.masked_fill_(~active, 0)
        action_host = actions[0].cpu().numpy()
        history[:, :-1] = history[:, 1:]
        history[:, -1] = action_host
        builder.append_actions(actions)
        if not np.array_equal(
            builder.histories.cpu().numpy().astype(np.uint16, copy=False),
            history,
        ):
            raise RuntimeError("MAPF-GPT resident action-history parity failed")
        token_hashes.append(_digest_arrays(tokens=reference_tokens))
        action_hashes.append(_digest_arrays(actions=action_host))
        simulator.update_actions(actions)
        simulator.step_sim_only()
    return {
        "steps": steps,
        "num_envs": 1,
        "num_agents": total_agents,
        "token_mismatches": 0,
        "history_mismatches": 0,
        "token_sha256s": token_hashes,
        "action_sha256s": action_hashes,
    }


def run_p0_resident(
    *,
    manifest: Path,
    scan_cell: str,
    checkpoint: Path,
    horizon: int = 120,
    microbatch_size: int = 256,
    parity_steps: int = 1,
    device: str = "cuda:0",
) -> dict[str, Any]:
    import torch

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    cell, batch = load_scan_cell(manifest, scan_cell)
    if horizon > batch.horizon:
        raise ValueError("horizon exceeds frozen input horizon")
    model, payload, model_size = _load_inference_model(
        checkpoint, device=device
    )
    temporal_parity = _resident_temporal_parity(
        model,
        batch,
        steps=parity_steps,
        microbatch_size=microbatch_size,
        device=device,
    )
    simulator, builder = _build_resident_components(
        batch, device=device, horizon=horizon
    )
    total_agents = batch.num_envs * batch.num_agents
    actions_flat = torch.empty(total_agents, dtype=torch.uint8, device=device)
    actions = actions_flat.view(batch.num_envs, batch.num_agents)
    trajectory = torch.empty(
        (horizon, batch.num_envs, batch.num_agents),
        dtype=torch.uint8,
        device=device,
    )

    # Warm only model kernels; do not advance the measured episode or initialize
    # its goal cache.
    warm_tokens = torch.full(
        (min(total_agents, microbatch_size), 256),
        66,
        dtype=torch.int32,
        device=device,
    )
    warm_actions = torch.empty(
        warm_tokens.shape[0], dtype=torch.uint8, device=device
    )
    _infer_actions(
        model, warm_tokens, warm_actions, microbatch_size=microbatch_size
    )
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_started = time.perf_counter()
    total_microbatches = 0
    for step in range(horizon):
        builder.build_tokens_from_state(
            simulator.cur_x,
            simulator.cur_y,
            simulator.goal_x,
            simulator.goal_y,
        )
        total_microbatches += _infer_actions(
            model,
            builder.tokens,
            actions_flat,
            microbatch_size=microbatch_size,
        )
        active_envs = ~(
            simulator.terminated.to(torch.bool)
            | simulator.truncated.to(torch.bool)
        )
        active_agents = (
            builder.active_mask.view(batch.num_envs, batch.num_agents)
            & active_envs[:, None]
        )
        actions.masked_fill_(~active_agents, 0)
        trajectory[step].copy_(actions)
        builder.append_actions(actions)
        simulator.update_actions(actions)
        simulator.step_sim_only()
    end_event.record()
    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - wall_started
    gpu_ms = float(start_event.elapsed_time(end_event))
    peak_allocated = int(torch.cuda.max_memory_allocated(torch.device(device)))
    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(device)))

    diagnostics = builder.diagnostics.detach().cpu().tolist()
    if any(diagnostics):
        raise RuntimeError(
            f"MAPF-GPT resident builder diagnostics are non-zero: {diagnostics}"
        )
    trajectory_host = trajectory.cpu().numpy()
    final_positions = torch.stack(
        (simulator.cur_x, simulator.cur_y), dim=-1
    ).to(torch.int16).cpu().numpy()
    arrived = simulator.arrived.to(torch.bool).cpu().numpy()
    terminated = simulator.terminated.to(torch.bool).cpu().numpy()
    truncated = simulator.truncated.to(torch.bool).cpu().numpy()
    step_counts = simulator.step_counts.cpu().numpy()
    env_steps = batch.num_envs * horizon
    agent_steps = env_steps * batch.num_agents
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "P0-MAPF-GPT-resident-inference",
        "scan_cell": scan_cell,
        "cell": cell,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_optimizer_step": payload.get(
            "optimizer_step", payload.get("iter_num")
        ),
        "model_size": model_size,
        "num_envs": batch.num_envs,
        "num_agents": batch.num_agents,
        "horizon": horizon,
        "microbatch_size": microbatch_size,
        "microbatches_per_step": total_microbatches / horizon,
        "wall_s": wall_s,
        "gpu_total_ms": gpu_ms,
        "wall_ms_per_batched_step": wall_s / horizon * 1000.0,
        "gpu_ms_per_batched_step": gpu_ms / horizon,
        "env_steps_s": env_steps / wall_s,
        "agent_steps_s": agent_steps / wall_s,
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "resident_loop_h2d_bytes": 0,
        "resident_loop_d2h_bytes": 0,
        "validation_in_timed_region": False,
        "monitoring_in_timed_region": False,
        "cold_goal_cache_in_timed_region": True,
        "action_selection": "argmax",
        "history_length": 5,
        "history_update_order": "predict_then_append_executed_action",
        "builder_diagnostics": diagnostics,
        "individual_success_rate": float(arrived.mean()),
        "complete_success_rate": float(terminated.mean()),
        "truncation_rate": float(truncated.mean()),
        "step_count_mean": float(step_counts.mean()),
        "trajectory_action_sha256": _digest_arrays(actions=trajectory_host),
        "final_state_sha256": _digest_arrays(
            positions=final_positions,
            arrived=arrived,
            terminated=terminated,
            truncated=truncated,
            step_counts=step_counts,
        ),
        "input_semantic_sha256": batch.semantic_sha256,
        "temporal_parity": temporal_parity,
    }


def _write(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: report.get(key)
        for key in (
            "status",
            "num_envs",
            "num_agents",
            "horizon",
            "wall_ms_per_batched_step",
            "env_steps_s",
            "agent_steps_s",
            "individual_success_rate",
            "complete_success_rate",
            "truncation_rate",
            "exception_type",
            "exception_message",
        )
        if report.get(key) is not None
    }
    (output_dir / "stdout.log").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one MAPF-GPT GPU-resident closed-loop row."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--microbatch-size", type=int, default=256)
    parser.add_argument("--parity-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_p0_resident(
            manifest=args.manifest,
            scan_cell=args.scan_cell,
            checkpoint=args.checkpoint,
            horizon=args.horizon,
            microbatch_size=args.microbatch_size,
            parity_steps=args.parity_steps,
            device=args.device,
        )
        _write(args.output_dir, report)
        print((args.output_dir / "stdout.log").read_text(encoding="utf-8").strip())
        return 0
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": (
                "capacity_failure"
                if "out of memory" in str(error).lower()
                else "failed"
            ),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "scan_cell": args.scan_cell,
        }
        _write(args.output_dir, report)
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_infer_actions", "run_p0_resident"]
