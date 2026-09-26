import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
for path in (PROJECT_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch

import grid_world_cpp as ext
from expert.expert_running import (
    RESET_FLAG_COL,
    LacamExpertPolicy,
    initial_refresh_flags,
    no_refresh_flags,
    resolve_expert_timeouts,
)
from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter
from mapf_cuda.models.checkpoints import load_magat_checkpoint
from mapf_cuda.simulation.grids import extract_single_env_grid
from mapf_cuda.simulation.pogema_envs import (
    build_random_standard_env,
    build_topology_standard_env,
)
from mapf_cuda.training.topology_async import _build_step_data
from expert.expert_running import put_maps_into_registry


def _strip_prefix_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _remap_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    replacements = (
        ("cnn.compressMLP.", "cnn.compress_mlp."),
        ("edge_attr_cnn.", "edge_attr_encoder.net."),
        ("actionsMLP.", "actions_mlp."),
    )
    remapped = {}
    for key, value in state_dict.items():
        mapped_key = key
        for old, new in replacements:
            if mapped_key.startswith(old):
                mapped_key = new + mapped_key[len(old) :]
                break
        remapped[mapped_key] = value
    return remapped


def _checkpoint_to_state_dict(checkpoint_path: str, device: str) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except RuntimeError as exc:
        if "TorchScript archive" not in str(exc):
            raise
        loaded = torch.jit.load(checkpoint_path, map_location=device)

    if hasattr(loaded, "state_dict") and not isinstance(loaded, dict):
        loaded = loaded.state_dict()
    if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
        loaded = loaded["model"]
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} must provide a state_dict dict, got {type(loaded).__name__}"
        )
    return loaded


def _load_checkpoint_into_runtime(runtime: MAGATRuntimeAdapter, checkpoint_path: str, device: str) -> None:
    state_dict = _checkpoint_to_state_dict(checkpoint_path, device)
    candidates = [state_dict]
    stripped = _strip_prefix_from_state_dict(state_dict, "_orig_mod.")
    if stripped != state_dict:
        candidates.append(stripped)

    remapped_candidates = []
    for candidate in candidates:
        remapped = _remap_checkpoint_keys(candidate)
        if remapped != candidate:
            remapped_candidates.append(remapped)
    candidates.extend(remapped_candidates)

    last_error = None
    for candidate in candidates:
        try:
            runtime.model.load_state_dict(candidate, strict=True)
            runtime.model.eval()
            return
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Checkpoint at {checkpoint_path} is not compatible with FixedMAGATPlusModel"
    ) from last_error


def _synchronize_device(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _validate_checkpoint_path(checkpoint: str) -> str:
    resolved = str(Path(checkpoint).expanduser().resolve())
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def _predict_model_actions(
    runtime: MAGATRuntimeAdapter,
    simulator,
    observations,
    expert_actions: np.ndarray,
    refresh_flags: np.ndarray,
    device: str,
) -> np.ndarray:
    raw_batch_np = _build_step_data(
        observations,
        expert_actions.astype(np.uint16, copy=False),
        env_id=0,
        reset_flag=refresh_flags,
    )
    raw_batch = torch.from_numpy(raw_batch_np.astype(np.int16, copy=False)).to(device)
    _synchronize_device(device)

    agents_to_update = raw_batch[raw_batch[:, RESET_FLAG_COL] != 0]
    if agents_to_update.shape[0] > 0:
        simulator.update_energy_maps(agents_to_update, agents_to_update.shape[0])
    simulator.refresh_compact_state_from_raw_batch(raw_batch)
    data = runtime.build_batch(simulator, raw_batch)

    with torch.no_grad():
        logits = runtime.model(data.x, data)
    _synchronize_device(device)
    return logits.argmax(dim=-1).to(torch.int64).cpu().numpy()


def _build_step_result(step_idx: int, expert_actions: np.ndarray, model_actions: np.ndarray) -> dict[str, Any]:
    matches = model_actions == expert_actions
    mismatches = [
        {
            "agent_idx": int(agent_idx),
            "expert_action": int(expert_actions[agent_idx]),
            "model_action": int(model_actions[agent_idx]),
        }
        for agent_idx in np.flatnonzero(~matches)
    ]
    correct = int(matches.sum())
    total = int(matches.shape[0])
    return {
        "step": step_idx,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "mismatches": mismatches,
        "expert_actions": expert_actions.tolist(),
        "model_actions": model_actions.tolist(),
    }


def _advance_expert_trajectory(env, policy: LacamExpertPolicy, expert_actions: np.ndarray):
    _, _, terminated, truncated, _ = env.step(expert_actions)
    if all(terminated) or all(truncated):
        env.reset()
        observations = env.env.unwrapped._obs()
        policy.reset_states(env)
        refresh_flags = initial_refresh_flags(len(observations))
    else:
        observations = env.env.unwrapped._obs()
        refresh_flags = no_refresh_flags(len(observations))
    return observations, refresh_flags


def compare_expert_model_accuracy(
    *,
    checkpoint: str,
    num_steps: int = 3,
    num_agents: int = 8,
    map_size: int = 32,
    density: float = 0.0,
    map_name: str | None = None,
    maps_path: str = "maps/maps.yaml",
    seed: int = 42,
    max_episode_steps: int = 64,
    device: str = "cuda:0",
    expert_timeouts=None,
) -> dict[str, Any]:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not str(device).startswith("cuda"):
        raise ValueError(f"This script requires a CUDA device, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison script.")

    checkpoint_path = _validate_checkpoint_path(checkpoint)
    if map_name:
        put_maps_into_registry(maps_path)
        env = build_topology_standard_env(
            num_agents=num_agents,
            map_name=map_name,
            seed=seed,
            max_episode_steps=max_episode_steps,
            collision_system="soft",
        )
    else:
        env = build_random_standard_env(
            num_agents=num_agents,
            map_size=map_size,
            density=density,
            seed=seed,
            max_episode_steps=max_episode_steps,
            collision_system="soft",
        )
    simulator = ext.StatelessGridWorldSimulator(
        extract_single_env_grid(env, device=device),
        num_agents,
        3,
    )
    runtime = MAGATRuntimeAdapter(device=device)
    load_magat_checkpoint(runtime, checkpoint_path, device)

    resolved_expert_timeouts = resolve_expert_timeouts(expert_timeouts)
    policy = LacamExpertPolicy(timeouts=resolved_expert_timeouts)
    policy.reset_states(env)

    observations = env.env.unwrapped._obs()
    refresh_flags = initial_refresh_flags(len(observations))

    per_step = []
    overall_correct = 0
    overall_total = 0

    for step_idx in range(num_steps):
        expert_actions = policy.act(observations).astype(np.int64, copy=False)
        model_actions = _predict_model_actions(
            runtime,
            simulator,
            observations,
            expert_actions,
            refresh_flags,
            device,
        )
        step_result = _build_step_result(step_idx, expert_actions, model_actions)
        per_step.append(step_result)
        overall_correct += step_result["correct"]
        overall_total += step_result["total"]
        observations, refresh_flags = _advance_expert_trajectory(env, policy, expert_actions)

    return {
        "checkpoint": checkpoint_path,
        "num_steps": num_steps,
        "num_agents": num_agents,
        "map_size": map_size,
        "density": density,
        "map_name": map_name,
        "maps_path": maps_path,
        "seed": seed,
        "max_episode_steps": max_episode_steps,
        "device": device,
        "expert_timeouts": list(resolved_expert_timeouts),
        "comparison_mode": "same-state expert-trajectory argmax accuracy",
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "per_step": per_step,
    }


def _format_mismatches(mismatches: list[dict[str, int]]) -> str:
    if not mismatches:
        return "none"
    return "; ".join(
        f"agent[{item['agent_idx']}] expert:{item['expert_action']} model:{item['model_action']}"
        for item in mismatches
    )


def _print_report(result: dict[str, Any]) -> None:
    print("Step-by-step accuracy (model argmax vs expert on the same expert-trajectory states):")
    for item in result["per_step"]:
        print(
            f"  step {item['step']}: {item['correct']}/{item['total']} correct "
            f"({item['accuracy'] * 100:.2f}%)"
        )
        print(f"    mismatches: {_format_mismatches(item['mismatches'])}")
    print(
        f"Overall: {result['overall_correct']}/{result['overall_total']} "
        f"({result['overall_accuracy'] * 100:.2f}%)"
    )


def _parse_expert_timeouts(raw_value: str | None):
    if raw_value is None or raw_value == "":
        return None
    return [float(part.strip()) for part in raw_value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed MAGAT+ argmax actions with expert actions on the first few expert-trajectory steps."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=8)
    parser.add_argument("--map_size", type=int, default=32)
    parser.add_argument("--density", type=float, default=0.0)
    parser.add_argument("--map_name", type=str, default="")
    parser.add_argument("--maps_path", type=str, default="maps/maps.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--expert_timeouts", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    result = compare_expert_model_accuracy(
        checkpoint=args.checkpoint,
        num_steps=args.num_steps,
        num_agents=args.num_agents,
        map_size=args.map_size,
        density=args.density,
        map_name=args.map_name or None,
        maps_path=args.maps_path,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        expert_timeouts=_parse_expert_timeouts(args.expert_timeouts),
    )
    _print_report(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()
