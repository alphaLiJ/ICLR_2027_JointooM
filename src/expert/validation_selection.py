from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

import grid_world_cpp as ext
from expert.controlled_arrived_ab import load_frozen_trajectory, replay_frozen_batches
from expert.expert_running import put_maps_into_registry
from mapf_cuda.simulation.grids import extract_single_env_grid
from mapf_cuda.training.topology_async import _build_pogema_topology_map_env


class FrozenValidationEvaluator:
    def __init__(
        self,
        trajectory_paths: Iterable[str | Path],
        *,
        device: str = "cuda:0",
    ):
        resolved_paths = [str(Path(path).expanduser().resolve()) for path in trajectory_paths]
        if not resolved_paths:
            raise ValueError("validation_trajectories must contain at least one path")
        self.device = device
        self._entries = [self._load_entry(path) for path in resolved_paths]

    def __call__(self, runtime, optimizer_step: int) -> dict[str, Any]:
        was_training = runtime.model.training
        runtime.model.eval()
        overall_correct = 0
        overall_total = 0
        nonstay_correct = 0
        nonstay_total = 0
        try:
            with torch.no_grad():
                for entry in self._entries:
                    simulator = entry["simulator"]
                    metrics = replay_frozen_batches(
                        raw_batches=entry["trajectory"].raw_batches,
                        simulator=simulator,
                        device=self.device,
                        predict_actions=lambda sim, raw_batch: self._predict_actions(runtime, sim, raw_batch),
                    )
                    overall_correct += int(metrics["overall_correct"])
                    overall_total += int(metrics["overall_total"])
                    nonstay_correct += int(metrics["nonstay_correct"])
                    nonstay_total += int(metrics["nonstay_total"])
        finally:
            runtime.model.train(was_training)
        return {
            "optimizer_step": int(optimizer_step),
            "val_overall_accuracy": (
                overall_correct / overall_total if overall_total else 0.0
            ),
            "val_nonstay_accuracy": (
                nonstay_correct / nonstay_total if nonstay_total else 0.0
            ),
            "validation_num_trajectories": len(self._entries),
            "validation_trajectory_sha256s": [entry["trajectory"].digest for entry in self._entries],
        }

    def _load_entry(self, path: str) -> dict[str, Any]:
        trajectory = load_frozen_trajectory(path, expected={})
        metadata = trajectory.metadata
        put_maps_into_registry(metadata["maps_path"])
        env = _build_pogema_topology_map_env(
            num_agents=metadata["num_agents"],
            map_name=metadata["map_name"],
            seed=metadata["seed"],
            max_episode_steps=metadata["max_episode_steps"],
            collision_system="soft",
        )
        simulator = ext.StatelessGridWorldSimulator(
            extract_single_env_grid(env, device=self.device),
            metadata["num_agents"],
            3,
        )
        return {
            "path": path,
            "trajectory": trajectory,
            "simulator": simulator,
        }

    def _predict_actions(self, runtime, simulator, raw_batch):
        batch = runtime.build_batch(simulator, raw_batch)
        logits = runtime.model(batch.x, batch)
        return logits.argmax(dim=-1).to(torch.int64).cpu().numpy()


def build_frozen_validation_evaluator(
    trajectory_paths: Iterable[str | Path],
    *,
    device: str = "cuda:0",
) -> FrozenValidationEvaluator:
    return FrozenValidationEvaluator(trajectory_paths, device=device)
