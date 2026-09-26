from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from expert.mapf_gpt_schema import MAPFGPT_CONTEXT_SIZE


def _validate_arrays(
    tokens: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    token_array = np.asarray(tokens)
    label_array = np.asarray(labels)
    if token_array.dtype != np.int32:
        raise ValueError(f"tokens must use int32 dtype, got {token_array.dtype}")
    if token_array.ndim != 2 or token_array.shape[1] != MAPFGPT_CONTEXT_SIZE:
        raise ValueError(
            f"tokens must have shape [N, {MAPFGPT_CONTEXT_SIZE}], got {token_array.shape}"
        )
    if label_array.dtype != np.int64:
        raise ValueError(f"labels must use int64 dtype, got {label_array.dtype}")
    if label_array.shape != (token_array.shape[0],):
        raise ValueError(
            f"labels must have shape [{token_array.shape[0]}], got {label_array.shape}"
        )
    if np.any(token_array < 0) or np.any(token_array > 66):
        raise ValueError("tokens must be in [0, 66]")
    if np.any(label_array < 0) or np.any(label_array > 4):
        raise ValueError("labels must be in [0, 4]")
    return token_array, label_array


def save_frozen_mapf_gpt_validation(
    path: str | Path,
    *,
    tokens: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> str:
    token_array, label_array = _validate_arrays(tokens, labels)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with tmp_path.open("wb") as output:
        np.savez_compressed(
            output,
            tokens=token_array,
            labels=label_array,
            metadata_json=np.asarray(metadata_json),
        )
    os.replace(tmp_path, resolved)
    return str(resolved)


class FrozenMapfGPTValidationEvaluator:
    def __init__(
        self,
        dataset_paths: Iterable[str | Path],
        *,
        device: str | torch.device = "cuda:0",
        batch_size: int = 256,
    ):
        resolved = [Path(path).expanduser().resolve() for path in dataset_paths]
        if not resolved:
            raise ValueError("validation_datasets must contain at least one path")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._entries = [self._load(path) for path in resolved]

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as payload:
            tokens, labels = _validate_arrays(payload["tokens"], payload["labels"])
            metadata = json.loads(str(payload["metadata_json"].item()))
        return {
            "path": str(path),
            "digest": digest,
            "tokens": torch.from_numpy(tokens.copy()),
            "labels": torch.from_numpy(labels.copy()),
            "metadata": metadata,
        }

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
                    tokens = entry["tokens"]
                    labels = entry["labels"]
                    for start in range(0, tokens.shape[0], self.batch_size):
                        end = min(start + self.batch_size, tokens.shape[0])
                        batch_tokens = tokens[start:end].to(self.device)
                        predictions = runtime.model.act(batch_tokens).to("cpu")
                        batch_labels = labels[start:end]
                        if predictions.shape != batch_labels.shape:
                            raise RuntimeError(
                                "MAPF-GPT validation prediction shape mismatch: "
                                f"predictions={tuple(predictions.shape)}, "
                                f"labels={tuple(batch_labels.shape)}"
                            )
                        matches = predictions.eq(batch_labels)
                        nonstay = batch_labels.ne(0)
                        overall_correct += int(matches.sum().item())
                        overall_total += int(batch_labels.numel())
                        nonstay_correct += int((matches & nonstay).sum().item())
                        nonstay_total += int(nonstay.sum().item())
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
            "validation_trajectory_sha256s": [
                entry["digest"] for entry in self._entries
            ],
        }


def build_frozen_mapf_gpt_validation_evaluator(
    dataset_paths: Iterable[str | Path],
    *,
    device: str | torch.device = "cuda:0",
    batch_size: int = 256,
) -> FrozenMapfGPTValidationEvaluator:
    return FrozenMapfGPTValidationEvaluator(
        dataset_paths, device=device, batch_size=batch_size
    )


__all__ = [
    "FrozenMapfGPTValidationEvaluator",
    "build_frozen_mapf_gpt_validation_evaluator",
    "save_frozen_mapf_gpt_validation",
]
