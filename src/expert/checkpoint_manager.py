from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


def model_state_sha256(model: Any) -> str:
    """Return a deterministic hash of names, metadata, and tensor bytes."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        # Optimizer-adjacent model buffers can be scalar tensors.  Flatten to
        # one element before reinterpreting bytes so scalar integer buffers
        # hash identically to non-scalar parameters.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class SavedCheckpoint:
    loss: float
    optimizer_step: int
    path: str
    val_overall_accuracy: float | None = None
    val_nonstay_accuracy: float | None = None

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "optimizer_step": self.optimizer_step,
            "path": self.path,
            "val_overall_accuracy": self.val_overall_accuracy,
            "val_nonstay_accuracy": self.val_nonstay_accuracy,
        }


class TopKCheckpointManager:
    """Checkpoint manager supporting latest-K or validation-ranked retention."""

    def __init__(
        self,
        checkpoint_dir: str | None,
        *,
        top_k: int = 3,
        save_interval_steps: int = 1000,
        mode_label: str = "",
        logger: Callable[[str], None] | None = None,
        selection_mode: str = "latest",
        validation_evaluator: Callable[[Any, int], dict[str, Any]] | None = None,
        latest_checkpoint_name: str = "ckpt_latest.pt",
    ):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if save_interval_steps <= 0:
            raise ValueError(
                "save_interval_steps must be positive, "
                f"got {save_interval_steps}"
            )
        if selection_mode not in {"latest", "validation_accuracy"}:
            raise ValueError(
                "selection_mode must be one of {'latest', 'validation_accuracy'}, "
                f"got {selection_mode!r}"
            )
        self.mode_label = mode_label
        self.top_k = int(top_k)
        self.save_interval_steps = int(save_interval_steps)
        self.logger = logger
        self.selection_mode = str(selection_mode)
        self.validation_evaluator = validation_evaluator
        self.keep_latest_checkpoint = self.selection_mode == "validation_accuracy"
        self.directory: str | None = None
        self.latest_checkpoint_path: str | None = None
        self._saved: list[SavedCheckpoint] = []
        self._seen_steps: dict[int, SavedCheckpoint | None] = {}
        self._last_validation_metrics: dict[str, Any] | None = None
        self._checkpoint_history: list[dict[str, Any]] = []
        self._created_at = time.perf_counter()
        if checkpoint_dir:
            resolved = str(Path(checkpoint_dir).expanduser().resolve())
            Path(resolved).mkdir(parents=True, exist_ok=True)
            self.directory = resolved
            self.latest_checkpoint_path = str(Path(resolved) / latest_checkpoint_name)
        if (
            self.directory is not None
            and self.selection_mode == "validation_accuracy"
            and self.validation_evaluator is None
        ):
            raise ValueError(
                "validation_evaluator is required when selection_mode='validation_accuracy'"
            )

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def maybe_save(
        self,
        *,
        loss: float,
        runtime,
        optimizer_step: int,
        extra_meta: dict | None = None,
        force: bool = False,
    ) -> SavedCheckpoint | None:
        if not self.enabled:
            return None
        step = int(optimizer_step)
        if not force and step % self.save_interval_steps != 0:
            return None
        if step in self._seen_steps:
            return self._seen_steps[step]

        validation_metrics = self._evaluate_validation(runtime, step)
        history_row = {
            "optimizer_step": step,
            "training_loss": float(loss),
            "checkpoint_elapsed_s": time.perf_counter() - self._created_at,
            **validation_metrics,
        }
        if extra_meta is not None and "training_elapsed_s" in extra_meta:
            history_row["training_elapsed_s"] = float(
                extra_meta["training_elapsed_s"]
            )
        self._checkpoint_history.append(history_row)
        payload = self._build_payload(
            loss=float(loss),
            runtime=runtime,
            optimizer_step=step,
            extra_meta=extra_meta,
            validation_metrics=validation_metrics,
        )
        self._write_latest_checkpoint(payload)

        if self.selection_mode == "latest":
            saved = self._save_latest_ranked_checkpoint(
                payload=payload,
                loss=float(loss),
                optimizer_step=step,
            )
            self._seen_steps[step] = saved
            return saved

        saved = self._save_validation_ranked_checkpoint(
            payload=payload,
            loss=float(loss),
            optimizer_step=step,
            validation_metrics=validation_metrics,
        )
        self._seen_steps[step] = saved
        return saved

    def snapshot(self) -> list[dict]:
        return [item.to_dict() for item in self._saved]

    def last_validation_metrics(self) -> dict[str, Any] | None:
        if self._last_validation_metrics is None:
            return None
        return dict(self._last_validation_metrics)

    def checkpoint_history(self) -> list[dict[str, Any]]:
        """Return every checkpoint/validation observation, including evicted rows."""
        return [dict(item) for item in self._checkpoint_history]

    def _evaluate_validation(self, runtime, optimizer_step: int) -> dict[str, Any]:
        if self.selection_mode != "validation_accuracy":
            self._last_validation_metrics = None
            return {}
        t0 = time.perf_counter()
        metrics = dict(self.validation_evaluator(runtime, optimizer_step) or {})
        metrics["validation_elapsed_s"] = time.perf_counter() - t0
        if "val_overall_accuracy" not in metrics:
            if "overall_accuracy" not in metrics:
                raise ValueError(
                    "validation_evaluator must return 'val_overall_accuracy' or 'overall_accuracy'"
                )
            metrics["val_overall_accuracy"] = float(metrics["overall_accuracy"])
        metrics["val_overall_accuracy"] = float(metrics["val_overall_accuracy"])
        if "val_nonstay_accuracy" in metrics and metrics["val_nonstay_accuracy"] is not None:
            metrics["val_nonstay_accuracy"] = float(metrics["val_nonstay_accuracy"])
        self._last_validation_metrics = metrics
        self._log(
            "[val] "
            f"step={optimizer_step} "
            f"overall_accuracy={metrics['val_overall_accuracy']:.6f}"
            + (
                " "
                f"nonstay_accuracy={metrics['val_nonstay_accuracy']:.6f}"
                if metrics.get("val_nonstay_accuracy") is not None
                else ""
            )
            + (
                " "
                f"trajectories={int(metrics['validation_num_trajectories'])}"
                if metrics.get("validation_num_trajectories") is not None
                else ""
            )
            + (
                " "
                f"elapsed_s={float(metrics['validation_elapsed_s']):.6f}"
                if metrics.get("validation_elapsed_s") is not None
                else ""
            )
        )
        return metrics

    def _build_payload(
        self,
        *,
        loss: float,
        runtime,
        optimizer_step: int,
        extra_meta: dict | None,
        validation_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        meta = dict(extra_meta or {})
        meta["checkpoint_interval"] = self.save_interval_steps
        payload = {
            "model": runtime.model.state_dict(),
            "optimizer": runtime.optimizer.state_dict(),
            "optimizer_step": int(optimizer_step),
            "loss": float(loss),
            "mode": self.mode_label,
            "selection_mode": self.selection_mode,
            "meta": meta,
            **validation_metrics,
        }
        checkpoint_state = getattr(runtime, "checkpoint_state", None)
        if callable(checkpoint_state):
            payload["runtime_state"] = checkpoint_state()
        return payload

    def _save_latest_ranked_checkpoint(
        self,
        *,
        payload: dict[str, Any],
        loss: float,
        optimizer_step: int,
    ) -> SavedCheckpoint:
        filename = f"ckpt_step{int(optimizer_step):08d}_loss{float(loss):.6f}.pt"
        final_path = Path(self.directory) / filename
        self._write_payload(final_path, payload)
        saved = SavedCheckpoint(
            loss=float(loss),
            optimizer_step=int(optimizer_step),
            path=str(final_path),
        )
        self._saved.append(saved)
        self._saved.sort(key=lambda item: item.optimizer_step)
        self._log(
            f"[ckpt] saved step={saved.optimizer_step} loss={saved.loss:.6f} path={saved.path}"
        )
        if len(self._saved) > self.top_k:
            evicted = self._saved.pop(0)
            if os.path.exists(evicted.path):
                os.remove(evicted.path)
            self._log(
                f"[ckpt] evicted step={evicted.optimizer_step} loss={evicted.loss:.6f} path={evicted.path}"
            )
        return saved

    def _save_validation_ranked_checkpoint(
        self,
        *,
        payload: dict[str, Any],
        loss: float,
        optimizer_step: int,
        validation_metrics: dict[str, Any],
    ) -> SavedCheckpoint | None:
        overall_accuracy = float(validation_metrics["val_overall_accuracy"])
        filename = (
            f"ckpt_step{int(optimizer_step):08d}_valacc{overall_accuracy:.6f}.pt"
        )
        final_path = Path(self.directory) / filename
        self._write_payload(final_path, payload)
        saved = SavedCheckpoint(
            loss=float(loss),
            optimizer_step=int(optimizer_step),
            path=str(final_path),
            val_overall_accuracy=overall_accuracy,
            val_nonstay_accuracy=(
                None
                if validation_metrics.get("val_nonstay_accuracy") is None
                else float(validation_metrics["val_nonstay_accuracy"])
            ),
        )
        self._saved.append(saved)
        self._saved.sort(key=self._validation_sort_key)
        if len(self._saved) > self.top_k:
            evicted = self._saved.pop(-1)
            if os.path.exists(evicted.path):
                os.remove(evicted.path)
            if evicted.path == saved.path:
                self._log(
                    "[ckpt] skipped "
                    f"step={saved.optimizer_step} "
                    f"val_overall_accuracy={saved.val_overall_accuracy:.6f}"
                )
                return None
            self._log(
                "[ckpt] evicted "
                f"step={evicted.optimizer_step} "
                f"val_overall_accuracy={evicted.val_overall_accuracy:.6f} "
                f"path={evicted.path}"
            )
        self._log(
            "[ckpt] saved "
            f"step={saved.optimizer_step} "
            f"val_overall_accuracy={saved.val_overall_accuracy:.6f} "
            f"path={saved.path}"
        )
        return saved

    def _write_latest_checkpoint(self, payload: dict[str, Any]) -> None:
        if not self.keep_latest_checkpoint or self.latest_checkpoint_path is None:
            return
        self._write_payload(Path(self.latest_checkpoint_path), payload)
        self._log(f"[ckpt] updated latest path={self.latest_checkpoint_path}")

    @staticmethod
    def _validation_sort_key(item: SavedCheckpoint) -> tuple[float, int, float]:
        accuracy = item.val_overall_accuracy
        if accuracy is None:
            accuracy = -math.inf
        return (-float(accuracy), -int(item.optimizer_step), float(item.loss))

    @staticmethod
    def _write_payload(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def __iter__(self) -> Iterable[SavedCheckpoint]:
        return iter(self._saved)
