"""Load historical MAGAT checkpoints into the current runtime model."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _strip_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _remap_historical_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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


def _read_state_dict(checkpoint_path: str, device: str) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except RuntimeError as exc:
        if "TorchScript archive" not in str(exc):
            raise
        loaded = torch.jit.load(checkpoint_path, map_location=device)

    if hasattr(loaded, "state_dict") and not isinstance(loaded, dict):
        loaded = loaded.state_dict()
    if (
        isinstance(loaded, dict)
        and "model" in loaded
        and isinstance(loaded["model"], dict)
    ):
        loaded = loaded["model"]
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} must provide a state_dict dict, "
            f"got {type(loaded).__name__}"
        )
    return loaded


def load_magat_checkpoint(runtime, checkpoint_path: str, device: str) -> None:
    """Load current, compiled, or original-MAGAT key layouts strictly."""

    state_dict = _read_state_dict(checkpoint_path, device)
    candidates = [state_dict]
    stripped = _strip_prefix(state_dict, "_orig_mod.")
    if stripped != state_dict:
        candidates.append(stripped)
    for candidate in tuple(candidates):
        remapped = _remap_historical_keys(candidate)
        if remapped != candidate:
            candidates.append(remapped)

    last_error = None
    for candidate in candidates:
        try:
            runtime.model.load_state_dict(candidate, strict=True)
            runtime.model.eval()
            return
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(
        f"Checkpoint at {checkpoint_path} is not compatible with "
        "FixedMAGATPlusModel"
    ) from last_error

