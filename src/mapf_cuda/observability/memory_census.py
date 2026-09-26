"""Allocator- and storage-level memory accounting for MAPF-CUDA.

The helpers in this module are observability-only.  They never clone or move
tensors, and capacity preflight does not attempt the allocation it evaluates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


ALLOWED_OWNERS = frozenset(
    {
        "static_map",
        "dynamic_simulator",
        "compact_state",
        "derived_state",
        "magat_builder",
        "mapf_gpt_builder",
        "model_parameters",
        "model_gradients",
        "optimizer_state",
        "model_ephemeral",
        "transport_gpu",
        "transport_host",
        "input",
    }
)

ALLOWED_LIFETIMES = frozenset(
    {
        "context",
        "instance",
        "episode",
        "goal_epoch",
        "frontier",
        "model_call",
        "run",
    }
)


@dataclass(frozen=True)
class TensorReference:
    """A named tensor together with its unique ownership interpretation."""

    name: str
    tensor: torch.Tensor
    owner: str
    lifetime: str
    kind: str = "tensor"

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"{self.name!r} is not a torch.Tensor")
        if self.owner not in ALLOWED_OWNERS:
            raise ValueError(
                f"unknown memory owner {self.owner!r}; expected one of "
                f"{sorted(ALLOWED_OWNERS)}"
            )
        if self.lifetime not in ALLOWED_LIFETIMES:
            raise ValueError(
                f"unknown lifetime {self.lifetime!r}; expected one of "
                f"{sorted(ALLOWED_LIFETIMES)}"
            )


def _device_label(tensor: torch.Tensor) -> str:
    device = tensor.device
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _storage_metadata(tensor: torch.Tensor) -> tuple[str | None, int]:
    if tensor.numel() == 0:
        return None, 0
    storage = tensor.untyped_storage()
    storage_bytes = int(storage.nbytes())
    if storage_bytes <= 0:
        return None, 0
    storage_id = f"{_device_label(tensor)}:{int(storage.data_ptr())}:{storage_bytes}"
    return storage_id, storage_bytes


def census_tensors(references: Iterable[TensorReference]) -> dict[str, Any]:
    """Return logical and unique-storage accounting for named tensor refs."""

    rows: list[dict[str, Any]] = []
    seen_storage: set[str] = set()
    owner_storage_bytes: defaultdict[str, int] = defaultdict(int)
    owner_logical_bytes: defaultdict[str, int] = defaultdict(int)
    device_storage_bytes: defaultdict[str, int] = defaultdict(int)
    logical_total = 0
    unique_storage_total = 0

    for reference in references:
        tensor = reference.tensor
        logical_bytes = int(tensor.numel()) * int(tensor.element_size())
        storage_id, storage_bytes = _storage_metadata(tensor)
        counted_unique = storage_id is not None and storage_id not in seen_storage
        if counted_unique:
            seen_storage.add(storage_id)
            unique_storage_total += storage_bytes
            owner_storage_bytes[reference.owner] += storage_bytes
            device_storage_bytes[_device_label(tensor)] += storage_bytes

        logical_total += logical_bytes
        owner_logical_bytes[reference.owner] += logical_bytes
        rows.append(
            {
                "name": reference.name,
                "owner": reference.owner,
                "lifetime": reference.lifetime,
                "kind": reference.kind,
                "device": _device_label(tensor),
                "dtype": str(tensor.dtype),
                "shape": [int(size) for size in tensor.shape],
                "logical_bytes": logical_bytes,
                "storage_id": storage_id,
                "storage_bytes": storage_bytes,
                "storage_offset": int(tensor.storage_offset()),
                "is_view": bool(
                    getattr(tensor, "_base", None) is not None
                    or int(tensor.storage_offset()) != 0
                    or (storage_bytes > 0 and logical_bytes != storage_bytes)
                ),
                "is_pinned": bool(tensor.is_pinned()) if tensor.device.type == "cpu" else False,
                "counted_unique_storage": bool(counted_unique),
            }
        )

    return {
        "schema_version": 1,
        "logical_bytes": logical_total,
        "unique_storage_bytes": unique_storage_total,
        "by_owner_logical_bytes": dict(sorted(owner_logical_bytes.items())),
        "by_owner_storage_bytes": dict(sorted(owner_storage_bytes.items())),
        "by_device_storage_bytes": dict(sorted(device_storage_bytes.items())),
        "rows": rows,
    }


def _named_parameters_with_aliases(module: torch.nn.Module):
    try:
        return module.named_parameters(recurse=True, remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return module.named_parameters(recurse=True)


def _named_buffers_with_aliases(module: torch.nn.Module):
    try:
        return module.named_buffers(recurse=True, remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return module.named_buffers(recurse=True)


def module_tensor_references(
    module: torch.nn.Module,
    *,
    prefix: str = "model",
    include_gradients: bool = False,
) -> list[TensorReference]:
    """Enumerate module parameters/buffers and optionally their gradients."""

    references: list[TensorReference] = []
    for name, parameter in _named_parameters_with_aliases(module):
        references.append(
            TensorReference(
                f"{prefix}.parameter.{name}",
                parameter,
                "model_parameters",
                "run",
                "parameter",
            )
        )
        if include_gradients and parameter.grad is not None:
            references.append(
                TensorReference(
                    f"{prefix}.gradient.{name}",
                    parameter.grad,
                    "model_gradients",
                    "run",
                    "gradient",
                )
            )
    for name, buffer in _named_buffers_with_aliases(module):
        references.append(
            TensorReference(
                f"{prefix}.buffer.{name}",
                buffer,
                "model_parameters",
                "run",
                "buffer",
            )
        )
    return references


def inventory_tensor_references(
    inventory: Mapping[str, torch.Tensor],
    *,
    classifications: Mapping[str, tuple[str, str]],
    prefix: str,
    kind: str = "tensor",
) -> list[TensorReference]:
    """Classify a native tensor inventory without allocating tensor copies.

    Native simulator inventories are deliberately strict: every returned key
    must have an explicit ``(owner, lifetime)`` entry.  This makes newly added
    persistent allocations visible to the S3 experiment instead of silently
    assigning them to a catch-all bucket.
    """

    unknown = sorted(set(inventory) - set(classifications))
    if unknown:
        raise ValueError(
            f"unclassified tensors in {prefix!r} inventory: {unknown}; "
            "update the ownership table before collecting memory evidence"
        )
    references: list[TensorReference] = []
    for name, tensor in inventory.items():
        owner, lifetime = classifications[name]
        references.append(
            TensorReference(
                name=f"{prefix}.{name}",
                tensor=tensor,
                owner=owner,
                lifetime=lifetime,
                kind=kind,
            )
        )
    return references


def _walk_tensor_values(value: Any, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _walk_tensor_values(nested, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _walk_tensor_values(nested, f"{prefix}.{index}")


def optimizer_tensor_references(
    optimizer: torch.optim.Optimizer,
    *,
    prefix: str = "optimizer",
) -> list[TensorReference]:
    references: list[TensorReference] = []
    for state_index, (_, state) in enumerate(optimizer.state.items()):
        for name, tensor in _walk_tensor_values(state, f"{prefix}.state.{state_index}"):
            references.append(
                TensorReference(
                    name,
                    tensor,
                    "optimizer_state",
                    "run",
                    "optimizer_state",
                )
            )
    return references


def allocator_snapshot(
    checkpoint: str,
    *,
    device: str | torch.device = "cuda:0",
    census: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("allocator snapshots require a CUDA device")
    allocated = int(torch.cuda.memory_allocated(resolved))
    reserved = int(torch.cuda.memory_reserved(resolved))
    tensor_bytes = (
        int(census.get("by_device_storage_bytes", {}).get(str(resolved), 0))
        if census is not None
        else None
    )
    residual = allocated - tensor_bytes if tensor_bytes is not None else None
    residual_ratio = (
        abs(residual) / allocated if residual is not None and allocated > 0 else None
    )
    return {
        "checkpoint": checkpoint,
        "device": str(resolved),
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
        "tensor_storage_bytes": tensor_bytes,
        "residual_bytes": residual,
        "residual_ratio": residual_ratio,
    }


def reconcile_after_cleanup(
    *,
    allocated_bytes: int,
    tensor_storage_bytes: int,
    cleanup_allocated_bytes: int,
    context_allocated_bytes: int = 0,
) -> dict[str, Any]:
    """Separate process-lifetime CUDA library storage from owned tensors.

    cuBLAS/cuDNN may retain allocator-backed workspaces after every Python
    tensor, model, and simulator has been released.  A post-cleanup allocator
    floor is therefore reported as library workspace rather than pretending it
    is an unenumerated tensor owner.
    """

    allocated = int(allocated_bytes)
    tensors = int(tensor_storage_bytes)
    cleanup = int(cleanup_allocated_bytes)
    context = int(context_allocated_bytes)
    if min(allocated, tensors, cleanup, context) < 0:
        raise ValueError("allocator reconciliation inputs must be non-negative")
    library_workspace = max(0, cleanup - context)
    owned_allocator_bytes = max(0, allocated - context - library_workspace)
    residual = owned_allocator_bytes - tensors
    ratio = abs(residual) / owned_allocator_bytes if owned_allocator_bytes else None
    return {
        "allocated_bytes": allocated,
        "context_allocated_bytes": context,
        "cleanup_allocated_bytes": cleanup,
        "library_workspace_floor_bytes": library_workspace,
        "owned_allocator_bytes": owned_allocator_bytes,
        "tensor_storage_bytes": tensors,
        "adjusted_residual_bytes": residual,
        "adjusted_residual_ratio": ratio,
    }


def build_capacity_estimate(
    *,
    num_envs: int,
    num_agents: int,
    map_width: int = 128,
    map_height: int = 128,
    magat_observation_diameter: int = 13,
) -> dict[str, Any]:
    for name, value in {
        "num_envs": num_envs,
        "num_agents": num_agents,
        "map_width": map_width,
        "map_height": map_height,
        "magat_observation_diameter": magat_observation_diameter,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    envs = int(num_envs)
    agents = int(num_agents)
    cells = int(map_width) * int(map_height)
    total_agents = envs * agents
    total_edges = envs * agents * agents
    components = {
        "derived.energy_maps": total_agents * cells,
        "magat.edge_index_storage": 2 * total_edges * 8,
        "magat.edge_attr_storage": 3 * total_edges * 4,
        "magat.node_storage": (
            total_agents * (4 * int(magat_observation_diameter) ** 2) * 4
        ),
        "compact.state_packed": total_agents * 8 * 2,
    }
    return {
        "schema_version": 1,
        "num_envs": envs,
        "num_agents": agents,
        "map_width": int(map_width),
        "map_height": int(map_height),
        "components": components,
        "mandatory_known_bytes": int(sum(components.values())),
    }


def preflight_capacity(
    *,
    required_bytes: int,
    safety_reserve_bytes: int,
    device: str | torch.device = "cuda:0",
    memory_info: tuple[int, int] | None = None,
) -> dict[str, Any]:
    required = int(required_bytes)
    reserve = int(safety_reserve_bytes)
    if required < 0 or reserve < 0:
        raise ValueError("required and safety-reserve bytes must be non-negative")
    if memory_info is None:
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
    else:
        free_bytes, total_bytes = memory_info
    free_bytes = int(free_bytes)
    total_bytes = int(total_bytes)
    requested_with_reserve = required + reserve
    shortfall = max(0, requested_with_reserve - free_bytes)
    return {
        "schema_version": 1,
        "device": str(torch.device(device)),
        "required_bytes": required,
        "safety_reserve_bytes": reserve,
        "requested_with_reserve_bytes": requested_with_reserve,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "shortfall_bytes": shortfall,
        "allowed": shortfall == 0,
    }


def ring_payload_bytes(
    *,
    depth: int,
    num_envs: int,
    num_agents: int,
    feature_dim: int = 8,
    element_size: int = 2,
) -> int:
    values = (depth, num_envs, num_agents, feature_dim, element_size)
    if any(int(value) <= 0 for value in values):
        raise ValueError("ring dimensions and element size must be positive")
    return int(depth) * int(num_envs) * int(num_agents) * int(feature_dim) * int(
        element_size
    )


__all__ = [
    "ALLOWED_LIFETIMES",
    "ALLOWED_OWNERS",
    "TensorReference",
    "allocator_snapshot",
    "build_capacity_estimate",
    "census_tensors",
    "inventory_tensor_references",
    "module_tensor_references",
    "optimizer_tensor_references",
    "preflight_capacity",
    "reconcile_after_cleanup",
    "ring_payload_bytes",
]
