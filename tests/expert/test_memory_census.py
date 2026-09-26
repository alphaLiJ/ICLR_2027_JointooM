from __future__ import annotations

import torch

from mapf_cuda.observability.memory_census import (
    TensorReference,
    build_capacity_estimate,
    census_tensors,
    inventory_tensor_references,
    module_tensor_references,
    preflight_capacity,
    reconcile_after_cleanup,
    ring_payload_bytes,
)


def test_census_deduplicates_views_by_backing_storage():
    backing = torch.arange(16, dtype=torch.float32)
    view = backing[4:12]

    report = census_tensors(
        [
            TensorReference("backing", backing, "model_parameters", "run"),
            TensorReference("view", view, "model_ephemeral", "model_call"),
        ]
    )

    assert report["logical_bytes"] == backing.numel() * 4 + view.numel() * 4
    assert report["unique_storage_bytes"] == backing.untyped_storage().nbytes()
    assert report["by_device_storage_bytes"]["cpu"] == backing.untyped_storage().nbytes()
    assert sum(row["counted_unique_storage"] for row in report["rows"]) == 1
    assert report["rows"][1]["is_view"] is True


def test_module_references_expose_tied_weights_without_double_counting():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.output = torch.nn.Linear(4, 8, bias=False)
            self.output.weight = self.embedding.weight

    module = Tied()
    refs = module_tensor_references(module)
    report = census_tensors(refs)

    parameter_rows = [row for row in report["rows"] if row["kind"] == "parameter"]
    assert {row["name"] for row in parameter_rows} == {
        "model.parameter.embedding.weight",
        "model.parameter.output.weight",
    }
    assert report["unique_storage_bytes"] == module.embedding.weight.numel() * 4


def test_zero_sized_tensor_has_no_unique_storage_charge():
    empty = torch.empty(0, dtype=torch.float32)
    report = census_tensors(
        [TensorReference("empty", empty, "model_ephemeral", "model_call")]
    )

    assert report["logical_bytes"] == 0
    assert report["unique_storage_bytes"] == 0
    assert report["rows"][0]["storage_id"] is None
    assert report["rows"][0]["counted_unique_storage"] is False


def test_native_inventory_requires_explicit_classification():
    inventory = {
        "known": torch.empty(4),
        "new_allocation": torch.empty(8),
    }

    try:
        inventory_tensor_references(
            inventory,
            classifications={"known": ("dynamic_simulator", "episode")},
            prefix="simulator",
        )
    except ValueError as error:
        assert "new_allocation" in str(error)
    else:  # pragma: no cover - assertion message is clearer than pytest.raises
        raise AssertionError("unclassified inventory entry was accepted")


def test_native_inventory_preserves_tensor_identity():
    tensor = torch.empty(4)
    references = inventory_tensor_references(
        {"state": tensor},
        classifications={"state": ("compact_state", "frontier")},
        prefix="simulator",
    )

    assert references[0].name == "simulator.state"
    assert references[0].tensor is tensor


def test_capacity_estimate_matches_current_128_map_layout():
    estimate = build_capacity_estimate(num_envs=1024, num_agents=512)

    assert estimate["components"]["derived.energy_maps"] == 8 * 1024**3
    assert estimate["components"]["magat.edge_index_storage"] == 4 * 1024**3
    assert estimate["components"]["magat.edge_attr_storage"] == 3 * 1024**3
    assert estimate["components"]["magat.node_storage"] == (
        1024 * 512 * (4 * 13 * 13) * 4
    )
    assert estimate["mandatory_known_bytes"] > 15 * 1024**3


def test_preflight_rejects_without_allocating_when_capacity_is_insufficient():
    decision = preflight_capacity(
        required_bytes=15 * 1024**3,
        safety_reserve_bytes=1024**3,
        memory_info=(15 * 1024**3, 16 * 1024**3),
    )

    assert decision["allowed"] is False
    assert decision["shortfall_bytes"] == 1024**3
    assert decision["free_bytes"] == 15 * 1024**3


def test_ring_payload_uses_eight_int16_fields_per_row():
    assert ring_payload_bytes(
        depth=128, num_envs=4, num_agents=256, feature_dim=8
    ) == 2 * 1024**2
    assert ring_payload_bytes(
        depth=128, num_envs=8, num_agents=256, feature_dim=8
    ) == 4 * 1024**2


def test_cleanup_floor_is_named_as_library_workspace_before_reconciliation():
    report = reconcile_after_cleanup(
        allocated_bytes=58 * 1024**2,
        tensor_storage_bytes=40 * 1024**2,
        cleanup_allocated_bytes=18 * 1024**2,
    )

    assert report["library_workspace_floor_bytes"] == 18 * 1024**2
    assert report["owned_allocator_bytes"] == 40 * 1024**2
    assert report["adjusted_residual_bytes"] == 0
    assert report["adjusted_residual_ratio"] == 0.0
