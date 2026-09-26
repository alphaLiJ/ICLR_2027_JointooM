from __future__ import annotations

import json

import torch
import pytest

from expert.fixed_magat_plus_runtime import FixedMAGATPlusModel
from mapf_cuda.evaluation.lagat_export import (
    LaGATDeploymentModel,
    export_lagat_checkpoint,
    make_parity_case,
    parity_audit,
)


def test_deployment_wrapper_matches_training_model_for_lagat_self_edges():
    torch.manual_seed(12)
    model = FixedMAGATPlusModel().eval()
    wrapper = LaGATDeploymentModel(model).eval()

    rows = parity_audit(model, wrapper, node_counts=(8, 16))

    assert all(row["allclose"] for row in rows)
    assert all(row["action_agreement"] == 1.0 for row in rows)


def test_export_round_trip_is_strictly_equivalent(tmp_path):
    torch.manual_seed(13)
    model = FixedMAGATPlusModel().eval()
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "compiled.pt"
    torch.save({"model": model.state_dict()}, checkpoint)

    audit = export_lagat_checkpoint(checkpoint, output)
    loaded = torch.jit.load(str(output), map_location="cpu")
    x, data = make_parity_case(24, seed=99)

    assert output.is_file()
    assert audit["state_tensor_count"] == 94
    assert all(row["allclose"] for row in audit["reload_parity"])
    assert loaded(x, data).shape == (24, 5)
    stored = json.loads(
        output.with_suffix(".pt.audit.json").read_text(encoding="utf-8")
    )
    assert stored["output_sha256"] == audit["output_sha256"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_traced_export_moves_to_cuda_without_device_constants(tmp_path):
    model = FixedMAGATPlusModel().eval()
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "compiled.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    export_lagat_checkpoint(checkpoint, output)
    loaded = torch.jit.load(str(output), map_location="cpu").eval().cuda()
    x, data = make_parity_case(8, seed=101)

    logits = loaded(x.cuda(), {key: value.cuda() for key, value in data.items()})

    assert logits.shape == (8, 5)
    assert logits.is_cuda
