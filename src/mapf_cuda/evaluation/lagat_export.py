"""Export trained MAGAT+ checkpoints for the LaGAT C++ inference contract.

LaGAT pre-materializes self edges in C++ because its upstream traced model
disables PyG's dynamic self-loop construction.  The training model, however,
adds self loops *after* encoding raw edge attributes.  Since that encoder is
non-linear, encoding the C++ mean raw attribute is not numerically equivalent
to averaging encoded attributes.

The deployment wrapper below accepts the unmodified LaGAT input contract,
discards its pre-materialized self edges, and delegates self-loop construction
to the exact graph layers used during training.  This preserves the learned
policy rather than introducing an export-only numerical change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from expert.fixed_magat_plus_runtime import (
    FixedMAGATPlusModel,
    RuntimePyGBatch,
)
from mapf_cuda.models.checkpoints import load_magat_checkpoint


class LaGATDeploymentModel(torch.nn.Module):
    """Traceable, training-faithful MAGAT+ model for LaGAT's C++ inputs."""

    def __init__(self, model: FixedMAGATPlusModel):
        super().__init__()
        self.model = copy.deepcopy(model)
        for graph_conv in self.model.gnn.graph_convs:
            graph_conv.gnn.add_self_loops = False

    def forward(
        self, x: torch.Tensor, data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        edge_index_with_self = data["edge_index"]
        edge_attr_with_self = data["edge_attr"]

        # LaGAT appends one precomputed self edge for every connected agent.
        # The training graph has no explicit self edges; PyG constructs them
        # after the non-linear edge encoder.  Recover that exact input here.
        non_self = edge_index_with_self[0] != edge_index_with_self[1]
        edge_index = edge_index_with_self[:, non_self]
        encoded_non_self = self.model.edge_attr_encoder(edge_attr_with_self[non_self])

        # Materialize the exact self-loop attributes that PyG creates during
        # training: mean *encoded* incoming attributes, or zero for an
        # isolated node.  All workspace constructors derive their device from
        # an input tensor so the CPU-traced graph remains movable to CUDA.
        num_nodes = x.shape[0]
        destination = edge_index[1]
        encoded_sum = encoded_non_self.new_zeros(
            (num_nodes, encoded_non_self.shape[1])
        )
        encoded_sum.index_add_(0, destination, encoded_non_self)
        degree = encoded_non_self.new_zeros((num_nodes, 1))
        degree.index_add_(
            0,
            destination,
            torch.ones_like(destination, dtype=encoded_non_self.dtype).unsqueeze(1),
        )
        self_attr = encoded_sum / degree.clamp_min(1)
        node_ids = torch.cumsum(
            torch.ones_like(x[:, 0, 0, 0], dtype=torch.int64), dim=0
        ) - 1
        self_edges = torch.stack((node_ids, node_ids))
        edge_index = torch.cat((edge_index, self_edges), dim=1)
        edge_attr = torch.cat((encoded_non_self, self_attr), dim=0)

        x = self.model.cnn(x)
        for graph_conv in self.model.gnn.graph_convs:
            x = graph_conv.gnn(x, edge_index, edge_attr)
            x = F.relu(x)
        x = F.relu(self.model.actions_mlp[0](x))
        return self.model.actions_mlp[1](x)


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_model(checkpoint_path: Path) -> FixedMAGATPlusModel:
    model = FixedMAGATPlusModel().cpu()
    load_magat_checkpoint(
        SimpleNamespace(model=model), str(checkpoint_path), "cpu"
    )
    model.eval()
    return model


def make_parity_case(
    num_nodes: int, *, seed: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create a deterministic LaGAT-style graph including raw self edges."""

    if num_nodes < 3:
        raise ValueError("parity cases require at least three nodes")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(num_nodes, 4, 13, 13, generator=generator)

    nodes = torch.arange(num_nodes, dtype=torch.int64)
    source = torch.cat((nodes, nodes, nodes))
    target = torch.cat(
        ((nodes + 1) % num_nodes, (nodes - 1) % num_nodes, (nodes + 2) % num_nodes)
    )
    edge_index = torch.stack((source, target))
    edge_attr = torch.randn(edge_index.shape[1], 3, generator=generator)

    # Match LaGAT policy.cpp: append each node's mean incoming raw attribute.
    self_attr = torch.stack(
        tuple(edge_attr[target == node].mean(dim=0) for node in nodes)
    )
    self_edges = torch.stack((nodes, nodes))
    return x, {
        "edge_index": torch.cat((edge_index, self_edges), dim=1),
        "edge_attr": torch.cat((edge_attr, self_attr), dim=0),
    }


def _training_logits(
    model: FixedMAGATPlusModel,
    x: torch.Tensor,
    cxx_data: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    non_self = cxx_data["edge_index"][0] != cxx_data["edge_index"][1]
    batch = RuntimePyGBatch(
        edge_index=cxx_data["edge_index"][:, non_self],
        edge_attr=cxx_data["edge_attr"][non_self],
    )
    return model(x, batch)


def parity_audit(
    training_model: FixedMAGATPlusModel,
    deployment_model: torch.nn.Module,
    *,
    node_counts: tuple[int, ...] = (8, 16, 32),
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case_index, num_nodes in enumerate(node_counts):
            x, data = make_parity_case(num_nodes, seed=9100 + case_index)
            reference = _training_logits(training_model, x, data)
            candidate = deployment_model(x, data)
            absolute = (reference - candidate).abs()
            close = torch.allclose(reference, candidate, atol=atol, rtol=rtol)
            action_agreement = float(
                (reference.argmax(dim=1) == candidate.argmax(dim=1))
                .to(torch.float32)
                .mean()
                .item()
            )
            row = {
                "num_nodes": num_nodes,
                "num_edges_with_self": int(data["edge_index"].shape[1]),
                "max_abs_error": float(absolute.max().item()),
                "mean_abs_error": float(absolute.mean().item()),
                "action_agreement": action_agreement,
                "allclose": bool(close),
            }
            rows.append(row)
            if not close or action_agreement != 1.0:
                raise RuntimeError(f"LaGAT export parity failed: {row}")
    return rows


def export_lagat_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    training_model = load_training_model(checkpoint_path)
    deployment = LaGATDeploymentModel(training_model).eval()
    trace_x, trace_data = make_parity_case(16, seed=9001)
    with torch.inference_mode(), torch.jit.optimized_execution(True):
        traced = torch.jit.trace(
            deployment,
            (trace_x, trace_data),
            check_trace=True,
            strict=True,
        )
        parity_rows = parity_audit(training_model, traced)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    reloaded = torch.jit.load(str(output_path), map_location="cpu").eval()
    reload_rows = parity_audit(training_model, reloaded)

    audit = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "state_tensor_count": len(training_model.state_dict()),
        "export_contract": "lagat-cxx-self-edges-to-training-pyg-self-loops-v1",
        "trace_parity": parity_rows,
        "reload_parity": reload_rows,
    }
    resolved_audit_path = (
        audit_path.expanduser().resolve()
        if audit_path is not None
        else output_path.with_suffix(output_path.suffix + ".audit.json")
    )
    resolved_audit_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained MAGAT+ checkpoint for LaGAT C++ inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    audit = export_lagat_checkpoint(args.checkpoint, args.output, audit_path=args.audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
