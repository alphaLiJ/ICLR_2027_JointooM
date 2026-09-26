#!/usr/bin/env python3
"""Emit deterministic CUDA transition hashes for old/new migration pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types


source_root = os.environ.get("MAPF_MIGRATION_SOURCE_ROOT")
if source_root:
    source_src = Path(source_root).resolve() / "src"
    sys.path.insert(0, str(source_src))
    # The historical ``expert`` tree was an implicit namespace.  Pin it here
    # so an editable installation of the clean regular package cannot win the
    # namespace-package resolution during the old-side comparison.
    expert_package = types.ModuleType("expert")
    expert_package.__path__ = [str(source_src / "expert")]
    expert_package.__package__ = "expert"
    sys.modules["expert"] = expert_package

from expert.benchmark_contract import FrozenTransitionBatch, load_frozen_transition_batch
from expert.transition_adapters import make_transition_adapter


def _digest_state(state) -> str:
    digest = hashlib.sha256()
    for name in (
        "positions",
        "goals",
        "arrived",
        "terminated",
        "truncated",
        "step_counts",
    ):
        value = getattr(state, name)
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_batch")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    source = load_frozen_transition_batch(args.frozen_batch)
    selected = source.take_envs(args.envs)
    batch = FrozenTransitionBatch(
        instance_ids=selected.instance_ids,
        grids=selected.grids,
        positions=selected.positions,
        goals=selected.goals,
        arrived=selected.arrived,
        active=selected.active,
        actions=selected.actions[: args.steps],
        horizon=args.steps,
    )
    adapter = make_transition_adapter("cuda", batch, "cuda:0")
    timings = []
    final_hash = None
    for _ in range(args.repetitions):
        adapter.reset_device_state()
        handles = adapter.prepare_actions(batch.actions)
        adapter.synchronize_transition()
        started = time.perf_counter()
        for handle in handles:
            adapter.step_transition(handle)
        adapter.synchronize_transition()
        timings.append(time.perf_counter() - started)
        observed = _digest_state(adapter.materialize_state())
        if final_hash is not None and observed != final_hash:
            raise RuntimeError("non-deterministic final-state hash")
        final_hash = observed

    timings.sort()
    median = timings[len(timings) // 2]
    print(
        json.dumps(
            {
                "input_sha256": batch.semantic_sha256,
                "consumed_input_sha256": adapter.consumed_input_sha256,
                "final_state_sha256": final_hash,
                "num_envs": batch.num_envs,
                "num_agents": batch.num_agents,
                "steps": batch.horizon,
                "repetitions": args.repetitions,
                "median_seconds": median,
                "agent_steps_per_second": (
                    batch.num_envs * batch.num_agents * batch.horizon / median
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
