from __future__ import annotations

import csv
import json

from experiments.renderers.render_s3_memory_decomposition import (
    allocator_rows,
    discover_runs,
    memory_component_rows,
    render_results_markdown,
)


def test_s3_renderer_deduplicates_inventory_and_uses_peak_allocator(tmp_path):
    run = tmp_path / "runs" / "S3-M1" / "rep-0"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "row_id": "S3-M1",
                "repetition": 0,
                "provenance": {
                    "code_commit": "a" * 40,
                    "input_sha256": "b" * 64,
                    "checkpoint_sha256": "c" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "capacity-model.json").write_text(
        json.dumps(
            {
                "estimate": {"components": {"x": 4}, "mandatory_known_bytes": 4},
                "preflight": {
                    "free_bytes": 10,
                    "total_bytes": 20,
                    "allowed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    with (run / "tensor-inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "checkpoint",
                "owner",
                "storage_bytes",
                "counted_unique_storage",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "checkpoint": "persistent_allocated",
                "owner": "derived_state",
                "storage_bytes": 1024,
                "counted_unique_storage": True,
            }
        )
        writer.writerow(
            {
                "checkpoint": "persistent_allocated",
                "owner": "derived_state",
                "storage_bytes": 1024,
                "counted_unique_storage": False,
            }
        )
    (run / "allocator-checkpoints.jsonl").write_text(
        json.dumps(
            {
                "checkpoint": "inference_peak",
                "peak_allocated_bytes": 4096,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runs = discover_runs(tmp_path / "runs")
    components = memory_component_rows(runs)
    allocators = allocator_rows(runs)
    markdown = render_results_markdown(runs, components, allocators)

    assert components[0]["storage_bytes"] == 1024
    assert allocators[0]["peak_allocated_bytes"] == 4096
    assert "S3-M1" in markdown
    assert "NVML was not used" in markdown
