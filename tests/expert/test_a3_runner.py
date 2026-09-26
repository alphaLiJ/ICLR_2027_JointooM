from __future__ import annotations

import json

import numpy as np


def _write_manifest_pool(tmp_path):
    from expert.benchmark_contract import (
        FrozenTransitionBatch,
        save_frozen_transition_batch,
    )

    batch = FrozenTransitionBatch(
        instance_ids=np.asarray([100, 101, 102, 103], dtype=np.int64),
        grids=np.zeros((4, 128, 128), dtype=np.uint8),
        positions=np.asarray(
            [[[2, 2]], [[3, 3]], [[4, 4]], [[5, 5]]], dtype=np.uint16
        ),
        goals=np.asarray(
            [[[2, 3]], [[3, 4]], [[4, 5]], [[5, 6]]], dtype=np.uint16
        ),
        arrived=np.zeros((4, 1), dtype=np.bool_),
        actions=np.full((2, 4, 1), 4, dtype=np.uint8),
        horizon=2,
    )
    pool = tmp_path / "pool.npz"
    save_frozen_transition_batch(pool, batch)
    manifest = {
        "schema_version": "benchmark-manifest-v1",
        "entries": [
            {
                "pool_name": "a2-random-d020-a001",
                "role": "a2_pool",
                "topology": "random",
                "density": 0.2,
                "num_agents": 1,
                "horizon": 2,
                "instance_ids": [100, 101, 102, 103],
                "frozen_batch_path": str(pool),
                "semantic_sha256": batch.semantic_sha256,
                "num_envs": 4,
                "grid_shape": [128, 128],
            }
        ],
        "a2_cells": [
            {
                "cell_id": "env-e2",
                "axis": "env",
                "topology": "random",
                "density": 0.2,
                "num_agents": 1,
                "env_count": 2,
                "pool_name": "a2-random-d020-a001",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, batch


def test_a3_cuda_profile_summary_counts_launches_and_unions_overlaps():
    from expert.a3_runner import _summarize_cuda_profile_events

    class TimeRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class Event:
        def __init__(self, name, device_type, start, end, count=1):
            self.name = name
            self.device_type = device_type
            self.time_range = TimeRange(start, end)
            self.count = count

    rows = _summarize_cuda_profile_events(
        [
            Event("cudaLaunchKernel", "cpu", 0.0, 1.0, count=2),
            Event("cudaLaunchKernelExC", "cpu", 1.0, 2.0),
            Event("cudaMemcpyAsync", "cpu", 2.0, 3.0),
            Event("kernel_a", "cuda", 10.0, 30.0),
            Event("kernel_b", "cuda", 20.0, 50.0),
            Event("kernel_c", "cuda", 70.0, 80.0),
        ],
        cuda_device_type="cuda",
        profiled_steps=2,
    )

    assert rows["cuda_launch_count"] == 3
    assert rows["cuda_launch_count_per_step"] == 1.5
    assert rows["profiled_gpu_active_time_us"] == 50.0
    assert rows["gpu_utilization_pct"] == 50.0 / 70.0 * 100.0
    assert rows["cuda_launch_count_source"] == "torch_profiler_separate_pass"


def test_a3_reports_exact_five_cumulative_stages(tmp_path):
    from expert.a3_runner import A3_STAGE_NAMES, run_builder_stage_scan

    manifest, _ = _write_manifest_pool(tmp_path)
    observed_ids = []

    def stage_executor(*, batch, backend, **_kwargs):
        observed_ids.extend(batch.instance_ids.tolist())
        return [
            {
                "stage": name,
                "wall_ms_per_step": index + 1.0,
                "gpu_ms_per_step": index + 0.5,
                "env_steps_s": 2.0,
                "agent_steps_s": 2.0,
                "peak_gpu_memory_bytes": 1,
                "graph_edges": 0,
                "average_graph_degree": 0.0,
                "token_count": None,
                "padding_fraction": None,
                "cuda_launch_count": 1,
                "cuda_launch_count_per_step": 1.0,
                "gpu_utilization_pct": 50.0,
                "consumed_input_sha256": batch.semantic_sha256,
            }
            for index, name in enumerate(A3_STAGE_NAMES)
        ]

    report = run_builder_stage_scan(
        manifest_path=manifest,
        scan_cell="env-e2",
        backend="magat",
        warmup_steps=0,
        repetitions=1,
        num_steps=2,
        stage_executor=stage_executor,
    )

    assert observed_ids == [100, 101]
    assert [row["stage"] for row in report["rows"]] == list(A3_STAGE_NAMES)
    assert all(row["consumed_input_sha256"] == report["consumed_input_sha256"] for row in report["rows"])


def test_a3_rejects_missing_or_reordered_stage(tmp_path):
    from expert.a3_runner import run_builder_stage_scan

    manifest, _ = _write_manifest_pool(tmp_path)

    def bad_executor(**_kwargs):
        return [{"stage": "transition"}]

    try:
        run_builder_stage_scan(
            manifest_path=manifest,
            scan_cell="env-e2",
            backend="magat",
            warmup_steps=0,
            repetitions=1,
            num_steps=2,
            stage_executor=bad_executor,
        )
    except RuntimeError as error:
        assert "exact five cumulative stages" in str(error)
    else:
        raise AssertionError("A3 accepted an incomplete stage decomposition")
