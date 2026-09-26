import json

import pytest

from expert.render_f2_producer_scaling_report import (
    AGENT_COUNTS,
    PRODUCER_COUNTS,
    _load_rows,
    _summarize,
)


def test_f2_summary_requires_complete_three_seed_matrix(tmp_path):
    root = tmp_path / "rows"
    for num_agents in AGENT_COUNTS:
        for num_producers in PRODUCER_COUNTS:
            for seed in (42, 43, 44):
                output = root / f"a{num_agents}-p{num_producers}-s{seed}"
                output.mkdir(parents=True)
                row = {
                    "experiment": "F2",
                    "status": "ok",
                    "accepted": True,
                    "num_agents": num_agents,
                    "train_batch_size": num_agents * 4,
                    "num_producer_processes": num_producers,
                    "seed": seed,
                    "total_wall_s": 10.0,
                    "samples_s": float(num_agents * num_producers),
                    "optimizer_steps_s": 2.0,
                    "consumer_wait_s": 5.0,
                    "consumer_train_step_s": 4.0,
                    "consumer_dma_sync_s": 0.1,
                    "consumer_gpu_builder_s": 0.2,
                    "worker_ringbuffer_write_s_max": 0.3,
                    "peak_gpu_memory_bytes": 100,
                }
                (output / "result.json").write_text(json.dumps(row))

    summary = _summarize(_load_rows([root]))

    assert len(summary) == 12
    row = next(
        item
        for item in summary
        if item["num_agents"] == 256 and item["num_producers"] == 8
    )
    assert row["median_consumer_wait_share"] == pytest.approx(0.5)
    assert row["speedup_vs_p1"] == pytest.approx(8.0)


def test_f2_loader_rejects_duplicate_cells(tmp_path):
    for root_name in ("first", "second"):
        output = tmp_path / root_name / "row"
        output.mkdir(parents=True)
        (output / "result.json").write_text(
            json.dumps(
                {
                    "experiment": "F2",
                    "status": "ok",
                    "accepted": True,
                    "num_agents": 128,
                    "num_producer_processes": 1,
                    "seed": 42,
                }
            )
        )

    with pytest.raises(ValueError, match="duplicate"):
        _load_rows([tmp_path / "first", tmp_path / "second"])
