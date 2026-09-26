from expert.render_f4_batch_granularity_report import (
    BATCH_ENVS,
    SEEDS,
    _load_rows,
    _summarize,
)


def test_f4_summary_reports_per_sample_speedup():
    rows = []
    for batch_envs in BATCH_ENVS:
        for seed in SEEDS:
            rows.append(
                {
                    "batch_envs": batch_envs,
                    "seed": seed,
                    "num_optimizer_steps": 4000 // batch_envs,
                    "total_wall_s": 10.0,
                    "samples_s": float(batch_envs),
                    "consumer_train_step_s": 8.0,
                    "consumer_wait_s": 1.0,
                    "peak_gpu_memory_bytes": 100 * batch_envs,
                }
            )

    summary = _summarize(rows)

    assert len(summary) == 4
    assert summary[-1]["train_batch_size"] == 2048
    assert summary[-1]["speedup_vs_batch256"] == 8.0


def test_f4_loader_accepts_legacy_f2_batch_metadata(tmp_path):
    root = tmp_path / "f2"
    output = root / "row"
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        """{
          "experiment": "F2",
          "status": "ok",
          "accepted": true,
          "num_agents": 256,
          "num_producer_processes": 8,
          "train_batch_size": 1024,
          "async_ring_buffer_steps": 128
        }"""
    )

    rows = _load_rows(tmp_path / "empty", [root])

    assert len(rows) == 1
    assert rows[0]["batch_envs"] == 4
