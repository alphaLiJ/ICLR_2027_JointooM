from expert.render_core_component_ablation_report import _summarize


def _row(cell, seed, steps):
    return {
        "cell": cell,
        "seed": seed,
        "num_steps": steps,
        "samples_processed": steps * 1024,
        "samples_s": 10.0,
        "total_wall_s": 2.0,
        "h2d_bytes": 100.0,
        "consumer_wait_s": 0.2,
        "consumer_train_step_s": 1.0,
        "peak_gpu_memory_bytes": 1000,
        "peak_host_memory_bytes": 2000,
    }


def test_paired_summary_keeps_budgets_separate():
    short = {
        cell: [_row(cell, seed, 100) for seed in (42, 43, 44)]
        for cell in ("s0_full_one_stage", "s1_compact_one_stage")
    }
    long = {
        cell: [_row(cell, seed, 1000) for seed in (42, 43, 44)]
        for cell in ("s1_compact_one_stage", "s3_compact_deep_ring")
    }

    summary = _summarize(short, long)

    assert [row["num_steps"] for row in summary] == [100, 100, 1000, 1000]
