from expert.render_f6_infra_factorial_report import (
    RING_DEPTHS,
    SEEDS,
    TRANSFER_MODES,
    _summarize,
)


def test_f6_summary_keeps_factorial_cells_separate():
    rows = []
    for mode in TRANSFER_MODES:
        for depth in RING_DEPTHS:
            for seed in SEEDS:
                rows.append(
                    {
                        "transfer_mode": mode,
                        "ring_depth": depth,
                        "seed": seed,
                        "samples_s": float(depth),
                        "total_wall_s": 10.0,
                        "consumer_wait_s": 2.0,
                        "consumer_dma_sync_s": 0.1,
                        "worker_ringbuffer_write_s_max": 1.0,
                    }
                )

    summary = _summarize(rows)

    assert len(summary) == len(TRANSFER_MODES) * len(RING_DEPTHS)
    assert {
        (row["transfer_mode"], row["ring_depth"]) for row in summary
    } == {
        (mode, depth) for mode in TRANSFER_MODES for depth in RING_DEPTHS
    }
