from expert.render_f3_ring_depth_report import (
    RING_DEPTHS,
    SEEDS,
    _summarize,
)


def test_f3_summary_computes_depth128_fraction():
    rows = []
    for depth in RING_DEPTHS:
        for seed in SEEDS:
            rows.append(
                {
                    "ring_depth": depth,
                    "seed": seed,
                    "total_wall_s": 10.0,
                    "samples_s": float(depth),
                    "consumer_wait_s": 5.0,
                    "worker_ringbuffer_write_s_max": 2.0,
                }
            )

    summary = _summarize(rows)

    assert len(summary) == len(RING_DEPTHS)
    assert summary[0]["throughput_fraction_of_depth128"] == 2 / 128
    assert summary[-1]["throughput_fraction_of_depth128"] == 1.0
