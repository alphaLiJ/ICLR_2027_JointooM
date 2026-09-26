from expert.render_f5_ring_skew_control_report import (
    CONDITIONS,
    RING_DEPTHS,
    SEEDS,
    _summarize,
)


def test_f5_summary_keeps_conditions_separate():
    rows = []
    for condition in CONDITIONS:
        for depth in RING_DEPTHS:
            for seed in SEEDS:
                rows.append(
                    {
                        "map_condition": condition,
                        "ring_depth": depth,
                        "seed": seed,
                        "samples_s": float(depth),
                        "total_wall_s": 10.0,
                        "consumer_wait_s": 2.0,
                        "worker_ringbuffer_write_s_max": 1.0,
                    }
                )

    summary = _summarize(rows)

    assert len(summary) == len(CONDITIONS) * len(RING_DEPTHS)
    assert {row["map_condition"] for row in summary} == set(CONDITIONS)
