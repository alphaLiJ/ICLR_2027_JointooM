from expert.minimal_core_ablation_runner import CELLS
from expert.render_core_ablation_report import SEEDS, _summarize


def test_core_ablation_summary_has_four_cells():
    rows = []
    for cell, (representation, depth) in CELLS.items():
        for seed in SEEDS:
            row = {
                "cell": cell,
                "seed": seed,
                "representation": representation,
                "ring_buffer_steps": depth,
                "samples_s": 10.0,
                "total_wall_s": 2.0,
                "h2d_bytes": 100.0,
                "consumer_wait_s": 0.2,
                "consumer_train_step_s": 1.0,
                "peak_gpu_memory_bytes": 1000,
                "peak_host_memory_bytes": 2000,
            }
            if representation == "full_host_materialized":
                row["consumer_host_builder_and_h2d_s"] = 0.5
            else:
                row["consumer_dma_sync_s"] = 0.1
                row["consumer_gpu_builder_s"] = 0.2
            rows.append(row)

    summary = _summarize(rows)

    assert len(summary) == 4
    assert {row["cell"] for row in summary} == set(CELLS)
