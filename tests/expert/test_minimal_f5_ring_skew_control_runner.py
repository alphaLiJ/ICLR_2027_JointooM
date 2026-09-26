from expert.minimal_f5_ring_skew_control_runner import (
    HOMOGENEOUS_MAP,
    run_f5_ring_skew_control,
)


def test_f5_repeats_one_topology_without_changing_other_controls(tmp_path):
    observed = {}

    def fake_engine(**kwargs):
        observed.update(kwargs)
        return {
            "total_wall_s": 2.0,
            "num_optimizer_steps": 4,
            "samples_processed": 4096,
            "worker_pids": list(range(8)),
            "worker_exitcodes": [0] * 8,
        }

    result = run_f5_ring_skew_control(
        ring_depth=8,
        seed=42,
        frontier_steps=2,
        output_dir=tmp_path / "row",
        engine=fake_engine,
    )

    assert result["experiment"] == "F5"
    assert result["map_condition"] == "homogeneous_topology"
    assert observed["map_names"] == (HOMOGENEOUS_MAP,) * 8
    assert observed["train_batch_size"] == 1024
    assert observed["num_producer_processes"] == 8
