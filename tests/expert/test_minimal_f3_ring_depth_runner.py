from expert.minimal_f3_ring_depth_runner import run_f3_ring_depth


def test_f3_forwards_fixed_saturating_configuration(tmp_path):
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

    result = run_f3_ring_depth(
        ring_depth=2,
        seed=42,
        frontier_steps=2,
        output_dir=tmp_path / "row",
        engine=fake_engine,
    )

    assert result["experiment"] == "F3"
    assert result["protocol"]["ring_capacity_steps"] == 2
    assert result["ring_payload_bytes"] == 2 * 2048 * 8 * 2
    assert observed["num_producer_processes"] == 8
    assert observed["train_batch_size"] == 1024
