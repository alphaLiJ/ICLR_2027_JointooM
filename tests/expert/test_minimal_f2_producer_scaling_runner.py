from __future__ import annotations


def test_f2_keeps_logical_frontier_and_optimizer_batch_fixed(tmp_path):
    from expert.minimal_f2_producer_scaling_runner import run_f2_scaling

    observed = {}

    def engine(**kwargs):
        observed.update(kwargs)
        return {
            "num_optimizer_steps": 6,
            "samples_processed": 6144,
            "total_wall_s": 2.0,
            "samples_s": 3072.0,
            "worker_pids": [101, 102],
            "worker_exitcodes": [0, 0],
            "consumer_wait_s": 0.5,
        }

    result = run_f2_scaling(
        num_producers=2,
        num_agents=256,
        seed=7,
        frontier_steps=3,
        output_dir=tmp_path / "f2",
        engine=engine,
    )

    assert observed["num_experts"] == 8
    assert observed["num_agents"] == 256
    assert observed["num_producer_processes"] == 2
    assert observed["batch_threshold"] == 2048
    assert observed["train_batch_size"] == 1024
    assert result["expected_optimizer_steps"] == 6
    assert result["expected_samples"] == 6144
    assert result["accepted"] is True
    assert result["optimizer_steps_s"] == 3.0
