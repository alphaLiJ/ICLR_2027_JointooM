from expert.minimal_f4_batch_granularity_runner import (
    run_f4_batch_granularity,
)


def test_f4_changes_only_complete_environments_per_update(tmp_path):
    observed = {}

    def fake_engine(**kwargs):
        observed.update(kwargs)
        return {
            "total_wall_s": 2.0,
            "num_optimizer_steps": 16,
            "samples_processed": 4096,
            "worker_pids": list(range(8)),
            "worker_exitcodes": [0] * 8,
        }

    result = run_f4_batch_granularity(
        batch_envs=1,
        seed=42,
        frontier_steps=2,
        output_dir=tmp_path / "row",
        engine=fake_engine,
    )

    assert result["experiment"] == "F4"
    assert result["train_batch_size"] == 256
    assert result["optimizer_steps_per_frontier"] == 8
    assert result["expected_optimizer_steps"] == 16
    assert observed["num_producer_processes"] == 8
    assert observed["async_ring_buffer_steps"] == 128
