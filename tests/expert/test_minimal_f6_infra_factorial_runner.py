from expert.minimal_f6_infra_factorial_runner import (
    run_f6_infra_factorial,
)


def test_f6_forwards_only_ring_and_transfer_factors(tmp_path):
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

    result = run_f6_infra_factorial(
        ring_depth=2,
        transfer_mode="sync",
        seed=42,
        frontier_steps=2,
        output_dir=tmp_path / "row",
        engine=fake_engine,
    )

    assert result["experiment"] == "F6"
    assert result["transfer_mode"] == "sync"
    assert result["ring_depth"] == 2
    assert observed["async_ring_buffer_steps"] == 2
    assert observed["transfer_mode"] == "sync"
    assert observed["train_batch_size"] == 1024
