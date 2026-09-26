from pathlib import Path

from expert.minimal_core_ablation_runner import CELLS, run_core_ablation


def _fake_engine(**kwargs):
    steps = int(kwargs["num_steps"])
    samples = steps * 1024
    return {
        "accepted": True,
        "num_optimizer_steps": steps,
        "samples_processed": samples,
        "total_wall_s": 2.0,
        "samples_s": samples / 2.0,
        "mean_loss": 0.5,
        "final_loss": 0.4,
        "loss_history": [0.4] * steps,
        "health_report": {"validation": {"valid": True, "violations": []}},
        "ring_buffer_steps": int(kwargs["ring_buffer_steps"]),
        "consumer_wait_s": 0.5,
        "consumer_host_builder_and_h2d_s": 0.4,
        "consumer_dma_sync_s": 0.1,
        "consumer_gpu_builder_s": 0.2,
        "consumer_train_step_s": 0.8,
        "h2d_bytes": samples * 16,
        "d2h_bytes": steps * 4,
    }


def test_core_ablation_matrix_has_expected_factors():
    assert CELLS == {
        "s0_full_one_stage": ("full_host_materialized", 1),
        "s1_compact_one_stage": ("compact_cuda_replay", 1),
        "s2_full_deep_ring": ("full_host_materialized", 128),
        "s3_compact_deep_ring": ("compact_cuda_replay", 128),
    }


def test_core_ablation_runner_records_clean_protocol(tmp_path: Path):
    result = run_core_ablation(
        cell="s3_compact_deep_ring",
        seed=42,
        num_steps=2,
        output_dir=tmp_path / "row",
        device="cpu",
        engine=_fake_engine,
    )

    assert result["accepted"]
    assert result["samples_processed"] == 2048
    assert result["expected_optimizer_steps"] == 2
    assert result["ring_buffer_steps"] == 128
    assert result["protocol"]["validation"] == "disabled"
    assert not result["monitoring_in_timed_region"]
