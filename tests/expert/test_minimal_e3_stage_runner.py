from __future__ import annotations


def test_e3_aggregate_reports_medians_and_memory_maxima():
    from expert.minimal_e3_stage_runner import _aggregate

    common = {
        "stage": "transition",
        "num_steps": 5,
        "cuda_launch_count_per_step": 1.0,
        "profiled_gpu_utilization_pct": 90.0,
        "profile_status": "ok",
        "profile_error": None,
    }
    rows = [
        {
            **common,
            "repetition": 0,
            "wall_ms_per_step": 2.0,
            "gpu_ms_per_step": 1.0,
            "env_steps_s": 10.0,
            "agent_steps_s": 20.0,
            "torch_peak_allocated_bytes": 100,
            "torch_peak_reserved_bytes": 200,
            "nvml_process_memory_bytes": 300,
        },
        {
            **common,
            "repetition": 1,
            "wall_ms_per_step": 4.0,
            "gpu_ms_per_step": 3.0,
            "env_steps_s": 6.0,
            "agent_steps_s": 12.0,
            "torch_peak_allocated_bytes": 400,
            "torch_peak_reserved_bytes": 500,
            "nvml_process_memory_bytes": 600,
        },
    ]

    summary = _aggregate(rows)

    assert len(summary) == 1
    assert summary[0]["median_wall_ms_per_step"] == 3.0
    assert summary[0]["median_gpu_ms_per_step"] == 2.0
    assert summary[0]["max_torch_peak_allocated_bytes"] == 400
    assert summary[0]["max_nvml_process_memory_bytes"] == 600
    assert summary[0]["cuda_launch_count_per_step"] == 1.0
