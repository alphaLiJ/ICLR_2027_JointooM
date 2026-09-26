"""Regression tests for optional NVML samples in E5 result rendering."""

from expert.minimal_e5_magat_runner import (
    _health_disposition as magat_health_disposition,
)
from expert.minimal_e5_magat_runner import _peak_nvml_bytes as magat_peak
from expert.minimal_e5_mapf_gpt_runner import (
    _health_disposition as mapf_gpt_health_disposition,
)
from expert.minimal_e5_mapf_gpt_runner import _peak_nvml_bytes as mapf_gpt_peak


def _metrics(*gpu_samples):
    return {
        "health_report": {
            "events": [{"gpu": sample} for sample in gpu_samples],
        }
    }


def test_peak_nvml_bytes_tolerates_unavailable_nvml():
    assert magat_peak(_metrics(None, None)) is None
    assert mapf_gpt_peak(_metrics(None, None)) is None


def test_peak_nvml_bytes_uses_available_samples():
    metrics = _metrics(
        None,
        {"memory_used_bytes": 1024},
        {"memory_used_bytes": 4096},
    )
    assert magat_peak(metrics) == 4096
    assert mapf_gpt_peak(metrics) == 4096


def test_health_disposition_treats_only_missing_telemetry_as_advisory():
    metrics = {
        "health_report": {
            "validation": {
                "valid": False,
                "violations": [
                    {"code": "collector_error"},
                    {"code": "gpu_metrics_missing"},
                ],
            }
        }
    }
    assert magat_health_disposition(metrics) == "telemetry_unavailable"
    assert mapf_gpt_health_disposition(metrics) == "telemetry_unavailable"


def test_health_disposition_keeps_pipeline_failure_fatal():
    metrics = {
        "health_report": {
            "validation": {
                "valid": False,
                "violations": [{"code": "gpu_stall"}],
            }
        }
    }
    assert magat_health_disposition(metrics) == "fail"
    assert mapf_gpt_health_disposition(metrics) == "fail"
