"""Side-effect-free snapshots for resident training pipelines."""

from __future__ import annotations

import subprocess
from typing import Any


def device_index(device: str | None) -> int:
    if device is None:
        return 0
    text = str(device).strip().lower()
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def pipeline_snapshot(
    pipeline: Any, *, fallback_block_size: int
) -> dict[str, int]:
    stats = {}
    getter = getattr(pipeline, "get_stats", None)
    if callable(getter):
        try:
            stats = dict(getter())
        except BaseException:
            stats = {}
    dma_worker = getattr(pipeline, "dma_worker", None)
    reserve_ptr = stats.get("reserve_ptr")
    if reserve_ptr is None:
        ring_buffer = getattr(pipeline, "ring_buffer", None)
        shared = getattr(ring_buffer, "shared_reserve_ptr", None)
        reserve_ptr = getattr(shared, "value", None)
    if reserve_ptr is None:
        reserve_ptr = getattr(
            dma_worker, "dma_read_ptr", getattr(pipeline, "compute_ptr", 0)
        )
    dma_ptr = stats.get("dma_read_ptr")
    if dma_ptr is None:
        dma_ptr = getattr(
            dma_worker, "dma_read_ptr", getattr(pipeline, "compute_ptr", 0)
        )
    compute_ptr = stats.get("compute_ptr", getattr(pipeline, "compute_ptr", 0))
    ring_capacity = stats.get(
        "capacity", getattr(pipeline, "capacity", fallback_block_size)
    )
    block_size = stats.get("stage_rows", fallback_block_size)
    return {
        "reserve_ptr": int(reserve_ptr),
        "dma_ptr": int(dma_ptr),
        "compute_ptr": int(compute_ptr),
        "ring_capacity": int(ring_capacity),
        "block_size": max(1, int(block_size)),
    }


def query_gpu_snapshot(*, device_index: int) -> dict[str, Any]:
    """Collect optional out-of-band GPU health data outside timed regions."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            observed_index = int(fields[0])
            utilization_pct = float(fields[1])
            memory_used_bytes = int(float(fields[2]) * 1024 * 1024)
            memory_total_bytes = int(float(fields[3]) * 1024 * 1024)
        except ValueError:
            continue
        if observed_index == int(device_index):
            return {
                "device_index": observed_index,
                "utilization_pct": utilization_pct,
                "memory_used_bytes": memory_used_bytes,
                "memory_total_bytes": memory_total_bytes,
                "progress_counter": 0,
            }
    raise RuntimeError(f"nvidia-smi did not report GPU {device_index}")
