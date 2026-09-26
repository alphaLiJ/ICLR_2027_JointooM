"""Fail-stop regression tests for complete-stage publication.

These tests deliberately exercise the narrow reserve-before-publish window.
The supported contract is fail-closed safety plus whole-run abort/restart; the
ring does not lease or reassign an incomplete environment block in place.
"""

import multiprocessing as mp
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from expert.expert_running import DMAWorker, FEATURE_DIM, RingBuffer


_HARD_EXIT_CODE = 73


def _reserve_then_hard_exit(ring: RingBuffer, reserved) -> None:
    ring._reserve_stage_block(env_id=0)
    reserved.set()
    os._exit(_HARD_EXIT_CODE)


def _make_stage_ring() -> RingBuffer:
    with patch("torch.Tensor.pin_memory", lambda self: self):
        return RingBuffer(capacity=8, num_envs=2, agents_per_env=2)


def _env_block(env_id: int) -> np.ndarray:
    block = np.zeros((2, FEATURE_DIM), dtype=np.uint16)
    block[:, 0] = env_id
    block[:, 1] = np.arange(2, dtype=np.uint16)
    return block


def test_hard_exit_after_reserve_keeps_partial_stage_invisible_and_restartable():
    ctx = mp.get_context("fork")
    ring = _make_stage_ring()
    reserved = ctx.Event()
    child = ctx.Process(target=_reserve_then_hard_exit, args=(ring, reserved))

    started = time.monotonic()
    child.start()
    assert reserved.wait(timeout=2.0)
    child.join(timeout=2.0)
    elapsed = time.monotonic() - started

    assert child.is_alive() is False
    assert child.exitcode == _HARD_EXIT_CODE
    assert elapsed < 2.0
    assert ring.stage_seq_tags[0] == 0
    assert ring.stage_ready_counts[0] == 0
    assert list(ring.env_write_seq) == [0, 0]
    assert sum(ring.ready_flags[:4]) == 0
    assert ring.is_stage_ready(0) is False

    # DMA admission uses the same complete-stage predicate and therefore must
    # not advance when the reserved sub-block was never published.
    with patch("torch.cuda.Stream", return_value=object()), patch(
        "torch.cuda.Event", return_value=MagicMock()
    ):
        dma = DMAWorker(
            ring,
            torch.zeros((8, FEATURE_DIM), dtype=torch.int16),
            batch_threshold=4,
            device="cpu",
        )
    assert dma.run_once() is None
    assert dma.dma_read_ptr == 0

    # Recovery is whole-run restart, not in-place lease reclamation.
    fresh_ring = _make_stage_ring()
    fresh_ring.reserve_and_write(_env_block(0))
    fresh_ring.reserve_and_write(_env_block(1))
    assert fresh_ring.is_stage_ready(0) is True
