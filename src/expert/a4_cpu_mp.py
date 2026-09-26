"""True-process CPU environment fan-out for the MAGAT A4 deployment study."""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from expert.benchmark_contract import FrozenTransitionBatch


def _receive(connection, *, worker_id: int, expected: str, timeout_s: float = 120.0):
    if not connection.poll(float(timeout_s)):
        raise TimeoutError(f"A4 worker {worker_id} did not send {expected}")
    message = connection.recv()
    if not isinstance(message, tuple) or not message:
        raise RuntimeError(f"A4 worker {worker_id} sent a malformed message")
    if message[0] == "ERROR":
        raise RuntimeError(f"A4 worker {worker_id} failed: {message[1]}")
    if message[0] != expected:
        raise RuntimeError(
            f"A4 worker {worker_id} sent {message[0]!r}, expected {expected!r}"
        )
    return message[1:]


def _worker_main(
    connection,
    worker_id: int,
    batch: FrozenTransitionBatch,
    max_episode_steps: int,
) -> None:
    """Own and step a disjoint CPU environment shard."""

    try:
        from expert.a4_runtime import _CpuEpisodeSet

        episodes = _CpuEpisodeSet(batch, int(max_episode_steps))
        connection.send(("READY", int(worker_id), os.getpid(), batch.semantic_sha256))
        command = connection.recv()
        if command != ("RUN",):
            raise RuntimeError(f"unexpected initial command {command!r}")
        builder_s = transition_s = 0.0
        for step in range(int(max_episode_steps)):
            started = time.perf_counter()
            references = episodes.build_references()
            builder_s += time.perf_counter() - started
            connection.send(
                (
                    "INPUT",
                    int(step),
                    references,
                    np.array(episodes.arrived, copy=True),
                )
            )
            command = connection.recv()
            if not isinstance(command, tuple) or len(command) != 3:
                raise RuntimeError(f"malformed action command {command!r}")
            phase, action_step, actions = command
            if phase != "ACTIONS" or int(action_step) != step:
                raise RuntimeError(f"unexpected action command {command[:2]!r}")
            started = time.perf_counter()
            episodes.step(np.asarray(actions, dtype=np.uint8), step)
            transition_s += time.perf_counter() - started
            locally_done = bool(np.all(episodes.terminated | episodes.truncated))
            connection.send(("STATUS", int(step), locally_done))
            control = connection.recv()
            if control == ("STOP",):
                break
            if control != ("CONTINUE",):
                raise RuntimeError(f"unexpected step control {control!r}")
        result = episodes.result()
        connection.send(
            (
                "DONE",
                result,
                {
                    "cpu_builder_s": float(builder_s),
                    "cpu_transition_s": float(transition_s),
                    "peak_host_memory_bytes": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    ),
                },
            )
        )
    except BaseException as exc:
        try:
            connection.send(("ERROR", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
        raise
    finally:
        connection.close()


def _combine_worker_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    array_keys = (
        "arrival_times",
        "arrived",
        "terminated",
        "truncated",
        "step_counts",
        "positions",
    )
    scalar_keys = ("requested_moves", "blocked_moves", "env_steps", "agent_steps")
    return {
        **{
            key: np.concatenate([np.asarray(result[key]) for result in results], axis=0)
            for key in array_keys
        },
        **{key: sum(int(result[key]) for result in results) for key in scalar_keys},
    }


def run_cpu_mp_engine(
    runtime,
    batch: FrozenTransitionBatch,
    *,
    max_episode_steps: int,
    device: str,
    num_workers: int,
) -> dict[str, Any]:
    """Run CPU transitions in spawned workers and one GPU model in the parent."""

    import torch

    from expert.a4_runtime import (
        _batch_references,
        _common_performance_metrics,
        _digest_arrays,
        _final_state_digest,
        _finish_health,
        _quality_metrics,
        _slice_batch,
        _start_health,
    )

    worker_count = min(int(num_workers), batch.num_envs)
    if worker_count <= 0:
        raise ValueError("A4 CPU MP requires at least one worker")
    shards = [
        part
        for part in np.array_split(np.arange(batch.num_envs), worker_count)
        if part.size
    ]
    context = mp.get_context("spawn")
    processes = []
    connections = []
    records: list[dict[str, Any]] = []
    collector = None
    try:
        for worker_id, indices in enumerate(shards):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_worker_main,
                args=(
                    child,
                    int(worker_id),
                    _slice_batch(batch, indices),
                    int(max_episode_steps),
                ),
                name=f"a4-cpu-env-{worker_id}",
            )
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        for worker_id, connection in enumerate(connections):
            ready_id, pid, _semantic_sha = _receive(
                connection, worker_id=worker_id, expected="READY"
            )
            if int(ready_id) != worker_id or int(pid) != int(processes[worker_id].pid):
                raise RuntimeError(f"A4 worker {worker_id} readiness identity mismatch")
            records.append(
                {
                    "worker_id": worker_id,
                    "pid": int(pid),
                    "parent_pid": os.getpid(),
                    "ready": True,
                }
            )

        def process_snapshot():
            workers = []
            for record, process in zip(records, processes):
                workers.append(
                    {
                        **record,
                        "alive": process.is_alive(),
                        "exit_code": process.exitcode,
                    }
                )
            return {"parent_pid": os.getpid(), "workers": workers}

        collector = _start_health(process_snapshot)
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for connection in connections:
            connection.send(("RUN",))

        trajectory = np.zeros(
            (max_episode_steps, batch.num_envs, batch.num_agents), dtype=np.uint8
        )
        h2d_bytes = d2h_bytes = 0
        model_s = 0.0
        executed_steps = 0
        for step in range(max_episode_steps):
            all_references = []
            arrived_parts = []
            for worker_id, connection in enumerate(connections):
                input_step, references, arrived = _receive(
                    connection, worker_id=worker_id, expected="INPUT"
                )
                if int(input_step) != step:
                    raise RuntimeError(f"A4 worker {worker_id} step identity mismatch")
                all_references.extend(references)
                arrived_parts.append(np.asarray(arrived, dtype=np.bool_))
            arrived = np.concatenate(arrived_parts, axis=0)
            data, copied = _batch_references(all_references, arrived, device=device)
            h2d_bytes += copied
            started = time.perf_counter()
            with torch.no_grad():
                logits = runtime.model(data.x, data)
            actions = logits.argmax(dim=-1).reshape(batch.num_envs, batch.num_agents)
            actions_host = actions.to(torch.uint8).cpu().numpy()
            torch.cuda.synchronize(torch.device(device))
            model_s += time.perf_counter() - started
            d2h_bytes += int(actions_host.nbytes)
            trajectory[step] = actions_host
            offset = 0
            for connection, indices in zip(connections, shards):
                count = int(indices.size)
                connection.send(
                    ("ACTIONS", int(step), actions_host[offset : offset + count])
                )
                offset += count
            local_done = []
            for worker_id, connection in enumerate(connections):
                status_step, done = _receive(
                    connection, worker_id=worker_id, expected="STATUS"
                )
                if int(status_step) != step:
                    raise RuntimeError(f"A4 worker {worker_id} status step mismatch")
                local_done.append(bool(done))
            executed_steps = step + 1
            stop = all(local_done) or executed_steps >= max_episode_steps
            for connection in connections:
                connection.send(("STOP",) if stop else ("CONTINUE",))
            if stop:
                break

        worker_results = []
        worker_timings = []
        for worker_id, connection in enumerate(connections):
            result, timings = _receive(connection, worker_id=worker_id, expected="DONE")
            worker_results.append(result)
            worker_timings.append(timings)
        for process in processes:
            process.join(timeout=30.0)
            if process.is_alive():
                raise TimeoutError(f"A4 worker {process.pid} did not exit")
            if process.exitcode != 0:
                raise RuntimeError(f"A4 worker {process.pid} exited with {process.exitcode}")
        torch.cuda.synchronize(torch.device(device))
        wall_s = time.perf_counter() - wall_started
        cpu_process_s = time.process_time() - cpu_started
        result = _combine_worker_results(worker_results)
        result["executed_steps"] = int(executed_steps)
        health = _finish_health(collector, expected_workers=worker_count)
        collector = None
        peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))
        return {
            **_quality_metrics(result, max_episode_steps),
            **_common_performance_metrics(
                wall_s=wall_s,
                batch=batch,
                result=result,
                h2d_bytes=h2d_bytes,
                d2h_bytes=d2h_bytes,
                peak_gpu_memory_bytes=peak_gpu,
                cpu_process_s=cpu_process_s,
                health=health,
            ),
            "accepted": bool(health["validation"]["valid"]),
            "observed_worker_pids": [int(process.pid) for process in processes],
            "health_report": health,
            "worker_cpu_builder_s": [
                float(item["cpu_builder_s"]) for item in worker_timings
            ],
            "worker_cpu_transition_s": [
                float(item["cpu_transition_s"]) for item in worker_timings
            ],
            "worker_peak_host_memory_bytes": [
                int(item["peak_host_memory_bytes"]) for item in worker_timings
            ],
            "worker_peak_host_memory_sum_bytes": sum(
                int(item["peak_host_memory_bytes"]) for item in worker_timings
            ),
            "model_and_action_readback_s": float(model_s),
            "trajectory_action_sha256": _digest_arrays(
                actions=trajectory[:executed_steps]
            ),
            "final_state_sha256": _final_state_digest(result, batch.goals),
        }
    finally:
        if collector is not None:
            try:
                collector.stop()
            except BaseException:
                pass
        for connection in connections:
            try:
                connection.close()
            except BaseException:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)


__all__ = ["run_cpu_mp_engine"]
