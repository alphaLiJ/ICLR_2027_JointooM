"""Small spawn target for the throughput-only POGEMA baseline."""

from __future__ import annotations


def pogema_throughput_worker(connection, worker_id: int, batch) -> None:
    """Execute fixed trajectories without hashes, snapshots, or parity replay."""

    try:
        from expert.transition_adapters import make_transition_adapter

        # ``spawn`` recreates NumPy allocations as writable arrays.  The fast
        # path trusts the parent-side batch validation, but action binding still
        # uses the read-only flag to distinguish the canonical schedule from a
        # caller-owned copy.
        for name in (
            "instance_ids",
            "grids",
            "positions",
            "goals",
            "arrived",
            "active",
            "actions",
        ):
            getattr(batch, name).flags.writeable = False
        adapter = make_transition_adapter(
            "pogema",
            batch,
            "cpu",
            verify_consumption=False,
        )
        connection.send(("BOOT", worker_id))
        while True:
            command = connection.recv()
            phase = command[0]
            if phase == "STOP":
                break
            if phase != "RESET":
                raise RuntimeError(f"unexpected worker command: {command!r}")
            repetition = int(command[1])
            adapter.reset_device_state()
            prepared = adapter.prepare_actions(batch.actions)
            adapter.synchronize_transition()
            connection.send(("READY", worker_id, repetition))

            command = connection.recv()
            if command != ("GO", repetition):
                raise RuntimeError(f"unexpected GO command: {command!r}")
            for handle in prepared:
                adapter.step_transition(handle)
            adapter.synchronize_transition()
            connection.send(("DONE", worker_id, repetition))
    except BaseException as exc:
        try:
            connection.send(("ERROR", worker_id, repr(exc)))
        finally:
            connection.close()
        raise
    connection.close()


__all__ = ["pogema_throughput_worker"]
