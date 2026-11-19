"""Native single-state joint-action labels; explicit failures, no wait fallback."""
import runtime
import ctypes
import os
from pathlib import Path
import time
import numpy as np

_LIB = {}


def initialize_worker(workers=4):
    import multiprocessing as mp
    allowed = sorted(os.sched_getaffinity(0))
    index = (mp.current_process()._identity[-1] - 1) % workers
    os.sched_setaffinity(0, allowed[index::workers])


def library(fixed_work=False):
    if fixed_work not in _LIB:
        name = 'native/fixed-work/liblacam_work.so' if fixed_work else 'native/liblacam_first.so'
        value = ctypes.CDLL(str(runtime.ART / name))
        value.run_lacam.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
        value.run_lacam.restype = ctypes.c_char_p
        if fixed_work:
            value.last_lacam_iterations.argtypes = []
            value.last_lacam_iterations.restype = ctypes.c_int
        _LIB[fixed_work] = value
    return _LIB[fixed_work]


def label_state(payload):
    started = time.perf_counter()
    grid = np.asarray(payload['grid'], dtype=np.uint8)
    rows = np.asarray(payload['rows'], dtype=np.uint16)
    # Source coordinates are official Pogema observation-padded coordinates.
    starts = rows[:, 2:4].astype(np.int32) - 5
    goals = rows[:, 4:6].astype(np.int32) - 5
    h, w = grid.shape
    body = '\n'.join(''.join('@' if c else '.' for c in row) for row in grid)
    map_text = f'type octile\nheight {h}\nwidth {w}\nmap\n{body}'
    scene = 'version 1\n' + ''.join(
        f'{i}\ttmp.map\t{h}\t{w}\t{p[1]}\t{p[0]}\t{g[1]}\t{g[0]}\t1\n'
        for i, (p, g) in enumerate(zip(starts, goals)))
    iterations = int(payload.get('iterations', 0))
    native = library(iterations > 0)
    budget = iterations if iterations > 0 else float(payload.get('timeout', 2.0))
    encoded = native.run_lacam(map_text.encode(), scene.encode(), len(rows), float(budget))
    text = encoded.decode()
    result = dict(index=payload.get('index'), success=False, agents=len(rows),
                  pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)))
    if iterations > 0:
        result.update(iteration_budget=iterations, search_iterations=native.last_lacam_iterations())
    if not text.startswith('ERROR'):
        lines = [line for line in text.strip().splitlines() if line]
        first = np.asarray([list(map(int, v.split(',')))[::-1]
                            for v in lines[0].split('|') if v], dtype=np.int32)
        second = first if len(lines) == 1 else np.asarray([
            list(map(int, v.split(',')))[::-1] for v in lines[1].split('|') if v], dtype=np.int32)
        from pogema import GridConfig
        moves = {tuple(v): i for i, v in enumerate(GridConfig().MOVES)}
        if np.array_equal(first, starts):
            actions = np.asarray([moves[tuple(d)] for d in second-first], dtype=np.int64)
            result.update(success=True, labels=actions.tolist(), plan_steps=len(lines)-1,
                          first_positions=first.tolist(), next_positions=second.tolist())
        else:
            result['error'] = 'planner first configuration differs from queried state'
    else:
        result['error'] = text
    result['seconds'] = time.perf_counter()-started
    return result
