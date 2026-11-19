from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label
import yaml
from bootstrap import SOURCE


@dataclass
class TaskBatch:
    name: str
    grid: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    cells: np.ndarray
    candidates: np.ndarray

    def subset(self, ids):
        return TaskBatch(self.name, self.grid, self.positions[ids].copy(),
                         self.goals[ids].copy(), self.cells, self.candidates[ids])


def make_tasks(name, agents, envs, seed, candidate_length=2048):
    catalog = yaml.safe_load((SOURCE / "maps/maps.yaml").read_text())
    if name in catalog:
        rows = catalog[name].strip().splitlines()
        grid = np.array([[c in "#@T" for c in row] for row in rows], dtype=np.uint8)
    elif name == "random-e2-seed20261108":
        grid = (np.random.default_rng(20261108).random((65, 65)) < .2).astype(np.uint8)
    elif name == "warehouse-e2-shelves65":
        grid = np.zeros((65, 65), dtype=np.uint8)
        for x in range(4, 59, 8):
            for y in range(4, 59, 12):
                grid[x:x+4, y:y+8] = 1
    else:
        raise KeyError(name)
    components, _ = label(grid == 0)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    cells = np.argwhere(components == sizes.argmax()).astype(np.int16)
    if len(cells) < 2 * agents:
        raise ValueError("not enough reachable cells")
    positions, goals, candidates = [], [], []
    for e in range(envs):
        rng = np.random.default_rng(np.random.SeedSequence([seed, e]))
        ids = rng.choice(len(cells), 2 * agents, replace=False)
        positions.append(cells[ids[:agents]])
        goals.append(cells[ids[agents:]])
        candidates.append(rng.integers(len(cells), size=(agents, candidate_length), dtype=np.int32))
    return TaskBatch(name, grid, np.array(positions), np.array(goals), cells, np.array(candidates))


def assign_goals(positions, goals, cursors, starts, cells, reached):
    """Same legal candidates and canonical agent order as the optional CUDA branch."""
    reserved = {tuple(g) for g, done in zip(goals, reached) if not done}
    for i in np.flatnonzero(reached):
        begin = int(starts[i, cursors[i] % starts.shape[1]])
        for offset in range(len(cells)):
            candidate = cells[(begin + offset) % len(cells)]
            point = tuple(candidate)
            if point != tuple(positions[i]) and point not in reserved:
                goals[i] = candidate
                reserved.add(point)
                cursors[i] += 1
                break
        else:
            raise RuntimeError("goal assignment exhausted reachable cells")
