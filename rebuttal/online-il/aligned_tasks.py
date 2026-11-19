import bootstrap
"""Original complete-decision tasks, preserving each environment's own map."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.ndimage import label


PAPER_INPUT = bootstrap.input_path('JOINTLOOM_DECISION_INPUT', bootstrap.SOURCE/'artifacts/rebuttal/inputs/a2-random-d020-a256.npz')
PAPER_CHECKPOINT = bootstrap.input_path('JOINTLOOM_DECISION_CHECKPOINT', bootstrap.SOURCE/'artifacts/rebuttal/checkpoints/magat.pt')


@dataclass
class AlignedTasks:
    name: str
    grids: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    initial_arrived: np.ndarray
    cells: np.ndarray
    cell_counts: np.ndarray
    candidates: np.ndarray
    source_stats: dict

    @property
    def grid(self):
        return self.grids[0]

    def cells_for_env(self, env_id):
        return self.cells[env_id, :self.cell_counts[env_id]]

    def subset(self, ids):
        return AlignedTasks(self.name, *[getattr(self, key)[ids].copy() for key in
            ('grids', 'positions', 'goals', 'initial_arrived', 'cells', 'cell_counts', 'candidates')], self.source_stats)


def load_paper_tasks(envs, seed=20261108, candidate_length=512):
    with np.load(PAPER_INPUT) as source:
        grids = source['grids'][:envs].astype(np.uint8)
        positions = source['positions'][:envs].astype(np.int16)
        goals = source['goals'][:envs].astype(np.int16)
        arrived = source['arrived'][:envs].astype(np.uint8)
    if len(grids) != envs:
        raise ValueError('requested more environments than the original pool')
    original_arrived = int(arrived.sum())
    unreachable = 0
    cell_lists = []
    candidates = []
    for e, grid in enumerate(grids):
        components, _ = label(grid == 0)
        ids = components[positions[e, :, 0], positions[e, :, 1]]
        target_ids = components[goals[e, :, 0], goals[e, :, 1]]
        unreachable += int(np.count_nonzero(ids != target_ids))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        cells = np.argwhere(components == sizes.argmax()).astype(np.int16)
        cell_lists.append(cells)
        rng = np.random.default_rng(np.random.SeedSequence([seed, e]))
        selected = rng.choice(len(cells), 2*positions.shape[1], replace=False)
        positions[e] = cells[selected[:positions.shape[1]]]
        goals[e] = cells[selected[positions.shape[1]:]]
        candidates.append(rng.integers(len(cells), size=(positions.shape[1], candidate_length), dtype=np.int32))
    padded = np.zeros((envs, max(map(len, cell_lists)), 2), dtype=np.int16)
    for e, cells in enumerate(cell_lists):
        padded[e, :len(cells)] = cells
    stats = dict(initial_state='common reachable resample on unchanged original maps', seed=seed,
        original_arrived_agents=original_arrived, original_unreachable_goals=unreachable)
    return AlignedTasks('original-random-d020-a256', grids, positions, goals, np.zeros_like(arrived),
        padded, np.array(list(map(len, cell_lists)), dtype=np.int32), np.stack(candidates), stats)


class AlignedResident:
    def __init__(self, tasks, mode, device='cuda:0', horizon=4096):
        import torch
        import grid_world_cpp as ext
        from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator
        self.tasks, self.mode, self.device = tasks, mode, device
        grids, extents = stack_grids_for_compiled_simulator(tasks.grids, device=device)
        self.sim = ext.GridWorldSimulator(grids, tasks.positions.shape[1], 0, grids.numel(),
            20261108, mode, horizon)
        self.sim.set_real_map_extents(extents)
        self.sim.pyg_builder_mode = 'local_gather'
        self.sim.pyg_local_gather_impl = 'auto'
        self.candidates = torch.as_tensor(tasks.candidates, device=device)
        self.cells = torch.as_tensor(tasks.cells, device=device)
        self.counts = torch.as_tensor(tasks.cell_counts, device=device)
        self.reset()

    def reset(self):
        import torch
        tasks = self.tasks
        self.sim.load_state(torch.as_tensor(tasks.positions, device=self.device),
            torch.as_tensor(tasks.goals, device=self.device),
            torch.as_tensor(tasks.initial_arrived if self.mode=='standard_mapf' else
                np.zeros_like(tasks.initial_arrived), device=self.device),
            torch.zeros(len(tasks.positions), dtype=torch.int32, device=self.device))
        if self.mode == 'lifelong':
            self.sim.set_goal_candidates(self.candidates, self.cells, self.counts)
        # load_state marks goals dirty. Leave initial distance construction timed.

    def result(self):
        import torch
        sim = self.sim
        arrived = sim.arrived.cpu().numpy() if self.mode=='standard_mapf' else (
            (sim.cur_x==sim.goal_x)&(sim.cur_y==sim.goal_y)).cpu().numpy().astype(np.uint8)
        cursors = sim.goal_cursors.cpu().numpy() if self.mode == 'lifelong' else (
            arrived.astype(np.int32) - self.tasks.initial_arrived)
        return dict(positions=torch.stack((sim.cur_x, sim.cur_y), -1).cpu().numpy(),
            goals=torch.stack((sim.goal_x, sim.goal_y), -1).cpu().numpy(),
            arrived=arrived, cursors=cursors)
