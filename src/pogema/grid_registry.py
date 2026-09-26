import numpy as np

from pogema.utils import check_grid, render_grid

GRID_STR_REGISTRY = {}


def in_registry(name):
    return name in GRID_STR_REGISTRY


def get_grid(name):
    if in_registry(name):
        return GRID_STR_REGISTRY[name]
    else:
        raise KeyError(f"Grid with name {name} not found")


class RegisteredGrid:
    FREE = 0
    OBSTACLE = 1

    def str_to_grid(self, grid_str):
        obstacles = []
        agents = {}
        targets = {}
        for idx, line in enumerate(grid_str.split()):
            row = []
            for char in line:
                if char == '.':
                    row.append(self.FREE)
                elif char == '#':
                    row.append(self.OBSTACLE)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = len(obstacles), len(row)
                    row.append(self.FREE)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = len(obstacles), len(row)
                    row.append(self.FREE)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {idx}")
            if row:
                if obstacles:
                    assert len(obstacles[-1]) == len(row), f"Wrong string size for row {idx};"
                obstacles.append(row)
        return obstacles, agents, targets

    @staticmethod
    def _normalize_positions(positions):
        if positions is None:
            return None
        return [list(position) for position in positions]

    @staticmethod
    def _normalize_semantic_positions(semantic_positions):
        if semantic_positions is None:
            return {}
        normalized = {}
        for symbol, positions in semantic_positions.items():
            normalized[symbol] = tuple(tuple(position) for position in positions)
        return normalized

    @staticmethod
    def _check_possible_positions(obstacles, positions, label):
        if positions is None:
            return
        height, width = obstacles.shape
        for x, y in positions:
            if not (0 <= x < height and 0 <= y < width):
                raise IndexError(f"{label} position {(x, y)} is out of bounds for {obstacles.shape}")
            if obstacles[x, y] != RegisteredGrid.FREE:
                raise KeyError(f"{label} position {(x, y)} is blocked by an obstacle")

    def __init__(
        self,
        name: str,
        grid_str: str = None,
        agents_positions: list = None,
        agents_targets: list = None,
        obstacles=None,
        possible_agents_xy: list = None,
        possible_targets_xy: list = None,
        semantic_positions: dict | None = None,
    ):
        self.name = name
        self.grid_str = grid_str
        self.agents_positions = agents_positions
        self.agents_targets = agents_targets
        self.possible_agents_xy = self._normalize_positions(possible_agents_xy)
        self.possible_targets_xy = self._normalize_positions(possible_targets_xy)
        self.semantic_positions = self._normalize_semantic_positions(semantic_positions)

        if grid_str is not None and obstacles is not None:
            raise ValueError("Provide either grid_str or obstacles, not both")
        if grid_str is None and obstacles is None:
            raise ValueError("RegisteredGrid requires either grid_str or obstacles")

        if grid_str is not None:
            self.obstacles, agents, targets = self.str_to_grid(grid_str)
            self.obstacles = np.array(self.obstacles, dtype=np.int32)
        else:
            self.obstacles = np.array(obstacles, dtype=np.int32, copy=True)
            agents, targets = {}, {}
            if self.obstacles.ndim != 2:
                raise ValueError(f"Expected 2D obstacle grid, got shape {self.obstacles.shape}")

        if agents_positions and agents:
            raise ValueError("Agents positions are already defined in the grid string!")
        if agents_targets and targets:
            raise ValueError("Agents targets are already defined in the grid string!")

        if agents:
            self.agents_xy = []
            for _, (x, y) in sorted(agents.items()):
                self.agents_xy.append([x, y])
        else:
            self.agents_xy = agents_positions

        if targets:
            self.targets_xy = []
            for _, (x, y) in sorted(targets.items()):
                self.targets_xy.append([x, y])
        else:
            self.targets_xy = agents_targets
        if in_registry(name):
            raise ValueError(f"Grid with name {self.name} already registered!")
        check_grid(self.obstacles, self.agents_xy, self.targets_xy)
        self._check_possible_positions(self.obstacles, self.possible_agents_xy, "Possible agent")
        self._check_possible_positions(self.obstacles, self.possible_targets_xy, "Possible target")

        register_grid(self)

    def get_obstacles(self):
        return self.obstacles

    def get_agents_xy(self):
        return self.agents_xy

    def get_targets_xy(self):
        return self.targets_xy

    def get_possible_agents_xy(self):
        if self.possible_agents_xy is None:
            return None
        return [list(position) for position in self.possible_agents_xy]

    def get_possible_targets_xy(self):
        if self.possible_targets_xy is None:
            return None
        return [list(position) for position in self.possible_targets_xy]

    def render(self):
        render_grid(obstacles=self.get_obstacles(), positions_xy=self.get_agents_xy(), targets_xy=self.get_targets_xy())


def register_grid(rg: RegisteredGrid):
    if in_registry(rg.name):
        raise KeyError(f"Grid with name {rg.name} already registered")
    GRID_STR_REGISTRY[rg.name] = rg
