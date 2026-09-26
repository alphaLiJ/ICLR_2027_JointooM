from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_CANONICAL_TOPOLOGY_MAP_FAMILIES = frozenset(
    {
        "maze",
        "warehouse",
        "room",
        "real",
        "random",
        "kiva",
        "test",
    }
)

_TOPOLOGY_MAP_FAMILY_ALIASES = {
    "maze": "maze",
    "mazes": "maze",
    "warehouse": "warehouse",
    "warehouses": "warehouse",
    "room": "room",
    "rooms": "room",
    "real": "real",
    "random": "random",
    "kiva": "kiva",
    "test": "test",
}


@dataclass(frozen=True)
class ParsedTopologyMapName:
    original_name: str
    raw_prefix: str
    family: str
    opaque_id: str


@dataclass(frozen=True)
class ParsedTopologyCatalogGrid:
    obstacles: np.ndarray
    semantic_positions: dict[str, tuple[tuple[int, int], ...]]


def normalize_topology_map_family(family: str) -> str:
    normalized = family.strip().lower()
    canonical = _TOPOLOGY_MAP_FAMILY_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            "Unsupported topology map family prefix "
            f"{family!r}; expected one of {sorted(_CANONICAL_TOPOLOGY_MAP_FAMILIES)}"
        )
    return canonical


def parse_topology_map_name(name: str) -> ParsedTopologyMapName:
    if not isinstance(name, str) or not name:
        raise ValueError("Topology map name must be a non-empty string")
    if name != name.lower():
        raise ValueError(f"Topology map name must be lowercase ASCII, got {name!r}")
    if " " in name:
        raise ValueError(f"Topology map name must not contain spaces, got {name!r}")

    prefix, sep, opaque_id = name.partition("-")
    if not sep:
        canonical = _TOPOLOGY_MAP_FAMILY_ALIASES.get(name)
        if canonical is None:
            raise ValueError(
                "Topology map name must follow '<family>-<opaque_id>', "
                f"got {name!r}"
            )
        return ParsedTopologyMapName(
            original_name=name,
            raw_prefix=name,
            family=canonical,
            opaque_id="",
        )

    if not opaque_id:
        raise ValueError(
            "Topology map name must follow '<family>-<opaque_id>', "
            f"got {name!r}"
        )

    return ParsedTopologyMapName(
        original_name=name,
        raw_prefix=prefix,
        family=normalize_topology_map_family(prefix),
        opaque_id=opaque_id,
    )


def _load_simple_topology_yaml(file_path: str | Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            if not line.startswith(" ") and ":" in line:
                if current_name is not None:
                    loaded[current_name] = "\n".join(current_lines)
                name, value = line.split(":", 1)
                current_name = name.strip()
                current_lines = []
                value = value.strip()
                if value and value not in {"|", "|-"}:
                    raise ValueError(
                        f"Unsupported YAML value form for map {current_name!r}: {value!r}"
                    )
                continue

            if current_name is None:
                raise ValueError("Encountered YAML content before any map name")
            if not line.startswith("  "):
                raise ValueError(
                    f"Expected indented topology row for map {current_name!r}, got {line!r}"
                )
            current_lines.append(line[2:])

    if current_name is not None:
        loaded[current_name] = "\n".join(current_lines)

    if not loaded:
        raise ValueError("Expected YAML topology file to contain at least one named map")

    return loaded


def load_named_topology_strings_from_yaml(file_path: str | Path) -> dict[str, str]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_topology_yaml(file_path)

    with open(file_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError("Expected YAML topology file to contain a name-to-grid mapping")

    result: dict[str, str] = {}
    for name, grid_str in loaded.items():
        if not isinstance(name, str):
            raise ValueError("Map names must be strings")
        if not isinstance(grid_str, str):
            raise ValueError(f"Map {name!r} must be stored as a grid string")
        result[name] = grid_str
    return result


def _normalize_topology_rows(grid_str: str) -> list[str]:
    normalized = textwrap.dedent(grid_str).strip("\n")
    rows: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        rows.append(line)

    if not rows:
        raise ValueError("Topology grid string must contain at least one non-empty row")

    expected_width = len(rows[0])
    for row_idx, row in enumerate(rows[1:], start=1):
        if len(row) != expected_width:
            raise ValueError(
                f"Non-rectangular topology grid: row 0 has width {expected_width}, "
                f"row {row_idx} has width {len(row)}"
            )

    return rows


def parse_topology_catalog_grid_string(grid_str: str) -> ParsedTopologyCatalogGrid:
    rows: list[list[int]] = []
    semantic_positions: dict[str, list[tuple[int, int]]] = {}

    for row_idx, line in enumerate(_normalize_topology_rows(grid_str)):
        row: list[int] = []
        for col_idx, char in enumerate(line):
            if char == ".":
                row.append(0)
            elif char in {"#", "@"}:
                row.append(1)
            elif ("a" <= char <= "z") or ("A" <= char <= "Z"):
                lowered = char.lower()
                semantic_positions.setdefault(lowered, []).append((row_idx, col_idx))
                row.append(0)
            else:
                raise ValueError(
                    f"Unsupported topology symbol {char!r} at row {row_idx}, col {col_idx}"
                )
        rows.append(row)

    return ParsedTopologyCatalogGrid(
        obstacles=np.asarray(rows, dtype=np.int32),
        semantic_positions={
            symbol: tuple(positions) for symbol, positions in semantic_positions.items()
        },
    )


def parse_topology_grid_string(grid_str: str) -> np.ndarray:
    rows: list[list[int]] = []
    for row_idx, line in enumerate(_normalize_topology_rows(grid_str)):
        row: list[int] = []
        for col_idx, char in enumerate(line):
            if char == ".":
                row.append(0)
            elif char == "#":
                row.append(1)
            elif ("a" <= char <= "z") or ("A" <= char <= "Z"):
                row.append(0)
            else:
                raise ValueError(
                    f"Unsupported topology symbol {char!r} at row {row_idx}, col {col_idx}"
                )
        rows.append(row)

    return np.asarray(rows, dtype=np.int32)


def embed_obstacles_in_compiled_profile(
    obstacles: np.ndarray,
    *,
    compiled_map_w: int,
    compiled_map_h: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if obstacles.ndim != 2:
        raise ValueError(f"Expected 2D obstacle grid, got shape {obstacles.shape}")

    real_map_w, real_map_h = obstacles.shape
    if real_map_w > compiled_map_w or real_map_h > compiled_map_h:
        raise ValueError(
            f"Obstacle grid shape {obstacles.shape} exceeds compiled profile "
            f"{compiled_map_w}x{compiled_map_h}"
        )

    padded = np.zeros((compiled_map_w, compiled_map_h), dtype=np.int32)
    padded[:real_map_w, :real_map_h] = obstacles.astype(np.int32, copy=False)

    if real_map_w < compiled_map_w:
        padded[real_map_w, :real_map_h] = 1
    if real_map_h < compiled_map_h:
        padded[:real_map_w, real_map_h] = 1
    if real_map_w < compiled_map_w and real_map_h < compiled_map_h:
        padded[real_map_w, real_map_h] = 1

    return padded, (real_map_w, real_map_h)


def stack_embedded_obstacle_grids(
    obstacle_grids: Iterable[np.ndarray],
    *,
    compiled_map_w: int,
    compiled_map_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    padded_grids: list[np.ndarray] = []
    extents: list[tuple[int, int]] = []
    for obstacles in obstacle_grids:
        padded, extent = embed_obstacles_in_compiled_profile(
            obstacles,
            compiled_map_w=compiled_map_w,
            compiled_map_h=compiled_map_h,
        )
        padded_grids.append(padded)
        extents.append(extent)

    if not padded_grids:
        raise ValueError("Need at least one obstacle grid to stack")

    return np.stack(padded_grids), np.asarray(extents, dtype=np.int32)


def build_real_map_extent_tensor(real_extents, *, device: str | None = None):
    import torch

    extents = np.asarray(real_extents, dtype=np.int32)
    if extents.ndim == 1:
        if extents.shape != (2,):
            raise ValueError("1D real extents must have shape (2,)")
        extents = extents[None, :]
    if extents.ndim != 2 or extents.shape[1] != 2:
        raise ValueError("real extents must have shape (n_envs, 2)")

    tensor = torch.from_numpy(extents.copy())
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.contiguous()
