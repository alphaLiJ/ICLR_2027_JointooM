import numpy as np
import pytest

from mapf_cuda.training.topology_async import (
    _apply_simulator_builder_policy,
    _select_topology_training_maps,
)
from expert.expert_running import put_maps_into_registry
from expert.profiled_topology_loader import (
    embed_obstacles_in_compiled_profile,
    load_named_topology_strings_from_yaml,
    parse_topology_map_name,
    parse_topology_catalog_grid_string,
    parse_topology_grid_string,
)
from pogema.grid import Grid
from pogema.grid_config import GridConfig
from pogema.grid_registry import GRID_STR_REGISTRY, RegisteredGrid


def test_load_named_topology_strings_from_yaml(tmp_path):
    maps_yaml = tmp_path / "maps.yaml"
    maps_yaml.write_text(
        "tiny: |-\n"
        "  .#.\n"
        "  ..#\n",
        encoding="utf-8",
    )

    loaded = load_named_topology_strings_from_yaml(maps_yaml)

    assert list(loaded.keys()) == ["tiny"]
    assert ".#." in loaded["tiny"]


def test_parse_topology_grid_string_supports_letters_as_free_cells():
    obstacles = parse_topology_grid_string(
        """
        .a.
        #A#
        ...
        """
    )

    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(obstacles, expected)


def test_embed_obstacles_in_compiled_profile_adds_boundary_walls():
    obstacles = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    padded, real_extent = embed_obstacles_in_compiled_profile(
        obstacles,
        compiled_map_w=5,
        compiled_map_h=6,
    )

    assert padded.shape == (5, 6)
    assert real_extent == (2, 3)
    np.testing.assert_array_equal(padded[:2, :3], obstacles)
    np.testing.assert_array_equal(padded[2, :3], np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(padded[:2, 3], np.ones(2, dtype=np.int32))
    assert padded[4, 5] == 0


def test_embed_obstacles_in_compiled_profile_rejects_oversized_maps():
    obstacles = np.zeros((6, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="exceeds compiled profile"):
        embed_obstacles_in_compiled_profile(
            obstacles,
            compiled_map_w=5,
            compiled_map_h=6,
        )


def test_parse_topology_catalog_grid_string_supports_kiva_semantics():
    parsed = parse_topology_catalog_grid_string(
        """
        .re
        @@@
        e.r
        """
    )

    expected = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(parsed.obstacles, expected)
    assert parsed.semantic_positions["r"] == ((0, 1), (2, 2))
    assert parsed.semantic_positions["e"] == ((0, 2), (2, 0))


def test_parse_topology_map_name_normalizes_known_family_aliases():
    parsed = parse_topology_map_name("mazes-s0_wc8_od55")

    assert parsed.family == "maze"
    assert parsed.raw_prefix == "mazes"
    assert parsed.opaque_id == "s0_wc8_od55"


def test_parse_topology_map_name_rejects_unknown_prefix():
    with pytest.raises(ValueError, match="Unsupported topology map family prefix"):
        parse_topology_map_name("unknown-demo-map")


def test_parse_topology_map_name_accepts_existing_bare_family_names():
    parsed = parse_topology_map_name("kiva")

    assert parsed.family == "kiva"
    assert parsed.raw_prefix == "kiva"
    assert parsed.opaque_id == ""


def test_put_maps_into_registry_registers_kiva_catalog_without_inline_agents(tmp_path):
    map_name = "__unit_test_kiva_semantic__"
    GRID_STR_REGISTRY.pop(map_name, None)
    maps_yaml = tmp_path / "maps.yaml"
    maps_yaml.write_text(
        f"{map_name}: |-\n"
        "  .re\n"
        "  @@@\n"
        "  e.r\n",
        encoding="utf-8",
    )

    put_maps_into_registry(maps_yaml)

    assert map_name in GRID_STR_REGISTRY
    reg = GRID_STR_REGISTRY[map_name]
    np.testing.assert_array_equal(
        reg.get_obstacles(),
        np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    assert reg.get_agents_xy() is None
    assert reg.get_targets_xy() is None
    assert reg.semantic_positions["r"] == ((0, 1), (2, 2))


def test_grid_uses_registry_possible_positions_from_preparsed_entry():
    map_name = "__unit_test_sampling_semantic__"
    GRID_STR_REGISTRY.pop(map_name, None)
    RegisteredGrid(
        name=map_name,
        obstacles=np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=np.int32,
        ),
        possible_agents_xy=[(0, 1)],
        possible_targets_xy=[(2, 2)],
    )

    grid = Grid(
        GridConfig(
            map_name=map_name,
            num_agents=1,
            obs_radius=1,
            max_episode_steps=8,
            seed=7,
        ),
        add_artificial_border=False,
    )

    assert [tuple(xy) for xy in grid.starts_xy] == [(0, 1)]
    assert [tuple(xy) for xy in grid.finishes_xy] == [(2, 2)]


def test_select_topology_training_maps_respects_requested_map_names(tmp_path):
    map_a = "__unit_topology_train_a__"
    map_b = "__unit_topology_train_b__"
    GRID_STR_REGISTRY.pop(map_a, None)
    GRID_STR_REGISTRY.pop(map_b, None)
    maps_yaml = tmp_path / "maps.yaml"
    maps_yaml.write_text(
        f"{map_a}: |-\n"
        "  ...\n"
        "  .#.\n"
        "  ...\n"
        f"{map_b}: |-\n"
        "  ..#\n"
        "  ...\n"
        "  #..\n",
        encoding="utf-8",
    )

    selected = _select_topology_training_maps(
        maps_path=maps_yaml,
        num_experts=1,
        map_names=[map_b],
    )

    assert [name for name, _ in selected] == [map_b]
    np.testing.assert_array_equal(
        selected[0][1],
        np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_select_topology_training_maps_supports_family_filter(tmp_path):
    map_maze = "mazes-unit-a"
    map_kiva = "kiva-unit-b"
    GRID_STR_REGISTRY.pop(map_maze, None)
    GRID_STR_REGISTRY.pop(map_kiva, None)
    maps_yaml = tmp_path / "maps.yaml"
    maps_yaml.write_text(
        f"{map_maze}: |-\n"
        "  ...\n"
        "  .#.\n"
        "  ...\n"
        f"{map_kiva}: |-\n"
        "  ..#\n"
        "  ...\n"
        "  #..\n",
        encoding="utf-8",
    )

    selected = _select_topology_training_maps(
        maps_path=maps_yaml,
        num_experts=1,
        map_families=["maze"],
    )

    assert [name for name, _ in selected] == [map_maze]


def test_select_topology_training_maps_rejects_insufficient_registered_maps(tmp_path):
    map_name = "__unit_topology_train_single__"
    GRID_STR_REGISTRY.pop(map_name, None)
    maps_yaml = tmp_path / "maps.yaml"
    maps_yaml.write_text(
        f"{map_name}: |-\n"
        "  ...\n"
        "  ...\n"
        "  ...\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Need at least 2 topology maps"):
        _select_topology_training_maps(
            maps_path=maps_yaml,
            num_experts=2,
        )


def test_select_topology_training_maps_rejects_insufficient_maps_after_family_filter(tmp_path):
    map_name = "kiva-unit-single"
    GRID_STR_REGISTRY.pop(map_name, None)
    maps_yaml = tmp_path / "maps.yaml"
    maps_yaml.write_text(
        f"{map_name}: |-\n"
        "  ...\n"
        "  ...\n"
        "  ...\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Need at least 1 topology maps"):
        _select_topology_training_maps(
            maps_path=maps_yaml,
            num_experts=1,
            map_families=["maze"],
        )


def test_apply_simulator_builder_policy_sets_requested_modes():
    class DummySimulator:
        def __init__(self):
            self._builder_mode = "auto"
            self._local_gather_impl = "auto"

        @property
        def pyg_builder_mode(self):
            return self._builder_mode

        @pyg_builder_mode.setter
        def pyg_builder_mode(self, value):
            self._builder_mode = value

        @property
        def pyg_local_gather_impl(self):
            return self._local_gather_impl

        @pyg_local_gather_impl.setter
        def pyg_local_gather_impl(self, value):
            self._local_gather_impl = value

        @property
        def resolved_pyg_builder_impl(self):
            if self._builder_mode != "local_gather":
                return self._builder_mode
            if self._local_gather_impl == "async_sm89plus":
                return "local_gather_async_sm89plus"
            return "local_gather_scalar"

    sim = DummySimulator()

    resolved = _apply_simulator_builder_policy(
        sim,
        pyg_builder_mode="local_gather",
        pyg_local_gather_impl="async_sm89plus",
    )

    assert sim.pyg_builder_mode == "local_gather"
    assert sim.pyg_local_gather_impl == "async_sm89plus"
    assert resolved == "local_gather_async_sm89plus"
