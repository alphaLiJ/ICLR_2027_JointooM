"""Small standard-MAPF POGEMA constructors used by experts and baselines."""

from __future__ import annotations

from pogema import GridConfig, pogema_v0


def build_random_standard_env(
    *,
    num_agents: int,
    map_size: int,
    density: float,
    seed: int,
    max_episode_steps: int = 256,
    collision_system: str = "soft",
):
    config = GridConfig(
        size=int(map_size),
        density=float(density),
        num_agents=int(num_agents),
        obs_radius=5,
        max_episode_steps=int(max_episode_steps),
        observation_type="MAPF",
        on_target="nothing",
        collision_system=collision_system,
        seed=int(seed),
    )
    env = pogema_v0(grid_config=config)
    env.reset()
    return env


def build_topology_standard_env(
    *,
    num_agents: int,
    map_name: str,
    seed: int,
    max_episode_steps: int = 256,
    collision_system: str = "soft",
):
    config = GridConfig(
        num_agents=int(num_agents),
        map_name=str(map_name),
        obs_radius=5,
        max_episode_steps=int(max_episode_steps),
        observation_type="MAPF",
        on_target="nothing",
        collision_system=collision_system,
        seed=int(seed),
    )
    env = pogema_v0(grid_config=config)
    env.reset()
    return env

