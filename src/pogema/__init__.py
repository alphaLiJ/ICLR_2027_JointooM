from pogema.grid_config import GridConfig

try:
    from gymnasium import register
except ModuleNotFoundError:
    register = None

try:
    from pogema.integrations.make_pogema import pogema_v0
    from pogema.svg_animation.animation_wrapper import AnimationMonitor, AnimationConfig
except ModuleNotFoundError:
    pogema_v0 = None
    AnimationMonitor = None
    AnimationConfig = None

try:
    from pogema.a_star_policy import AStarAgent, BatchAStarAgent
except ModuleNotFoundError:
    AStarAgent = None
    BatchAStarAgent = None

__version__ = '1.4.0'

__all__ = [
    'GridConfig',
    'pogema_v0',
    'AStarAgent', 'BatchAStarAgent',
    "AnimationMonitor", "AnimationConfig",
]

if register is not None and pogema_v0 is not None:
    register(
        id="Pogema-v0",
        entry_point="pogema.integrations.make_pogema:make_single_agent_gym",
    )
