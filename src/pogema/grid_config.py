import sys
from typing import Optional, Union, Any
from pydantic import field_validator, model_validator, ValidationInfo

from pogema.utils import CommonSettings

from typing_extensions import Literal


class GridConfig(CommonSettings):
    on_target: Literal['finish', 'nothing', 'restart'] = 'finish'
    seed: Optional[int] = None
    task_seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size: int = 8
    density: float = 0.3
    obs_radius: int = 5
    agents_xy: Optional[list] = None
    targets_xy: Optional[list] = None
    num_agents: Optional[int] = None
    possible_agents_xy: Optional[list] = None
    possible_targets_xy: Optional[list] = None
    collision_system: Literal['block_both', 'priority', 'soft'] = 'priority'
    persistent: bool = False
    observation_type: Literal['POMAPF', 'MAPF', 'default'] = 'default'
    map: Optional[Union[list, str]] = None

    map_name: Optional[str] = None

    integration: Literal['SampleFactory', 'PyMARL', 'rllib', 'gymnasium', 'PettingZoo'] = None
    max_episode_steps: int = 64
    auto_reset: Optional[bool] = None

    @model_validator(mode='before')
    @classmethod
    def map_validation(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        
        v = data.get('map')
        if v is not None:
            if isinstance(v, str):
                v, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy = cls.str_map_to_list(
                    v, data.get('FREE'), data.get('OBSTACLE')
                )
                if agents_xy and targets_xy and data.get('agents_xy') is not None and data.get('targets_xy') is not None:
                    raise KeyError("""Can't create task. Please provide agents_xy and targets_xy only once.
                    Either with parameters or with a map.""")
                if (agents_xy or targets_xy) and (possible_agents_xy or possible_targets_xy):
                    raise KeyError("""Can't create task. Mark either possible locations or precise ones.""")
                elif agents_xy and targets_xy:
                    data['agents_xy'] = agents_xy
                    data['targets_xy'] = targets_xy
                    data['num_agents'] = len(agents_xy)
                elif (data.get('agents_xy') is None or data.get('targets_xy') is None) and possible_agents_xy and possible_targets_xy:
                    data['possible_agents_xy'] = possible_agents_xy
                    data['possible_targets_xy'] = possible_targets_xy
            
            height = len(v)
            width = 0
            area = 0
            for line in v:
                width = max(width, len(line))
                area += len(line)
            
            data['map'] = v
            data['size'] = max(width, height)
            data['width'] = width
            data['height'] = height
            data['density'] = sum([sum(line) for line in v]) / area

        v_agents = data.get('num_agents')
        if v_agents is None:
            if data.get('agents_xy'):
                v_agents = len(data.get('agents_xy'))
            else:
                v_agents = 1
        assert 1 <= v_agents <= 10000000, "num_agents must be in [1, 10000000]"
        data['num_agents'] = v_agents

        width_provided = data.get('width') is not None
        height_provided = data.get('height') is not None
        
        if width_provided and not height_provided:
            raise ValueError("Invalid dimension configuration. Please provide height.")
        elif not width_provided and height_provided:
            raise ValueError("Invalid dimension configuration. Please provide width.")
        
        if not width_provided and not height_provided:
            data['width'] = data.get('size', 8)
            data['height'] = data.get('size', 8)
        if 'size' not in data or data.get('size') != max(data.get('width'), data.get('height')):
            data['size'] = max(data.get('width'), data.get('height'))

        width = data.get('width')
        height = data.get('height')
        
        if width is not None and height is not None:
            agents_xy = data.get('agents_xy')
            if agents_xy is not None:
                cls.check_positions(agents_xy, width, height)

            targets_xy = data.get('targets_xy')
            if targets_xy is not None:
                first_element = targets_xy[0]
                if isinstance(first_element[0], (list, tuple)):
                    for agent_goals in targets_xy:
                        cls.check_positions(agent_goals, width, height)
                else:
                    cls.check_positions(targets_xy, width, height)
        
        return data

    @field_validator('seed', 'task_seed')
    @classmethod
    def seed_initialization(cls, v):
        assert v is None or (0 <= v < sys.maxsize), "seed must be in [0, " + str(sys.maxsize) + ']'
        return v

    def get_effective_task_seed(self) -> Optional[int]:
        return self.task_seed if self.task_seed is not None else self.seed

    @staticmethod
    def _validate_dimension(v, field_name):
        if v is not None:
            if field_name == 'size':
                assert 2 <= v <= 4096, f"{field_name} must be in [2, 4096]"
            else:
                assert 1 <= v <= 4096, f"{field_name} must be in [1, 4096]"
        return v

    @field_validator('size')
    @classmethod
    def size_restrictions(cls, v):
        return cls._validate_dimension(v, 'size')

    @field_validator('width')
    @classmethod
    def width_restrictions(cls, v):
        return cls._validate_dimension(v, 'width')

    @field_validator('height')
    @classmethod
    def height_restrictions(cls, v):
        return cls._validate_dimension(v, 'height')

    @field_validator('density')
    @classmethod
    def density_restrictions(cls, v):
        assert 0.0 <= v <= 1, "density must be in [0, 1]"
        return v

    @field_validator('agents_xy')
    @classmethod
    def agents_xy_validation(cls, v):
        if v is not None:
            if not isinstance(v, (list, tuple)):
                raise ValueError("agents_xy must be a list")
            for position in v:
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    raise ValueError("Position must be a list/tuple of length 2")
                if not all(isinstance(coord, int) for coord in position):
                    raise ValueError("Position coordinates must be integers")
        return v

    @field_validator('targets_xy')
    @classmethod
    def targets_xy_validation(cls, v, info: ValidationInfo):
        if v is not None:
            if not v or not isinstance(v, (list, tuple)):
                raise ValueError("targets_xy must be a list")
            
            first_element = v[0]
            if not isinstance(first_element, (list, tuple)):
                raise ValueError("Invalid targets_xy format")
            
            if isinstance(first_element[0], (list, tuple)):
                for agent_goals in v:
                    if not isinstance(agent_goals, (list, tuple)) or len(agent_goals) < 2:
                        raise ValueError("Each agent must have at least two goals in the sequence")
                    for position in agent_goals:
                        if not isinstance(position, (list, tuple)) or len(position) != 2:
                            raise ValueError("Position must be a list/tuple of length 2")
                        if not all(isinstance(coord, int) for coord in position):
                            raise ValueError("Position coordinates must be integers")
            else:
                on_target = info.data.get('on_target', 'finish')
                if on_target == 'restart':
                    raise ValueError("on_target='restart' requires goal sequences, not single goals. Use format: targets_xy: [[[x1,y1],[x2,y2]], [[x3,y3],[x4,y4]]]")
                for position in v:
                    if not isinstance(position, (list, tuple)) or len(position) != 2:
                        raise ValueError("Position must be a list/tuple of length 2")
                    if not all(isinstance(coord, int) for coord in position):
                        raise ValueError("Position coordinates must be integers")
        return v

    @staticmethod
    def check_positions(v, width, height):
        for position in v:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError("Position must be a list/tuple of length 2")
            x, y = position
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("Position coordinates must be integers")
            if not (0 <= x < height and 0 <= y < width):
                raise IndexError(f"Position is out of bounds! {position} is not in [{0}, {height}] x [{0}, {width}]")

    @field_validator('obs_radius')
    @classmethod
    def obs_radius_must_be_positive(cls, v):
        assert 1 <= v <= 128, "obs_radius must be in [1, 128]"
        return v

    @field_validator('possible_agents_xy')
    @classmethod
    def possible_agents_xy_validation(cls, v):
        return v

    @field_validator('possible_targets_xy')
    @classmethod
    def possible_targets_xy_validation(cls, v):
        return v

    @staticmethod
    def str_map_to_list(str_map, free, obstacle):
        obstacles = []
        agents = {}
        targets = {}
        possible_agents_xy = []
        possible_targets_xy = []
        special_chars = {'@', '$', '!'}

        for row_idx, line in enumerate(str_map.split()):
            row = []
            for col_idx, char in enumerate(line):
                position = (row_idx, col_idx)

                if char == '.':
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif char == '#':
                    row.append(obstacle)
                elif char in special_chars:
                    row.append(free)
                    if char == '@':
                        possible_agents_xy.append(position)
                    elif char == '$':
                        possible_targets_xy.append(position)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {row_idx}")

            if row:
                assert len(obstacles[-1]) == len(row) if obstacles else True, f"Wrong string size for row {row_idx};"
                obstacles.append(row)

        agents_xy = [[x, y] for _, (x, y) in sorted(agents.items())]
        targets_xy = [[x, y] for _, (x, y) in sorted(targets.items())]

        assert len(targets_xy) == len(agents_xy), "Mismatch in number of agents and targets."

        if not any(char in special_chars for char in str_map):
            possible_agents_xy, possible_targets_xy = None, None

        return obstacles, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy

    def update_config(self, **kwargs):
        current_values = self.model_dump()
        
        if 'size' in kwargs:
            current_values.pop('width', None)
            current_values.pop('height', None)
        elif 'width' in kwargs or 'height' in kwargs:
            current_values.pop('size', None)
        current_values.update(kwargs)
        new_instance = GridConfig(**current_values)
        
        for field_name, field_value in new_instance.__dict__.items():
            setattr(self, field_name, field_value)