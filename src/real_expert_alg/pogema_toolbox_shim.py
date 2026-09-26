"""
Local shim for pogema_toolbox.algorithm_config.AlgoBase
Provides a pydantic BaseModel with sensible defaults for algorithm configs.
"""
from pydantic import BaseModel, ConfigDict
from typing import Literal


class AlgoBase(BaseModel):
    """Base class for algorithm configurations."""
    model_config = ConfigDict(extra='forbid')
