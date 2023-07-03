"""Ensure gymnasium spaces are accessible if you import whynot.gym as gym."""
from gymnasium.spaces.space import Space
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.multi_discrete import MultiDiscrete
from gymnasium.spaces.multi_binary import MultiBinary
from gymnasium.spaces.tuple import Tuple
from gymnasium.spaces.dict import Dict

from gymnasium.spaces.utils import flatdim
from gymnasium.spaces.utils import flatten
from gymnasium.spaces.utils import unflatten

__all__ = [
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "Tuple",
    "Dict",
    "flatdim",
    "flatten",
    "unflatten",
]
