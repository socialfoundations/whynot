"""Momoh and Fugenschuh Zika simulator initialization."""

from whynot.simulators.zika.simulator import (
    Config,
    dynamics,
    Intervention,
    simulate,
    State,
)
from whynot.simulators.zika.experiments import *
from whynot.simulators.zika.environments import *

SUPPORTS_CAUSAL_GRAPHS = True
