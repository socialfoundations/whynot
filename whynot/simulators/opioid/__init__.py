"""Opioid initialization."""

from whynot.simulators.opioid.simulator import (
    Config,
    Intervention,
    dynamics,
    simulate,
    State,
)
from whynot.simulators.opioid.experiments import *
from whynot.simulators.opioid.environments import *

SUPPORTS_CAUSAL_GRAPHS = True
