"""DICE initialization."""

from whynot.simulators.dice.simulator import Config, Intervention, simulate, State
from whynot.simulators.dice.experiments import *

# Currently, we cannot trace through the dynamics defined in Pyomo.
SUPPORTS_CAUSAL_GRAPHS = False
