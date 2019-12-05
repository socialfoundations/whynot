"""Experiments on the DICE model."""
import numpy as np

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import dice

__all__ = ["get_experiments", "RCT"]


def get_experiments():
    """Return all of the Lotka-Volterra experiments."""
    return [RCT]


def sample_initial_states():
    """Sample an initial state for the DICE model."""
    # Initial state is currently deterministic. Randomness
    # comes from stochastic dynamics.
    return dice.State()


def dice_outcome_extractor(run):
    """Return the atmospheric temperature of the final state."""
    return run.states[-1].TATM


###########################
# RCT Experiment
###########################
@parameter(
    name="propensity",
    default=0.5,
    values=np.linspace(0.05, 0.5, 10),
    description="Probability of treatment",
)
def rct_propensity(propensity):
    """Return constant propensity for RCT."""
    return propensity


#: An RCT probing the effect of using optimal carbon prices on atmospheric temperature
RCT = DynamicsExperiment(
    name="dice_rct",
    description="A RCT to determine effect of optimal carbon prices on atmospheric temperature.",
    simulator=dice,
    simulator_config=dice.Config(ifopt=0, numPeriods=10),
    intervention=dice.Intervention(ifopt=1),
    state_sampler=sample_initial_states,
    propensity_scorer=rct_propensity,
    outcome_extractor=dice_outcome_extractor,
    covariate_builder=lambda run: run.initial_state.values(),
)
