"""Experiments for Zika simulator."""

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import zika


__all__ = ["get_experiments", "ZikaRCT"]


def get_experiments():
    """Return all experiments for zika."""
    return [ZikaRCT]


def sample_initial_states(rng):
    """Sample initial state by randomly perturbing the default state."""
    state = zika.State().values()
    state *= rng.uniform(low=0.95, high=1.05, size=state.shape)
    return zika.State(*state)


def mixed_treatment_intervention():
    """Apply mixed strategy on all controls starting at time 0."""
    return zika.Intervention(
        time=0,
        treated_bednet_use=0.5,
        condom_use=0.5,
        treatment_of_infected=0.5,
        indoor_spray_use=0.5,
    )


##################
# RCT Experiment
##################
# pylint: disable-msg=invalid-name
#: Experiment on effect of mixed treatment policy on infections in 20 days
ZikaRCT = DynamicsExperiment(
    name="ZikaRCT",
    description="Study effect of mixed treatment policy on infections in 20 days.",
    simulator=zika,
    simulator_config=zika.Config(start_time=0, end_time=20),
    intervention=mixed_treatment_intervention(),
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=lambda run: run[19].symptomatic_humans,
    covariate_builder=lambda run: run.initial_state.values(),
)
