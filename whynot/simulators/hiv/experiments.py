"""Experiments for HIV simulator."""

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import hiv


__all__ = ["get_experiments", "HIVRCT", "HIVConfounding"]


def get_experiments():
    """Return all experiments for HIV."""
    return [HIVRCT, HIVConfounding]


def sample_initial_states(rng):
    """Sample initial state by randomly perturbing the default state."""
    state = hiv.State()
    state.uninfected_T1 *= rng.uniform(low=0.95, high=1.05)
    state.infected_T1 *= rng.uniform(low=0.95, high=1.05)
    state.uninfected_T2 *= rng.uniform(low=0.95, high=1.05)
    state.infected_T2 *= rng.uniform(low=0.95, high=1.05)
    state.free_virus *= rng.uniform(low=0.95, high=1.05)
    state.immune_response *= rng.uniform(low=0.95, high=1.05)
    return state


##################
# RCT Experiment
##################
# pylint: disable-msg=invalid-name
#: Experiment on effect of increased drug efficacy on infected macrophages (cells/ml)
HIVRCT = DynamicsExperiment(
    name="HIVRCT",
    description="Study effect of increased drug efficacy on infected macrophages (cells/ml).",
    simulator=hiv,
    simulator_config=hiv.Config(epsilon_1=0.1, start_time=0, end_time=150),
    intervention=hiv.Intervention(time=100, epsilon_1=0.5),
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=lambda run: run[149].infected_T2,
    covariate_builder=lambda run: run.initial_state.values(),
)


##########################
# Confounding Experiments
##########################


@parameter(
    name="treatment_bias",
    default=0.9,
    description="Treatment probability bias between more infected and less infected units.",
)
def hiv_confounded_propensity(untreated_run, treatment_bias):
    """Probability of treating each unit.

    We are more likely to treat units with high immune response and free virus.
    """
    if (
        untreated_run.initial_state.immune_response > 10
        and untreated_run.initial_state.free_virus > 1
    ):
        return treatment_bias

    return 1.0 - treatment_bias


# pylint: disable-msg=invalid-name
#: Experiment on effect of increased drug efficacy on infected macrophages with confounding
HIVConfounding = DynamicsExperiment(
    name="HIVConfounding",
    description=(
        "Study effect of increased drug efficacy on infected macrophages (cells/ml). "
        "Units with high immune response and free virus are more likely to be treated."
    ),
    simulator=hiv,
    simulator_config=hiv.Config(epsilon_1=0.1, start_time=0, end_time=150),
    intervention=hiv.Intervention(time=100, epsilon_1=0.5),
    state_sampler=sample_initial_states,
    propensity_scorer=hiv_confounded_propensity,
    outcome_extractor=lambda run: run[149].infected_T2,
    covariate_builder=lambda intervention, run: run.initial_state.values(),
)
