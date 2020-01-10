"""Experiments on the Lotka-Volterra model."""
import numpy as np

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import lotka_volterra


__all__ = ["get_experiments", "RCT", "Confounding", "UnobservedConfounding"]


def get_experiments():
    """Return all of the Lotka-Volterra experiments."""
    return [RCT, Confounding, UnobservedConfounding]


def sample_initial_states(rng):
    """Sample an initial state for the LV model."""
    rabbits = rng.randint(10, 100)
    # Ensure the number of rabbits is greater than number of foxes.
    foxes = rng.uniform(0.1, 0.8) * rabbits
    return lotka_volterra.State(rabbits=rabbits, foxes=foxes)


@parameter(
    name="window",
    default=10,
    values=[5, 10, 20],
    description="Number of years to inspect outcome.",
)
@parameter(
    name="outcome_year",
    default=80,
    values=[30, 50, 80],
    description="Year to measure outcome.",
)
def lv_outcome_extractor(run, window, outcome_year):
    """Return the minimum fox population over the window before the outcome year."""
    return np.min(
        [run[year].foxes for year in range(outcome_year - window, outcome_year)]
    )


###########################
# RCT Experiment
###########################
@parameter(
    name="propensity",
    default=0.1,
    values=np.linspace(0.05, 0.5, 10),
    description="Probability of treatment",
)
def rct_propensity(propensity):
    """Return constant propensity for RCT."""
    return propensity


#: RCT experiment to determine effect of reducing number of caught rabbits that create a fox.
RCT = DynamicsExperiment(
    name="lotka_volterra_rct",
    description="A RCT to determine effect of reducing rabbits needed to sustain a fox.",
    simulator=lotka_volterra,
    simulator_config=lotka_volterra.Config(fox_growth=0.75),
    intervention=lotka_volterra.Intervention(time=30, fox_growth=0.4),
    state_sampler=sample_initial_states,
    propensity_scorer=rct_propensity,
    outcome_extractor=lv_outcome_extractor,
    covariate_builder=lambda run: run.initial_state.values(),
)


##########################
# Confounding Experiments
##########################
@parameter(
    name="propensity",
    default=0.9,
    values=np.linspace(0.5, 0.99, 10),
    description="Probability of treatment for group with low fox population.",
)
def confounded_propensity_scores(untreated_run, propensity=0.9):
    """Return confounded treatment assignment probability.

    Treatment increases fox population growth. Therefore, we're assume
    treatment is more likely for runs with low initial fox population.
    """
    if untreated_run.initial_state.foxes < 20:
        return propensity
    return 1.0 - propensity


# pylint: disable-msg=invalid-name
#: Observational experiment with confounding due to initial fox population.
Confounding = DynamicsExperiment(
    name="lotka_volterra_confounding",
    description=(
        "Determine effect of reducing rabbits needed to sustain a fox."
        "Treament confounded by initial fox population."
    ),
    simulator=lotka_volterra,
    simulator_config=lotka_volterra.Config(fox_growth=0.75),
    intervention=lotka_volterra.Intervention(time=30, fox_growth=0.4),
    state_sampler=sample_initial_states,
    propensity_scorer=confounded_propensity_scores,
    outcome_extractor=lv_outcome_extractor,
    covariate_builder=lambda run: run.initial_state.values(),
)


# pylint: disable-msg=invalid-name
#: Observational experiment with unobserved confounding due to initial fox population.
UnobservedConfounding = DynamicsExperiment(
    name="lotka_volterra_unobserved_confounding",
    description=(
        "Determine effect of reducing rabbits needed to sustain a fox."
        "Treament confounded by initial fox population, which is not observed."
    ),
    simulator=lotka_volterra,
    simulator_config=lotka_volterra.Config(fox_growth=0.75),
    intervention=lotka_volterra.Intervention(time=30, fox_growth=0.4),
    state_sampler=sample_initial_states,
    propensity_scorer=confounded_propensity_scores,
    outcome_extractor=lv_outcome_extractor,
    covariate_builder=lambda run: np.array([run.initial_state.rabbits]),
)
