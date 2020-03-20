"""Experiments for world3 simulator."""
import numpy as np

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import world3

__all__ = [
    "get_experiments",
    "PollutionRCT",
    "PollutionConfounding",
    "PollutionUnobservedConfounding",
    "PollutionMediation",
]


def get_experiments():
    """Return all experiments for world3."""
    return [
        PollutionRCT,
        PollutionConfounding,
        PollutionUnobservedConfounding,
        PollutionMediation,
    ]


def sample_initial_states(rng):
    """Sample initial state by randomly perturbing the default state."""

    def random_scale(scale, base=10):
        """Make 10**(-scale) to 10**scale smaller/bigger uniformly."""
        return rng.choice(np.logspace(-scale, scale, base=base, num=50))

    state = world3.State()
    state.population_0_to_14 *= random_scale(0.33)
    state.population_15_to_44 *= random_scale(0.33)
    state.population_45_to_64 *= random_scale(0.33)
    state.population_65_and_over *= random_scale(0.33)
    state.industrial_capital *= random_scale(0.33)
    state.service_capital *= random_scale(0.33)
    state.arable_land *= random_scale(0.33)
    state.potentially_arable_land *= random_scale(0.33)
    state.urban_industrial_land *= random_scale(0.33)
    state.nonrenewable_resources *= random_scale(0.33)
    state.persistent_pollution *= random_scale(0.33)
    state.land_fertility *= random_scale(0.26)

    return state


##################
# RCT Experiment
##################
# pylint: disable-msg=invalid-name
#: An RCT experiment to study the effect of decreases in pollution generation on population.
PollutionRCT = DynamicsExperiment(
    name="PollutionRCT",
    description="Study effect of intervening in 1975 to decrease pollution generation.",
    simulator=world3,
    simulator_config=world3.Config(persistent_pollution_generation_factor=1.0),
    intervention=world3.Intervention(
        time=1975, persistent_pollution_generation_factor=0.85
    ),
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=lambda run: run[2050].total_population,
    covariate_builder=lambda run: run.initial_state.values(),
)


##########################
# Confounding Experiments
##########################
@parameter(
    name="treatment_bias",
    default=0.8,
    values=np.linspace(0.5, 1.0, 5),
    description="Treatment probability bias between low and high pollution runs.",
)
def pollution_confounded_propensity(intervention, untreated_runs, treatment_bias):
    """Probability of treating each unit.

    To generate confounding, we are more likely to treat worlds with high pollution.
    """

    def persistent_pollution(run):
        return run[intervention.time].persistent_pollution

    pollution = [persistent_pollution(run) for run in untreated_runs]
    upper_quantile = np.quantile(pollution, 0.9)

    def treatment_prob(idx):
        if pollution[idx] > upper_quantile:
            return treatment_bias
        return 1.0 - treatment_bias

    return np.array([treatment_prob(idx) for idx in range(len(untreated_runs))])


# pylint: disable-msg=invalid-name
#: An observational experiment with confounding. Polluted states are more likely to be treated.
PollutionConfounding = DynamicsExperiment(
    name="PollutionConfounding",
    description=(
        "Study effect of intervening to decrease pollution on total population. Confounding "
        "arises becauses states with high pollution are more likely "
        "to receive treatment."
    ),
    simulator=world3,
    simulator_config=world3.Config(persistent_pollution_generation_factor=1.0),
    intervention=world3.Intervention(
        time=1975, persistent_pollution_generation_factor=0.85
    ),
    state_sampler=sample_initial_states,
    propensity_scorer=pollution_confounded_propensity,
    outcome_extractor=lambda run: run[2050].total_population,
    covariate_builder=lambda intervention, run: run[intervention.time].values(),
)


# pylint: disable-msg=invalid-name
#: An observational experiment with unobserved confounding.
PollutionUnobservedConfounding = DynamicsExperiment(
    name="PollutionUnobservedConfounding",
    description=(
        "Study effect of intervening to decrease pollution.  Confounding "
        "arises becauses states with high pollution are more likely "
        "to receive treatment. However, only variables that directly affect "
        "treatment assignment are observed."
    ),
    simulator=world3,
    simulator_config=world3.Config(persistent_pollution_generation_factor=1.0),
    intervention=world3.Intervention(
        time=1975, persistent_pollution_generation_factor=0.85
    ),
    state_sampler=sample_initial_states,
    propensity_scorer=pollution_confounded_propensity,
    outcome_extractor=lambda run: run[2050].total_population,
    covariate_builder=lambda intervention, run: np.array(
        [run[intervention.time].persistent_pollution]
    ),
)


########################
# Mediation Experiments
########################
@parameter(
    name="mediation_year",
    default=2015,
    values=[1980, 2000, 2020, 2045],
    description="What year to get mediating variables",
)
@parameter(
    name="num_mediators",
    default=11,
    values=[0, 3, 5, 7, 10],
    description="number of mediators to include.",
)
def mediation_covariates(intervention, run, mediation_year, num_mediators):
    """Build the causal dataset."""
    # Just use all of the states at the moment of treatment assignment
    confounders = run[intervention.time].values()
    mediators = run[mediation_year].values()[:num_mediators]
    return np.concatenate([confounders, mediators])


# pylint: disable-msg=invalid-name
#: An observational experiment with mediation from states after intervention.
PollutionMediation = DynamicsExperiment(
    name="PollutionMediation",
    description=(
        "Study effect of intervening to decrease pollution.  Confounding "
        "arises becauses states with high pollution are more likely "
        "to receive treatment. Mediation arises since future states "
        "are also observed."
    ),
    simulator=world3,
    simulator_config=world3.Config(persistent_pollution_generation_factor=1.0),
    intervention=world3.Intervention(
        time=1975, persistent_pollution_generation_factor=0.85
    ),
    state_sampler=sample_initial_states,
    propensity_scorer=pollution_confounded_propensity,
    outcome_extractor=lambda run: run[2050].total_population,
    covariate_builder=mediation_covariates,
)
