"""Experiments for world2 simulator."""
import numpy as np

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import world2

__all__ = ["get_experiments", "RCT", "BiasedTreatment", "Mediation"]


def get_experiments():
    """Return all experiments for world2."""
    return [RCT, BiasedTreatment, Mediation]


def sample_initial_states(rng):
    """Sample an initial world2 state by perturbing the default."""
    state = world2.State()
    state.population *= rng.uniform(0.75, 2)
    state.natural_resources *= rng.uniform(0.25, 10.0)
    state.capital_investment *= rng.uniform(0.5, 2.0)
    state.pollution *= rng.uniform(0.5, 2)
    state.capital_investment_in_agriculture *= rng.uniform(0.5, 1.5)
    state.initial_natural_resources = state.natural_resources
    return state


###########################
# RCT Experiment
###########################
#: RCT experiment for world2.
RCT = DynamicsExperiment(
    name="world2_rct",
    description="RCT to measure the effect of increasing capital investment in 1970.",
    simulator=world2,
    simulator_config=world2.Config(),
    intervention=world2.Intervention(time=1970, capital_investment_generation=0.06),
    state_sampler=sample_initial_states,
    propensity_scorer=0.25,
    outcome_extractor=lambda run: run[2000].population,
    covariate_builder=lambda run: run.initial_state.values(),
)


###########################
# Treatment Bias Experiment
###########################
@parameter(
    name="treatment_bias",
    default=0.2,
    values=[0.1, 0.2, 0.5, 0.9],
    description="How much to bias treatment assignment",
)
@parameter(
    name="treatment_percentage",
    default=0.2,
    values=[0.05, 0.2, 0.5, 0.9],
    description="What fraction of the units should be treated.",
)
def biased_treatment_propensity(untreated_runs, treatment_bias, treatment_percentage):
    """Compute propensity scores as a function of the bias."""
    # Sort runs by initial quantity of natural resources
    sorted_idxs = sorted(
        range(len(untreated_runs)),
        key=lambda idx: untreated_runs[idx].initial_state.natural_resources,
        reverse=True,
    )

    # More likely to treat units with top initial quantity of natural resources
    top_resource_idxs = sorted_idxs[int(treatment_percentage * len(sorted_idxs)) :]

    propensities = (1.0 - treatment_bias) * np.ones(len(untreated_runs))
    propensities[top_resource_idxs] = treatment_bias

    return propensities


# pylint: disable-msg=invalid-name
#: Observational experiment with confounding for world2.
BiasedTreatment = DynamicsExperiment(
    name="world2_biased_treatment",
    description="Experiment with treatment bias on World 2.",
    simulator=world2,
    simulator_config=world2.Config(),
    intervention=world2.Intervention(time=1970, capital_investment_generation=0.06),
    state_sampler=sample_initial_states,
    propensity_scorer=biased_treatment_propensity,
    outcome_extractor=lambda run: run[2000].population,
    covariate_builder=lambda run: run.initial_state.values(),
)


######################
# Mediation Experiment
######################
@parameter(
    name="birth_rate_intervention",
    default=0.02,
    values=[0.039, 0.02, 0.01],
    description="Decrease in birth rate in the intervention year.",
)
def mediation_intervention(birth_rate_intervention):
    """Return the treatment config for the mediation experiment.

    Provides an example showing how the framework supports parameterized
    configurations.
    """
    base_birth_rate = world2.Config().birth_rate
    assert np.less_equal(birth_rate_intervention, base_birth_rate)
    return world2.Intervention(time=1970, birth_rate=birth_rate_intervention)


def mediation_propensity_scores(intervention, untreated_runs):
    """Probability of treating each unit.

    Units with the largest populations are more likely to be treated.
    """
    populations = [run[intervention.time].population for run in untreated_runs]
    upper_quantile = np.quantile(populations, 0.9)
    propensities = 0.05 * np.ones(len(untreated_runs))
    propensities[populations > upper_quantile] = 0.9
    return propensities


@parameter(
    name="mediation_year",
    default=1980,
    values=[1980, 2000, 2030, 2030],
    description="Year to choose the mediation states.",
)
@parameter(
    name="num_mediators",
    default=3,
    values=[0, 1, 2, 3, 4],
    description="How many mediating states to choose.",
)
def mediation_covariates(run, intervention, mediation_year, num_mediators):
    """Return covariates including mediators."""
    confounders = run[intervention.time].values()
    mediators = run[mediation_year].values()
    # Use only a subset of the mediators to avoid blocking all paths
    return np.concatenate([confounders, mediators[:num_mediators]])


def mediation_outcome_extractor(run, config, intervention):
    return world2.quality_of_life(
        state=run[2030], time=2030, config=config, intervention=intervention
    )


# pylint: disable-msg=invalid-name
#: Observational experiment with mediation for world2.
Mediation = DynamicsExperiment(
    name="world2_mediation",
    description="Create mediation by adding covariates after intervention.",
    simulator=world2,
    simulator_config=world2.Config(),
    intervention=mediation_intervention,
    state_sampler=sample_initial_states,
    propensity_scorer=mediation_propensity_scores,
    outcome_extractor=mediation_outcome_extractor,
    covariate_builder=mediation_covariates,
)

if __name__ == "__main__":
    RCT.run(num_samples=10)
