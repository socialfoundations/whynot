"""Basic experiments for opioid simulator."""
import numpy as np

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import opioid

__all__ = [
    "get_experiments",
    "RCT",
    "Confounding",
    "UnobservedConfounding",
    "Mediation",
]


def get_experiments():
    """Return all of the experiments for opioid simulator."""
    return [RCT, Confounding, UnobservedConfounding, Mediation]


def sample_initial_states(rng):
    """Sample initial states based on estimated uncertainty in Chen paper."""

    def sample(mean, std_dev):
        return mean + std_dev * rng.randn()

    nonmedical_users = sample(10029859, 100 * 329807)
    oud_users = sample(1369218, 100 * 116347)
    illicit_users = sample(328731, 100 * 60446)

    return opioid.State(
        nonmedical_users=nonmedical_users,
        oud_users=oud_users,
        illicit_users=illicit_users,
    )


@parameter(
    name="nonmedical_incidence_delta",
    default=-0.113,
    values=[-0.073, -0.113],
    description="Percent decrease in new nonmedical users of prescription opioids.",
)
def opioid_intervention(nonmedical_incidence_delta):
    """Return treatment config. Changes incidence of nonmedical abuse."""
    return opioid.Intervention(
        time=2015, nonmedical_incidence=nonmedical_incidence_delta
    )


def overdose_deaths(config, intervention, run):
    """Compute overdose deaths from the decade after intervention."""
    return opioid.simulator.compute_overdose_deaths(
        run, intervention.time, intervention.time + 10, config, intervention
    )


########################
# Randomized Experiment
#######################
#: RCT on the opioid simulator to see how reducing nonmedical opioid use reduces overdose deaths.
RCT = DynamicsExperiment(
    name="opioid_rct",
    description=(
        "Randomized experiment reducing nonmedical incidence of " "opioid use in 2015."
    ),
    simulator=opioid,
    simulator_config=opioid.Config(),
    intervention=opioid_intervention,
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=overdose_deaths,
    covariate_builder=lambda run: run.initial_state.values(),
)


########################
# Confounded experiments
########################
@parameter(
    name="propensity",
    default=0.9,
    values=[0.5, 0.6, 0.7, 0.9, 0.99],
    description="Probability of treatment assignment in positive group.",
)
def confounded_propensity_scores(config, intervention, untreated_runs, propensity):
    """Rollouts with top 10% of illicit deaths in 2015 more likely to receive treatment."""
    num_samples = len(untreated_runs)

    # More likely to assign treatment when illicit overdose
    # deaths are high in 2015
    illicit_deaths = [
        opioid.simulator.compute_illicit_deaths(run, 2015, config, intervention)
        for run in untreated_runs
    ]

    # Get top 10% of cities based on illicit overdose deaths
    sorted_idxs = sorted(range(num_samples), key=lambda idx: illicit_deaths[idx])
    top_death_idxs = sorted_idxs[int(num_samples * 0.9) :]

    propensity_scores = (1.0 - propensity) * np.ones(num_samples)
    propensity_scores[top_death_idxs] = propensity

    return propensity_scores


# pylint: disable-msg=invalid-name
#: Observational version of RCT with confounding on observed nonmedical use overdose deaths.
Confounding = DynamicsExperiment(
    name="confounded_opioid_experiment",
    description="Opioid confounding by treating runs with high numbers of nonmedical use deaths.",
    simulator=opioid,
    simulator_config=opioid.Config(),
    intervention=opioid_intervention,
    state_sampler=sample_initial_states,
    propensity_scorer=confounded_propensity_scores,
    outcome_extractor=overdose_deaths,
    covariate_builder=lambda intervention, run: run[intervention.time].values(),
)


# pylint: disable-msg=invalid-name
#: Observational version of RCT with unobserved confounding on nonmedical use overdose deaths.
UnobservedConfounding = DynamicsExperiment(
    name="unobserved_confounding_opioid_experiment",
    description=(
        "Opioid unobserved confounding by treating runs with high "
        "numbers of illicit deaths and omitting this variables."
    ),
    simulator=opioid,
    simulator_config=opioid.Config(),
    intervention=opioid_intervention,
    state_sampler=sample_initial_states,
    propensity_scorer=confounded_propensity_scores,
    outcome_extractor=overdose_deaths,
    covariate_builder=lambda intervention, run: np.array(
        [run[intervention.time].oud_users, run[intervention.time].nonmedical_users]
    ),
)


########################
# Mediation experiments
########################
@parameter(
    name="mediation_year",
    default=2020,
    values=[2017, 2019, 2020, 2021, 2024],
    description="Year of mediating variables--proxy for mediation strength." "",
)
def mediation_covariates(run, mediation_year):
    """Construct covariates for mediation experiment.

    Mediators are future state variables. Block backdoor paths with
    state variables in intervention year.
    """
    confounders = run[2015].values()
    mediation_state = run[mediation_year]
    mediators = np.array([mediation_state.nonmedical_users, mediation_state.oud_users])
    return np.concatenate([confounders, mediators])


# pylint: disable-msg=invalid-name
#: Observational version of RCT with mediation induced by observing future states.
Mediation = DynamicsExperiment(
    name="opioid_mediation",
    description="Mediation on opioid experiment by including future state.",
    simulator=opioid,
    simulator_config=opioid.Config(),
    intervention=opioid_intervention,
    state_sampler=sample_initial_states,
    propensity_scorer=confounded_propensity_scores,
    outcome_extractor=overdose_deaths,
    covariate_builder=mediation_covariates,
)

if __name__ == "__main__":
    RCT.run(num_samples=10)
