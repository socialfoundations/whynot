"""Experiments for lending simulator."""

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import delayed_impact
from whynot.simulators.delayed_impact.simulator import INV_CDFS, GROUP_SIZE_RATIO

__all__ = ["get_experiments", "CreditBureauExperiment"]


def get_experiments():
    """Return all experiments for delayed impact simulator."""
    return [CreditBureauExperiment]


################################
# Credit Intervention Experiment
################################
def sample_initial_states(rng):
    """Sample initial states according to FICO data on group membership."""
    group = int(rng.uniform() < GROUP_SIZE_RATIO[1])
    # Compute score via inverse CDF trick
    score = INV_CDFS[group](rng.uniform())
    return delayed_impact.State(group=group, credit_score=score)


def creditscore_threshold(score):
    """Alternate credit bureau scoring policy."""
    return max(score, 600)


def extract_outcomes(run):
    """Outcome is both the score change Delta after 1 step."""
    return run.states[1].credit_score - run.states[0].credit_score


@parameter(
    name="threshold_g0", default=650, description="Lending threshold for group 0",
)
@parameter(
    name="threshold_g1", default=650, description="Lending threshold for group 1",
)
def construct_config(threshold_g0, threshold_g1):
    """Experimental config is parameterized by the lending thresholds."""
    return delayed_impact.Config(
        start_time=0, end_time=1, threshold_g0=threshold_g0, threshold_g1=threshold_g1
    )


#: Effects of interventions on credit scoring on group score changes
CreditBureauExperiment = DynamicsExperiment(
    name="CreditBureauExperiment",
    description="Intervention on the credit scoring mechanism.",
    simulator=delayed_impact,
    # Run for a single time step
    simulator_config=construct_config,
    # Change the threshold on the first step.
    intervention=delayed_impact.Intervention(
        credit_scorer=creditscore_threshold, time=0
    ),
    state_sampler=sample_initial_states,
    # All units are treated
    propensity_scorer=1.0,
    outcome_extractor=extract_outcomes,
    # Only covariate is group membership
    covariate_builder=lambda run: run.initial_state.group,
)
