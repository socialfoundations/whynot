"""Experiments for lending simulator."""

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import lending

from whynot.simulators.lending.simulator import INV_CDFS, GROUP_SIZE_RATIO

__all__ = ["get_experiments", "LendingRCT"]


def get_experiments():
    """Return all experiments for lending simulator."""
    return [LendingRCT]


################################
# Credit Intervention Experiment
################################


def sample_initial_states(rng):
    """Sample initial states according to FICO data on group membership."""
    group = int(rng.uniform() < GROUP_SIZE_RATIO[1])
    # Compute score via inverse CDF trick
    score = INV_CDFS[group](rng.uniform())
    return lending.State(group=group, score=score, profits=0)


def creditscore_threshold(score):
    """Alternate credit bureau scoring policy."""
    return max(score, 600)


def compute_score_changes(run):
    """Compute the score change between the first and second states."""
    return run.states[1].score - run.states[0].score


LendingRCT = DynamicsExperiment(
    name="CreditScoreIntervention",
    description="Intervention on the credit scoring mechanism.",
    simulator=lending,
    # Run for a single time step
    simulator_config=lending.Config(start_time=0, end_time=1),
    intervention=lending.Intervention(credit_scorer=creditscore_threshold),
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=compute_score_changes,
    # Only covariate is group membership
    covariate_builder=lambda run: run.initial_state.group,
)
