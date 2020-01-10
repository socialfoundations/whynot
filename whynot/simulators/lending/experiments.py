"""Experiments for lending simulator."""

from whynot.dynamics import DynamicsExperiment
from whynot.framework import parameter
from whynot.simulators import lending

from whynot.simulators.lending import data_utils

__all__ = ["get_experiments", "LendingRCT"]


import os

dir_path = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(dir_path, "data")
INV_CDFS, LOAN_REPAY_PROBS, _, GROUP_SIZE_RATIO, _, _ = data_utils.get_data_args(
    datapath
)


def get_experiments():
    """Return all experiments for lending simulator."""
    return [LendingRCT]


def sample_initial_states(rng):
    """Sample initial states according to FICO data on group membership."""
    # Load demographic information
    prob_A_equals_1 = GROUP_SIZE_RATIO[-1]
    group = int(rng.uniform() < prob_A_equals_1)
    # Compute score via inverse CDF trick
    score = INV_CDFS[group](rng.uniform())
    return lending.State(group=group, score=score, profits=0)


def repayment_rates(group, score):
    return (
        LOAN_REPAY_PROBS[0](score) ** (1 - group) * LOAN_REPAY_PROBS[1](score) ** group
    )


def creditscore_threshold(score):
    """Alternate credit bureau scoring policy."""
    return max(score, 600)


def outcome_extractor(run):
    """Compute the score change between the first and second states."""
    return run.states[1].score - run.states[0].score


##################
# RCT Experiment
##################
# pylint: disable-msg=invalid-name
LendingRCT = DynamicsExperiment(
    name="CreditScoreIntervention",
    description="Intervention on the credit scoring mechanism.",
    simulator=lending,
    simulator_config=lending.Config(repayment_rate=repayment_rates, end_time=1),
    intervention=lending.Intervention(credit_scorer=creditscore_threshold),
    state_sampler=sample_initial_states,
    propensity_scorer=0.5,
    outcome_extractor=outcome_extractor,
    covariate_builder=lambda run: run.initial_state.group,  # only covariate is group membership
)
