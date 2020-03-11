"""Unit tests for setup and intervention in delayed impact simulator."""
import numpy as np
import statsmodels

from whynot.simulators.delayed_impact.simulator import *


def test_lending_policy():
    config = Config(threshold_g0=600, threshold_g1=650)
    assert lending_policy(config, group=0, score=599) == 0
    assert lending_policy(config, group=0, score=600) == 1
    assert lending_policy(config, group=0, score=660) == 1
    assert lending_policy(config, group=1, score=649) == 0
    assert lending_policy(config, group=1, score=650) == 1
    assert lending_policy(config, group=1, score=660) == 1


def test_update_score():
    config = Config(
        repayment_score_change=50,
        default_score_change=-80,
        min_score=300,
        max_score=800,
    )

    score = 600
    # No change if loan not approved
    assert update_score(config, score, loan_approved=False, repaid=True) == score
    assert update_score(config, score, loan_approved=False, repaid=False) == score
    assert update_score(config, score, loan_approved=True, repaid=True) == score + 50
    assert update_score(config, score, loan_approved=True, repaid=False) == score - 80

    score = 325
    assert update_score(config, score, loan_approved=True, repaid=False) == 300

    score = 780
    assert update_score(config, score, loan_approved=True, repaid=True) == 800


def test_update_profits():
    config = Config(repayment_utility=10, default_utility=-40)

    profits = 10
    assert (
        update_profits(config, profits, loan_approved=True, repaid=False)
        == profits - 40
    )
    assert (
        update_profits(config, profits, loan_approved=True, repaid=True) == profits + 10
    )
    assert update_profits(config, profits, loan_approved=False, repaid=True) == profits
    assert update_profits(config, profits, loan_approved=False, repaid=False) == profits


def check_repayment_rate(repayments, prob):
    """Ensure that repayment has approximately the correct proportion of repaid."""
    total_repaid = np.sum(repayments)
    ci_low, ci_upp = statsmodels.stats.proportion.proportion_confint(
        count=total_repaid, nobs=len(repayments), alpha=0.001, method="beta"
    )
    assert ci_low <= prob <= ci_upp


def test_repayment_fcn():
    rng = np.random.RandomState(1234)
    score = 600

    # Group 0
    for group in range(2):
        repayments = [determine_repayment(rng, group, score) for _ in range(500)]
        check_repayment_rate(repayments, LOAN_REPAY_PROBS[group](score))
