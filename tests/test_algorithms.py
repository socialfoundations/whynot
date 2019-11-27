"""Tests for causal inference methods."""
import numpy as np
import pytest

import whynot as wn
from whynot.algorithms import ols, propensity_score_matching, propensity_weighted_ols


def generate_dataset(num_samples, num_features, true_ate=1.0, seed=1234):
    """Generate an observational dataset."""
    np.random.seed(seed)

    features = [0.25 * np.random.randn(num_samples, 1) for _ in range(num_features)]
    covariates = np.concatenate(features, axis=1)

    # Logisitic treatment probability
    arg = np.sum(covariates, axis=1) + np.random.randn(num_samples)
    prob = np.exp(arg) / (1.0 + np.exp(arg))
    treatment = np.random.binomial(1, prob)

    # Outcome is confounded by treatment
    # ATE is true_ate since the covariates are zero-mean.
    outcome = (np.sum(covariates, axis=1) + true_ate) * treatment
    outcome += np.random.randn(num_samples)

    return covariates, treatment.ravel(), outcome.ravel()


@pytest.mark.parametrize(
    "estimator",
    [ols, propensity_score_matching, propensity_weighted_ols],
    ids=["ols", "propensity_score_matching", "propensity_weighted_ols"],
)
def test_estimator(estimator):
    """Verify the estimator correctly computes treatment effects."""
    num_samples = 5000
    num_features = 3
    true_ate = 8

    covariates, treatment, outcome = generate_dataset(
        num_samples=num_samples, num_features=num_features, true_ate=true_ate
    )

    result = propensity_score_matching.estimate_treatment_effect(
        covariates, treatment, outcome
    )
    assert result.ci[0] <= result.ate <= result.ci[1]
    assert result.ci[0] <= true_ate <= result.ci[1]
    assert true_ate - 0.1 <= result.ate <= true_ate + 0.1


def test_causal_suite():
    """Integration test for the causal suite."""
    num_samples = 200
    num_features = 15

    covariates, treatment, outcomes = generate_dataset(
        num_samples=num_samples, num_features=num_features
    )

    result = wn.causal_suite(covariates, treatment, outcomes)
    assert "ols" in result
    assert "propensity_score_matching" in result
    assert "propensity_weighted_ols" in result
