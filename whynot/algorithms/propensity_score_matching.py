"""Average treatment effect estimation with propensity score matching."""
from time import perf_counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

from whynot.framework import InferenceResult


def estimate_treatment_effect(covariates, treatment, outcome):
    """Estimate treatment effects using propensity score matching.

    Parameters
    ----------
        covariates: `np.ndarray`
            Array of shape [num_samples, num_features] of features
        treatment:  `np.ndarray`
            Binary array of shape [num_samples]  indicating treatment status for each
            sample.
        outcome:  `np.ndarray`
            Array of shape [num_samples] containing the observed outcome for each sample.

    Returns
    -------
        result: `whynot.framework.InferenceResult`
            InferenceResult object for this procedure

    """
    start_time = perf_counter()

    # Compute propensity scores with logistic regression model.
    features = sm.add_constant(covariates, has_constant="add")
    logit = sm.Logit(treatment, features)
    model = logit.fit(disp=0)
    propensity_scores = model.predict(features)

    matched_treatment, matched_outcome, matched_weights = get_matched_dataset(
        treatment, propensity_scores, outcome
    )

    ate = compute_ate(matched_outcome, matched_treatment, matched_weights)

    # Bootstrap confidence intervals
    samples = []
    num_units = len(matched_treatment)
    for _ in range(1000):
        sample_idxs = np.random.choice(num_units, size=num_units, replace=True)
        samples.append(
            compute_ate(
                matched_outcome[sample_idxs],
                matched_treatment[sample_idxs],
                matched_weights[sample_idxs],
            )
        )
    conf_int = (np.quantile(samples, 0.025), np.quantile(samples, 0.975))
    stop_time = perf_counter()

    return InferenceResult(
        ate=ate,
        stderr=None,
        ci=conf_int,
        individual_effects=None,
        elapsed_time=stop_time - start_time,
    )


def get_matched_dataset(treatment, scores, outcome, num_neighbors=5):
    """Construct matched dataset for both control and treatment based on the provided score.

    Parameters
    ----------
        treatment:  `np.ndarray`
            Binary array of shape [num_samples]  indicating treatment status for each
            sample.
        scores:  `np.ndarray`
            Array of shape [num_samples] containing the score to match on for
            each sample, e.g. propensity scores.
        outcome:  `np.ndarray`
            Array of shape [num_samples] containing the observed outcome for each sample.
        num_neighbors: int
            (Optional) Number of units to match with.

    Returns
    -------
        matched_treatment:  `np.ndarray`
            Binary array indicating treatment status for each sample in the
            matched dataset.
        matched_outcome:  `np.ndarray`
            Array containing the observed outcome for each sample in the matched
            dataset.
        weights:  `np.ndarray`
            Array containing the weight for each sample in the matched dataset.
            The original units all have weight 1, and the matched units have
            uniform weight 1 / num_neighbors.

    """
    # Split into treatment and control groups
    num_samples = len(outcome)
    treated_scores = scores[treatment == 1.0].reshape(-1, 1)
    treated_outcomes = outcome[treatment == 1.0]
    control_scores = scores[treatment == 0.0].reshape(-1, 1)
    control_outcomes = outcome[treatment == 0.0]

    def match(base_scores, target_scores):
        """Match units in base to target and return the matched indices."""
        search = NearestNeighbors(metric="euclidean", n_neighbors=num_neighbors)
        search.fit(base_scores)
        matched = search.kneighbors(target_scores, return_distance=False)
        return matched.ravel()

    # Match control units to treatment and treatment units to controls.
    matched_control = match(control_scores, treated_scores)
    matched_treated = match(treated_scores, control_scores)

    # Construct an augmented dataset with the matched units
    matched_outcome = np.concatenate(
        [outcome, treated_outcomes[matched_treated], control_outcomes[matched_control]]
    )
    matched_treatment = np.concatenate(
        [treatment, np.ones_like(matched_treated), np.zeros_like(matched_control)]
    )
    weights = np.ones_like(matched_outcome)
    weights[num_samples:] = 1.0 / num_neighbors

    return matched_treatment, matched_outcome, weights


def compute_ate(outcome, treatment, weights):
    """Compute the (weighted) ATE."""
    treated_weights = weights[treatment == 1.0] / np.sum(weights[treatment == 1.0])
    control_weights = weights[treatment == 0.0] / np.sum(weights[treatment == 0.0])

    treated_outcome = np.sum(outcome[treatment == 1.0] * treated_weights)
    control_outcome = np.sum(outcome[treatment == 0.0] * control_weights)

    return treated_outcome - control_outcome
