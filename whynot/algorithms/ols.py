"""Ordinary least-squares based-estimators for causal inference."""
from time import perf_counter

import numpy as np
import statsmodels.api as sm

from whynot.framework import InferenceResult


def estimate_treatment_effect(covariates, treatment, outcome):
    """Run ordinary least squares to estimate causal effect of treatment variable on the outcome Y.

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
    features = np.copy(covariates)
    treatment = treatment.reshape(-1, 1)
    features = np.concatenate([treatment, features], axis=1)
    features = sm.add_constant(features, prepend=True, has_constant="add")

    # Only time model fitting, not preprocessing
    start_time = perf_counter()
    model = sm.OLS(outcome, features)
    results = model.fit()
    stop_time = perf_counter()

    # Treatment is the second variable (first is the constant offset)
    ate = results.params[1]
    stderr = results.bse[1]
    conf_int = (ate - 1.96 * stderr, ate + 1.96 * stderr)
    return InferenceResult(
        ate=ate,
        stderr=stderr,
        ci=conf_int,
        individual_effects=None,
        elapsed_time=stop_time - start_time,
    )
