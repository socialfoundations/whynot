"""Average treatment effect estimation from propensity weighted least squares."""
from time import perf_counter

import numpy as np
import statsmodels.api as sm

from whynot.framework import InferenceResult


def estimate_treatment_effect(covariates, treatment, outcome):
    """Estimate treatment effects using propensity weighted least-squares.

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
    features = sm.add_constant(covariates, prepend=True, has_constant="add")
    logit = sm.Logit(treatment, features)
    model = logit.fit(disp=0)
    propensities = model.predict(features)

    # IP-weights
    treated = treatment == 1.0
    untreated = treatment == 0.0
    weights = treated / propensities + untreated / (1.0 - propensities)

    treatment = treatment.reshape(-1, 1)
    features = np.concatenate([treatment, covariates], axis=1)
    features = sm.add_constant(features, prepend=True, has_constant="add")

    model = sm.WLS(outcome, features, weights=weights)
    results = model.fit()
    stop_time = perf_counter()

    # Treatment is the second variable (after the constant offset)
    ate = results.params[1]
    stderr = results.bse[1]
    conf_int = tuple(results.conf_int()[1])

    return InferenceResult(
        ate=ate,
        stderr=stderr,
        ci=conf_int,
        individual_effects=None,
        elapsed_time=stop_time - start_time,
    )
