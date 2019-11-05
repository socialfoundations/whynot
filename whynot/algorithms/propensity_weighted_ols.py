"""Average treatment effect estimation from propensity weighted least squares."""
from time import perf_counter

import pandas as pd

from causality.estimation.parametric import InverseProbabilityWeightedLS
from whynot.framework import InferenceResult


def estimate_treatment_effect(covariates, treatment, outcome):
    """Estimate treatment effects using propensity weighted least-squares.

    Wrapper around causality.estimation.parametric.InverseProbabilityWeightedLS

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
    data = pd.DataFrame({'treatment': treatment, 'outcome': outcome})

    covdict = {}
    for j in range(covariates.shape[1]):
        name = str(j)
        data[name] = covariates[:, j]
        covdict[name] = 'c'

    ipweighter = InverseProbabilityWeightedLS()
    ate = ipweighter.estimate_ATE(data, 'treatment', 'outcome', covdict)
    stop_time = perf_counter()

    return InferenceResult(ate=ate[1], stderr=None, ci=(ate[0], ate[2]),
                           individual_effects=None,
                           elapsed_time=stop_time - start_time)
