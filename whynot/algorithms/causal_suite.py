"""Algorithms for causal inference."""
import whynot

try:
    import whynot_estimators

    ESTIMATORS_INSTALLED = True
except ImportError:
    ESTIMATORS_INSTALLED = False


def causal_suite(covariates, treatment, outcome, verbose=False):
    """Run a collection of causal inference algorithms on the observational dataset.

    By default, the suite only runs estimators implemented in Python
        - Ordinary least squares (ols)
        - Propensity score matching
        - Propensity weighted least squares

    Depending on the estimators installed in whynot_estimators, the suite additionally runs:
        - An IP weighting estimator (ip_weighting)
        - Matching estimators with mahalanobis distance metrics
        - Causal Forest (causal_forest)
        - TMLE          (tmle)

    Parameters
    ----------
        covariates :    `np.ndarray`
            Array of shape `num_samples` x `num_features`.
        treatment :     np.ndarray
            Boolean array of shape [num_samples] indicating treatment status for each sample.
        outcome :       np.ndarray
            Array of shape [num_sample] containing the observed outcome for each sample.
        verbose : bool
            If True, print incremental messages as estimators are executed.

    Returns
    -------
        results: dict
            Dictionary with keys denoting the name of each method and
            values the corresponding :class:`InferenceResult`:

                results['causal_forest'] -> inference_results

    """
    # Map from algorithm name to estimation function.
    methods = {
        "ols": whynot.ols,
        "propensity_score_matching": whynot.propensity_score_matching,
        "propensity_weighted_ols": whynot.propensity_weighted_ols,
    }
    if ESTIMATORS_INSTALLED:
        additional_methods = {
            "ip_weighting": whynot_estimators.ip_weighting,
            "mahalanobis_matching": whynot_estimators.matching,
            "causal_forest": whynot_estimators.causal_forest,
            "tmle": whynot_estimators.tmle,
        }
        installed = {}
        for name, estimator in additional_methods.items():
            if estimator.is_installed():
                installed[name] = estimator
        methods.update(installed)
    elif verbose:
        print("Only the base estimators are installed.")
        print(
            "To install additional additional estimators, `pip install whynot_estimators.`"
        )

    results = {}
    for name, estimator in methods.items():
        if verbose:
            print(f"Running estimator: {name}")
        try:
            inference_result = estimator.estimate_treatment_effect(
                covariates, treatment, outcome
            )
        # pylint:disable-msg=broad-except
        except Exception:
            print(f"Estimator {name} failed to run... skipping!")
            continue
        results[name] = inference_result

    return results
