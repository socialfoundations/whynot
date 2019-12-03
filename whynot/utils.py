"""Utility functions used by all of the simulators."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools

import numpy as np
from tqdm.auto import tqdm

import whynot as wn


def pretty_print(experiment, dataset, results):
    """Print the results of running the causal suite on data from an experiment.

    Parameters
    ----------
        experiment: `whynot.dynamics.DynamicsExperiment` or `whynot.framework.GenericExperiment`
            Experiment object used to generate the causal dataset
        dataset: `whynot.framework.Dataset`
            Dataset object passed to the causal suite.
        results:    dict
            Dictionary of results returned running `whynot.causal_suite` on the dataset

    """
    print("Name: ", experiment.name)
    print("Description: ", experiment.description)
    for method, estimate in results.items():
        print(f"Method: {method:<25} \t\t Estimate: {estimate.ate:2.2e}")
    print(f"{' ':<30} \t\t\t Ground Truth: {dataset.sate:2.2e}")


def parallelize(func, arg_lst, show_progress=False, max_workers=None):
    """Parallel execution of function func across a list of arguments.

    The function func and all of the arguments must be pickable. Func is
    executed on each elements of arg_list as func(*args)

    Parameters
    ----------
        func:
            Function to repeatedly execute, must be pickable.
        arg_lst: iterable
            Iterator of unnamed arguments. Each element arg is passed as func(*arg).
        show_progress: bool
            Whether or not to display a progress bar.
        max_workers: int
            Maximum number of parallel processes to execute simultaneously.

    Returns
    -------
        results: list
            List of outcomes of running func(*arg) on each arg in arg_list.
            Results are in the same order as the input arg_list.

    """

    def display(range_obj):
        if show_progress:
            range_obj = tqdm(range_obj)
        return range_obj

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in arg_lst:
            futures.append(executor.submit(func, *args))
        for future in display(futures):
            data = future.result()
            results.append(data)
    return results


def parallel_run_estimators(causal_datasets):
    """Run causal suite in parallel for repeated trials of a causal experiment.

    Parameters
    ----------
        causal_datasets: dict
            Dictionary mapping an experiment setting to a list of datasets
            representing repeated trials of the experiment.

    Returns
    -------
        all_estimates: dict
            Dictionary mapping estimators name and experimental setting
            to a list of `wn.InferenceResult` objects for each trial,
            e.g. all_estimates['ols'][200][3] is the InferenceResult for ols
            on the 3rd trial of the experiment with setting 200.

    """
    all_estimates = defaultdict(lambda: defaultdict(list))
    for key, trials in causal_datasets.items():
        parallel_args = [
            (dataset.covariates, dataset.treatments, dataset.outcomes)
            for dataset in trials
        ]
        all_trial_estimates = parallelize(
            wn.causal_suite, parallel_args, show_progress=True
        )
        for estimates in all_trial_estimates:
            for method, estimate in estimates.items():
                all_estimates[method][key].append(estimate)
    return all_estimates


def sample_size_experiment(
    experiment, sample_sizes, num_trials, parameters=None, seeds=None, verbose=False
):
    """Repeatedly run an experiment at different sample sizes.

    All of the datasets are generate sequentially, and the estimators are run in
    parallel.

    Parameters
    ----------
        experiment: `whynot.dynamics.DynamicsExperiment` or `whynot.framework.GenericExperiment`
            Instantiated experiment object.
        sample_sizes: list
            List of sample sizes to run the experiment
        num_trials: int
            How many trials to run each experiment for a fixed sample size.
        parameters: dict
            Dictionary of {param_name: param_value} fixing non-varying parameters
            for the experiment.
        (optional) seeds: list
           List of random seeds to use for each trial. If specified, should have
           length num_trials.
        (optional) verbose: bool
            Print status updates.

    Returns
    -------
        estimates: dict
            Dictionary mapping each method to a dictionary of sample_size to
            `whynot.framework.InferenceResults` for each trial at the given
            sample size.

                estimates[method_name] = {
                    sample_size1: [estimates_for_sample_size_1],
                    sample_size2: [estimates_for_sample_size_2],
                    ...}

        sample_ates: dict
            Dictionary mapping sample_size to the sample_ate for each trial.

                sample_ates = {
                    sample_size1: [sample_ates_for_sample_size_1],
                    sample_size2: [sample_ates_for_sample_size_2],
                    ...}

    """
    if seeds is None:
        seeds = [None] * num_trials
    assert len(seeds) == num_trials
    if parameters is None:
        parameters = {}

    if verbose:
        print("Generating causal datasets...")
    datasets = defaultdict(list)
    all_sates = defaultdict(list)
    for (sample_size, seed) in itertools.product(sample_sizes, seeds):
        dataset = experiment.run(num_samples=sample_size, seed=seed, **parameters)
        all_sates[sample_size].append(dataset.sate)
        datasets[sample_size].append(dataset)

    if verbose:
        print("Running estimators...")
    return parallel_run_estimators(datasets), all_sates


def parameter_sweep_experiment(
    experiment,
    sample_size,
    num_trials,
    parameter_name,
    parameter_values,
    fixed_parameters=None,
    seeds=None,
    verbose=False,
):
    """Repeatedly run an experiment for different values of a parameter.

    All of the datasets are generate sequentially, and the estimators are run in
    parallel.

    Parameters
    ----------
        experiment: `whynot.dynamics.DynamicsExperiment` or `whynot.framework.GenericExperiment`
            Instantiated experiment object.
        sample_size: int
            Sample size to use for all experiments.
        num_trials: int
            How many trials to run each experiment for a fixed parameter setting.
        parameter_name: str
            Name of the parameter to vary.
        parameter_values: list
            List of values of the parameter to vary
        fixed_parameters: dict
            Dictionary of {param_name: param_value} fixing non-varying parameters
            for the experiment.
        (optional) seeds: list
           List of random seeds to use for each trial. If specified, should have
           length num_trials.
        (optional) verbose: bool
            Print status updates.

    Returns
    -------
        estimates: dict
            Dictionary mapping each method to a dictionary of parameter_value to
            `whynot.framework.InferenceResults` for each trial at the given
            sample size.

                estimates[method_name] = {
                    parameter_value1: [estimates_for_parameter_value1],
                    parameter_value2: [estimates_for_parameter_value2],
                    ...}

        sample_ates: dict
            Dictionary mapping parameter_value to the sample_ate for each trial.

    """
    if seeds is None:
        seeds = [None] * num_trials
    assert len(seeds) == num_trials
    if fixed_parameters is None:
        fixed_parameters = {}

    if verbose:
        print("Generating causal datasets...")
    datasets = defaultdict(list)
    sample_ates = defaultdict(list)
    for (parameter_value, seed) in itertools.product(parameter_values, seeds):
        parameters = copy.deepcopy(fixed_parameters)
        parameters[parameter_name] = parameter_value
        dataset = experiment.run(num_samples=sample_size, seed=seed, **parameters)
        sample_ates[parameter_value].append(dataset.sate)
        datasets[parameter_value].append(dataset)
    if verbose:
        print("Running estimators...")
    return parallel_run_estimators(datasets), sample_ates


def summarize_errors(estimates, sample_ates, metric):
    """Summarize estimator errors for a parameter or sample size sweep.

    Currently, this function only supports summaries for ATE estimation.
    This function should be used in conjunction with parameter_sweep_experiment
    and sample_size_experiment.

    Parameters
    ----------
        estimates: dict
            Dictionary mapping method_names to a dictionary of experiment
            settings and `whynot.InferenceResults` as returned by
            parameter_sweep_experiment.
        sample_ates: dict
            Dictionary mapping experiment settings to sample ates.
        metric: str
            One of 'relative_error' or 'absolute_error' for reporting results.

    Returns
    -------
        summary: dict
            Dictionary mapping method name to a tuple of (means, stds), where
            means is a list of mean error for each experimental setting, and
            similarly for standard deviation.

    """

    def score(est, sate):
        if metric == "relative_error":
            return np.abs((est - sate) / sate)

        if metric == "absolute_error":
            return np.abs(est - sate)

        raise NotImplementedError

    summary = {}
    for method, results in estimates.items():
        means, stds = [], []
        for setting, inferences in results.items():
            scores = []
            for inference, sample_ate in zip(inferences, sample_ates[setting]):
                scores.append(score(inference.ate, sample_ate))
            means.append(np.mean(scores))
            stds.append(np.std(scores) / np.sqrt(len(scores)))
        summary[method] = (means, stds)
    return summary
