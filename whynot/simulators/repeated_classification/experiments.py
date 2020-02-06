"""Experiments for repeated classification simulator."""

from whynot.dynamics import DynamicsExperiment
from whynot.simulators import repeated_classification

import whynot.traceable_numpy as np

__all__ = ["get_experiments", "TwoGaussiansExperiment"]


def get_experiments():
    """Return all experiments for repeated classification simulator."""
    return [TwoGaussiansExperiment]


################################
# Two Gaussians Experiment
################################
def sample_initial_states(rng):
    """Sample initial states according to two Gaussians."""
    config = construct_config()
    features, labels, classifier_params, risks = repeated_classification.simulator.compute_state_values(
        populations=config.baseline_growth,
        init_params=np.array([0, 0]),
        config=config,
        rng=rng,
    )
    init_state = repeated_classification.State(
        expected_populations=config.baseline_growth,
        populations=config.baseline_growth,
        features=features,
        labels=labels,
        classifier_params=classifier_params,
        risks=risks,
    )
    return init_state


def extract_outcomes(run):
    """Outcome is both the score change Delta after 1 step."""
    final_expected_populations = run.states[-1].expected_populations
    # return sum(final_expected_populations)
    return min(final_expected_populations) / max(final_expected_populations)


def zero_one_loss(predicted, actual):
    """Zero-one loss."""
    return 1 - (predicted == actual)


def linear_retention(x):
    """Linearly decreasing user retention function, complementing zero-one
    loss."""
    return 1.0 - x


def left_gaussian_dist(population_size, rng):
    mean = [-1, 0]
    cov = 0.1 * np.eye(2)
    features = rng.multivariate_normal(mean, cov, population_size)
    labels = features[:, 1] >= 2/3 * (features[:, 0] + 1)
    return features, labels


def right_gaussian_dist(population_size, rng):
    mean = [1, 0]
    cov = 0.1 * np.eye(2)
    features = rng.multivariate_normal(mean, cov, population_size)
    labels = features[:, 1] >= -2/3 * (features[:, 0] - 1)
    return features, labels


def linear_classifier_2d(features, params, rng):
    return features.T[1] >= params[1] * features.T[0] + params[0]


def construct_config():
    """Experimental config."""
    return repeated_classification.Config(
        K=2,
        baseline_growth=np.array([1000, 1000]),
        group_distributions=[left_gaussian_dist, right_gaussian_dist],
        classifier_func=linear_classifier_2d,
        loss=zero_one_loss,
        user_retention=linear_retention,
    )


TwoGaussiansExperiment = DynamicsExperiment(
    name="TwoGaussiansExperiment",
    description="",  # TODO: description
    simulator=repeated_classification,
    simulator_config=construct_config,
    intervention=repeated_classification.Intervention(time=0),  # TODO: change to DRO
    state_sampler=sample_initial_states,
    # All units are treated
    propensity_scorer=0.5,
    outcome_extractor=extract_outcomes,
    covariate_builder=lambda run: 0,  # TODO: change
)
