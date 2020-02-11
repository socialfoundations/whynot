"""Experiments for repeated classification simulator."""

from scipy.optimize import minimize
from sklearn.svm import SVC

from whynot.dynamics import DynamicsExperiment
from whynot.simulators import repeated_classification
import whynot.traceable_numpy as np


__all__ = ["get_experiments", "TwoGaussiansExperiment", "MedianEstimationExperiment"]


def get_experiments():
    """Return all experiments for repeated classification simulator."""
    return [TwoGaussiansExperiment, MedianEstimationExperiment]


################################
# Two Gaussians Experiment
################################
def sample_initial_states_gaussians(rng):
    """Sample initial states according to two Gaussians."""
    config = construct_config_gaussians()
    features, labels, classifier_params, risks = repeated_classification.simulator.compute_state_values(
        populations=config.baseline_growth,
        init_params=np.array([0, 1, 0]),
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
    """Outcome is the maximal single-group risk at the final step."""
    final_risks = run.states[-1].risks
    return max(final_risks)


def zero_one_loss(predicted, actual):
    """Zero-one loss."""
    return 1 - (predicted == actual)


def linear_retention(x):
    """Linearly decreasing user retention function, complementing zero-one
    loss."""
    return 1.0 - x


def left_gaussian_dist(population_size, rng):
    mean = [-1, 0]
    cov = 0.15 * np.eye(2)
    features = rng.multivariate_normal(mean, cov, population_size)
    labels = features[:, 1] >= 2/3 * (features[:, 0] + 1)
    return features, labels.astype(int)


def right_gaussian_dist(population_size, rng):
    mean = [1, 0]
    cov = 0.15 * np.eye(2)
    features = rng.multivariate_normal(mean, cov, population_size)
    labels = features[:, 1] >= -2/3 * (features[:, 0] - 1)
    return features, labels.astype(int)


def linear_classifier_2d(features, params, rng):
    predictions = np.dot(features, params[:-1]) + params[-1] >= 0
    return predictions.astype(int)


def empirical_risk_minimization(classifier_func, loss, features, labels,
                                init_params, rng, **kwargs):
    # TODO: docstring
    def objective(theta):
        return np.mean(loss(classifier_func(features, theta, rng), labels))

    if 'method' not in kwargs:
        kwargs['method'] = 'Powell'

    if len(np.shape(init_params)) == 1:
        init_params = [init_params]
    best_value = float('inf')
    best_param = None
    for param in init_params:
        result = minimize(
            objective,
            x0=param,
            **kwargs
        )
        if result.fun < best_value:
            best_value, best_param = result.fun, result.x

    if not np.shape(best_param):
        best_param = np.array([best_param])

    return best_param


def train_svm(classifier_func, loss, features, labels, init_params, rng):
    model = SVC(C=0.1, kernel='linear').fit(features, labels)
    return np.concatenate([model.coef_.flatten(), model.intercept_.flatten()])


def construct_config_gaussians():
    """Experimental config."""
    return repeated_classification.Config(
        K=2,
        baseline_growth=np.array([1000, 800]),
        group_distributions=[left_gaussian_dist, right_gaussian_dist],
        classifier_func=linear_classifier_2d,
        loss=zero_one_loss,
        user_retention=linear_retention,
        train_classifier=empirical_risk_minimization,
        # train_classifier=train_svm,
    )


TwoGaussiansExperiment = DynamicsExperiment(
    name="TwoGaussiansExperiment",
    description="",  # TODO: description
    simulator=repeated_classification,
    simulator_config=construct_config_gaussians,
    intervention=repeated_classification.Intervention(time=0),  # TODO: change to DRO
    state_sampler=sample_initial_states_gaussians,
    propensity_scorer=0.5,
    outcome_extractor=extract_outcomes,
    covariate_builder=lambda run: 0,  # TODO: change
)


################################
# Median Estimation Experiment
################################
def sample_initial_states_median(rng):
    """Sample initial states according to median estimation setup."""
    config = construct_config_median()
    features, labels, classifier_params, risks = repeated_classification.simulator.compute_state_values(
        populations=config.baseline_growth,
        init_params=np.array([0]),
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


def l1_loss(predicted, actual):
    """l1-norm loss."""
    return np.abs(predicted - actual)


def exponential_retention(x):
    """Exponentially decreasing user retention function."""
    return np.exp(-x)


def constant(features, params, rng):
    return np.tile(params[0], len(features))


def left_gaussian_dist_1d(population_size, rng):
    features = rng.normal(-1, 0.2, (population_size, 1))
    return features, features.flatten()


def right_gaussian_dist_1d(population_size, rng):
    features = rng.normal(1, 0.2, (population_size, 1))
    return features, features.flatten()


def construct_config_median():
    """Experimental config."""
    return repeated_classification.Config(
        K=2,
        baseline_growth=np.array([1000, 500]),
        group_distributions=[left_gaussian_dist_1d, right_gaussian_dist_1d],
        classifier_func=constant,
        loss=l1_loss,
        user_retention=exponential_retention,
        train_classifier=empirical_risk_minimization,
    )


MedianEstimationExperiment = DynamicsExperiment(
    name="MedianEstimationExperiment",
    description="",  # TODO: description
    simulator=repeated_classification,
    simulator_config=construct_config_median,
    intervention=repeated_classification.Intervention(time=0),  # TODO: change to DRO
    state_sampler=sample_initial_states_median,
    propensity_scorer=0.5,
    outcome_extractor=extract_outcomes,
    covariate_builder=lambda run: 0,  # TODO: change
)
