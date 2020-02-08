"""Experiments for repeated classification simulator."""

from scipy.optimize import minimize
from sklearn.svm import SVC

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
    result = minimize(
        objective,
        x0=init_params,
        **kwargs
    )
    return result.x


def train_svm(classifier_func, loss, features, labels, init_params, rng):
    model = SVC(C=0.1, kernel='linear').fit(features, labels)
    return np.concatenate([model.coef_.flatten(), model.intercept_.flatten()])


def construct_config():
    """Experimental config."""
    return repeated_classification.Config(
        K=2,
        baseline_growth=np.array([1000, 1000]),
        group_distributions=[left_gaussian_dist, right_gaussian_dist],
        classifier_func=linear_classifier_2d,
        loss=zero_one_loss,
        user_retention=linear_retention,
        # train_classifier=empirical_risk_minimization,
        train_classifier=train_svm,
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
