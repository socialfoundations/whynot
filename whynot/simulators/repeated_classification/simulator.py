"""Implementation of repeated loss minimization simulator based on Hashimoto et
al.

Hashimoto, T., Srivastava, M., Namkoong, H., & Liang, P. (2018, July).
Fairness without demographics in repeated loss minimization. In International
Conference on Machine Learning. (https://arxiv.org/abs/1806.08010)
"""
import copy
import dataclasses
from typing import Any, Callable, List

import whynot as wn
import whynot.traceable_numpy as np
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention


def poisson_population_sampler(expected_populations, rng):
    """Poisson process to sample number of individuals in each group according
    to expected group populations"""
    return rng.poisson(expected_populations)


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics."""

    #: Number of groups
    K: int

    #: Positive lower bound on smallest group proportion
    min_proportion: float

    #: Expected group-k baseline population growth per step, length K
    baseline_growth: np.ndarray

    #: Group-k distribution of features and labels, length K
    group_distributions: List[Callable]

    #: Predict individual label from input features and learned params
    classifier_func: Callable

    #: Loss function for individual predictions
    loss: Callable

    #: Fraction of users retained, a function of (per-group) loss
    user_retention: Callable

    #: Algorithm to learn classifier parameters
    train_classifier: Callable

    #: Random process to determine group sizes from expected group sizes
    population_sampler: Callable = poisson_population_sampler

    #: Simulation start step (in rounds)
    start_time: float = 0

    #: Simulation end step (in rounds)
    end_time: float = 500

    #: Simulator step size
    delta_t: float = 1


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the repeated classification simulator."""

    #: Expected population of group k at current time step, length K
    expected_populations: np.ndarray

    #: Actual population of group k at current time step, length K
    populations: np.ndarray

    #: Length n x d, n current total population, d number of features
    features: np.ndarray

    # True labels for each individual, length n
    labels: np.ndarray

    #: Estimated parameters for classifier
    classifier_params: Any

    #: Average loss within group k, length K
    risks: np.ndarray


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parametrization of an intervention in the repeated classification
    model.

    Examples
    --------
    >>> # Change the user retention function to `other_retention_func`
    >>> Intervention(time=0, user_retention=other_retention_func)

    """

    def __init__(self, time=100, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention (days)
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def compute_state_values(populations, init_params, config, rng):
    features, labels = [], []
    for pop, dist in zip(populations, config.group_distributions):
        features_k, labels_k = dist(pop, rng)
        features.append(features_k)
        labels.append(labels_k)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    classifier_params = config.train_classifier(
        config=config,
        features=features,
        labels=labels,
        init_params=init_params,
        rng=rng,
    )

    risks = np.array([])
    index = 0
    for i in populations:
        features_k = features[index:index + i]
        labels_k = labels[index:index + i]
        index += i
        risk_k = np.mean(
            config.loss(
                config.classifier_func(features_k, classifier_params, rng),
                labels_k
            )
        )
        risks = np.append(risks, risk_k)

    return features, labels, classifier_params, risks


def dynamics(state, time, config, intervention=None, rng=None):
    """Update equations for the repeated classification simulation.

    Performs one step of classification and group adjustment accordingly.

    Parameters
    ----------
        state: np.ndarray, list, or tuple
            State of the dynamics
        time: int
            Current time step
        config: whynot.simulators.repeated_classification.Config
            Configuration object controlling the simulation parameters
        intervention: whynot.simulators.repeated_classification.Intervention
            Intervention object specifying when and how to update the dynamics
        rng: np.RandomState
            Seed random number generator for all randomness (optional)

    Returns
    -------
        state: np.ndarray, list, or tuple
            State after one time step.

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    if rng is None:
        rng = np.random.RandomState(None)

    (
        expected_populations,
        populations,
        features,
        labels,
        classifier_params,
        risks,
    ) = state

    new_expected_populations = expected_populations * config.user_retention(risks) + config.baseline_growth
    new_populations = config.population_sampler(new_expected_populations, rng)

    new_features, new_labels, new_classifier_params, new_risks = compute_state_values(
        populations=new_populations,
        init_params=classifier_params,  # initialize with previous parameters
        config=config,
        rng=rng,
    )

    return (
        new_expected_populations,
        new_populations,
        new_features,
        new_labels,
        new_classifier_params,
        new_risks,
    )


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the repeated classification simulator.

    Parameters
    ----------
        initial_state:  `whynot.simulators.repeated_classification.State`
            Initial State object, which is used as x_{t_0} for the simulator.
        config:  `whynot.simulators.repeated_classification.Config`
            Config object that encapsulates the parameters that define the dynamics.
        intervention: `whynot.simulators.repeated_classification.Intervention`
            Intervention object that specifies what, if any, intervention to perform.
        seed: int
            Seed to set internal randomness.

    Returns
    -------
        run: `whynot.dynamics.Run`
            Rollout of the model.

    """
    rng = np.random.RandomState(seed)

    # Iterate the discrete dynamics
    times = [config.start_time]
    states = [initial_state]
    state = copy.deepcopy(initial_state)
    np.set_printoptions(precision=4)
    print()
    for step in range(config.start_time, config.end_time):
        state = State(*dynamics(state.values(), step, config, intervention, rng))
        states.append(state)
        times.append(step + 1)
        if step % 20 == 0:
            print('{}: expected pops {}, risks {}, params {}'.format(
                step,
                state.expected_populations,
                state.risks,
                state.classifier_params,
            ))

    return wn.dynamics.Run(states=states, times=times)


if __name__ == "__main__":
    print(simulate(State(), Config()))
