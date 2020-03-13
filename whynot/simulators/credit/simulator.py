"""Implementation of the Perdomo et. al model of strategic classification.

The data is from the Kaggle Give Me Some Credit dataset:

    https://www.kaggle.com/c/GiveMeSomeCredit/data,

and the dynamics are taken from:

    Perdomo, Juan C., Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt.
    "Performative Prediction." arXiv preprint arXiv:2002.06673 (2020).
"""
import copy
import dataclasses
from typing import Callable, List

import whynot as wn
import whynot.traceable_numpy as np
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState
from whynot.simulators.credit.dataloader import CreditData


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of Credit simulator dynamics.

    Examples
    --------
    >>> # Configure simulator for run for 10 iterations
    >>> config = Config(start_time=0, end_time=10, delta_t=1)

    """

    # Dynamics parameters
    #: Subset of the features that can be manipulated by the agent
    changeable_features: np.ndarray = np.array([0, 5, 7])

    #: Model how much the agent adapt her features in response to a classifier
    epsilon: float = 0.1

    #: Parameters for logistic regression classifier used by the institution
    theta: np.ndarray = np.ones((11, 1))

    #: L2 penalty on the logistic regression loss
    l2_penalty: float = 0.0

    #: Whether or not dynamics have memory
    memory: bool = False

    # Simulator book-keeping
    #: Start time of the simulator
    start_time: int = 0
    #: End time of the simulator
    end_time: int = 5
    #: Spacing of the evaluation grid
    delta_t: int = 1


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the Credit model."""

    #: Matrix of agent features (see https://www.kaggle.com/c/GiveMeSomeCredit/data)
    features: np.ndarray = np.ones((13, 11))

    #: Vector indicating whether or not the agent experiences financial distress
    labels: np.ndarray = np.zeros((13, 1))

    def values(self):
        """Return the state as a dictionary of numpy arrays."""
        return {name: getattr(self, name) for name in self.variable_names()}


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the Credit model.

    An intervention changes a subset of the configuration variables in the
    specified year. The remaining variables are unchanged.

    Examples
    --------
    >>> # Starting at time 25, update the classifier to random chance.
    >>> config = Config()
    >>> Intervention(time=25, theta=np.zeros_like(config.theta))

    """

    def __init__(self, time=30, **kwargs):
        """Specify an intervention in credit.

        Parameters
        ----------
            time: int
                Time of intervention in simulator dynamics.
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def strategic_logistic_loss(config, features, labels, theta):
    """Evaluate the performative loss for logistic regression classifier."""

    config = config.update(Intervention(theta=theta))

    # Compute adjusted data
    strategic_features = agent_model(features, config)

    # compute log likelihood
    num_samples = strategic_features.shape[0]
    logits = strategic_features @ config.theta
    log_likelihood = (1.0 / num_samples) * np.sum(
        -1.0 * np.multiply(labels, logits) + np.log(1 + np.exp(logits))
    )

    # Add regularization (without considering the bias)
    regularization = (config.l2_penalty / 2.0) * np.linalg.norm(config.theta[:-1]) ** 2

    return log_likelihood + regularization


def agent_model(features, config):
    """Compute agent reponse to the classifier and adapt features accordingly.

    TODO: For now, the best-response model corresponds to best-response with
    linear utility and quadratic costs. We should expand this to cover a rich
    set of agent models beyond linear/quadratic, and potentially beyond
    best-response.
    """
    # Move everything by epsilon in the direction towards better classification
    strategic_features = np.copy(features)
    theta_strat = config.theta[config.changeable_features].flatten()
    strategic_features[:, config.changeable_features] -= config.epsilon * theta_strat
    return strategic_features


def dynamics(state, time, config, intervention=None):
    """Perform one round of interaction between the agents and the credit scorer.

    Parameters
    ----------
        state: whynot.simulators.credit.State
            Agent state at time TIME
        time: int
            Current round of interaction
        config: whynot.simulators.credit.Config
            Configuration object controlling the interaction, e.g. classifier
            and agent model
        intervention: whynot.simulators.credit.Intervention
            Intervention object specifying when and how to update the dynamics.

    Returns
    -------
        state: whynot.simulators.credit.State
            Agent state after one step of strategic interaction.

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    if config.memory:
        features, labels = state
    else:
        features, labels = CreditData.features, CreditData.labels

    # Update features in response to classifier. Labels are fixed.
    strategic_features = agent_model(features, config)
    return strategic_features, labels


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Credit model.

    Parameters
    ----------
        initial_state: whynot.credit.State
        config: whynot.credit.Config
            Base parameters for the simulator run
        intervention: whynot.credit.Intervention
            (Optional) Parameters specifying a change in dynamics
        seed: int
            Unused since the simulator is deterministic.

    Returns
    -------
        run: whynot.dynamics.Run
            Simulator rollout

    """
    # Iterate the discrete dynamics
    times = [config.start_time]
    states = [initial_state]
    state = copy.deepcopy(initial_state)

    for step in range(config.start_time, config.end_time):
        next_state = dynamics(state.values(), step, config, intervention)
        state = State(*next_state)
        states.append(state)
        times.append(step + 1)

    return wn.dynamics.Run(states=states, times=times)


if __name__ == "__main__":
    print(simulate(State(), Config(end_time=2)))
