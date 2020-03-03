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


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of Credit simulator dynamics.
    
    Examples
    --------

    """
    # Dynamics parameters
    #: Subset of the features that can be manipulated by the agent
    changeable_features: np.ndarray = np.array([0, 5, 7])

    #: Model how much the agent adapt her features in response to a classifier
    epsilon: float = 0.1

    #: Parameters for logistic regression classifier used by the institution
    theta: np.ndarray = np.ones((11, 1))

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
        """Return the state as a list of numpy array."""
        return [getattr(self, name) for name in self.variable_names()]


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the Credit model.

    An intervention changes a subset of the configuration variables in the
    specified year. The remaining variables are unchanged.

    Examples
    --------
    >>> # Starting at time 25, update the classifier to random chance.
    >>> Intervention(time=25, classifier=lambda x: return 0.5)

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


def evaluate_loss(features, labels, config, l2_penalty=0.0):
    """Evaluate the performative loss for logistic regression"""

    # Compute adjusted data
    strategic_features = agent_model(features, config)

    # compute log likelihood
    logits = strategic_features @ config.theta
    log_likelihood = np.sum(
        -1.0 * np.multiply(labels, logits) + np.log(1 + np.exp(logits))
    )

    log_likelihood /= strategic_features.shape[0]

    # Add regularization (without considering the bias)
    regularization = l2_penalty / 2.0 * np.linalg.norm(config.theta[:-1]) ** 2

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
    strategic_features[:, config.changeable_features] += -config.epsilon * theta_strat
    return strategic_features


def dynamics(state, time, config, intervention=None):
    """Performs one round of interaction between the agents and the credit scorer.
    
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

    features, labels = state

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
    rng = np.random.RandomState(seed)

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
    features, labels = State().values()
    print(evaluate_loss(features, labels, Config(), l2_penalty=0.1))
