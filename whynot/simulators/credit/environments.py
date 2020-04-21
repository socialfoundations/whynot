"""Interactive environment for the credit simulator."""
import copy

import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.credit import (
    Config,
    Intervention,
    simulate,
    strategic_logistic_loss,
    State,
)


def compute_reward(intervention, state, config):
    """Compute the reward based on the observed state and choosen intervention."""
    return strategic_logistic_loss(
        config, state.features, state.labels, intervention.updates["theta"],
    )


def compute_intervention(action, time):
    """Return intervention that changes the classifier parameters to action."""
    return Intervention(time=time, theta=action)


def credit_action_space(initial_state):
    """Return action space for credit simulator.
    
    The action space is the vector of possible logistic regression
    parameters, which depends on the dimensionality of the features.
    """
    num_features = initial_state.features.shape[1]
    return spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float64)


def credit_observation_space(initial_state):
    """Return observation space for credit simulator.
    
    The observation space is the vector of possible datasets, which
    must have the same dimensions as the initial state.
    """
    return spaces.Dict(
        {
            "features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=initial_state.features.shape,
                dtype=np.float64,
            ),
            "labels": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=initial_state.labels.shape,
                dtype=np.float64,
            ),
        }
    )


def build_credit_env(config=None, initial_state=None):
    """Construct credit environment that is parameterized by the initial state.

    This allows the user to specify different datasets other than the default
    Credit dataset.
    """
    if config is None:
        config = Config()
    elif not isinstance(config, Config):
        raise ValueError(f"Config must be an instance of {type(Config())}")
    if initial_state is None:
        initial_state = State()
    elif not isinstance(initial_state, State):
        raise ValueError(f"Initial state must be an instance of {type(State())}")

    config.base_state = copy.deepcopy(initial_state)

    return ODEEnvBuilder(
        simulate_fn=simulate,
        config=config,
        # The initial state is the baseline features and labels in the credit dataset
        initial_state=initial_state,
        # Action space is classifiers with the same number of parameters are
        # features.
        action_space=credit_action_space(initial_state),
        # Observation space is the strategically adapted features and labels
        observation_space=credit_observation_space(initial_state),
        timestep=1,
        intervention_fn=compute_intervention,
        reward_fn=compute_reward,
    )


register(
    id="Credit-v0",
    entry_point=build_credit_env,
    max_episode_steps=100,
    reward_threshold=0,
)
