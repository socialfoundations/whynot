"""Interactive environment for the credit simulator."""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.credit import (
    agent_model,
    CreditData,
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


CreditEnv = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    # The initial state is the baseline features and labels in the credit dataset
    initial_state=State(features=CreditData.features, labels=CreditData.labels),
    # Action space is classifiers with the same number of parameters are
    # features.
    action_space=spaces.Box(
        low=-np.inf, high=np.inf, shape=(CreditData.num_features,), dtype=np.float64
    ),
    # Observation space is the strategically adapted features and labels
    observation_space=spaces.Dict(
        {
            "features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=CreditData.features.shape,
                dtype=np.float64,
            ),
            "labels": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=CreditData.labels.shape,
                dtype=np.float64,
            ),
        }
    ),
    timestep=1,
    intervention_fn=compute_intervention,
    reward_fn=compute_reward,
)

register(
    id="Credit-v0", entry_point=CreditEnv, max_episode_steps=100, reward_threshold=0,
)
