"""Interactive environment for the credit simulator."""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnv
from whynot.simulators.credit import (
    CreditData,
    Config,
    evaluate_loss,
    Intervention,
    simulate,
    State,
)


class CreditEnv(ODEEnv):
    """Environment based on Credit simulator."""

    def __init__(self, config=None):
        """Construct a repeated classification task, similar to the Perdomo et. al paper."""
        if config is None:
            config = Config()

        # The initial state is the baseline features and labels in the credit dataset
        initial_state = State(features=CreditData.features, labels=CreditData.labels)
        action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(CreditData.num_features,)
        )

        # Observation space is the strategically adapted features and labels
        observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=initial_state.features.shape),
            spaces.Box(low=-np.inf, high=np.inf, shape=initial_state.labels.shape),
        ]

        super(CreditEnv, self).__init__(
            simulate, config, action_space, observation_space, initial_state, timestep=1
        )

    def _get_intervention(self, action):
        """Return the intervention in the simulator required to take action."""
        return Intervention(time=self.time, theta=action)

    def _get_reward(self, intervention, state):
        """Compute the reward based on the observed state and choosen intervention."""
        intervention_config = self.config.update(intervention)
        return evaluate_loss(
            state.features, state.labels, intervention_config, l2_penalty=0.0
        )
