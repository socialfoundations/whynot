"""Interactive environment for the credit simulator."""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnv
from whynot.simulators.credit import (
    agent_model,
    CreditData,
    Config,
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

    def evaluate_logistic_loss(self, features, labels, theta, l2_penalty=0.0):
        """Evaluate the performative loss for logistic regression classifier."""

        config = self.config.update(Intervention(theta=theta))

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

    def _get_reward(self, intervention, state):
        """Compute the reward based on the observed state and choosen intervention."""
        return self.evaluate_logistic_loss(
            state.features, state.labels, intervention.updates["theta"], l2_penalty=0.0
        )
