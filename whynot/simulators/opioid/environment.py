"""Reinforcement learning environments for the opioid epidemic simulator."""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnv
from whynot.simulators.opioid import Config, Intervention, simulate, State


class OpioidEnv(ODEEnv):
    """Environment based on ODE Opioid simulator."""

    def __init__(self, config=None):
        """Construct the treatment control problem for the original Adams paper."""
        if config is None:
            config = Config()
        initial_state = State()
        # In this environment we define 4 actions:
        #   - Do nothing
        #   - Reduce nonmedical opioid use by 5%
        #   - Reduce illicit opioid use by 5%
        #   - Reduce both by 5%
        action_space = spaces.Discrete(4)
        # The state is (nonmedical_users, oud_useres, illicit_users).
        state_dim = State.num_variables()
        state_space_low = np.zeros(state_dim)
        state_space_high = np.inf * np.ones(state_dim)
        observation_space = spaces.Box(state_space_low, state_space_high)

        super(OpioidEnv, self).__init__(
            simulate, config, action_space, observation_space, initial_state
        )

    def _get_intervention(self, action):
        """Return the intervention in the simulator required to tack action."""
        action_to_intervention_map = {
            0: Intervention(
                time=self.time, nonmedical_incidence=0.0, illicit_incidence=0.0
            ),
            1: Intervention(
                time=self.time, nonmedical_incidence=0.0, illicit_incidence=0.05
            ),
            2: Intervention(
                time=self.time, nonmedical_incidence=0.05, illicit_incidence=0.0
            ),
            3: Intervention(
                time=self.time, nonmedical_incidence=0.05, illicit_incidence=0.05
            ),
        }
        return action_to_intervention_map[action]

    def _get_reward(self, intervention, state):
        """Compute the reward based on the observed state and choosen intervention."""
        # pylint: disable-msg=invalid-name
        # Penalty for overdose death.
        C = 1000
        # Costs for reducing nonmedical users and illicit users per user.
        C1 = 1
        C2 = 5
        deaths = state.nonmedical_users * self.config.nonmedical_overdose
        deaths += state.oud_users * self.config.oud_overdose[self.time]
        deaths += state.illicit_users * self.config.illicit_overdose[self.time]
        reward = -C * deaths
        reward -= (
            C1 * intervention.updates["nonmedical_incidence"] * state.nonmedical_users
        )
        reward -= C2 * intervention.updates["illicit_incidence"] * state.illicit_users
        return reward
