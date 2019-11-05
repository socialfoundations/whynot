"""Implementation of Adams et al.'s ODE simulator for HIV treatment.

Adams, Brian Michael, et al. Dynamic multidrug therapies for HIV: Optimal and
STI control approaches. North Carolina State University. Center for Research in
Scientific Computation, 2004. APA.

https://pdfs.semanticscholar.org/c030/127238b1dbad2263fba6b64b5dec7c3ffa20.pdf
"""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnv
from whynot.simulators.hiv import Config, Intervention, simulate, State


class HIVEnv(ODEEnv):
    """Environment based on ODE HIV simulator."""

    def __init__(self, config=None):
        """Construct the treatment control problem for the original Adams paper."""
        if config is None:
            config = Config()
        initial_state = State()
        # In this environment there are 4 actions defined by
        # epsilon_1 = 0 or 0.7 and epsilon_2 = 0 or 0.3.
        action_space = spaces.Discrete(4)
        # The state is (uninfected_T1, infected_T1, uninfected_T2, infected_T2,
        # free_virus, immune_response) in units (cells/ml, cells/ml, cells/ml,
        # cells/ml, copies/ml, cells/ml)
        state_dim = State.num_variables()
        state_space_low = np.zeros(state_dim)
        state_space_high = np.inf * np.ones(state_dim)
        observation_space = spaces.Box(state_space_low, state_space_high)

        super(HIVEnv, self).__init__(simulate, config, action_space,
                                     observation_space, initial_state)

    def _get_intervention(self, action):
        """Return the intervention in the simulator required to tack action."""
        action_to_intervention_map = {
            0: Intervention(time=self.time, epsilon_1=0.0, epsilon_2=0.0),
            1: Intervention(time=self.time, epsilon_1=0.0, epsilon_2=0.3),
            2: Intervention(time=self.time, epsilon_1=0.7, epsilon_2=0.0),
            3: Intervention(time=self.time, epsilon_1=0.7, epsilon_2=0.3),
        }
        return action_to_intervention_map[action]

    def _get_reward(self, intervention, state):
        """Compute the reward based on the observed state and choosen intervention."""
        # pylint: disable-msg=invalid-name
        # Constants chosen according to the reference paper
        Q, R1, R2, S = 0.1, 20000., 20000., 1000.
        reward = S * state.immune_response - Q * state.free_virus
        reward -= R1 * (intervention.updates['epsilon_1'] ** 2)
        reward -= R2 * (intervention.updates['epsilon_2'] ** 2)
        return reward
