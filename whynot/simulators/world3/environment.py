"""Reinforcment learning for world3."""
from itertools import product

import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnv
from whynot.simulators.world3 import Config, Intervention, simulate, State


class World3Env(ODEEnv):
    """Environment based on ODE World3 simulator."""

    def __init__(self, config=None):
        """Initialize a the world3 environment."""
        if config is None:
            # A smaller delta_t improves numerical stability.
            config = Config(delta_t=0.5)
        initial_state = State()
        # In this environment there are 6 actions defined by
        # nonrenewable_resource_usage_factor in 0.8, 1.0, 1.2
        # and persistent_pollution_generation_factor in 0.8, 1.0, 1.2
        action_space = spaces.Discrete(9)

        num_states = State.num_variables()
        state_space_low = np.zeros(num_states)
        state_space_high = np.inf * np.ones(num_states)
        observation_space = spaces.Box(state_space_low, state_space_high)

        super(World3Env, self).__init__(simulate, config, action_space,
                                        observation_space, initial_state,
                                        timestep=config.delta_t)

    def _get_intervention(self, action):
        """Return the intervention needed to take action in the simulator."""
        resource_usages = [0.8, 1.0, 1.2]
        pollution_generations = [0.8, 1.0, 1.2]

        action_to_intervention_map = {}
        actions = product(resource_usages, pollution_generations)
        for idx, (resource_usage, pollution_generation) in enumerate(actions):
            action_to_intervention_map[idx] = Intervention(
                self.time,
                nonrenewable_resource_usage_factor=resource_usage,
                persistent_pollution_generation_factor=pollution_generation)
        return action_to_intervention_map[action]

    def _get_reward(self, intervention, state):
        """Return the intervention needed to take action in the simulator."""
        # Constants are made up, but chosen so that all quantities have
        # the same magnitude for the initial state.

        # High population and capital are good, high pollution is bad.
        reward = 1e-11 * state.industrial_capital - 5e-7 * state.persistent_pollution
        reward += 2e-8 * state.total_population

        # Taking any action other than the "default" is costly
        reward -= 0.3 * (1. - intervention.updates['nonrenewable_resource_usage_factor']) ** 2
        reward -= 0.5 * (1. - intervention.updates['persistent_pollution_generation_factor']) ** 2

        return reward
