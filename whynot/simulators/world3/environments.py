"""Reinforcment learning for world3."""
from itertools import product

import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.world3 import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention needed to take action in the simulator."""
    resource_usages = [0.8, 1.0, 1.2]
    pollution_generations = [0.8, 1.0, 1.2]
    action_list = list(product(resource_usages, pollution_generations))
    resource_use, pollution_gen = action_list[action]
    return Intervention(
        time,
        nonrenewable_resource_usage_factor=resource_use,
        persistent_pollution_generation_factor=pollution_gen,
    )


def get_reward(intervention, state):
    """Return the intervention needed to take action in the simulator."""
    # Constants are made up, but chosen so that all quantities have
    # the same magnitude for the initial state.

    # High population and capital are good, high pollution is bad.
    reward = 1e-11 * state.industrial_capital - 5e-7 * state.persistent_pollution
    reward += 2e-8 * state.total_population

    # Taking any action other than the "default" is costly
    resource_usage = intervention.updates["nonrenewable_resource_usage_factor"]
    pollution_generation = intervention.updates[
        "persistent_pollution_generation_factor"
    ]
    reward -= 0.3 * (1.0 - resource_usage) ** 2
    reward -= 0.5 * (1.0 - pollution_generation) ** 2

    return reward


def observation_space():
    """Return the model observation space."""
    num_states = State.num_variables()
    state_space_low = np.zeros(num_states)
    state_space_high = np.inf * np.ones(num_states)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


World3Env = ODEEnvBuilder(
    simulate_fn=simulate,
    # Smaller delta_t improves numerical stability
    config=Config(delta_t=0.5),
    initial_state=State(),
    # In this environment there are 9 actions defined by
    # nonrenewable_resource_usage and pollution_generation_factor.
    action_space=spaces.Discrete(9),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)

register(
    id="world3-v0", entry_point=World3Env, max_episode_steps=400, reward_threshold=1e5,
)
