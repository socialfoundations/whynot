"""Implementation of Momoh and Fügenschuh environment for Zika prevention.

Momoh, Abdulfatai A., and Armin Fügenschuh. "Optimal control of intervention
strategies and cost effectiveness analysis for a Zika virus model." Operations
Research for Health Care 18 (2018): 99-111.

https://www.sciencedirect.com/science/article/pii/S2211692316301084#!
"""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.zika import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention in the simulator required to take action."""
    treated_bednet_use, condom_use, treatment_of_infected, indoor_spray_use = action
    return Intervention(
        time=time,
        treated_bednet_use=treated_bednet_use,
        condom_use=condom_use,
        treatment_of_infected=treatment_of_infected,
        indoor_spray_use=indoor_spray_use,
    )


def get_reward(intervention, state, time):
    """Compute the reward based on the observed state and choosen intervention."""
    A_1, A_2, A_3 = 60, 500, 60
    C_1, C_2, C_3, C_4 = 25, 20, 30, 40
    discount = 4.0 / 365

    cost = (
        A_1 * state.asymptomatic_humans
        + A_2 * state.symptomatic_humans
        + A_3 * state.mosquito_population
    )

    cost += 0.5 * (
        C_1 * intervention.updates["treated_bednet_use"] ** 2
        + C_2 * intervention.updates["condom_use"] ** 2
        + C_3 * intervention.updates["treatment_of_infected"] ** 2
        + C_4 * intervention.updates["indoor_spray_use"] ** 2
    )
    return -cost * np.exp(-discount * time)


def observation_space():
    """Return observation space, the positive orthant."""
    state_dim = State.num_variables()
    state_space_low = np.zeros(state_dim)
    state_space_high = np.inf * np.ones(state_dim)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


def action_space():
    """Return action space.

    There are four control variables in the model:
        - Treated bednet use
        - Condom use
        - Direct treatment of infected humans
        - Indoor residual spray use.

    """
    return spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)


ZikaEnv = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    initial_state=State(),
    action_space=action_space(),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)

register(
    id="Zika-v0", entry_point=ZikaEnv, max_episode_steps=200, reward_threshold=1e10,
)
