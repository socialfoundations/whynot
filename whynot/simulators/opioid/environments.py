"""Reinforcement learning environments for the opioid epidemic simulator."""
import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.opioid import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention in the simulator required to take action."""
    action_to_intervention_map = {
        0: Intervention(time=time, nonmedical_incidence=0.0, illicit_incidence=0.0),
        1: Intervention(time=time, nonmedical_incidence=0.0, illicit_incidence=0.05),
        2: Intervention(time=time, nonmedical_incidence=0.05, illicit_incidence=0.0),
        3: Intervention(time=time, nonmedical_incidence=0.05, illicit_incidence=0.05),
    }
    return action_to_intervention_map[action]


def get_reward(intervention, state, config, time):
    """Compute the reward based on the observed state and choosen intervention."""
    # pylint: disable-msg=invalid-name
    # Penalty for overdose death.
    C = 1000
    # Costs for reducing nonmedical users and illicit users per user.
    C1 = 1
    C2 = 5
    deaths = state.nonmedical_users * config.nonmedical_overdose
    deaths += state.oud_users * config.oud_overdose[time]
    deaths += state.illicit_users * config.illicit_overdose[time]
    reward = -C * deaths
    reward -= C1 * intervention.updates["nonmedical_incidence"] * state.nonmedical_users
    reward -= C2 * intervention.updates["illicit_incidence"] * state.illicit_users
    return reward


def observation_space():
    """Return the observation space. The state is (nonmedical_users, oud_useres, illicit_users)."""
    state_dim = State.num_variables()
    state_space_low = np.zeros(state_dim)
    state_space_high = np.inf * np.ones(state_dim)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


OpioidEnv = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    initial_state=State(),
    # In this environment we define 4 actions:
    #   - Do nothing
    #   - Reduce nonmedical opioid use by 5%
    #   - Reduce illicit opioid use by 5%
    #   - Reduce both by 5%
    action_space=spaces.Discrete(4),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)

register(
    id="opioid-v0",
    entry_point=OpioidEnv,
    # The simulator starts in 2002 and ends in 2030.
    max_episode_steps=28,
    reward_threshold=0,
)
