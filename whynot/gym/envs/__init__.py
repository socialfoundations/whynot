"""Environments based on WhyNot simulators."""

from whynot.gym.envs.registration import registry, register, make, spec
from whynot.gym.envs.ode_env import ODEEnv

print("registering")
register(
    id='HIV-v0',
    entry_point='whynot.simulators.hiv.environment:HIVEnv',
    max_episode_steps=400,
    reward_threshold=1e10,
)


register(
    id='world3-v0',
    entry_point='whynot.simulators.world3.environment:World3Env',
    max_episode_steps=400,
    reward_threshold=1e5,
)
