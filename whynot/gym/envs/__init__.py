"""Environments based on WhyNot simulators."""

from whynot.gym.envs.registration import registry, register, make, spec
from whynot.gym.envs.ode_env import ODEEnv

register(
    id="HIV-v0",
    entry_point="whynot.simulators.hiv.environment:HIVEnv",
    max_episode_steps=400,
    reward_threshold=1e10,
)


register(
    id="world3-v0",
    entry_point="whynot.simulators.world3.environment:World3Env",
    max_episode_steps=400,
    reward_threshold=1e5,
)


register(
    id='opioid-v0',
    entry_point='whynot.simulators.opioid.environment:OpioidEnv',
    # The simulator starts in 2002 and ends in 2030.
    max_episode_steps=28,
    reward_threshold=0,
)
