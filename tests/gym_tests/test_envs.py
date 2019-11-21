import pytest
import numpy as np

from whynot.gym import envs
from spec_list import spec_list

# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.parametrize("spec", spec_list)
def test_env(spec):
    # Capture warnings
    with pytest.warns(None) as warnings:
        env = spec.make()

    # Check that dtype is explicitly declared for gym.Box spaces
    for warning_msg in warnings:
        assert not "autodetected dtype" in str(warning_msg.message)

    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), "Reset observation: {!r} not in space".format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), "Step observation: {!r} not in space".format(
        observation
    )
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    env.close()


# Run a longer rollout on some environments


def test_random_rollout():
    for env in [envs.make("HIV-v0"), envs.make("world3-v0")]:

        def agent(ob):
            return env.action_space.sample()

        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            print("Observation: ", ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done:
                break
        env.close()
