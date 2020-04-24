import dataclasses
import pytest

import numpy as np

import whynot as wn
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
@pytest.mark.parametrize("spec", ["HIV-v0", "world3-v0", "opioid-v0"])
def test_random_rollout(spec):
    for env in [envs.make(spec), envs.make(spec), envs.make(spec)]:

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

@pytest.mark.parametrize("spec", ["HIV-v0", "world3-v0", "opioid-v0", "Zika-v0"])
def test_config(spec):
    """Test setting simulator config via gym.make"""
    base_env = envs.make(spec)
    base_config = base_env.config
    new_config = dataclasses.replace(base_config, delta_t=-100)
    new_env = envs.make(spec, config=new_config)
    assert base_env.config.delta_t == base_config.delta_t
    assert new_env.config.delta_t == new_config.delta_t





def test_credit_config():
    """Set simulator config for Credit sim via gym.make"""
    base_features = wn.credit.Config().changeable_features
    new_features = np.array([0, 1, 2])
    base_env = envs.make("Credit-v0")
    config = wn.credit.Config(changeable_features=new_features)
    env = envs.make("Credit-v0", config=config)
    assert np.allclose(base_env.config.changeable_features, base_features)
    assert np.allclose(env.config.changeable_features, new_features)
    base_env.close()
    env.close()

def test_credit_initial_state():
    """Test initial state for Credit sim via gym.make"""
    base_env = envs.make("Credit-v0")
    original_state = base_env.reset()
    features, labels = original_state["features"], original_state["labels"]
    subsampled_state = wn.credit.State(features[:100], labels[:100])

    env = envs.make("Credit-v0", initial_state=subsampled_state)

    assert np.allclose(env.initial_state.features, subsampled_state.features)
    assert np.allclose(env.initial_state.labels, subsampled_state.labels)

    ob = env.reset()
    assert np.allclose(ob["features"], subsampled_state.features)
    assert np.allclose(ob["labels"], subsampled_state.labels)
    for idx in range(10):
        print(env.observation_space)
        print(ob["features"].shape, ob["labels"].shape)
        assert env.observation_space.contains(ob)
        a = env.action_space.sample()
        assert env.action_space.contains(a)
        (ob, _reward, done, _info) = env.step(a)
        if done:
            break
    ob = env.reset()
    assert np.allclose(ob["features"], subsampled_state.features)
    assert np.allclose(ob["labels"], subsampled_state.labels)
    env.close()
