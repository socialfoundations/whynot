# -*- coding: utf-8 -*-
import whynot.gym as gym
from whynot.gym import error, envs
from whynot.gym.envs import registration
import whynot.simulators as simulators


class ArgumentEnv(gym.Env):
    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


gym.register(
    id="test.ArgumentEnv-v0",
    entry_point="test_registration:ArgumentEnv",
    kwargs={"arg1": "arg1", "arg2": "arg2",},
)


def test_make():
    env = envs.make("HIV-v0")
    assert env.spec.id == "HIV-v0"
    assert isinstance(env.unwrapped, simulators.hiv.environment.HIVEnv)


def test_make_with_kwargs():
    env = envs.make("test.ArgumentEnv-v0", arg2="override_arg2", arg3="override_arg3")
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.arg1 == "arg1"
    assert env.arg2 == "override_arg2"
    assert env.arg3 == "override_arg3"


def test_spec():
    spec = envs.spec("HIV-v0")
    assert spec.id == "HIV-v0"


def test_missing_lookup():
    registry = registration.EnvRegistry()
    registry.register(id="Test-v1", entry_point=None)
    try:
        registry.spec("Unknown-v1")
    except error.UnregisteredEnv:
        pass
    else:
        assert False


def test_malformed_lookup():
    registry = registration.EnvRegistry()
    try:
        registry.spec(u"“Breakout-v0”")
    except error.Error as e:
        assert "malformed environment ID" in "{}".format(
            e
        ), "Unexpected message: {}".format(e)
    else:
        assert False


if __name__ == "__main__":
    test_missing_lookup()
