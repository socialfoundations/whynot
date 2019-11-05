from whynot.gym import envs, logger
import os


def should_skip_env_spec_for_tests(spec):
    # We skip tests for envs that require dependencies or are otherwise
    # troublesome to run frequently
    ep = spec.entry_point
    if ep.startswith('gym.envs.ode.hiv'):
        return True
    return False


spec_list = [spec for spec in sorted(envs.registry.all(
), key=lambda x: x.id) if spec.entry_point is not None and not should_skip_env_spec_for_tests(spec)]
