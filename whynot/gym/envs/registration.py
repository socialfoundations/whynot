"""Global registry of environments, for consistency with openai gym."""
import importlib

from gym.envs.registration import EnvRegistry

# Keep for consistency with original API
# pylint:disable-msg=invalid-name
# Have a global registry
registry = EnvRegistry()


# pylint:disable-msg=redefined-builtin
def register(id, **kwargs):
    """Register the environment."""
    return registry.register(id, **kwargs)


def make(id, **kwargs):
    """Build the environment."""
    return registry.make(id, **kwargs)


def spec(id):
    """View the spec for the environment."""
    return registry.spec(id)


warn_once = True
