"""Global registry of environments, for consistency with openai gym."""
import importlib

from gym.envs.registration import EnvRegistry


# The naming format is (Env-Name-v0).

def load(name):
    """Load the class specified by name, where name is mod_name:attr."""
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    func = getattr(mod, attr_name)
    return func


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
