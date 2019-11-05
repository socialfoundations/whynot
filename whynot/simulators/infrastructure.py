"""Base classes for config, state, and intervention."""

import dataclasses
import numpy as np


class BaseConfig:
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics."""

    # Start and end time.
    start_time: float = 0
    end_time: float = 0
    # Time elapsed between measurements of the system.
    delta_t: float = 1.

    def update(self, intervention):
        """Generate a new config after applying the intervention."""
        return dataclasses.replace(self, **intervention.updates)

    @classmethod
    def parameter_names(cls):
        """Return the parameters of the simulator, excluding start, end, and delta_t."""
        excluded = ["start_time", "end_time", "delta_t"]
        return [f.name for f in dataclasses.fields(cls) if f.name not in excluded]


@dataclasses.dataclass
class BaseState:
    # pylint: disable-msg=too-few-public-methods
    """State of the simulator."""

    @classmethod
    def num_variables(cls):
        """Count the number of variables."""
        return len(dataclasses.fields(cls))

    @classmethod
    def variable_names(cls):
        """List the variable names."""
        return [f.name for f in dataclasses.fields(cls)]

    def values(self):
        """Return the state as a numpy array."""
        # This ensures the values are shallow copied, unlike
        # dataclasses.astuple, which is necessary for graph tracing.
        return np.array([getattr(self, name) for name in self.variable_names()])


class BaseIntervention:
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention to a config."""

    def __init__(self, config_class, time, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention.
            config_class:
                The Config class, a child class of dataclass.
            kwargs: dict
                Only valid keyword arguments are parameters of the config class.

        """
        self.time = time
        config_args = set(f.name for f in dataclasses.fields(config_class))
        for arg in kwargs:
            if arg not in config_args:
                raise TypeError(f"__init__() got an unexpected keyword argument {arg}!")
        self.updates = kwargs
