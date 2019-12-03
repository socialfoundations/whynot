"""Simulation code for Lotka-Volterra model."""
import dataclasses

import numpy as np
from scipy.integrate import odeint

import whynot as wn
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of Lotka-Volterra dynamics.

    Examples
    --------
    >>> # Configure the simulator so each caught rabbit creates 2 foxes
    >>> lotka_volterra.Config(fox_growth=0.5)

    """

    # Dynamics parameters
    #: Natural growth rate of rabbits, when there's no foxes.
    rabbit_growth: float = 1.0
    #: Natural death rate of rabbits, due to predation.
    rabbit_death: float = 0.1
    #: Natural death rate of fox, when there's no rabbits.
    fox_death: float = 1.5
    #: Factor describing how many caught rabbits create a new fox.
    fox_growth: float = 0.75

    # Simulator book-keeping
    #: Start time of the simulator (in years).
    start_time: float = 0
    #: End time of the simulator (in years).
    end_time: float = 100
    #: Spacing of the evaluation grid
    delta_t: float = 1.0


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the Lotka-Volterra model."""

    #: Number of rabbits.
    rabbits: float = 10.0
    #: Number of foxes.
    foxes: float = 5.0


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the Lotka-Volterra model.

    An intervention changes a subset of the configuration variables in the
    specified year. The remaining variables are unchanged.

    Examples
    --------
    >>> # Starting at time 25, set fox_growth to 0.4 (leave other variables unchanged)
    >>> Intervention(time=25, fox_growth=0.4)

    """

    def __init__(self, time=30, **kwargs):
        """Specify an intervention in lotka_volterra.

        Parameters
        ----------
            time: int
                Time of intervention in simulator dynamics.
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def dynamics(state, time, config, intervention=None):
    """Noisy Lotka-Volterra equations for 2 species.

    Equations from:
        https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html

    Parameters
    ----------
        state: np.ndarray or tuple or list
            Instantenous system state
        time: float
            Time to evaluate the derivative
        config: whynot.simulators.lotka_volterra.Config
            Configuration parameters for the dynamics
        intervention: whynot.simulators.lotka_volterra.Intervention
            (Optional) Parameters specifying when/how to update the dynamics

    Returns
    -------
        ds_dt: np.ndarray
            Derivative of the state with respect to time, evaluated at state and time.

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    rabbits, foxes = state

    delta_rabbits = (
        config.rabbit_growth * rabbits - config.rabbit_death * rabbits * foxes
    )

    delta_foxes = (
        -config.fox_death * foxes
        + config.fox_growth * config.rabbit_death * rabbits * foxes
    )

    ds_dt = np.array([delta_rabbits, delta_foxes])
    return ds_dt


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Lotka volterra model.

    Parameters
    ----------
        initial_state: whynot.lotka_volterra.State
        config: whynot.lotka_volterra.Config
            Base parameters for the simulator run
        intervention: whynot.lotka_volterra.Intervention
            (Optional) Parameters specifying a change in dynamics
        seed: int
            Unused since the simulator is deterministic.

    Returns
    -------
        run: whynot.dynamics.Run
            Simulator rollout

    """
    # pylint: disable-msg=unused-argument
    t_eval = np.arange(
        config.start_time, config.end_time + config.delta_t, config.delta_t
    )

    solution = odeint(
        dynamics,
        y0=dataclasses.astuple(initial_state),
        t=t_eval,
        args=(config, intervention),
        rtol=1e-4,
        atol=1e-4,
    )

    states = [initial_state] + [State(*state) for state in solution[1:]]
    return wn.dynamics.Run(states=states, times=t_eval)
