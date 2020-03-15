"""Implementation of Momoh and Fügenschuh ODE simulator for Zika prevention.

Momoh, Abdulfatai A., and Armin Fügenschuh. "Optimal control of intervention
strategies and cost effectiveness analysis for a Zika virus model." Operations
Research for Health Care 18 (2018): 99-111.

https://www.sciencedirect.com/science/article/pii/S2211692316301084#!
"""

import dataclasses
import numpy as np

from scipy.integrate import odeint

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics."""
    # Simulation parameters
    #: Recruitment rate of humans into susceptible population
    lambda_h: float = 0.000011
    #: Natural human death rate
    mu_h: float = 1. / (360 * 60)
    #: Recovered humans loss of immunity
    varphi_h: float = 0.02
    #: Spontaneous individual recovery
    phi: float = 0.05
    #: Rate of recovery with temporary immunity
    nu: float = 0.023
    #: Breakthrough rate of humans from exposed to infected
    alpha_h: float = 0.2
    #: Disease induced death rate
    delta_h: float = 0.003
    #: Relative human-human contact rate of asymptomatic infected
    c: float = 0.05
    #: Relative human-human contact rate of symptomatic infected
    kappa: float = 0.05
    #: Prob of transmission per contact by asymptomatic infected through sexual activity
    beta_a: float = 0.6
    #: Prob of transmission per contact by symptomatic infected through sexual activity
    beta_s: float = 0.3
    #: Prob of transmission per contact by an infectious mosquito
    beta_1: float = 0.4
    #: Prob of transmission per contact by an infected human
    beta_2: float = 0.5
    #: Per capital biting rate of mosquitos
    epsilon: float = 0.5
    #: Contact rate of mosquito per human per unit time
    rho: float = 0.1
    #: Recruitment rate of mosquitoes
    lambda_v: float = 0.071
    #: Natural death rate of mosquitoes
    mu_v: float = 1. / 14.
    #: Breakthrough rate of mosquitoes from exposed to infectious
    alpha_v: float = 0.1
    #: Rate constant due to indoor residual spray
    theta: float = 0.75
    #: Rate constant due to treatment effort
    tau: float = 0.15

    # Treatment parameters
    #: Control measure through use of treated bednets
    treated_bednet_control: float = 0.
    #: Control measure through use of condoms
    condom_use_control: float = 0.
    #: Control measure through treatment of infected humans
    treatment_control: float = 0.
    #: Control measure through use of indoor residual spray
    indoor_spray_control: float = 0.

    #: Simulation start time (in day)
    start_time: float = 0
    #: Simulation end time (in days)
    end_time: float = 100
    #: How frequently to measure simulator state
    delta_t: float = 0.05
    #: solver relative tolerance
    rtol: float = 1e-6
    #: solver absolute tolerance
    atol: float = 1e-6


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the Zika simulator.

    The default state corresponds 
    """
    #: sh, ah, ih, rh, sv, ev, iv, nh, nv
    #: Number of susceptible humans
    susceptible_humans: float = 750.
    #: Number of asymptomatic infected human
    asymptomatic_humans: float = 250.
    #: Number of symptomatic infected humans
    symptomatic_humans: float = 10.
    #: Number of recovered humans
    recovered_humans: float = 20.
    #: Number of susceptible mosquitoes
    susceptible_humans: float = 10000.
    #: Number of exposed mosquitoes
    exposed_mosquitos: float = 500.
    #: Number of infectious mosquitoes
    infectious_mosquites: float = 100.
    #: Total number of human
    human_population: float = 1030.
    #: Total number of mosquitoes
    mosquito_population: float = 10600


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the Zika model.

    Examples
    --------
    >>> # Starting in step 100, set epsilon_1 to 0.7 (leaving other variables unchanged)
    >>> Intervention(time=100, epsilon_1=0.7)

    """

    def __init__(self, time=100, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention (days)
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def dynamics(state, time, config, intervention=None):
    """Update equations for the Zika simulaton.

    Parameters
    ----------
        state:  np.ndarray, list, or tuple
            State of the dynamics
        time:   float
        config: hiv.Config
            Simulator configuration object that determines the coefficients
        intervantion: hiv.Intervention
            Simulator intervention object that determines when/how to update the
            dynamics.

    Returns
    -------
        ds_dt: list
            Derivative of the dynamics with respect to time

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    # Keep notation roughly consistent with the original paper
    # pylint: disable-msg=invalid-name
    (
        S_h,  # susceptible humans
        A_h,  # asymptomatic infected humans
        I_h,  # symptomatic infected humans
        R_h,  # recovered humans
        S_v,  # susceptible mosquitoes
        E_v,  # exposed mosquitoes
        I_v,  # infected mosquitoes
        N_h,  # total number of humans
        N_v,  # total number of mosquitoes
    ) = state

    # mu_1, mu_2, mu_3, mu_4 (
    #treated_bednet_control: float = 0.
    ##: Control measure through use of condoms
    #condom_use_control: float = 0.
    ##: Control measure through treatment of infected humans
    #treatment_control: float = 0.
    ##: Control measure through use of indoor residual spray
    #indoor_spray_control: float = 0.
    
    # Suspectible humans 


    ds_dt = [
        dS_h,
        dA_h,
        dI_h,
        dR_h,
        dS_v,
        dE_v,
        dI_v,
        dN_h,
        dN_v,
    ]
    return ds_dt


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Zika simulator model.

    The simulation starts at initial_state at time 0, and evolves the state
    using dynamics whose parameters are specified in config.

    Parameters
    ----------
        initial_state:  `whynot.simulators.zika.State`
            Initial State object, which is used as x_{t_0} for the simulator.
        config:  `whynot.simulators.zika.Config`
            Config object that encapsulates the parameters that define the dynamics.
        intervention: `whynot.simulators.zika.Intervention`
            Intervention object that specifies what, if any, intervention to perform.
        seed: int
            Seed to set internal randomness. The simulator is deterministic, so
            the seed parameter is ignored.

    Returns
    -------
        run: `whynot.dynamics.Run`
            Rollout of the model.

    """
    # Simulator is deterministic, so seed is ignored
    # pylint: disable-msg=unused-argument
    t_eval = np.arange(
        config.start_time, config.end_time + config.delta_t, config.delta_t
    )

    solution = odeint(
        dynamics,
        y0=dataclasses.astuple(initial_state),
        t=t_eval,
        args=(config, intervention),
        rtol=config.rtol,
        atol=config.atol,
    )

    states = [initial_state] + [State(*state) for state in solution[1:]]
    return wn.dynamics.Run(states=states, times=t_eval)


if __name__ == "__main__":
    print(simulate(State(), Config()))