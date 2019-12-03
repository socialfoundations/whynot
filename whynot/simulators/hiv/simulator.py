"""Implementation of Adams et al.'s ODE simulator for HIV treatment.

Adams, Brian Michael, et al. Dynamic multidrug therapies for HIV: Optimal and
STI control approaches. North Carolina State University. Center for Research in
Scientific Computation, 2004. APA.

https://pdfs.semanticscholar.org/c030/127238b1dbad2263fba6b64b5dec7c3ffa20.pdf
"""

import dataclasses
import numpy as np

from scipy.integrate import odeint

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics.

    Examples
    --------
    # Run the simulation for 200 days with infected cell death rate 0.3
    hiv.Config(duration=200, delta=0.3)

    """

    # pylint: disable-msg=invalid-name
    # pylint: disable-msg=too-many-instance-attributes
    #: Target cell type 1 production (source) rate
    lambda_1: float = 10000.0
    #: Target cell type 2 production (source) rate
    lambda_2: float = 31.98
    #: Target cell type 1 death rate
    d_1: float = 0.01
    #: Target cell type 2 death rate
    d_2: float = 0.01
    #: Population 1 infection rate
    k_1: float = 8e-7
    #: Population 2 infection rate
    k_2: float = 1e-4
    #: Treatment efficacy reduction in population 2
    f: float = 0.34
    #: Infected cell death rate
    delta: float = 0.7
    #: Immune-induced clearance rate for population 1
    m_1: float = 1e-5
    #: Immune-induced clearance rate for population 2
    m_2: float = 1e-5
    #: Virions produced per infected cell
    N_T: float = 100
    #: Virus natural death rate
    c: float = 13
    #: Average number virions infecting a type 1 cell
    rho_1: float = 1
    #: Average number virions infecting a type 2 cell
    rho_2: float = 1
    #: Immune effector production (source) rate
    lambda_E: float = 1
    #: Maximum birth rate for immune effectors
    b_E: float = 0.3
    #: Saturation constant for immune effector birth
    K_B: float = 100
    #: Maximum death rate for immune effectors
    d_E: float = 0.25
    #: Saturation constant for immune effector death
    K_D: float = 500
    #: Natural death rate for immune effectors
    delta_E: float = 0.1

    # Treatment regimes-- only consider constant
    # treatments for now. The paper also references
    # a constant "fully efficacious" treatment with
    # e_1 = 0.7 and e_2 = 0.3.
    #: Drug efficacy
    epsilon_1: float = 0.0
    #: Efficacy of protease inhibitors
    epsilon_2: float = 0.0

    #: Simulation start time (in day)
    start_time: float = 0
    #: Simulation end time (in days)
    end_time: float = 400
    #: How frequently to measure simulator state
    delta_t: float = 0.05
    #: solver relative tolerance
    rtol: float = 1e-6
    #: solver absolute tolerance
    atol: float = 1e-6


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the HIV simulator.

    The default state corresponds to an early infection state, defined by Adams
    et al. The early infection state is designed based on an unstable uninfected
    steady state by 1) adding one virus particle per ml of blood plasma, and 2)
    adding low levels of infected T-cells.
    """

    # pylint: disable-msg=invalid-name
    #: Uninfected CD4+ T-lymphocytes (cells/ml)
    uninfected_T1: float = 1e6
    #: Infected CD4+ T-lymphocytes (cells/ml)
    infected_T1: float = 1e-4
    #: Uninfected macrophages (cells/ml)
    uninfected_T2: float = 3198
    #: Infected macrophages (cells/ml)
    infected_T2: float = 1e-4
    #: Free virus (copies/ml)
    free_virus: float = 1
    #: Immune response CTL E (cells/ml)
    immune_response: float = 10


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the HIV model.

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
    """Update equations for the HIV simulaton.

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

    # Keep notation roughly consistent with the Adams paper.
    # pylint: disable-msg=invalid-name
    (
        uninfected_T1,
        infected_T1,
        uninfected_T2,
        infected_T2,
        free_virus,
        immune_response,
    ) = state

    delta_uninfected_T1 = (
        config.lambda_1
        - config.d_1 * uninfected_T1
        - (1 - config.epsilon_1) * config.k_1 * free_virus * uninfected_T1
    )

    delta_infected_T1 = (
        (1 - config.epsilon_1) * config.k_1 * free_virus * uninfected_T1
        - config.delta * infected_T1
        - config.m_1 * immune_response * infected_T1
    )

    delta_uninfected_T2 = (
        config.lambda_2
        - config.d_2 * uninfected_T2
        - (1 - config.f * config.epsilon_1) * config.k_2 * free_virus * uninfected_T2
    )

    delta_infected_T2 = (
        (1 - config.f * config.epsilon_1) * config.k_2 * free_virus * uninfected_T2
        - config.delta * infected_T2
        - config.m_2 * immune_response * infected_T2
    )

    delta_virus = (
        (1 - config.epsilon_2) * config.N_T * config.delta * (infected_T1 + infected_T2)
        - config.c * free_virus
        - free_virus
        * (
            (1.0 - config.epsilon_1) * config.rho_1 * config.k_1 * uninfected_T1
            + (
                (1.0 - config.f * config.epsilon_1)
                * config.rho_2
                * config.k_2
                * uninfected_T2
            )
        )
    )

    delta_immune_response = (
        config.lambda_E
        + (
            (config.b_E * (infected_T1 + infected_T2))
            / (infected_T1 + infected_T2 + config.K_B)
        )
        * immune_response
        - (
            (config.d_E * (infected_T1 + infected_T2))
            / (infected_T1 + infected_T2 + config.K_D)
        )
        * immune_response
        - config.delta_E * immune_response
    )

    ds_dt = [
        delta_uninfected_T1,
        delta_infected_T1,
        delta_uninfected_T2,
        delta_infected_T2,
        delta_virus,
        delta_immune_response,
    ]
    return ds_dt


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Adams HIV simulator model.

    The simulation starts at initial_state at time 0, and evolves the state
    using dynamics whose parameters are specified in config.

    Parameters
    ----------
        initial_state:  `whynot.simulators.hiv.State`
            Initial State object, which is used as x_{t_0} for the simulator.
        config:  `whynot.simulators.hiv.Config`
            Config object that encapsulates the parameters that define the dynamics.
        intervention: `whynot.simulators.hiv.Intervention`
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
