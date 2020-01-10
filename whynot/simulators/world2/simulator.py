"""World 2 dynamics simulator."""

import dataclasses
import numpy as np

from scipy.integrate import odeint

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention
from whynot.simulators.world2 import tables


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Dynamics and simulation parameters for World2.

    Default values correspond to the standard run of World2.

    Examples
    --------
    >>> # Set the initial birth rate to 0.05
    >>> world2.config(birth_rate=0.05)

    """

    # The #: comments allow Sphinx to autodocument these parameters.
    #:
    birth_rate: float = 0.04
    #:
    death_rate: float = 0.028
    #:
    effective_capital_investment_ratio: float = 1.0
    #:
    natural_resources_usage: float = 1.0
    #:
    land_area: float = 135e6
    #:
    population_density: float = 26.5
    #:
    food_coefficient: float = 1.0
    #:
    food_normal: float = 1.0
    #:
    capital_investment_agriculture: float = 0.3
    #:
    capital_investment_generation: float = 0.05
    #:
    capital_investment_discard: float = 0.025
    #:
    pollution_standard: float = 3.6e9
    #:
    pollution: float = 1.0
    #:
    intervention_time: float = 1970
    #:
    capital_investment_in_agriculture_frac_adj_time: float = 15.0
    #:
    quality_of_life_standard: float = 1.0

    #: Time to initialize the simulation (in years).
    start_time: float = 1900
    #: Time to end the simulation (in years).
    end_time: float = 2100
    #: Time (in years) elapsed on each update of the discrete dynamics.
    delta_t: float = 0.2
    #: solver relative tolerance
    rtol: float = 1e-6
    #: solver absolute tolerance
    atol: float = 1e-6


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Class to specify interventions in World2 dynamics.

    An intervention is a subset of the configuration variables,
    and only variables passed into the constructor are modified.

    Examples
    --------
    >>> # Set the death rate to 0.06 in 1980. All other parameters unchanged.
    >>> world2.Intervention(time=1980, death_rate=0.06)

    """

    def __init__(self, time=1970, **kwargs):
        """Construct an intervention object.

        Parameters
        ----------
            time: float
                time (in years) to intervene in the simulator.
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """World2 state. Defaults correspond to the initial state in the standard run of World2."""

    #: Total population stock
    population: float = 1.65e9
    #: Stock of natural resources
    natural_resources: float = 3.6e9 * 250
    #: Total capital investment stock
    capital_investment: float = 0.4e9
    #: Pollution stock
    pollution: float = 0.2e9
    #: Fraction of capital investment in agriculture
    capital_investment_in_agriculture: float = 0.2
    #: Initial natural resources
    initial_natural_resources: float = 3.6e9 * 250


def world2_intermediate_variables(state, config):
    """Update equations for the world2 simulaton.

    Parameters
    ----------
        state:  np.ndarray, list, or tuple
            State of the dynamics
        config: world2.Config
            Simulator configuration object that determines the coefficients

    Returns
    -------
        intermediate_variables: list
            Intermediate variables needed for world2 simulation.

    """
    (
        population,
        natural_resources,
        capital_investment,
        pollution,
        capital_investment_in_agriculture,
        initial_natural_resources,
    ) = state

    capital_investment_ratio = capital_investment / population
    crowding_ratio = population / (config.land_area * config.population_density)
    pollution_ratio = pollution / config.pollution_standard
    food_ratio = (
        tables.FOOD_POTENTIAL_FROM_CAPITAL_INVESTMENT[
            (
                capital_investment_ratio
                * capital_investment_in_agriculture
                / config.capital_investment_agriculture
            )  # capital investment ratio in agriculture
        ]  # capital investment food potential
        * tables.FOOD_FROM_CROWDING[crowding_ratio]  # crowding multiplier
        * tables.FOOD_FROM_POLLUTION[pollution_ratio]  # pollution multiplier
        * config.food_coefficient
        / config.food_normal
    )
    standard_of_living = (
        (
            capital_investment_ratio
            * tables.NATURAL_RESOURCE_EXTRACTION[
                natural_resources
                / initial_natural_resources  # fraction of natural resources remaining
            ]  # natural resources extraction multiplier
            * (1.0 - capital_investment_in_agriculture)
            / (1.0 - config.capital_investment_agriculture)
        )  # effective capital investment ratio
        / config.effective_capital_investment_ratio
    )
    death_rate_per_year = (
        config.death_rate
        * tables.DEATH_RATE_FROM_MATERIAL[standard_of_living]  # material multiplier
        * tables.DEATH_RATE_FROM_POLLUTION[pollution_ratio]  # pollution multiplier
        * tables.DEATH_RATE_FROM_FOOD[food_ratio]  # food multiplier
        * tables.DEATH_RATE_FROM_CROWDING[crowding_ratio]  # crowding multiplier
    )
    birth_rate_per_year = (
        config.birth_rate
        * tables.BIRTH_RATE_FROM_MATERIAL[standard_of_living]  # material multiplier
        * tables.BIRTH_RATE_FROM_POLLUTION[pollution_ratio]  # pollution multiplier
        * tables.BIRTH_RATE_FROM_FOOD[food_ratio]  # food multiplier
        * tables.BIRTH_RATE_FROM_CROWDING[crowding_ratio]  # crowding multiplier
    )
    natural_resources_usage_rate = (
        population
        * config.natural_resources_usage
        * tables.NATURAL_RESOURCES_FROM_MATERIAL[
            standard_of_living
        ]  # material multiplier
    )
    capital_investment_rate = (
        population
        * tables.CAPITAL_INVESTMENT_MULTIPLIER_TABLE[
            standard_of_living
        ]  # material multiplier
        * config.capital_investment_generation
    ) - capital_investment * config.capital_investment_discard
    pollution_rate = (  # pollution generation
        population
        * tables.POLLUTION_FROM_CAPITAL[capital_investment_ratio]  # capital multiplier
        * config.pollution
    ) - (  # pollution absorption
        pollution
        / tables.POLLUTION_ABSORPTION_TIME_TABLE[pollution_ratio]  # absorption time
    )

    intermediate_variables = [
        capital_investment_ratio,
        crowding_ratio,
        pollution_ratio,
        food_ratio,
        standard_of_living,
        death_rate_per_year,
        birth_rate_per_year,
        natural_resources_usage_rate,
        capital_investment_rate,
        pollution_rate,
    ]
    return intermediate_variables


def dynamics(state, time, config, intervention=None):
    """Update equations for the world2 simulaton.

    Parameters
    ----------
        state:  np.ndarray, list, or tuple
            State of the dynamics
        time:   float
        config: world2.Config
            Simulator configuration object that determines the coefficients
        intervention: world2.Intervention
            Simulator intervention object that determines when/how to update the
            dynamics.

    Returns
    -------
        ds_dt: list
            Derivative of the dynamics with respect to time

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    (
        population,
        natural_resources,
        capital_investment,
        pollution,
        capital_investment_in_agriculture,
        initial_natural_resources,
    ) = state

    (
        capital_investment_ratio,
        crowding_ratio,
        pollution_ratio,
        food_ratio,
        standard_of_living,
        death_rate_per_year,
        birth_rate_per_year,
        natural_resources_usage_rate,
        capital_investment_rate,
        pollution_rate,
    ) = world2_intermediate_variables(state, config)

    # Population
    delta_population = (birth_rate_per_year - death_rate_per_year) * population

    # Natural resources (negative since this is usage)
    delta_natural_resources = -natural_resources_usage_rate

    # Capital_investment
    delta_capital_investment = capital_investment_rate

    # Pollution
    delta_pollution = pollution_rate

    # Investment in agriculture
    delta_capital_investment_in_agriculture = (
        tables.CAPITAL_FRACTION_INDICATE_BY_FOOD_RATIO_TABLE[food_ratio]
        * tables.CAPITAL_INVESTMENT_FROM_QUALITY[
            (
                tables.QUALITY_OF_LIFE_FROM_MATERIAL[standard_of_living]
                / tables.QUALITY_OF_LIFE_FROM_FOOD[food_ratio]
            )  # life quality ratio
        ]  # capital investment from quality ratio
        - capital_investment_in_agriculture
    ) / config.capital_investment_in_agriculture_frac_adj_time

    ds_dt = [
        delta_population,
        delta_natural_resources,
        delta_capital_investment,
        delta_pollution,
        delta_capital_investment_in_agriculture,
        1,  # initial_natural_resources does not change
    ]
    return ds_dt


def simulate(initial_state, config, intervention=None, seed=None):
    """Run a simulation of the world2 dynamics from the given initial state.

    Parameters
    ----------
        initial_state: whynot.simulators.world2.State
            State object to initialize the simulation
        config: whynot.simulators.world2.Config
            Configuration object to determine coefficients of the dynamics.
        intervention: whynot.simulators.world2.Intervention
            (Optional) Specify what, if any, intervention to perform during execution.
        seed: int
            (Optional) The simulator is deterministic, and the seed parameter is ignored.

    Returns
    -------
        run: whynot.dynamics.Run
            Rollout sequence of states and measurement times produced by the simulator.

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


def quality_of_life(state, time, config, intervention=None):
    """Get the world2 quality of life metric derived from a state.

    Parameters
    ----------
        state:  whynot.simulators.world2.State or iterable representing a state.
            State of the dynamics
        time:   float
        config: world2.Config
            Simulator configuration object that determines the coefficients
        intervention: world2.Intervention
            Simulator intervention object that determines when/how to update the
            dynamics.

    Returns
    -------
        qol: float
            Quality of life metric, computed according to the world2 simulation.

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    if type(state) == wn.simulators.world2.State:
        state = (
            state.population,
            state.natural_resources,
            state.capital_investment,
            state.pollution,
            state.capital_investment_in_agriculture,
            state.initial_natural_resources,
        )

    (
        _,
        crowding_ratio,
        pollution_ratio,
        food_ratio,
        standard_of_living,
        _,
        _,
        _,
        _,
        _,
    ) = world2_intermediate_variables(state, config)

    qol = (
        config.quality_of_life_standard
        * tables.QUALITY_OF_LIFE_FROM_MATERIAL[
            standard_of_living
        ]  # material multiplier
        * tables.QUALITY_OF_LIFE_FROM_CROWDING[crowding_ratio]  # crowding multiplier
        * tables.QUALITY_OF_LIFE_FROM_FOOD[food_ratio]  # food multiplier
        * tables.QUALITY_OF_LIFE_FROM_POLLUTION[pollution_ratio]  # pollution multiplier
    )

    return qol
