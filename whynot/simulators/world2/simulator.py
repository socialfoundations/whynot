"""World 2 dynamics simulator."""
import copy
import dataclasses

import numpy as np

import whynot as wn
from whynot.simulators.infrastructure import BaseConfig, BaseState, BaseIntervention
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
    #: Time (in years) elapsed on each update of the discrete dynamics (forward euler).
    delta_t: float = 0.2


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
    #: (Derived) quality of life metric
    quality_of_life: float = 1.0


#################
# Dynamics
#################
class WorldDynamics():
    """Class encapsulating world2 dynamics."""

    def __init__(self, config, intervention, initial_state):
        """Initialize world2 dynamics at initial_state with config."""
        self._state = copy.deepcopy(initial_state)
        self.initial_natural_resources = initial_state.natural_resources
        self.intervention = intervention

        config = copy.deepcopy(config)
        self.preintervention_config = config
        if intervention:
            self.postintervention_config = config.update(intervention)
        else:
            self.postintervention_config = config
        self.config = config

    def __str__(self):
        """Print all of the state variables."""
        template = "Population: {}\nNatural Resources: {}\nCapital Investment: {}\n"
        template += "Pollution: {}\nFraction Capital Investment in Agriculture: {}\n"

        return template.format(*[f.name for f in dataclasses.fields(self._state)])

    @property
    def state(self):
        """Return a copy of the current world2 state."""
        return copy.deepcopy(self._state)

    @property
    def capital_investment_ratio(self):
        """Return current capital investment ratio."""
        return self._state.capital_investment / self._state.population

    @property
    def crowding_ratio(self):
        """Return current crowding ratio."""
        return self._state.population / (self.config.land_area * self.config.population_density)

    @property
    def pollution_ratio(self):
        """Return current pollution ratio."""
        return self._state.pollution / self.config.pollution_standard

    @property
    def food_ratio(self):
        """Return current food ratio."""
        capital_investment_ratio_in_agr = (self.capital_investment_ratio
                                           * self._state.capital_investment_in_agriculture
                                           / self.config.capital_investment_agriculture)

        pollution_multiplier = tables.FOOD_FROM_POLLUTION[self.pollution_ratio]
        crowding_multiplier = tables.FOOD_FROM_CROWDING[self.crowding_ratio]
        capital_investment_food_potential = \
            tables.FOOD_POTENTIAL_FROM_CAPITAL_INVESTMENT[capital_investment_ratio_in_agr]

        food_ratio = (capital_investment_food_potential
                      * crowding_multiplier
                      * pollution_multiplier
                      * self.config.food_coefficient
                      / self.config.food_normal)

        return food_ratio

    @property
    def standard_of_living(self):
        """Return the current standard of living."""
        # Natural resources
        frac_natural_resources_remaining = (self._state.natural_resources /
                                            self.initial_natural_resources)
        natural_resources_extraction_multiplier = \
            tables.NATURAL_RESOURCE_EXTRACTION[frac_natural_resources_remaining]

        effective_capital_investment_ratio = (
            self.capital_investment_ratio
            * natural_resources_extraction_multiplier
            * (1.0 - self._state.capital_investment_in_agriculture)
            / (1.0 - self.config.capital_investment_agriculture))

        return effective_capital_investment_ratio / self.config.effective_capital_investment_ratio

    @property
    def death_rate_per_year(self):
        """Return the fraction of people who die in the current year."""
        material_multiplier = tables.DEATH_RATE_FROM_MATERIAL[self.standard_of_living]
        pollution_multiplier = tables.DEATH_RATE_FROM_POLLUTION[self.pollution_ratio]
        food_multiplier = tables.DEATH_RATE_FROM_FOOD[self.food_ratio]
        crowding_multiplier = tables.DEATH_RATE_FROM_CROWDING[self.crowding_ratio]

        death_rate_per_year = (
            self.config.death_rate
            * material_multiplier
            * pollution_multiplier
            * food_multiplier
            * crowding_multiplier)

        return death_rate_per_year

    @property
    def birth_rate_per_year(self):
        """Return the fraction of people born in the current year."""
        material_multiplier = tables.BIRTH_RATE_FROM_MATERIAL[self.standard_of_living]
        pollution_multiplier = tables.BIRTH_RATE_FROM_POLLUTION[self.pollution_ratio]
        food_multiplier = tables.BIRTH_RATE_FROM_FOOD[self.food_ratio]
        crowding_multiplier = tables.BIRTH_RATE_FROM_CROWDING[self.crowding_ratio]

        birth_rate_per_year = (
            self.config.birth_rate
            * material_multiplier
            * pollution_multiplier
            * food_multiplier
            * crowding_multiplier)

        return birth_rate_per_year

    @property
    def natural_resources_usage_rate(self):
        """Return the rate of natural resource usage at the current time."""
        material_multiplier = tables.NATURAL_RESOURCES_FROM_MATERIAL[self.standard_of_living]

        natural_resources_usage_rate = (
            self._state.population
            * self.config.natural_resources_usage
            * material_multiplier)

        return natural_resources_usage_rate

    @property
    def capital_investment_rate(self):
        """Return the rate of capital investment at the current time."""
        material_multiplier = tables.CAPITAL_INVESTMENT_MULTIPLIER_TABLE[self.standard_of_living]

        capital_investment_generation = (
            self._state.population
            * material_multiplier
            * self.config.capital_investment_generation)

        capital_investment_discard = (
            self._state.capital_investment
            * self.config.capital_investment_discard)

        return capital_investment_generation - capital_investment_discard

    @property
    def pollution_rate(self):
        """Return the current rate of pollution."""
        capital_multiplier = tables.POLLUTION_FROM_CAPITAL[self.capital_investment_ratio]

        pollution_generation = (
            self._state.population
            * capital_multiplier
            * self.config.pollution)

        absorption_time = tables.POLLUTION_ABSORPTION_TIME_TABLE[self.pollution_ratio]
        pollution_absorption = self._state.pollution / absorption_time

        return pollution_generation - pollution_absorption

    @property
    def quality_of_life(self):
        """Return the current quality of life meta-statistic."""
        material_multiplier = tables.QUALITY_OF_LIFE_FROM_MATERIAL[self.standard_of_living]
        crowding_multiplier = tables.QUALITY_OF_LIFE_FROM_CROWDING[self.crowding_ratio]
        food_multiplier = tables.QUALITY_OF_LIFE_FROM_FOOD[self.food_ratio]
        pollution_multiplier = tables.QUALITY_OF_LIFE_FROM_POLLUTION[self.pollution_ratio]

        quality_of_life = (
            self.config.quality_of_life_standard
            * material_multiplier
            * crowding_multiplier
            * food_multiplier
            * pollution_multiplier)

        return quality_of_life

    def capital_invest_agr_frac_delta(self, delta_t):
        """Return the change in the fraction of captial invested in agriculture."""
        capital_fraction_indicated_by_food_ratio =\
            tables.CAPITAL_FRACTION_INDICATE_BY_FOOD_RATIO_TABLE[self.food_ratio]

        quality_of_life_material = tables.QUALITY_OF_LIFE_FROM_MATERIAL[self.standard_of_living]
        quality_of_life_food = tables.QUALITY_OF_LIFE_FROM_FOOD[self.food_ratio]

        life_quality_ratio = quality_of_life_material / quality_of_life_food
        capital_investment_from_quality_ratio = \
            tables.CAPITAL_INVESTMENT_FROM_QUALITY[life_quality_ratio]

        delta = (delta_t / self.config.capital_investment_in_agriculture_frac_adj_time) \
            * (capital_fraction_indicated_by_food_ratio
               * capital_investment_from_quality_ratio
               - self._state.capital_investment_in_agriculture)

        return delta

    def step(self, time, delta_t):
        """Advance the simulation forward delta_t steps, starting at the given time."""
        if self.intervention and time >= self.intervention.time:
            self.config = self.postintervention_config
        else:
            self.config = self.preintervention_config

        # Population
        deaths = self.death_rate_per_year * self._state.population * delta_t
        births = self.birth_rate_per_year * self._state.population * delta_t
        population_delta = births - deaths

        # Natural resources (negative since this is usage)
        natural_resources_delta = -self.natural_resources_usage_rate * delta_t

        # Capital_investment
        capital_investment_delta = self.capital_investment_rate * delta_t

        # Pollution
        pollution_delta = self.pollution_rate * delta_t

        # Investment in agriculture
        capital_investment_in_agr_frac_delta = self.capital_invest_agr_frac_delta(delta_t)

        # Update the state variables. Gather deltas first to ensure the updates
        # are atomic, e.g. since changing the population will also change the
        # resources delta.
        self._state.population += population_delta
        self._state.natural_resources += natural_resources_delta
        self._state.capital_investment += capital_investment_delta
        self._state.pollution += pollution_delta
        self._state.capital_investment_in_agriculture += capital_investment_in_agr_frac_delta
        self._state.quality_of_life = self.quality_of_life


def simulate(initial_state, config, intervention=None, seed=None):
    """Run a simulation of the world2 dynamics from the given initial state.

    The system of ODEs is solved using the forward-euler method.

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
        run: whynot.framework.Run
            Sequence of states and measurement times produced by the simulator.

    """
    # pylint: disable=unused-argument
    world = WorldDynamics(config, intervention, initial_state)

    states = [initial_state]
    times = np.arange(config.start_time, config.end_time + config.delta_t, config.delta_t)
    for time in times[:-1]:
        world.step(time, config.delta_t)
        states.append(world.state)
    return wn.framework.Run(states=states, times=times)
