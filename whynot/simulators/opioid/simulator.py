"""Implementation of Chen et al. systems dynamic model for opioid abuse.

The model implementation and parameterization is taken from:
    Chen Q, Larochelle MR, Weaver DT, et al. Prevention of Prescription Opioid
    Misuse and Projected Overdose Deaths in the United States. JAMA Network
    Open. 2019;2(2):e187621. doi:10.1001/jamanetworkopen.2018.7621
"""
import copy
import dataclasses

from scipy.integrate import odeint

import whynot as wn
import whynot.traceable_numpy as np
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState


class TimeVaryingParam:  # pylint: disable=too-few-public-methods
    """Encapsulate a time-varying parameter obtained from a joinpoint model."""

    def __init__(
        self,
        baseline,
        alpha_1,
        joinpoint=None,
        alpha_2=None,
        intervention_year=None,
        intervention_val=0.0,
        start_year=2002,
        stabilize_year=None,
    ):
        """Specify a joinpoint model for a time-varying parameter.

        Parameters
        ----------
            baseline: float
                Baseline value at initial time point
            alpha_1: float
                Annual percentage change (APC) before joinpoint
            joinpoint: float
                (Optional) time indicating signifcant change in APC
            alpha_2: float
                (Optional) Annual percentage change (APC) after joinpoint
            intervention_year: float
                (Optional) Year to intervene on the APC
            intervention_val: float
                (Optional) New value for the APC after intevention. Default 0 is
                no change.
            start_year: float
                (Optional) initial time point
            stabilize_year: float
                (Optional) Year (after 2015) to set APC decrease to 0.

        """
        if joinpoint is None and alpha_2 is not None:
            raise ValueError("Must specify joinpoint to use alpha_2")

        if stabilize_year is not None and stabilize_year <= 2015:
            raise ValueError("stabilize_year must be after 2015")

        if intervention_year is not None:
            if stabilize_year is not None:
                raise ValueError(
                    "Cannot use stabilization and intervention simultaneously"
                )
            if joinpoint is not None and intervention_year < joinpoint:
                raise ValueError("Cannot intervene before joinpoint")

        self.baseline = baseline
        self.start_year = start_year
        self.alpha_1 = alpha_1
        self.joinpoint = joinpoint
        self.alpha_2 = alpha_2
        self.stabilize_year = stabilize_year
        self.intervention_year = intervention_year
        self.intervention_val = intervention_val

    def get_boxable(self):
        """Return the underlying parameter that can be boxed for dependency tracing."""
        return self.baseline

    def set_boxable(self, box):
        """Set the underlying parameter as a box for dependency tracing."""
        self.baseline = box

    def _unstabilized_value(self, time):
        """Return value of parameter without taking stabilization into account."""
        # pylint:disable-msg=no-member
        running_value = self.baseline
        if self.joinpoint is None or time <= self.joinpoint:
            return running_value * np.power(1.0 + self.alpha_1, time - self.start_year)

        running_value *= np.power(1.0 + self.alpha_1, self.joinpoint - self.start_year)

        if self.intervention_year is None or time <= self.intervention_year:
            return running_value * np.power(1.0 + self.alpha_2, time - self.joinpoint)

        running_value *= np.power(
            1.0 + self.alpha_2, self.intervention_year - self.joinpoint
        )

        return running_value * np.power(
            1.0 + self.intervention_val, time - self.intervention_year
        )

    def _current_rate(self, time):
        """Return current APC."""
        if self.joinpoint is None or time <= self.joinpoint:
            return self.alpha_1
        if self.intervention_year is None or time <= self.intervention_year:
            return self.alpha_2
        return self.intervention_val

    def __getitem__(self, time):
        """Retrieve the parameter value at time t."""
        if self.stabilize_year is None or time < 2015:
            return self._unstabilized_value(time)

        running_val = self._unstabilized_value(2015)
        rate = self._current_rate(time)

        # Decrease at a constant rate until stabilize year
        decay_per_year = rate / (self.stabilize_year - 2015)
        for year in range(2016, self.stabilize_year + 1):
            rate -= decay_per_year
            if time <= year:
                break
            running_val *= 1.0 + rate

        # Handle fractional values
        if time <= self.stabilize_year:
            # pylint:disable-msg=no-member
            remainder = time - np.floor(time)
            running_val *= (1.0 + rate) ** remainder

        return running_val


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of the dynamics for opioid epidemic simulator.

    By default, the confiruation uses the mean parameter values specified in
    eTable 4 and the pessimistic-case scenario (overdose mortality and incidence
    for illicit use stabilizes in 2025) and considers the scenario with no
    improvement in incidence after 2015.

    Examples
    --------
    >>> # Standard run of the dynamics with slightly rate of overdose
    >>> # deaths from nonmedial opioids.
    >>> opioid.Config(start_time=2002, end_time=2030, nonmedical_overdose=0.002)

    """

    # Dynamics parameters
    #: Annual incidence of nonmedical prescription opioid use.
    nonmedical_incidence: TimeVaryingParam = TimeVaryingParam(
        baseline=2496835, alpha_1=0.0016, joinpoint=2008, alpha_2=-0.0747
    )

    #: Annual incidence of illicit opioid use from sources other than prescription opioids.
    illicit_incidence: TimeVaryingParam = TimeVaryingParam(
        baseline=13489, alpha_1=0.235, stabilize_year=2025
    )

    #: Annual transition rate of non-medical opioid use to opioid use disorder.
    nonmedical_to_oud: float = 0.06

    #: Annual transition rate of users with opioid user disorder to illicit opioid users.
    oud_to_illicit: TimeVaryingParam = TimeVaryingParam(baseline=0.071, alpha_1=0.042)

    #: Annual transition rate of nonmedial to illicit opioid users
    nonmedical_to_illicit: float = 0.002

    #: Overdose mortality rate for nonmedial prescription opioid use.
    nonmedical_overdose: float = 0.001

    # Decrease baseline slightly from paper to match the reported
    # calibration targets.
    #: Overdose mortality rate for opioid use disorder.
    oud_overdose: TimeVaryingParam = TimeVaryingParam(
        baseline=0.003, alpha_1=0.224, joinpoint=2007, alpha_2=0.03
    )
    #: Overdose mortality rate for illicit opioid use.
    illicit_overdose: TimeVaryingParam = TimeVaryingParam(
        baseline=0.005,
        alpha_1=0.011,
        joinpoint=2011,
        alpha_2=0.356,
        stabilize_year=2025,
    )
    #: "Exit rate" from nonmedical users group. (Either stop using or die from non-opioid causes).
    nonmedical_exit: float = 0.177
    #: "Exit rate" from opioid user disorder group.
    oud_exit: TimeVaryingParam = TimeVaryingParam(baseline=0.311, alpha_1=-0.021)
    #: "Exit rate" of illicit opioid users.
    illicit_exit: float = 0.319

    # Simulator configuration
    #: Time to start the simulation (in years)
    start_time: float = 2002
    #: Time to end the simulation (in years)
    end_time: float = 2030
    #: Resolution of the recorded simulator output (in years)
    delta_t: float = 0.1

    def update(self, intervention):
        """Return a new config after applying intervention."""
        config = copy.deepcopy(self)
        if "nonmedical_incidence" in intervention.updates:
            nonmedical_incidence = intervention.updates["nonmedical_incidence"]
            config.nonmedical_incidence.intervention_year = intervention.time
            config.nonmedical_incidence.intervention_val = nonmedical_incidence
        if "illicit_incidence" in intervention.updates:
            illicit_incidence = intervention.updates["illicit_incidence"]
            config.illicit_incidence.intervention_year = intervention.time
            config.illicit_incidence.intervention_val = illicit_incidence
        return config


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Intervention parameters for opioid epidemic simulator.

    Currently, the simulator only supports interventions to reduce
    nonmedical and illicit opioid use because these are the only interventions
    considered in the originl Chen et al. paper.

    Examples
    --------
    >>> # Intervene in 2017 to reduce nonmedical incidence by 5%
    >>> opioid.Intervention(nonmedical_incidence=-0.05, time=2017)

    """

    def __init__(self, time=2015, nonmedical_incidence=None, illicit_incidence=None):
        """Construct an intervention object.

        Parameters
        ----------
            time: float
                time (in years) to intervene in the simulator.
            nonmedical_incidence: float
                Percent change in nonmedial opioid use (range -1. to 1.)
            illicit_incidence: float
                Percent change in illicit opioid use (range -1. to 1.)

        """
        kwargs = {}
        if nonmedical_incidence is not None:
            kwargs["nonmedical_incidence"] = nonmedical_incidence
        if illicit_incidence is not None:
            kwargs["illicit_incidence"] = illicit_incidence

        super(Intervention, self).__init__(Config, time, **kwargs)


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the opioid simulator.

    Default values correspond to the mean initial state used in the Chen paper.
    """

    #: Non-medical users of prescription opioids
    nonmedical_users: float = 10029859
    #: Non-medical users of prescription opioid with prescription opioid use disorder.
    oud_users: float = 1369218
    #: Illicit opioid users (e.g. users of heroin/fentanyl as opposed to presciption opioids)
    illicit_users: float = 328371


def compute_overdose_deaths(run, start_year, end_year, config, intervention):
    """Compute number of opioid overdose deaths from start_year to end_year.

    Parameters
    ----------
        run: whynot.dynamics.Run
            Run object produced by running simulate for the opioid simulator
        start_year: int
            Year to start counting overdose deaths
        end_year: int
            Final year included in overdose death count
        config: whynot.simulator.opioid.Config
            Configuration used to generate `run`
        intervention: whynot.simulator.opioid.Intervention
            Intervention object to generate `run` when calling `simulate`

    Returns
    -------
        deaths: int
            Total number of overdose deaths across nonmedical users, users with
            OUD, and illicit opioid users.

    """
    if intervention:
        config = config.update(intervention)

    deaths = 0.0
    for outcome_year in range(start_year, end_year + 1):
        state = run[outcome_year]
        nonmedical_deaths = state.nonmedical_users * config.nonmedical_overdose
        oud_deaths = state.oud_users * config.oud_overdose[outcome_year]
        illicit_deaths = state.illicit_users * config.illicit_overdose[outcome_year]
        deaths += nonmedical_deaths + oud_deaths + illicit_deaths
    return deaths


def compute_illicit_deaths(run, outcome_year, config, intervention):
    """Compute number of overdose deaths for illicit use in year.

    Parameters
    ----------
        run: whynot.dynamics.Run
            Run object produced by running simulate for the opioid simulator
        outcome_year: int
            Year to measure illicit opioid use overdose deaths
        config: whynot.simulator.opioid.Config
            Configuration used to generate `run` when calling `simulate`
        intervention: whynot.simulator.opioid.Intervention
            Intervention object to generate `run` when calling `simulate`

    Returns
    -------
        deaths: int
            Number of overdose deaths for illicit opioid use in outcome_year.

    """
    if intervention:
        config = config.update(intervention)

    return run[outcome_year].illicit_users * config.illicit_overdose[outcome_year]


def dynamics(state, time, config, intervention=None):
    """State dynamics for the Chen et al. opioid epidemic model.

    Parameters
    ----------
        state: np.ndarray or tuple or list
            Instantenous system state
        time: float
            Time to evaluate the derivative
        config: whynot.simulators.opioid.Config
            Configuration parameters for the dynamics
        intervention: whynot.simulators.opioid.Intervention
            (Optional) Parameters specifying when/how to update the dynamics

    Returns
    -------
        ds_dt: np.ndarray
            Derivative of the state with respect to time, evaluated at state and time.

    """
    # In the opioid simulator, if there is an intervention,
    # it is applied for every time step of the dynamics (since
    # interventions are specified on TimeVaryingParams which internally track
    # the year).
    if intervention:
        config = config.update(intervention)

    nonmedical_users, oud_users, illicit_users = state

    # Update non-medical opioid use cases
    nonmedical_increase = config.nonmedical_incidence[time]
    nonmedical_decrease = nonmedical_users * (
        config.nonmedical_exit
        + config.nonmedical_overdose
        + config.nonmedical_to_oud
        + config.nonmedical_to_illicit
    )

    nonmedical_delta = nonmedical_increase - nonmedical_decrease

    # Update opioid use disorder cases
    oud_increase = config.nonmedical_to_oud * nonmedical_users
    oud_decrease = oud_users * (
        config.oud_exit[time] + config.oud_overdose[time] + config.oud_to_illicit[time]
    )
    oud_delta = oud_increase - oud_decrease

    # Update illicit opiate use cases
    illicit_increase = config.illicit_incidence[time] + (
        config.oud_to_illicit[time] * oud_users
        + config.nonmedical_to_illicit * nonmedical_users
    )

    illicit_decrease = illicit_users * (
        config.illicit_exit + config.illicit_overdose[time]
    )

    illicit_delta = illicit_increase - illicit_decrease

    return [nonmedical_delta, oud_delta, illicit_delta]


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Chen et al. model opioid epidemic model.

    Parameters
    ----------
        initial_state: whynot.simulator.opioid.State
            Initial state of the dynamics
        config: whynot.simulator.opioid.Config
            Config object to determine simulation dynamics.
        intervention: whynot.simulator.opioid.Intervention
            (Optional) Intervention object to determine what, if any,
            intervention to perform during the rollout of the dynamics.
        seed: int
            The simulator is deterministic, so the seed parameter is ignored.

    Returns
    -------
        run: whynot.dynamics.Run
            Run object produced by running simulate for the opioid simulator

    """
    # pylint: disable-msg=unused-argument
    # pylint:disable-msg=no-member
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
