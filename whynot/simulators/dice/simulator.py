"""Simulator for Nordhaus' 2007 DICE model."""
from contextlib import contextmanager
import copy
import dataclasses
import os
import sys

import numpy as np
import pyomo
from pyomo.environ import (
    RangeSet,
    ConcreteModel,
    NonNegativeReals,
    Reals,
    Var,
    Constraint,
    Objective,
    summation,
    log,
    maximize,
)

import whynot as wn
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState


@dataclasses.dataclass
class Config(BaseConfig):
    """Parameter values in the DICE model.

    Default values correspond to the base run of the 2013 version.
    """

    # pylint:disable-msg=too-many-public-methods,invalid-name
    # pylint:disable-msg=too-many-instance-attributes
    #: Number of time periods to run the simulation.
    numPeriods: int = 60
    #: Number of year for each time period.
    tstep: int = 5

    # Preferences
    #: Elasticity of marginal utility of consumption.
    elasmu: float = 1.45
    #: Initial rate of social time preference per year.
    prstp: float = 0.015

    # Population and technology
    #: Capital elasticity in production function.
    gama: float = 0.3
    #: Initial world population (millions)
    pop0: float = 6838
    #: Growth rate to calibrate 2050 population projection.
    popadj: float = 0.134
    #: Asymptotic population (millions)
    popasym: float = 10500
    #: Depreciation rate on capital (per year)
    dk: float = 0.1
    #: Initial world gross output (trill 2005 USD)
    q0: float = 63.69
    #: Initial capital value (trill 2005 USD)
    k0: float = 135
    #: Initial level of total factor productivity
    a0: float = 3.8
    #: Initial growth rate for total factor productivity (per 5 years)
    ga0: float = 0.079
    #: Decline rate of total factor productivity (per 5 years)
    dela: float = 0.006

    # Emissions
    #: Initial growth rate of sigma (per year)
    gsigma1: float = -0.01
    #: Decline rate of decarbonization (per period)
    dsig: float = -0.001
    #: Carbon emissions from land 2010 (GtCO2 per year)
    eland0: float = 3.3
    #: Decline rate of land emissions (per period)
    deland: float = 0.2
    #: Industrial emissions 2010 (GtC02 per year)
    e0: float = 33.61
    #: Initial emissions control rate for base case 2010
    miu0: float = 0.039

    # Carbon cycle
    #: Initial concentration in atmosphere 2010 (GtC)
    mat0: float = 830.4
    #: Initial concentration in upper strata 2010 (GtC)
    mu0: float = 1527
    #: Initial concentration in lower strata 2010 (GtC)
    ml0: float = 10010
    #: Equilibrium concentration atmosphere (GtC)
    mateq: float = 588
    #: Equilibrium concentration in upper strata (GtC)
    mueq: float = 1350
    #: Equilibrium concentration in lower strata (GtC)
    mleq: float = 10000
    #: Carbon cycle transition matrix.
    b12: float = 0.088
    #: Carbon cycle transition matrix.
    b23: float = 0.0025

    # Climate model
    #: Equilibrium temperature impact (oC per doubling CO2)
    t2xco2: float = 2.9
    #: 2010 forcings of non-CO2 CHG (Wm-2)
    fex0: float = 0.25
    #: 2100 forcings of non-CO2 CHG (Wm-2)
    fex1: float = 0.7
    #: Initial lower stratum temperature change (C from 1900)
    tocean0: float = 0.0068
    #: Initial atmospheric temperature change (C from 1900)
    tatm0: float = 0.8
    #: Climate equation coefficient for upper level
    c1: float = 0.098
    #: Tranfer coefficient upper to lower stratum
    c3: float = 0.088
    #: Transfer coefficient for lower level
    c4: float = 0.025
    #: Forcings of equilibrium CO2 doubling (Wm-2)
    fco22x: float = 3.8

    # Climate damage parameters calibrates for quadratic at 2.5 C for 2105
    #: Damage intercept
    a1: float = 0
    #: Damage quadratic term
    a2: float = 0.00267
    #: Damage exponent
    a3: float = 2

    # Abatement cost
    #: Exponent of control cost function
    expcost2: float = 2.8
    #: Cost of backstop 2005$ for tCO2 2010
    pback: float = 344
    #: Initial cost decline backstop cost per period
    gback: float = 0.025
    #: Upper limit on control rate after 2150
    limmiu: float = 1.2
    #: Period before which no emissions controls base
    tnopol: float = 45
    #: Initial base carbon price (2005$ per tCO2)
    cprice0: float = 1
    #: Growth rate of base carbon price per year.
    gcprice: float = 0.02
    #: Period at which to have full participation.
    periodfullpart: int = 21
    #: Fraction of emissions under control in 2010
    partfract2010: float = 1
    #: Fraction of emissions under control at full time
    partfractfull: float = 1

    # Availability of fossil fuels
    #: Maximum cumulative extraction fossil fuels (GtC)
    fosslim: float = 6000

    # Scaling and inessential parameters
    #: Multiplicative scaling coefficient
    scale1: float = 0.016408662
    #: Additive scaling coefficient
    scale2: float = -3855.106895

    #: Whether or not to use optimization to set the carbon price.
    ifopt: int = 1

    # Parameters for long-run consistency of carbon cycle
    @property
    def b11(self):
        """Carbon cycle transition matrix."""
        return 1 - self.b12

    @property
    def b21(self):
        """Carbon cycle transition matrix."""
        return self.b12 * self.mateq / self.mueq

    @property
    def b22(self):
        """Carbon cycle transition matrix."""
        return 1 - self.b21 - self.b23

    @property
    def b32(self):
        """Carbon cycle transition matrix."""
        return self.b23 * self.mueq / self.mleq

    @property
    def b33(self):
        """Carbon cycle transition matrix."""
        return 1 - self.b32

    @property
    def sig0(self):
        """Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)."""
        return self.e0 / (self.q0 * (1 - self.miu0))

    @property
    def lam(self):
        """Climate model parameter."""
        return self.fco22x / self.t2xco2

    @property
    def L(self):  # pylint:disable-msg=invalid-name
        """Level of population and labor."""
        pop = {1: self.pop0}
        for time in range(1, self.numPeriods):
            pop[time + 1] = pop[time] * ((self.popasym / pop[time]) ** self.popadj)
        return pop

    def ga(self, time):  # pylint:disable-msg=invalid-name
        """Growth rate of productivity."""
        return self.ga0 * np.exp(-self.dela * 5.0 * (time - 1))

    def al(self, time):  # pylint:disable-msg=invalid-name
        """Level of total factor productivity."""
        if time == 1:
            return self.a0
        return self.al(time - 1) / (1 - self.ga(time - 1))

    def gsig(self, time):
        """Change in sigma (cumulative improvement of energy efficiency)."""
        if time == 1:
            return self.gsigma1
        return self.gsig(time - 1) * ((1 + self.dsig) ** self.tstep)

    def sigma(self, time):
        """CO2-equivalent-emissions output ratio."""
        if time == 1:
            return self.sig0
        return self.sigma(time - 1) * np.exp(self.gsig(time - 1) * self.tstep)

    def pbacktime(self, time):
        """Backstop price."""
        return self.pback * (1 - self.gback) ** (time - 1)

    def cost1(self, time):
        """Cost adjusted for backstop."""
        return self.pbacktime(time) * self.sigma(time) / self.expcost2 / 1000.0

    def etree(self, time):
        """Emissions from deforestation."""
        return self.eland0 * (1 - self.deland) ** (time - 1)

    def rr(self, time):  # pylint:disable-msg=invalid-name
        """Average utility social discount rate."""
        return 1.0 / ((1 + self.prstp) ** (self.tstep * (time - 1)))

    def forcoth(self, time):
        """Exogenous forcing for other greenhouse gases."""
        if time < 19:
            return self.fex0 + (1 / 18) * (self.fex1 - self.fex0) * (time)
        return self.fex0 + (self.fex1 - self.fex0)

    @property
    def optlrsav(self):
        """Optimal long-run savings rate used for transversality."""
        return (
            (self.dk + 0.004) / (self.dk + 0.004 * self.elasmu + self.prstp) * self.gama
        )

    def partfract(self, time):
        """Fraction of emissions in control regime."""
        if time == 1:
            return self.partfract2010

        if time > self.periodfullpart:
            return self.partfractfull

        return (
            self.partfract2010
            + (self.partfractfull - self.partfract2010)
            * (time - 1)
            / self.periodfullpart
        )

    def cpricebase(self, time):
        """Carbon price in base case."""
        return self.cprice0 * (1 + self.gcprice) ** (5 * (time - 1))

    def update(self, intervention):
        """Generate a new config object after applying the intervention."""
        return dataclasses.replace(self, **intervention.updates)


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Encapsulate an intervention in DICE."""

    def __init__(self, **kwargs):
        """Construct an intervention in DICE.

        Currently, all interventions are performed at the first time step.

        Parameters
        ----------
            kwargs: dict
                Only valid keyword arguments are parameters of
                `whynot.simulators.dice.Config`.

        """
        super(Intervention, self).__init__(Config, time=0, **kwargs)


@dataclasses.dataclass
class State(BaseState):  # pylint:disable-msg=too-few-public-methods
    """State variables of the DICE simulator.

    Default values are extracted from the first time step of a run of the
    DICE model using optimization to set the carbon price.

    """

    # pylint:disable-msg=invalid-name
    # pylint:disable-msg=too-many-instance-attributes
    #: Emission control rate GHGs
    MIU: float = 0.038976322
    #: Radiative forcing in watts per m2
    FORC: float = 2.167363097
    #: Temperature of atmosphere in degrees C
    TATM: float = 0.8
    #: Temperature of lower oceans in degrees C
    TOCEAN: float = 0.0068
    #: Carbon concentration in atmosphere GtC
    MAT: float = 830.4
    #: Carbon concentration in shallow oceans GtC
    MU: float = 1527
    #: Carbon concentration in lower oceans GtC
    ML: float = 10010
    #: CO2-equivalent emissions GtC
    E: float = 36.85382682
    EIND: float = 33.55382682
    #: Consumption trillions US dollars
    C: float = 46.98638871
    #: Capital stock trillions US dollars
    K: float = 135
    #: Per capita consumption thousands US dollars
    CPC: float = 6.871364246
    #: Investment trillions US dollars
    I: float = 16.48646319
    #: Gross savings rate as fraction of gross world product
    S: float = 0.259740388
    #: Real interest rate per annum
    RI: float = 0.052124994
    #: Gross world product net of abatement and damages
    Y: float = 63.4728519
    #: Gross world product Gross of abatement and damages
    YGROSS: float = 63.58198682
    #: Output net damages equation
    YNET: float = 63.47333792
    #: Damages (trillions 2005 USD per year)
    DAMAGES: float = 0.108648899
    #: Damages as fraction of gross output
    DAMFRAC: float = 0.0017088
    #: Cost of emissions reductions
    ABATECOST: float = 0.000486016
    #: Marginal cost of abatement (2005$ per ton CO2)
    MCABATE: float = 0.999999958
    #: Cumulative industrial carbon emissions GtC
    CCA: float = 90
    #: One period utility function
    PERIODU: float = 0.288713996
    #: Carbon price (2005$ per ton of CO2)
    CPRICE: float = 0.999999958
    #: Period utility
    CEMUTOTPER: float = 1974.226305

    @property
    def variables(self):
        """Return names of all model variables."""
        return [var.name for var in dataclasses.fields(self)]

    @property
    def nonnegative_variables(self):
        """Return names of all nonnegative variables."""
        return ["MIU", "TATM", "MAT", "MU", "ML", "Y", "YGROSS", "C", "K", "I"]

    def __getitem__(self, var_name):
        """Look up variables via state[name]."""
        return getattr(self, var_name)

    def __setitem__(self, name, value):
        """Set state variables via state[name] = value."""
        setattr(self, name, value)


def add_constraint(model, func):
    """Add a time-varying constraint to pyomo model."""
    constraint_name = func.__name__
    model.add_component(constraint_name, Constraint(model.time, rule=func))


def get_ipopt_solver():
    """Construct a platform specific IPOPT solver."""
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    if sys.platform == "linux":
        executable_path = os.path.join(cur_dir, "ipopt_bin", "ipopt-linux64")
    elif sys.platform == "darwin":
        executable_path = os.path.join(cur_dir, "ipopt_bin", "ipopt-osx")
    else:
        raise ValueError(f"No IPOPT binaries for platform {sys.platform}")
    return pyomo.opt.SolverFactory("ipopt", executable=executable_path)


def initialize_model(model, initial_state, config):
    """Create model variables and set the initial state."""
    var_bounds = generate_bounds(config)

    # Create and initialize model variables
    for var_name in initial_state.variables:
        domain = (
            NonNegativeReals
            if (var_name in initial_state.nonnegative_variables)
            else Reals
        )
        bounds = var_bounds.get(var_name, None)
        model.add_component(var_name, Var(model.time, domain=domain, bounds=bounds))

        # Explicitly set the initial variable value only if the initial
        # state doesn't match the default state.
        initial_value = initial_state[var_name]
        default_value = State()[var_name]
        if initial_value != default_value:
            # Clamp the first time step to the initialized value
            getattr(model, var_name)[1].fix(initial_value)


def generate_bounds(config):
    """Generate variable bound functions for initialization."""

    def miu_bounds(_, time):
        if time == 1:
            return (1e-20, None)
        if time < 30:
            return (1e-20, 1)
        return (1e-20, config.limmiu * config.partfract(time))

    def tatm_bounds(_, time):
        if time == 1:
            return (config.tatm0, config.tatm0)
        return (None, 40.0)

    def tocean_bounds(_, time):
        if time == 1:
            return (config.tocean0, config.tocean0)
        return (-1, 20.0)

    def mat_bounds(_, time):
        if time == 1:
            return (config.mat0, config.mat0)
        return (10, None)

    def mu_bounds(_, time):
        if time == 1:
            return (config.mu0, config.mu0)
        return (100.0, None)

    def ml_bounds(_, time):
        if time == 1:
            return (config.ml0, config.ml0)
        return (1000.0, None)

    def c_bounds(*_):
        return (2.0, None)

    def k_bounds(_, time):
        if time == 1:
            return (config.k0, config.k0)
        return (1.0, None)

    def cpc_bounds(*_):
        return (0.01, None)

    def cca_bounds(_, time):
        if time == 1:
            return (90, 90)
        return (None, config.fosslim)

    def s_bounds(_, time):
        if time <= 50:
            return (None, None)
        return (config.optlrsav, config.optlrsav)

    def cprice_bounds(_, time):
        """Use base carbon price if base, otherwise optimized.

        Warning if parameters are changed, the next equation might make base case infeasible
        If so, reduce tnopol so that don't run out of resources.
        """
        if config.ifopt == 0:
            return (None, config.cpricebase(time))
        if time == 1:
            return (None, config.cpricebase(1))
        if time > config.tnopol:
            return (None, 1000)
        return (None, None)

    variable_bounds = {
        "MIU": miu_bounds,
        "TATM": tatm_bounds,
        "TOCEAN": tocean_bounds,
        "MAT": mat_bounds,
        "MU": mu_bounds,
        "ML": ml_bounds,
        "C": c_bounds,
        "K": k_bounds,
        "CPC": cpc_bounds,
        "S": s_bounds,
        "CCA": cca_bounds,
        "CPRICE": cprice_bounds,
    }
    return variable_bounds


def add_emissions_dynamics(model, config):
    """Add emissions and damages constraints."""

    def eeq(model, time):
        """Emissions equation."""
        return model.E[time] == model.EIND[time] + config.etree(time)

    add_constraint(model, eeq)

    def eindeq(model, time):
        """Industrial emissions."""
        return model.EIND[time] == config.sigma(time) * model.YGROSS[time] * (
            1 - (model.MIU[time])
        )

    add_constraint(model, eindeq)

    def ccaeq(model, time):
        """Cumulative carbon emissions."""
        if time < config.numPeriods:
            return model.CCA[time + 1] == model.CCA[time] + model.EIND[time] * 5 / 3.666
        return Constraint.Skip

    add_constraint(model, ccaeq)

    def foreq(model, time):
        return model.FORC[time] == config.fco22x * (
            (log((model.MAT[time] / 588.000)) / log(2))
        ) + config.forcoth(time)

    add_constraint(model, foreq)

    def damfraceq(model, time):
        return model.DAMFRAC[time] == (config.a1 * model.TATM[time]) + (
            config.a2 * model.TATM[time] ** config.a3
        )

    add_constraint(model, damfraceq)

    def dameq(model, time):
        return model.DAMAGES[time] == model.YGROSS[time] * model.DAMFRAC[time]

    add_constraint(model, dameq)

    def abateeq(model, time):
        return model.ABATECOST[time] == model.YGROSS[time] * config.cost1(time) * (
            model.MIU[time] ** config.expcost2
        ) * (config.partfract(time) ** (1 - config.expcost2))

    add_constraint(model, abateeq)

    def mcabateeq(model, time):
        return model.MCABATE[time] == config.pbacktime(time) * model.MIU[time] ** (
            config.expcost2 - 1
        )

    add_constraint(model, mcabateeq)

    def carbpriceeq(model, time):
        return model.CPRICE[time] == config.pbacktime(time) * (
            model.MIU[time] / config.partfract(time)
        ) ** (config.expcost2 - 1)

    add_constraint(model, carbpriceeq)


def add_climate_dynamics(model, config):
    """Impose dynamics of climate and carbon cycle via constraints."""

    def mmat(model, time):
        """Atmospheric concentration equation."""
        if time < config.numPeriods:
            return model.MAT[time + 1] == model.MAT[time] * config.b11 + model.MU[
                time
            ] * config.b21 + (model.E[time] * (5 / 3.666))
        return Constraint.Skip

    add_constraint(model, mmat)

    def mml(model, time):
        """Lower ocean concentration."""
        if time < config.numPeriods:
            return (
                model.ML[time + 1]
                == model.ML[time] * config.b33 + model.MU[time] * config.b23
            )
        return Constraint.Skip

    add_constraint(model, mml)

    def mmu(model, time):
        """Shallow ocean concentration."""
        if time < config.numPeriods:
            return (
                model.MU[time + 1]
                == model.MAT[time] * config.b12
                + model.MU[time] * config.b22
                + model.ML[time] * config.b32
            )
        return Constraint.Skip

    add_constraint(model, mmu)

    def tatmeq(model, time):
        """Temperature-climate equation for atmosphere."""
        if time < config.numPeriods:
            return model.TATM[time + 1] == model.TATM[time] + config.c1 * (
                (
                    model.FORC[time + 1]
                    - (config.fco22x / config.t2xco2) * model.TATM[time]
                )
                - (config.c3 * (model.TATM[time] - model.TOCEAN[time]))
            )
        return Constraint.Skip

    add_constraint(model, tatmeq)

    def toceaneq(model, time):
        """Temperature-climate equation for lower oceans."""
        if time < config.numPeriods:
            return model.TOCEAN[time + 1] == model.TOCEAN[time] + config.c4 * (
                model.TATM[time] - model.TOCEAN[time]
            )
        return Constraint.Skip

    add_constraint(model, toceaneq)


def add_economic_dynamics(model, config):
    """Add dynamics constraints on economic variables."""

    def ygrosseq(model, time):
        return model.YGROSS[time] == (
            config.al(time) * (config.L[time] / 1000) ** (1 - config.gama)
        ) * (model.K[time] ** config.gama)

    add_constraint(model, ygrosseq)

    def yneteq(model, time):
        return model.YNET[time] == model.YGROSS[time] * (1 - model.DAMFRAC[time])

    add_constraint(model, yneteq)

    def yyeq(model, time):
        return model.Y[time] == model.YNET[time] - model.ABATECOST[time]

    add_constraint(model, yyeq)

    def cc(model, time):  # pylint:disable-msg=invalid-name
        return model.C[time] == model.Y[time] - model.I[time]

    add_constraint(model, cc)

    def cpce(model, time):
        return model.CPC[time] == 1000 * model.C[time] / config.L[time]

    add_constraint(model, cpce)

    def seq(model, time):
        return model.I[time] == model.S[time] * model.Y[time]

    add_constraint(model, seq)

    def kkeq(model, time):
        if time < config.numPeriods:
            return (
                model.K[time + 1]
                <= (1 - config.dk) ** config.tstep * model.K[time]
                + config.tstep * model.I[time]
            )
        return Constraint.Skip

    add_constraint(model, kkeq)

    def rieq(model, time):
        if time < config.numPeriods:
            return (
                model.RI[time]
                == (1 + config.prstp)
                * (model.CPC[time + 1] / model.CPC[time])
                ** (config.elasmu / config.tstep)
                - 1
            )
        return Constraint.Skip

    add_constraint(model, rieq)


def add_utility_dynamics(model, config):
    """Add constraints on utility variables."""

    def cemutotpereq(model, time):
        return model.CEMUTOTPER[time] == model.PERIODU[time] * config.L[
            time
        ] * config.rr(time)

    add_constraint(model, cemutotpereq)

    def perideq(model, time):
        return (
            model.PERIODU[time]
            == ((model.C[time] * 1000 / config.L[time]) ** (1 - config.elasmu) - 1)
            / (1 - config.elasmu)
            - 1
        )

    add_constraint(model, perideq)

    def utiliteq(model):
        return (
            model.UTILITY
            == config.tstep * config.scale1 * summation(model.CEMUTOTPER)
            + config.scale2
        )

    model.util = Constraint(rule=utiliteq)


@contextmanager
def silence_stdout():
    """Supress stdout output while active."""
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def simulate(initial_state, config, intervention=None, seed=None, stochastic=True):
    """Simulate a run of the DICE model.

    Parameters
    ----------
        initial_state: whynot.simulators.dice.State
            Initial state of for a run of the DICE model
        config: whynot.simulators.dice.Config
            Configuration object to set parameters of the dynamics.
        intervention:  whynot.simulators.dice.Intervention
            (Optional) What, if any, intervention to perform during execution.
        seed: int
            (Optional) Random seed for all model randomness.
        stochastic: bool
            (Optional) Whether or not to apply a random perturbation to the dynamics.

    Returns
    -------
        run: whynot.dynamics.Run
            Sequence of states and corresponding timesteps generated during execution.

    """
    if intervention:
        config = config.update(intervention)

    # Initialize model
    rng = np.random.RandomState(seed)
    model = ConcreteModel()
    model.time = RangeSet(1, config.numPeriods, 1)

    # Generate additional randomness using uncertainty ranges
    # for configuration parameters from the original DICE model.
    if stochastic:
        config = copy.deepcopy(config)
        config.t2xco2 *= rng.uniform(low=0.9, high=1.1)
        config.fosslim *= rng.uniform(low=0.9, high=1.1)
        config.limmiu *= rng.uniform(low=0.9, high=1.1)
        config.dsig *= rng.uniform(low=0.9, high=1.1)
        config.pop0 *= rng.uniform(low=0.9, high=1.1)

    # Initialize model and set initial state
    initialize_model(model, initial_state, config)

    # Add constraints governing model dynamics
    add_emissions_dynamics(model, config)
    add_climate_dynamics(model, config)
    add_economic_dynamics(model, config)

    # Set objective
    model.UTILITY = Var(domain=Reals)
    add_utility_dynamics(model, config)
    model.OBJ = Objective(rule=lambda m: m.UTILITY, sense=maximize)

    # Solve the model
    solver = get_ipopt_solver()
    with silence_stdout():
        _ = solver.solve(
            model,
            tee=True,
            symbolic_solver_labels=True,
            keepfiles=False,
            options={"max_iter": 99900, "halt_on_ampl_error": "yes", "print_level": 0},
        )

    # Read out values from the optimized model
    states, times = [initial_state], [0]
    for time in range(1, config.numPeriods + 1):
        state = State()
        for var in state.variables:
            state[var] = getattr(model, var)[time].value
        states.append(state)
        times.append(time)

    return wn.dynamics.Run(states=states, times=times)


if __name__ == "__main__":
    simulate(State(MAT=600), Config(numPeriods=10), stochastic=False)
