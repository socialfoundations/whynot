"""Tests for methods for working with dynamical systems."""
import copy
import dataclasses

import numpy as np
import pytest
import statsmodels

import whynot as wn
from whynot import parameter


def test_baseconfig():
    """Test the basic configuration object."""

    @dataclasses.dataclass
    class Config(wn.dynamics.BaseConfig):
        param1: float = 0
        param2: float = 23

    config = Config()

    assert config.parameter_names() == ["param1", "param2"]

    intervention = wn.dynamics.BaseIntervention(Config, 1970, param1=19)

    assert config.param1 == 0
    updated = config.update(intervention)
    assert config.param1 == 0
    assert config.param2 == 23
    assert updated.param1 == 19
    assert updated.param2 == 23


def test_basestate():
    """Test the basic state object."""

    @dataclasses.dataclass
    class State(wn.dynamics.BaseState):
        state1: float = 0
        state2: float = 1
        state3: float = 3

    assert State.num_variables() == 3
    assert State.variable_names() == ["state1", "state2", "state3"]

    state2 = [2]
    state = State(state2=state2)
    assert state.num_variables() == 3

    values = state.values()
    assert values[0] == 0
    assert values[1] == [2]
    assert values[2] == 3

    # Ensure values are shallow copied
    assert values[1] is state2


def test_framework_run():
    """Test the whynot.Run object"""
    states = ["start_state", 2, 3, 4, "final_state"]
    times = [4, 5, 6, 10, 23]

    run = wn.dynamics.Run(states=states, times=times)

    assert run.initial_state == "start_state"
    assert run.initial_time == 4
    assert run.final_state == "final_state"
    assert run.final_time == 23
    assert run[6] == 3

    # Find the state closest to time 11
    assert run[11] == 4

    # Throw error on construction if not the same length!
    with pytest.raises(ValueError):
        wn.dynamics.Run(states=[1, 2, 3], times=[0, 1])


@dataclasses.dataclass
class ToyConfig:
    num_steps: int = 10


@dataclasses.dataclass
class Intervention(ToyConfig):
    year: int = 0
    effect_size: int = 1


@dataclasses.dataclass
class ToyState:
    val: int = 0


class ToySimulator:
    """Simple simulator for testing purposes.

    Dynamics are to add effect_size to state at each time step if
    the run is treated. Otherwise, the run is constant starting at
    the initial_state value.

    """

    @staticmethod
    def simulate(initial_state, config, intervention=None, seed=None):
        """If treated, add a constant offset at each timestep."""
        states = []
        times = list(range(config.num_steps))

        state = initial_state
        for time in times:
            states.append(copy.deepcopy(state))
            if intervention and time >= intervention.year:
                state.val += intervention.effect_size
        return wn.dynamics.Run(states=states, times=times)


TEST_SIMULATOR = ToySimulator()


def check_shapes(dataset, num_samples, num_features):
    """Ensure the output has the right size."""
    assert dataset.covariates.shape == (num_samples, num_features)
    assert dataset.treatments.shape == (num_samples,)
    assert dataset.outcomes.shape == (num_samples,)
    assert dataset.true_effects.shape == (num_samples,)


def check_rct_assigment(assignment, prob):
    """Ensure that RCT assignment has approximately the correct proportion of treated."""
    total_treated = np.sum(assignment)
    ci_low, ci_upp = statsmodels.stats.proportion.proportion_confint(
        count=total_treated, nobs=len(assignment), alpha=0.001, method="beta"
    )
    assert ci_low <= prob <= ci_upp


### Helper methods for DynamicsExperiment tests
def basic_state_sampler():
    """Initial state is 0 for all rollouts."""
    return ToyState(val=0)


def correlated_state_sampler(num_samples):
    """Sample correlated initial states."""
    states = []
    for sample_idx in range(num_samples):
        states.append(ToyState(val=sample_idx))
    return states


def basic_outcome_extractor(run):
    """Look at state value after 5 steps."""
    return run[5].val


def basic_covariate_builder(run):
    """Return the final state of the run."""
    return np.array(run.states[-1].val)


def build_experiment(
    state_sampler=basic_state_sampler,
    config=ToyConfig(num_steps=10),
    intervention=Intervention(year=0, effect_size=1),
    propensity_scorer=0.5,
    outcome_extractor=basic_outcome_extractor,
    covariate_builder=basic_covariate_builder,
):
    """Construct a dynamics experiment instance with sane defaults."""
    return wn.dynamics.DynamicsExperiment(
        name="basic_test",
        description="Sanity check.",
        simulator=TEST_SIMULATOR,
        simulator_config=config,
        intervention=intervention,
        state_sampler=state_sampler,
        propensity_scorer=propensity_scorer,
        outcome_extractor=outcome_extractor,
        covariate_builder=covariate_builder,
    )


# DynamicsExperiment class tests.
def test_dynamics_experiment_basic():
    """Sanity check: does the class work?"""

    basic_exp = build_experiment()

    # Check outputs are the right sizes
    dataset = basic_exp.run(num_samples=10)
    check_shapes(dataset, 10, 1)

    dataset = basic_exp.run(num_samples=200)
    check_shapes(dataset, 200, 1)

    # Make sure the true effect is correctly computed
    assert np.allclose(5, dataset.true_effects)

    # Make sure the assignment and outcome are consistent
    assert np.allclose(0, dataset.outcomes[dataset.treatments == 0.0])
    assert np.allclose(5, dataset.outcomes[dataset.treatments == 1.0])

    # Make sure that incorrect parameters are flagged
    with pytest.raises(ValueError):
        basic_exp.run(num_samples=20, parameter=1234)


def test_state_sampler():
    """Test the state sampler function."""

    def initial_state_covariates(run):
        """Outcome is the initial state of the run."""
        return run.initial_state.val

    # Ensure the state sampler correctly uses randomness
    def random_initial_state(rng):
        """Initial state is 0 for all rollouts."""
        if rng.uniform() < 0.5:
            return ToyState(val=0)
        return ToyState(val=1)

    exp = build_experiment(
        state_sampler=random_initial_state, covariate_builder=initial_state_covariates
    )
    dataset0 = exp.run(num_samples=100, seed=1234)
    dataset1 = exp.run(num_samples=100, seed=1234)

    assert np.allclose(dataset0.covariates, dataset1.covariates)

    # Ensure the state sampler correctly deals with correlated state sampling
    exp = build_experiment(
        state_sampler=correlated_state_sampler,
        covariate_builder=initial_state_covariates,
    )
    dataset = exp.run(num_samples=100, seed=1234)
    assert np.allclose(np.expand_dims(np.arange(100), axis=1), dataset.covariates)

    # Test that parameters play nicely with state sampler
    @parameter(name="init_val", default=2)
    def parameterized_sampler(init_val):
        return ToyState(val=init_val)

    exp = build_experiment(
        state_sampler=parameterized_sampler, covariate_builder=initial_state_covariates
    )

    # Ensure default works
    dataset = exp.run(num_samples=10)
    assert np.allclose(dataset.covariates, 2)

    # Ensure varying parameterization works
    dataset = exp.run(num_samples=10, init_val=3.14)
    assert np.allclose(dataset.covariates, 3.14)

    dataset = exp.run(num_samples=10, init_val=1234)
    assert np.allclose(dataset.covariates, 1234)

    with pytest.raises(ValueError):
        exp.run(num_samples=10, random_val=123)


def test_config():
    """Test simulator config methods."""
    # Make sure config is actually effective
    def covariate_builder(run):
        return len(run.times)

    config = ToyConfig(num_steps=20)
    exp = build_experiment(config=config, covariate_builder=covariate_builder)
    dataset = exp.run(num_samples=15)
    assert np.allclose(20, dataset.covariates)

    # Check that config can be a function
    config = lambda: ToyConfig(num_steps=23)
    exp = build_experiment(config=config, covariate_builder=covariate_builder)
    dataset = exp.run(num_samples=15)
    assert np.allclose(23, dataset.covariates)

    # Check that these functions can be parameterized
    @parameter(name="num_steps", default=10)
    def pconfig(num_steps):
        return ToyConfig(num_steps=num_steps)

    exp = build_experiment(config=pconfig, covariate_builder=covariate_builder)

    # Default
    dataset = exp.run(num_samples=10)
    assert np.allclose(10, dataset.covariates)

    # Vary the parameter.
    dataset = exp.run(num_samples=10, num_steps=7)
    assert np.allclose(7, dataset.covariates)
    dataset = exp.run(num_samples=10, num_steps=56)
    assert np.allclose(56, dataset.covariates)

    # Check that bad argument flagging works for configs.
    with pytest.raises(ValueError):
        exp.run(num_samples=10, random_arg=123)


def test_intervention():
    """Test intervention methods."""
    intervention = Intervention(effect_size=2)
    exp = build_experiment(intervention=intervention)
    dataset = exp.run(num_samples=20)
    assert np.allclose(dataset.true_effects, 10)

    intervention = lambda: Intervention(year=1)
    exp = build_experiment(intervention=intervention)
    dataset = exp.run(num_samples=20)
    assert np.allclose(dataset.true_effects, 4)

    @parameter(name="effect_size", default=2)
    def pintervention(effect_size):
        return Intervention(effect_size=effect_size)

    exp = build_experiment(intervention=pintervention)

    # Default
    dataset = exp.run(num_samples=20)
    assert np.allclose(dataset.true_effects, 10)

    # Varying parameter
    dataset = exp.run(num_samples=20, effect_size=3)
    assert np.allclose(dataset.true_effects, 15)
    dataset = exp.run(num_samples=20, effect_size=5)
    assert np.allclose(dataset.true_effects, 25)

    # Check that bad argument flagging works for interventions
    with pytest.raises(ValueError):
        exp.run(num_samples=10, random_arg=123)


def test_propensity_scorer_rct():
    """Basic tests of propensity scorer functionality in RCT setting."""

    # Test to ensure propensity is (approximately) correct when using a constant
    exp = build_experiment(propensity_scorer=0.3)
    dataset = exp.run(num_samples=1000, seed=1234)
    check_rct_assigment(dataset.treatments, 0.3)

    exp = build_experiment(propensity_scorer=0.8)
    dataset = exp.run(num_samples=1000, seed=1234)
    check_rct_assigment(dataset.treatments, 0.8)

    # Make sure providing a function for the propensity score also works
    exp = build_experiment(propensity_scorer=lambda: 0.3)
    dataset = exp.run(num_samples=1000, seed=1234)
    check_rct_assigment(dataset.treatments, 0.3)

    # Only floats between 0 and 1 are valid constant propensities
    with pytest.raises(ValueError):
        exp = build_experiment(propensity_scorer="random_string")
        exp.run(num_samples=1000, seed=1234)

    with pytest.raises(ValueError):
        exp = build_experiment(propensity_scorer=2.0)
        exp.run(num_samples=1000, seed=1234)

    # Test parameterizing the propensity scorer
    @parameter(name="propensity", default=0.2)
    def parameterized_scorer(propensity):
        return propensity

    exp = build_experiment(propensity_scorer=parameterized_scorer)

    # Check default
    dataset = exp.run(num_samples=1000, seed=1234)
    check_rct_assigment(dataset.treatments, 0.2)

    # Check using parameter
    dataset = exp.run(num_samples=1000, seed=1234, propensity=0.65)
    check_rct_assigment(dataset.treatments, 0.65)

    # Check error handing
    with pytest.raises(ValueError):
        exp.run(num_samples=10, seed=1234, random_arg=1234)

    # Tests specifying control_config and treatment_config
    config = ToyConfig(num_steps=13)

    def config_scorer(config):
        assert config.num_steps == 13
        return 0.2

    exp = build_experiment(propensity_scorer=config_scorer, config=config)
    dataset = exp.run(num_samples=100, seed=1234)
    check_rct_assigment(dataset.treatments, 0.2)

    intervene = Intervention(year=4)

    def intervention_scorer(intervention):
        assert intervention.year == 4
        return 0.2

    exp = build_experiment(
        propensity_scorer=intervention_scorer, intervention=intervene
    )
    dataset = exp.run(num_samples=100, seed=1234)
    check_rct_assigment(dataset.treatments, 0.2)

    def both_scorer(config, intervention):
        assert config.num_steps == 13
        assert intervention.year == 4
        return 0.2

    exp = build_experiment(
        propensity_scorer=both_scorer, config=config, intervention=intervene
    )
    dataset = exp.run(num_samples=100, seed=1234)
    check_rct_assigment(dataset.treatments, 0.2)


def test_propensity_scorer_observation():
    """Test propensity scorer in the observational setting."""

    # Test that specificying control_run, treatment_run gives you the correct
    # runs from the model.
    def crun_scorer(untreated_run):
        # Control runs are constant
        start = untreated_run.initial_state.val
        assert np.allclose(np.array([s.val for s in untreated_run.states]), start)
        return 0.5

    exp = build_experiment(
        state_sampler=correlated_state_sampler, propensity_scorer=crun_scorer
    )
    dataset = exp.run(num_samples=100, seed=1234)
    check_rct_assigment(dataset.treatments, 0.5)

    def trun_scorer(treated_run):
        start = treated_run.initial_state.val
        run_vals = np.array([s.val for s in treated_run.states])
        assert np.allclose(start + np.arange(10), run_vals)
        return 0.5

    exp = build_experiment(
        state_sampler=correlated_state_sampler, propensity_scorer=trun_scorer
    )
    dataset = exp.run(num_samples=100, seed=1234)
    check_rct_assigment(dataset.treatments, 0.5)

    # Check propensity scores with interference works
    def cruns_scorer(untreated_runs):
        # Make sure that we see all of the runs
        # control runs should start with 1, 2, 3, 4, ..., num_samples
        start_vals = sorted([run.initial_state.val for run in untreated_runs])
        assert np.allclose(np.arange(len(untreated_runs)), start_vals)

        # Make sure they are control runs (i.e. all constant)
        for run in untreated_runs:
            assert np.allclose(
                np.array([s.val for s in run.states]), run.initial_state.val
            )

        # Only treat first half of runs
        n = len(untreated_runs)
        propensities = np.zeros((n,))
        propensities[: n // 2] = 1.0
        return propensities

    n = 100
    exp = build_experiment(
        state_sampler=correlated_state_sampler, propensity_scorer=cruns_scorer
    )
    dataset = exp.run(num_samples=n, seed=1234)
    assert np.allclose(dataset.treatments[: n // 2], 1)
    assert np.allclose(dataset.treatments[n // 2 :], 0)

    def truns_scorer(treated_runs):
        # Make sure that we see all of the runs
        # control runs should start with 1, 2, 3, 4, ..., num_samples
        start_vals = sorted([run.initial_state.val for run in treated_runs])
        assert np.allclose(np.arange(len(treated_runs)), start_vals)

        # Make sure they are control runs (i.e. all constant)
        for run in treated_runs:
            assert np.allclose(
                np.array([s.val for s in run.states]),
                run.initial_state.val + np.arange(10),
            )

        # Only treat first half of runs
        n = len(treated_runs)
        propensities = np.zeros((n,))
        propensities[: n // 2] = 1.0
        return propensities

    exp = build_experiment(
        state_sampler=correlated_state_sampler, propensity_scorer=truns_scorer
    )
    dataset = exp.run(num_samples=n, seed=1234)
    assert np.allclose(dataset.treatments[: n // 2], 1)
    assert np.allclose(dataset.treatments[n // 2 :], 0)


def test_outcome_extractor():
    """Test the outcome extractor."""

    def compute_outcome(run):
        """Look at state value after 5 steps."""
        return run[5].val

    # Make sure compute outcome works as expected
    exp = build_experiment(outcome_extractor=compute_outcome)
    dataset = exp.run(num_samples=100, seed=1234)
    assert np.allclose(dataset.outcomes[dataset.treatments == 0], 0)
    assert np.allclose(dataset.outcomes[dataset.treatments == 1], 5)

    # Compose compute_outcome with parameters
    @parameter(name="observation_step", default=3)
    def parameter_compute_outcome(run, observation_step):
        """Look at state value after observation_step steps."""
        return run[observation_step].val

    exp = build_experiment(outcome_extractor=parameter_compute_outcome)

    # Check defaults
    dataset = exp.run(num_samples=100, seed=1234)
    assert np.allclose(dataset.outcomes[dataset.treatments == 0], 0)
    assert np.allclose(dataset.outcomes[dataset.treatments == 1], 3)

    # Check parameterized version
    dataset = exp.run(num_samples=100, seed=1234, observation_step=8)
    assert np.allclose(dataset.outcomes[dataset.treatments == 0], 0)
    assert np.allclose(dataset.outcomes[dataset.treatments == 1], 8)

    # Make sure error checking works
    with pytest.raises(ValueError):
        exp.run(num_samples=10, random_param=123)


def test_covariate_builder():
    """Simple tests for covariate builder function."""

    @parameter(name="step", default=4)
    def build_covariates(config, intervention, run, step):
        """Return the final state of the run."""
        # Make sure we're getting the correct config/intervention
        assert config.num_steps == 10 and intervention.effect_size == 1
        return run.states[step].val

    exp = build_experiment(covariate_builder=build_covariates)

    # Check defaults
    dataset = exp.run(num_samples=100, seed=1234)
    assert np.allclose(dataset.covariates[dataset.treatments == 0], 0)
    assert np.allclose(dataset.covariates[dataset.treatments == 1], 4)

    # Check parameter
    dataset = exp.run(num_samples=100, seed=1234, step=7)
    assert np.allclose(dataset.covariates[dataset.treatments == 0], 0)
    assert np.allclose(dataset.covariates[dataset.treatments == 1], 7)

    # Make sure error checking works
    with pytest.raises(ValueError):
        exp.run(num_samples=10, random_param=123)
