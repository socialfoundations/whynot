"""Simple integration and determinism tests for all simulators."""
import numpy as np
import pytest

import whynot as wn


def check_shapes(dataset, num_samples):
    """Ensure datast has the right size."""
    assert (
        len(dataset.covariates.shape) == 2
        and dataset.covariates.shape[0] == num_samples
    )
    assert dataset.treatments.shape == (num_samples,)
    assert dataset.outcomes.shape == (num_samples,)
    assert dataset.true_effects.shape == (num_samples,)


@pytest.mark.parametrize(
    "simulator",
    [wn.hiv, wn.lotka_volterra, wn.opioid, wn.world2, wn.world3],
    ids=["hiv", "lotka_volterra", "opioid", "world2", "world3",],
)
def test_initial_states(simulator):
    """For ODE simulators, ensure the iniitial_state is returned by reference in run."""
    initial_state = simulator.State()
    config = simulator.Config()
    run = simulator.simulate(initial_state, config)
    assert len(run.states) == len(run.times)
    assert run.initial_state is initial_state
    assert np.allclose(config.delta_t, run.times[1] - run.times[0])


@pytest.mark.parametrize(
    "simulator,intervention_param,intervention_val",
    [
        (wn.hiv, None, None),
        (wn.lotka_volterra, None, None),
        (wn.opioid, "nonmedical_incidence", -0.1),
        (wn.world2, None, None),
        (wn.world3, None, None),
    ],
    ids=["hiv", "lotka_volterra", "opioid", "world2", "world3",],
)
def test_intervention(simulator, intervention_param, intervention_val):
    """For ODE simulators, ensure test config.intervention."""
    initial_state = simulator.State()
    config = simulator.Config(delta_t=1.0)
    intervention_time = (config.start_time + config.end_time) / 2

    if intervention_param is None:
        intervention_param = config.parameter_names()[0]
        intervention_val = 5.0 * getattr(config, intervention_param)

    kwargs = {intervention_param: intervention_val}
    intervention = simulator.Intervention(time=intervention_time, **kwargs)

    # Run the simulator with and without the intervention
    # Ensure the states match up to the intervention time.
    untreated_run = simulator.simulate(
        initial_state, config, intervention=None, seed=1234
    )
    treated_run = simulator.simulate(
        initial_state, config, intervention=intervention, seed=1234
    )

    for idx, time in enumerate(untreated_run.times):
        if time < intervention_time:
            assert np.allclose(
                untreated_run.states[idx].values(), treated_run.states[idx].values()
            )
        elif time >= intervention_time + config.delta_t:
            # Should make sure the interventions have *some* effect
            assert not np.allclose(
                untreated_run.states[idx].values(), treated_run.states[idx].values()
            )


def integration_test(simulator, num_samples=15):
    """Check if the simulator generates data that is well-formed and can be used for inference."""
    for experiment in simulator.get_experiments():
        params = experiment.get_parameters()
        print(params)
        dataset = experiment.run(num_samples=num_samples, seed=1234, **params.sample())
        check_shapes(dataset, num_samples)

        # Sanity check to make sure data is well-formed.
        _ = wn.algorithms.ols.estimate_treatment_effect(
            dataset.covariates, dataset.treatments, dataset.outcomes
        )


def determinism_test(simulator, num_samples=15):
    """Check to make sure all experiments are deterministic."""
    for experiment in simulator.get_experiments():
        dataset1 = experiment.run(num_samples=num_samples, seed=1234)
        dataset2 = experiment.run(num_samples=num_samples, seed=1234)

        assert np.allclose(dataset1.covariates, dataset2.covariates)
        assert np.allclose(dataset1.treatments, dataset2.treatments)
        assert np.allclose(dataset1.outcomes, dataset2.outcomes)
        assert np.allclose(dataset1.true_effects, dataset2.true_effects)


# pylint:disable-msg=missing-docstring
def test_civil_violence_integration():
    integration_test(wn.civil_violence, num_samples=5)


def test_civil_violence_determinism():
    determinism_test(wn.civil_violence, num_samples=5)


def test_dice_integration():
    integration_test(wn.dice)


def test_dice_determinism():
    determinism_test(wn.dice)


def test_hiv_integration():
    integration_test(wn.hiv)


def test_hiv_determinism():
    determinism_test(wn.hiv)


def test_lalonde_integration():
    # 445 units in the LaLonde dataset
    integration_test(wn.lalonde, num_samples=445)


def test_lalonde_determinism():
    determinism_test(wn.lalonde)


def test_lotka_volterra_integration():
    integration_test(wn.lotka_volterra)


def test_lotka_volterra_determinism():
    determinism_test(wn.lotka_volterra)


def test_opioid_integration():
    integration_test(wn.opioid)


def test_opioid_determinism():
    determinism_test(wn.opioid)


def test_schelling_integration():
    integration_test(wn.schelling, num_samples=5)


def test_schelling_determinism():
    determinism_test(wn.schelling, num_samples=5)


def test_world2_integration():
    integration_test(wn.world2)


def test_world2_determinism():
    determinism_test(wn.world2)


def test_world3_integration():
    integration_test(wn.world3, num_samples=25)


def test_world3_determinism():
    determinism_test(wn.world3, num_samples=25)
