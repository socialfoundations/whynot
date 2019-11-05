"""Simple integration and determinism tests for all simulators."""
import numpy as np

import whynot as wn


def check_shapes(dataset, num_samples):
    """Ensure datast has the right size."""
    assert len(dataset.covariates.shape) == 2 and dataset.covariates.shape[0] == num_samples
    assert dataset.treatments.shape == (num_samples,)
    assert dataset.outcomes.shape == (num_samples,)
    assert dataset.true_effects.shape == (num_samples,)

def integration_test(simulator, num_samples=15):
    """Check if the simulator generates data that is well-formed and can be used for inference."""
    for experiment in simulator.get_experiments():
        params = experiment.get_parameters()
        print(params)
        dataset = experiment.run(num_samples=num_samples, seed=1234, **params.sample())
        check_shapes(dataset, num_samples)

        # Sanity check to make sure data is well-formed.
        _ = wn.algorithms.ols.estimate_treatment_effect(
            dataset.covariates, dataset.treatments, dataset.outcomes)

def determinism_test(simulator, num_samples=15):
    """Check to make sure all experiments are deterministic."""
    for experiment in simulator.get_experiments():
        dataset1 = experiment.run(num_samples=num_samples, seed=1234)
        dataset2 = experiment.run(num_samples=num_samples, seed=1234)

        assert np.allclose(dataset1.covariates, dataset2.covariates)
        assert np.allclose(dataset1.treatments, dataset2.treatments)
        assert np.allclose(dataset1.outcomes, dataset2.outcomes)
        assert np.allclose(dataset1.true_effects, dataset2.true_effects)

def initial_state_test(simulator):
    """For ODE simulators, ensure the iniitial_state is returned in run."""
    initial_state = simulator.State()
    config = simulator.Config()
    run = simulator.simulate(initial_state, config)
    assert len(run.states) == len(run.times)
    assert run.initial_state is initial_state


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

def test_initial_state():
    initial_state_test(wn.hiv)
    initial_state_test(wn.lotka_volterra)
    initial_state_test(wn.opioid)
    initial_state_test(wn.world2)
    initial_state_test(wn.world3)
