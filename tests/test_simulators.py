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
    [
        wn.delayed_impact,
        wn.hiv,
        wn.lotka_volterra,
        wn.opioid,
        wn.world2,
        wn.world3,
        wn.zika,
    ],
    ids=[
        "delayed_impact",
        "hiv",
        "lotka_volterra",
        "opioid",
        "world2",
        "world3",
        "zika",
    ],
)
def test_dynamics_initial_state(simulator):
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
        (wn.delayed_impact, "credit_scorer", lambda score: min(score, 600)),
        (wn.hiv, None, None),
        (wn.lotka_volterra, None, None),
        (wn.opioid, "nonmedical_incidence", -0.1),
        (wn.world2, None, None),
        (wn.world3, None, None),
        (wn.zika, "treated_bednet_use", 0.5),
    ],
    ids=[
        "delayed_impact",
        "hiv",
        "lotka_volterra",
        "opioid",
        "world2",
        "world3",
        "zika",
    ],
)
def test_dynamics_intervention(simulator, intervention_param, intervention_val):
    """For ODE simulators, ensure test config.intervention."""
    initial_state = simulator.State()
    config = simulator.Config(delta_t=1.0)
    intervention_time = (config.start_time + config.end_time) // 2

    if intervention_param is None:
        intervention_param = config.parameter_names()[0]
        intervention_val = 5.0 * getattr(config, intervention_param)

    kwargs = {intervention_param: intervention_val}
    intervention = simulator.Intervention(time=intervention_time, **kwargs)

    # Run the simulator with and without the intervention
    # Ensure the states match up to the intervention time and diverge
    # thereafter.
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


@pytest.mark.parametrize(
    "simulator,num_samples",
    [
        (wn.civil_violence, 5),
        (wn.delayed_impact, 10),
        (wn.dice, 10),
        (wn.hiv, 10),
        (wn.lalonde, 445),
        (wn.lotka_volterra, 10),
        (wn.opioid, 10),
        (wn.schelling, 5),
        (wn.world2, 10),
        (wn.world3, 10),
        (wn.zika, 10),
    ],
    ids=[
        "civil_violence",
        "delayed_impact",
        "dice",
        "hiv",
        "lalonde",
        "lotka_volterra",
        "opioid",
        "schelling",
        "world2",
        "world3",
        "zika",
    ],
)
def test_simulator_experiments(simulator, num_samples):
    """Test simulator experiments for determinism and proper outcome size."""
    for experiment in simulator.get_experiments():
        params = experiment.get_parameters()
        print(params)
        sampled_params = params.sample(seed=1234)
        dataset1 = experiment.run(num_samples=num_samples, seed=1234, **sampled_params)

        # Sanity check to make sure data is well-formed.
        check_shapes(dataset1, num_samples)

        # Rerun the experiment for determinism.
        dataset2 = experiment.run(num_samples=num_samples, seed=1234, **sampled_params)
        assert np.allclose(dataset1.covariates, dataset2.covariates)
        assert np.allclose(dataset1.treatments, dataset2.treatments)
        assert np.allclose(dataset1.outcomes, dataset2.outcomes)
        assert np.allclose(dataset1.true_effects, dataset2.true_effects)


@pytest.mark.parametrize(
    "simulator,num_samples",
    [
        (wn.civil_violence, 5),
        (wn.delayed_impact, 10),
        (wn.dice, 10),
        (wn.hiv, 10),
        (wn.lalonde, 445),
        (wn.lotka_volterra, 10),
        (wn.opioid, 10),
        (wn.schelling, 5),
        (wn.world2, 10),
        (wn.world3, 10),
        (wn.zika, 10),
    ],
    ids=[
        "civil_violence",
        "delayed_impact",
        "dice",
        "hiv",
        "lalonde",
        "lotka_volterra",
        "opioid",
        "schelling",
        "world2",
        "world3",
        "zika",
    ],
)
def test_parallelize(simulator, num_samples):
    """Test that parallelized runs are identical to non-parallelized."""
    for experiment in simulator.get_experiments():
        unparallelized = experiment.run(
            num_samples=num_samples, parallelize=False, seed=1234
        )
        parallelized = experiment.run(
            num_samples=num_samples, parallelize=True, seed=1234
        )

        assert np.allclose(unparallelized.covariates, parallelized.covariates)
        assert np.allclose(unparallelized.treatments, parallelized.treatments)
        assert np.allclose(unparallelized.outcomes, parallelized.outcomes)
        assert np.allclose(unparallelized.true_effects, parallelized.true_effects)
