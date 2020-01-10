"""Causal experiments for the Schelling model."""
import copy
import dataclasses

import numpy as np
import pandas as pd

from whynot.framework import GenericExperiment, parameter
from whynot.simulators import schelling
import whynot.utils as utils

__all__ = ["get_experiments", "RCT"]


def get_experiments():
    """Return all experiments for Schelling."""
    return [RCT]


################
# RCT Experiment
################
def run_simulator(control_config, treatment_config, seed):
    """Run the simulator in both the control and treatment conditions using a common seed."""
    control_segregation = schelling.simulate(control_config, rollouts=5, seed=seed)
    treatment_segregation = schelling.simulate(treatment_config, rollouts=5, seed=seed)
    return control_segregation, treatment_segregation


@parameter(
    name="education_pc",
    default=0.3,
    values=np.arange(0.0, 0.99, 0.05),
    description="Random fraction of the population to educate",
)
@parameter(
    name="education_boost",
    default=-1,
    values=[-1, -2, -3, -4],
    description="How much education decreases homophily",
)
def run_schelling(
    education_pc,
    education_boost,
    num_samples=100,
    seed=None,
    show_progress=False,
    parallelize=True,
):
    """Run a basic RCT experiment on Schelling.

    Each unit in the experiment is a grid or "community" in the Schelling model.
    Treatment corresponds to randomly educating some fraction of the
    agents in the grid to decrease their homophiliy. The measure outcome is the
    fraction of segregated agents on the grids. In this experiment, treatment is
    randomly assigned.

    Parameters
    ----------
        education_pc: float
            What percentrage of the population of the treated units should be "educated"
        education_boost: int
            How much receiving "education" changes the homophily of an agent
            Values in [-1, -2, -3, -4]
        num_samples: int
            How many units to sample for the experiment
        seed: int
            (Optional) Seed global randomness for the experiment
        show_progress: bool
            (Optional) Whether or not to print a progress bar
        parallelize: bool
            (Optional) Whether or not to use parallelism during the experiment

    Returns
    -------
        covariates: np.ndarray
            Array of shape [num_samples, num_features], observed covariates for
            each unit.
        treatment: np.ndarray
            Array of shape [num_samples], treatment assignment for each unit
        outcomes: np.ndarray
            Array of shape [num_samples], observed outcome for each unit
        ground_truth: np.ndarray
            Array of shape [num_samples], unit level treatment effects

    """
    rng = np.random.RandomState(seed)

    configs = []
    for _ in range(num_samples):
        config = schelling.Config()
        config.height = 10
        config.width = 10
        config.homophily = 5
        config.density = rng.uniform(0.05, 0.6)
        config.minority_pc = rng.uniform(0.05, 0.45)
        config.education_boost = education_boost

        # By default, no education.
        config.education_pc = 0.0
        treatment_config = copy.deepcopy(config)
        # Treatment corresponds to education
        treatment_config.education_pc = education_pc

        seed = rng.randint(0, 99999)
        configs.append((config, treatment_config, seed))

    if parallelize:
        runs = utils.parallelize(run_simulator, configs, show_progress=show_progress)
    else:
        runs = [run_simulator(*args) for args in configs]

    control_outcomes = np.array([run[0] for run in runs])
    treatment_outcomes = np.array([run[1] for run in runs])

    # Potential confounders are density and minority pc
    dataframe = pd.DataFrame([dataclasses.asdict(config[0]) for config in configs])
    covariates = dataframe[["density", "minority_pc"]].values

    # First pass: Randomly assign treatment
    # We could only treat "cities" with high density and
    # high minority pc
    treatment = (rng.rand(num_samples) < 0.25).astype(np.int32)
    treatment_idxs = np.where(treatment == 1.0)[0]

    outcomes = np.copy(control_outcomes)
    outcomes[treatment_idxs] = treatment_outcomes[treatment_idxs]

    treatment_effects = treatment_outcomes - control_outcomes
    return (covariates, treatment, outcomes), treatment_effects


#: RCT experiment to understand the effect of homophily reduction on total segregation.
RCT = GenericExperiment(
    name="schelling_rct",
    description=(
        "Test the efficacy of an education program to reduce "
        "homophiliy of agents in the Schelling model."
    ),
    run_method=run_schelling,
)
