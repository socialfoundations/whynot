"""Basic set of experiments on the LaLonde dataset."""
import os

import numpy as np
import pandas as pd

from whynot.framework import GenericExperiment, parameter

__all__ = ["get_experiments", "RandomResponse"]


##################
# Helper functions
##################


def load_dataset():
    """Load the LaLonde dataset."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, "lalonde.csv")

    lalonde = pd.read_csv(data_path, index_col=0)

    # Remove outcome
    lalonde = lalonde.drop("re78", axis=1)

    return lalonde.rename(columns={"treat": "treatment"})


########################
# Experiment definitions
########################
def get_experiments():
    """Return all of the LaLonde experiments."""
    return [RandomResponse]


@parameter(
    name="hidden_dim",
    default=32,
    values=[8, 16, 32, 64, 128, 256, 512],
    description="hidden dimension of 2-layer ReLu network response.",
)
@parameter(
    name="alpha_scale",
    default=0.01,
    values=np.linspace(1e-4, 10, 10),
    description="Scale of the hidden-layer weights.",
)
def run_lalonde(
    num_samples,
    hidden_dim,
    alpha_scale,
    seed=None,
    parallelize=True,
    show_progress=False,
):
    # pylint:disable-msg=unused-argument
    """Generate data from the LaLonde dataset with a random response function.

    The covariates and treatment are both specified by the dataset, and the
    response function is a random 2-layer neural network with ReLu.

    Parameters
    ----------
        num_samples: int
            This parameter is ignored since the LaLonde dataset size is fixed.
        hidden_dim: int
            Hidden dimension of the relu network.
        alpha_scale: float
            Standard deviation of the final layer weights.
        seed: int
            Random seed used for all internal randomness
        parallelize: bool
            Ignored, but included for consistency with GenericExperiment API.
        show_progress: False
            Ignored, but included for consistency with GenericExperiment API.

    """
    rng = np.random.RandomState(seed)

    dataset = load_dataset()
    treatment = dataset.treatment.values.astype(np.int64)
    covariates = dataset.drop("treatment", axis=1).values

    # Define the networks
    num_inputs = covariates.shape[1]
    control_config = {
        "W": 0.05 * rng.randn(num_inputs, hidden_dim),
        "alpha": alpha_scale * rng.randn(hidden_dim, 1),
    }

    treatment_config = {
        "W": 0.05 * rng.randn(num_inputs, hidden_dim),
        "alpha": alpha_scale * rng.randn(hidden_dim, 1),
    }

    def get_effect(features, treatment):
        if treatment:
            config = treatment_config
        else:
            config = control_config
        return np.maximum(features.dot(config["W"]), 0).dot(config["alpha"])[:, 0]

    control_outcomes = get_effect(covariates, treatment=False)
    treatment_outcomes = get_effect(covariates, treatment=True)

    outcomes = np.copy(control_outcomes)
    treatment_idxs = np.where(treatment == 1.0)
    outcomes[treatment_idxs] = treatment_outcomes[treatment_idxs]

    return (covariates, treatment, outcomes), treatment_outcomes - control_outcomes


# pylint: disable-msg=invalid-name
#: Experiment simulating an outcome function on top of fixed LaLonde covariates.
RandomResponse = GenericExperiment(
    name="lalonde",
    description=(
        "An experiment on the LaLone dataset with fixed covariates "
        "and random 2-layer Relu NN for the response."
    ),
    run_method=run_lalonde,
)
