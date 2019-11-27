"""Experiments for the incarceration simulator."""

import numpy as np

from whynot.framework import GenericExperiment
from whynot.simulators import incarceration

__all__ = ["get_experiments", "SentenceLength"]


def get_experiments():
    """Return all available experiments for incarceration simulator.

    Returns
    -------
    list
        List of available experiments.

    """
    return [SentenceLength]


def months_in_prison(popu):
    """Return array of months in prison for each person in population."""
    return np.array([person["months_in_prison"] for person in popu.values()])


def covariates_from_population(popu):
    """Create covariates from population of agents.

    The extracted covariates are:
        * sex (1 female, 0 male)
        * age
        * number of children
        * number of siblings
        * number of friends
        * age at first birth
        * had partner (1 yes, 0 no)

    Parameters
    ----------
    popu : dict
        Dictionary specifying population of agents.

    Returns
    -------
    covariates: np.array

    """
    covariates = []
    for person in popu.values():
        covariates.append(
            [
                1 if person["sex"] == "f" else 0,
                person["death"][0] - person["birth"],
                len(person["children"]),
                len(person["siblings"]),
                len(person["friends"]),
                person["age_at_first_birth"],
                1 if person["partner"] > 0 else 0,
            ]
        )

    return np.array(covariates)


def treatments_from_population(popu):
    """Return treatment vector.

    Parameters
    ----------
    popu : dict
        Dictionary specifying population of agents.

    Returns
    -------
    treatments: np.array

    """
    treatments = []
    for person in popu.values():
        if person["harsh_sentence"]:
            treatments.append(np.mean(person["harsh_sentence"]))
        else:
            treatments.append(0)
    return np.array(treatments)


def population_to_data(popu):
    """Convert simulated population to tabular data.

    Parameters
    ----------
    popu : dict
        Dictionary specifying population of agents.

    Returns
    -------
    covariates: np.array, [num_samples, num_features]
        Covariates as numpy array.
    treatments: Binary np.array, [num_samples,]
        Treatment vector as numpy array.
    outcomes: np.array [num_samples,]
        Outcome vector as numpy arrays.

    """
    covariates = covariates_from_population(popu)
    treatments = treatments_from_population(popu)
    outcomes = months_in_prison(popu)

    return covariates, treatments, outcomes


def run_sentence_length_trial(num_samples, seed, show_progress=False, parallelize=True):
    """Run sentence experiment on incarceration simulator.

    Experiment about the effect of harsher sentences on total prison time in the
    population. Individuals are randomly assigned sentences drawn from either a
    harsher sentence distribution or a more lenient distribution. Treatment value
    in [0, 1] corresponds to the number of times an individual received a harsh
    sentence.

    Outcomes corresponds to the total amount of time an individual in the
    population spent in prison.

    The ground truth gives two vectors, one for prison times under the harsh
    policy only and one under the lenient policy only.

    Parameters
    ----------
    num_samples : int
        Unused since the simulator does not allow for varying number of agents.
    seed : int
        Random seed used for simulation
    show_progress: bool
        Show a progress bar for each run of the simulator.
    parallelize: bool
        Unused, the experiment always runs sequentially.

    Returns
    -------
    covariates: np.array, [num_samples, state_dim]
    treatment: Binary np.array, [num_samples,]
    outcomes: np.array [num_samples,]
    ground_truth_effect:

    """
    # pylint:disable-msg=unused-argument
    config = incarceration.Config(random_seed=seed, random_sentence_type=True)
    popu_random = incarceration.simulate(config, show_progress)
    covariates, treatment, outcome = population_to_data(popu_random)

    config = incarceration.Config(
        random_seed=seed, random_sentence_type=False, harsh_sentence=True
    )
    popu_harsh = incarceration.simulate(config, show_progress)

    config = incarceration.Config(
        random_seed=seed, random_sentence_type=False, harsh_sentence=False
    )
    popu_lenient = incarceration.simulate(config, show_progress)

    ground_truth = (months_in_prison(popu_harsh), months_in_prison(popu_lenient))

    return (covariates, treatment, outcome), ground_truth


# pylint: disable-msg=invalid-name
#: Randomized trial with interference between units.
SentenceLength = GenericExperiment(
    name="sentence_length_rct",
    description="Randomized controlled trial with interference between units.",
    run_method=run_sentence_length_trial,
)
