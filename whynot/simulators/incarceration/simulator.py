"""Incarceration simulator.

Agent-based model for incarceration dynamics. The simulator models incarceration
as an infectious property that can be passed on along social ties. Infection
probabilities are based on survey data. The simulator is based on Lum, Swarup,
Eubank, and Hawdon (2014). URL: https://doi.org/10.1098/rsif.2014.0409

The simulation happens on top of a fixed set of agents provided by the authors
of the study. The number of agents cannot be varied.

"""

import os
import dataclasses
import random
import gzip
import pickle

import numpy as np
from tqdm.auto import tqdm


@dataclasses.dataclass
class Config:
    # pylint: disable-msg=too-few-public-methods
    """Metaparameters for a single run of simulator.

    Attributes
    ----------
    random_seed : int
        Controls all randomness of the simulation
    min_age : int
        Minimum age for incarceration
    start_iter : int
        Start iteration of simulation
    end_iter : int
        End iteration of simulation
    percent : float
        Initial incarceration rate
    random_sentence_type : bool
        Assign lenient or harsh sentence randomly
    random_sentence_bias : float
        Probability of picking harsh sentence
    mean_sentence_harsh : float
        Mean sentence length (months) for harsh assignment
    mean_sentence_lenient : float
        Mean sentence length (months) for lenient assignment

    """

    random_seed: int = 1
    min_age: int = 15
    start_iter: int = 100
    end_iter: int = 200
    percent: float = 0.01
    harsh_sentence: bool = False
    random_sentence_type: bool = False
    random_sentence_bias: float = 0.5
    consistent_sentence_length: bool = True
    mean_sentence_harsh: float = 17.0
    mean_sentence_lenient: float = 14.0


def infect(person_sex, relation_type, relation_sex):
    """Generate infection based on relation characteristics.

    Parameters
    ----------
    person_sex : string
        Sex of person ('m' or 'f')
    relation_type : string
        Type of social relation  ('parent', 'sibling', 'partner', 'child')
    relation_sex : string
        Sex of related person ('m' or 'f')

    Returns
    -------
    infected : int
        1 if infected, 0 otherwise

    """
    infection_probability_month = {
        "f": {
            "parent": {"f": 0.000849988733768181, "m": 0.0112878570024662,},
            "sibling": {"f": 0.00801193753900653, "m": 0.0332053842229949,},
            "partner": {"*": 0.0043472740358963},
            "child": {"*": 0.0169602401420906},
        },
        "m": {
            "parent": {"f": 0.00347339838166261, "m": 0.0113344842544054,},
            "sibling": {"f": 0.00436688659218365, "m": 0.0301729987453868,},
            "partner": {"*": 0.00078339990345766},
            "child": {"*": 0.00634223110220566},
        },
        "*": {"*": {"*": 1.675041946602729e-05}},
    }

    probability = infection_probability_month[person_sex][relation_type][relation_sex]
    return np.random.binomial(1, probability)


def valid_age(person, i, min_age):
    """Verify age requirement of person in given iteration."""
    return person["birth"] <= (i - min_age) and person["death"] >= i


def initialize(config):
    """Load population and generate initial incarceration state."""
    random.seed(config.random_seed)

    population_pickle = os.path.join(os.path.dirname(__file__), "population.pkl.gz")
    popu = pickle.load(gzip.open(population_pickle, "rb"))

    alive = []
    for num, person in popu.items():
        person["num"] = num
        person["months_in_prison"] = 0
        person["harsh_sentence"] = []
        if valid_age(person, config.start_iter, config.min_age):
            alive.append(person)
    num_infect = round(config.percent * len(alive))
    potentials = []
    for person in alive:
        if config.start_iter - person["birth"] <= 45:
            potentials.append(person)
    infected = random.sample(potentials, num_infect)
    for person in infected:
        sentence, harsh_sentence = generate_sentence(person, -1, 0, config)
        person["incarcerated"] = sentence
        person["harsh_sentence"].append(harsh_sentence)

    return popu


def spread_infection(popu, person, itr, month, config):
    """Pass on infection."""
    # ensure same infection patterns regardless of sentence intervention
    np.random.seed(
        hash((config.random_seed, person["num"], itr, month)) % (2 ** 32 - 1)
    )

    sex = person["sex"]

    for sibling in person["siblings"]:
        popu[sibling]["infected"] += infect(sex, "sibling", popu[sibling]["sex"])

    if person["partner"] >= 0 and person["iter_married"] >= itr:
        partner = person["partner"]
        popu[partner]["infected"] += infect(popu[partner]["sex"], "partner", "*")

    for child in person["children"]:
        if valid_age(popu[child], itr, config.min_age):
            popu[child]["infected"] += infect(sex, "child", "*")

    for friend in person["friends"]:
        popu[friend]["infected"] += infect(sex, "sibling", popu[friend]["sex"])

    for parent in person["parents"]:
        popu[parent]["infected"] += infect(sex, "parent", popu[parent]["sex"])


def generate_sentence(person, itr, month, config):
    """Generate sentence based on harshness setting."""
    # generate both sentences to ensure consistent counterfactuals
    # offset random seed to avoid interference with infection random seed
    np.random.seed(
        hash((1, config.random_seed, person["num"], itr, month)) % (2 ** 32 - 1)
    )

    gamma_parameter = 1.2

    lenient_mean = config.mean_sentence_lenient
    gamma_sample = np.random.gamma(gamma_parameter, lenient_mean / gamma_parameter)
    lenient_sentence = np.random.poisson(gamma_sample)

    if config.consistent_sentence_length:
        penalty = config.mean_sentence_harsh - config.mean_sentence_lenient
        harsh_sentence = lenient_sentence + penalty
    else:
        harsh_mean = config.mean_sentence_harsh
        gamma_sample = np.random.gamma(gamma_parameter, harsh_mean / gamma_parameter)
        harsh_sentence = np.random.poisson(gamma_sample)

    if config.random_sentence_type:
        if np.random.binomial(1, config.random_sentence_bias):
            return harsh_sentence, True
        return lenient_sentence, False

    if config.harsh_sentence:
        return harsh_sentence, True
    return lenient_sentence, False


def assign_sentence(person, itr, month, config):
    """Assign sentence."""
    if valid_age(person, itr, config.min_age):
        sentence, harsh_sentence = generate_sentence(person, itr, month, config)
        person["incarcerated"] = sentence
        person["harsh_sentence"].append(harsh_sentence)


def simulate(config, show_progress=False):
    """Simulate incarceration contagion dynamics.

    Parameters
    ----------
    config : Config
        Config object specifying simulation parameters.

    Returns
    -------
    dict
        Dictionary specifying simulated population of agents.

    """
    popu = initialize(config)

    agents = popu.values()

    def display(range_obj):
        if show_progress:
            range_obj = tqdm(range_obj)
        return range_obj

    # these are in years. need to work in terms of months
    for itr in display(range(config.start_iter, config.end_iter)):

        for month in range(12):

            # infection step
            for person in agents:

                # random infection, not due to contagion
                if valid_age(person, itr, config.min_age):
                    person["infected"] += infect("*", "*", "*")

                # infect connected people
                if person["incarcerated"] > 0:
                    person["incarcerated"] -= 1
                    person["months_in_prison"] += 1
                    spread_infection(popu, person, itr, month, config)

            # sentencing step
            for person in agents:
                if person["infected"] and not person["incarcerated"]:
                    assign_sentence(person, itr, month, config)
                person["infected"] = 0

    return popu
