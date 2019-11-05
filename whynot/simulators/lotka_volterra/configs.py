"""Default configurations for Lotka-Volterra simulation."""

DEFAULT_CONFIG = {
    "policy_year": 30,
    "growth_rate0": (1.5, 1.5),
    "carrying_capacity0": 100,
    "interaction01": (0.2, 0.2),
    "growth_rate1": (1.5, 1.5),
    "carrying_capacity1": 200,
    "interaction10": (1.6, 1.6),
}

DEFAULT_INITIAL_STATE = {
    "population0": 25,
    "population1": 80,
}

# Small change in the interaction coefficient
# after 30 years.
INTERVENTION_CONFIG = {
    "policy_year": 30,
    "growth_rate0": (1.5, 1.5),
    "carrying_capacity0": 100,
    "interaction01": (0.2, 0.2),
    "growth_rate1": (1.5, 1.5),
    "carrying_capacity1": 200,
    "interaction10": (1.6, 1.5),
}
