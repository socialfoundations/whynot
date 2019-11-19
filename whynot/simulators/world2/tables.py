"""Collection of tabular functions used in the world2 simulation."""
import numpy as np


class Table:  # pylint: disable=too-few-public-methods
    """Generic tabular function used in world2."""

    def __init__(self, inputs, outputs):
        """Set the fixed inputs and outputs of the function."""
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, query):
        """Given query, returns the interpolated function value."""
        return np.interp(query, self.inputs, self.outputs)


BIRTH_RATE_FROM_MATERIAL = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.2, 1.0, 0.85, 0.75, 0.7, 0.7]
)

BIRTH_RATE_FROM_CROWDING = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.05, 1.0, 0.9, 0.7, 0.6, 0.55]
)

BIRTH_RATE_FROM_FOOD = Table([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.6, 1.9, 2.0])

BIRTH_RATE_FROM_POLLUTION = Table(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [1.02, 0.9, 0.7, 0.4, 0.25, 0.15, 0.1]
)

DEATH_RATE_FROM_MATERIAL = Table(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    [3.0, 1.8, 1.0, 0.8, 0.7, 0.6, 0.53, 0.5, 0.5, 0.5, 0.5],
)

DEATH_RATE_FROM_POLLUTION = Table(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [0.92, 1.3, 2.0, 3.2, 4.8, 6.8, 9.2]
)

DEATH_RATE_FROM_FOOD = Table(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    [30.0, 3.0, 2.0, 1.4, 1.0, 0.7, 0.6, 0.5, 0.5],
)

DEATH_RATE_FROM_CROWDING = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.9, 1.0, 1.2, 1.5, 1.9, 3.0]
)

FOOD_FROM_CROWDING = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2.4, 1.0, 0.6, 0.4, 0.3, 0.2]
)

FOOD_POTENTIAL_FROM_CAPITAL_INVESTMENT = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.5, 1.0, 1.4, 1.7, 1.9, 2.05, 2.2]
)

CAPITAL_INVESTMENT_MULTIPLIER_TABLE = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 1.0, 1.8, 2.4, 2.8, 3.0]
)

FOOD_FROM_POLLUTION = Table(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [1.02, 0.9, 0.65, 0.35, 0.2, 0.1, 0.05]
)

POLLUTION_FROM_CAPITAL = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.05, 1.0, 3.5, 5.4, 7.4, 8.0]
)

POLLUTION_ABSORPTION_TIME_TABLE = Table(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [0.6, 2.5, 5.0, 8.0, 11.5, 15.5, 20.0]
)

CAPITAL_FRACTION_INDICATE_BY_FOOD_RATIO_TABLE = Table(
    [0.0, 0.5, 1.0, 1.5, 2.0], [1.0, 0.6, 0.3, 0.15, 0.1]
)

QUALITY_OF_LIFE_FROM_MATERIAL = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.2, 1.0, 1.7, 2.3, 2.7, 2.9]
)

QUALITY_OF_LIFE_FROM_CROWDING = Table(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    [2.0, 1.3, 1.0, 0.75, 0.55, 0.45, 0.38, 0.3, 0.25, 0.22, 0.2],
)

QUALITY_OF_LIFE_FROM_FOOD = Table([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.8, 2.4, 2.7])

QUALITY_OF_LIFE_FROM_POLLUTION = Table(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [1.04, 0.85, 0.6, 0.3, 0.15, 0.05, 0.02]
)

NATURAL_RESOURCE_EXTRACTION = Table(
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.15, 0.5, 0.85, 1.0]
)

NATURAL_RESOURCES_FROM_MATERIAL = Table(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [0.0, 1.0, 1.8, 2.4, 2.9, 3.3, 3.6, 3.8, 3.9, 3.95, 4.0],
)

CAPITAL_INVESTMENT_FROM_QUALITY = Table(
    [0.0, 0.5, 1.0, 1.5, 2.0], [0.7, 0.8, 1.0, 1.5, 2.0]
)
