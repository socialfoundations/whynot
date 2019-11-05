"""Unit tests for setup and intervention in world2 simulator."""
from whynot.simulators.world2.simulator import *


def test_intervention():
    """Simple test if intervention is correctly handled."""
    config = Config(birth_rate=0.2, pollution=0.1)
    intervention = Intervention(pollution=37, time=1960)
    initial_state = State()

    world = WorldDynamics(config, intervention, initial_state)

    assert world.config.birth_rate == 0.2 and world.config.pollution == 0.1
    world.step(time=1950, delta_t=0.2)
    assert world.config.birth_rate == 0.2 and world.config.pollution == 0.1
    world.step(time=1980, delta_t=0.2)
    assert world.config.birth_rate == 0.2 and world.config.pollution == 37
    world.step(time=2000, delta_t=0.2)
    assert world.config.birth_rate == 0.2 and world.config.pollution == 37
    world.step(time=1950, delta_t=0.2)
    assert world.config.birth_rate == 0.2 and world.config.pollution == 0.1
