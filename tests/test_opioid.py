"""Unit tests for opioid epidemic simulator."""
import whynot as wn


def test_config():
    """Ensure intervention update works as expected."""
    intervention = wn.opioid.Intervention(time=2021, nonmedical_incidence=-0.12)
    config = wn.opioid.Config()
    assert config.nonmedical_incidence.intervention_val == 0.0
    config = config.update(intervention)
    assert config.nonmedical_incidence.intervention_val == -0.12

    intervention = wn.opioid.Intervention(time=2021, illicit_incidence=1.2)
    config = wn.opioid.Config()
    assert config.illicit_incidence.intervention_val == 0.0
    config = config.update(intervention)
    assert config.illicit_incidence.intervention_val == 1.2
