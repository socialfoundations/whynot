"""Credit simulator initialization."""

from whynot.simulators.credit.simulator import (
    agent_model,
    Config,
    dynamics,
    Intervention,
    simulate,
    State,
)

from whynot.simulators.credit.dataloader import CreditData

SUPPORTS_CAUSAL_GRAPHS = True
