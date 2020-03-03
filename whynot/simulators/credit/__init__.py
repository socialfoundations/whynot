"""Credit simulator initialization."""

from whynot.simulators.credit.simulator import (
    Config,
    evaluate_loss,
    dynamics,
    Intervention,
    simulate,
    State,
)

from whynot.simulators.credit.dataloader import CreditData

SUPPORTS_CAUSAL_GRAPHS = True
