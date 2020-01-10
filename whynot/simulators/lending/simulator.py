"""Implementation of lending in dynamics simulator based on Liu et al.

Liu, L., Dean, S., Rolf, E., Simchowitz, M., & Hardt, M. (2018, July). Delayed
Impact of Fair Machine Learning. In International Conference on Machine
Learning. (https://arxiv.org/abs/1803.04383)
"""
import copy
import dataclasses
import numpy as np
from typing import Callable

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics.

    Examples
    --------

    """
    #: Repayment rate function p(A, X) \in [0, 1].
    repayment_rate: Callable

    #: Maps the true credit score to the reported score
    credit_scorer: Callable = lambda x: x

    #: Lending threshold for group 0
    threshold_g0: float = 650
    #: Lending threshold for group 1
    threshold_g1: float = 650

    #: Bank repayment utility
    repayment_utility: float = 1.0
    #: Bank default utility
    default_utility: float = -4.0

    #: Applicant's score change after repayment
    repayment_score_change: float = 75
    #: Applican't score change after default
    default_score_change: float = -150

    #: Minimum credit score
    min_score: int = 350
    max_score: int = 800

    #: Simulation start step (in rounds)
    start_time: float = 0

    #: Simulation end step (in rounds)
    end_time: float = 1

    #: Simulator step size (Unused)
    delta_t: float = 1


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the lending simulator."""
    #: Group membership (sensitive attribute) 0 or 1
    group: int
    #: Agent credit score
    score: int
    #: Running total of the banks profit/loss for the agent
    profits: list


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the lending model.

    Examples
    --------

    """

    def __init__(self, time=100, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention (days)
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def lending_policy(config, group, score):
    """Determine whether or not a bank gives a loan."""
    # P(T = 1 | X, A=j) = 1     if X >= tau_j
    #                     0     otherwise.
    return (score > config.threshold_g0) ** (1 - group) * (
        score > config.threshold_g1
    ) ** group


def determine_repayment(rng, config, group, score):
    """Determine whether or not the bank repays the loan."""
    repayment_rate = config.repayment_rate(group, score)
    # Sample a Bernouli with the Gumbel-max trick to allow
    # dependency graph tracing.
    uniform = rng.uniform()
    return (
        np.log(repayment_rate / (1 - repayment_rate)) + np.log(uniform / (1 - uniform))
    ) > 0.5


def update_score(config, score, loan_approved, repaid):
    """Update the agent's score after a round."""
    score_change = (
        config.repayment_score_change ** repaid ** loan_approved
        * config.default_score_change ** (1 - repaid) ** loan_approved
        * 0.0 ** (1 - loan_approved)
    )

    return score + score_change


def update_profits(config, profits, loan_approved, repaid):
    """Update the banks profit for the individual."""
    profit_change = (
        config.repayment_utility ** repaid ** loan_approved
        * config.default_utility ** (1 - repaid) ** loan_approved
        * 0.0 ** (1 - loan_approved)
    )
    return profits + profit_change


def dynamics(rng, state, time, config, intervention=None):
    """Update equations for the lending simulaton.

    Parameters
    ----------
    Returns
    -------

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    group, score, individual_profits = state.values()

    # Credit bureau measures the agent's score
    measured_score = config.credit_scorer(score)

    # Bank decides whether or not to extend the user a loan
    loan_approved = lending_policy(config, group, measured_score)

    # The user (potentially) repays the loan
    repaid = determine_repayment(rng, config, group, score)

    # The credit score updates in response
    new_score = update_score(config, score, loan_approved, repaid)

    new_profits = update_profits(config, individual_profits, loan_approved, repaid)

    return State(group, new_score, new_profits)


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the lending simulator.

    The simulation starts at initial_state, representing an agent before
    interacting with the lending institution. The simulator evolves the agent
    state through (repeated) interaction between the agent and the lender.  The
    dynamics encapsulate how the lending decisions and policies effect both the
    agent and the lender's profit. The parameters of the dynamics, e.g. the
    lending thresholds or the repayment model, are specified in the Config.

    Parameters
    ----------
        initial_state:  `whynot.simulators.lending.State`
            Initial State object, which is used as x_{t_0} for the simulator.
        config:  `whynot.simulators.lending.Config`
            Config object that encapsulates the parameters that define the dynamics.
        intervention: `whynot.simulators.lending.Intervention`
            Intervention object that specifies what, if any, intervention to perform.
        seed: int
            Seed to set internal randomness.

    Returns
    -------
        run: `whynot.dynamics.Run`
            Rollout of the model.

    """
    rng = np.random.RandomState(seed)

    # Iterate the discrete dynamics
    state = copy.deepcopy(initial_state)
    times = [config.start_time]
    states = [state]
    for step in range(config.start_time, config.end_time):
        state = dynamics(rng, state, step, config, intervention)
        states.append(state)
        times.append(step + 1)

    return wn.dynamics.Run(states=states, times=times)


if __name__ == "__main__":
    print(simulate(State(), Config()))
