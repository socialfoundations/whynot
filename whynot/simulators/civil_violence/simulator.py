"""Simulate and analyze runs from the civil violence model."""
import dataclasses

from mesa.datacollection import DataCollector
import numpy as np

from whynot.simulators.civil_violence.model import CivilViolenceModel


@dataclasses.dataclass
class Agent:
    # pylint: disable-msg=too-few-public-methods
    """Covariates for a single agent in the simulation.

    Examples
    --------
    >>> # An agent with low risk aversion and high hardship
    >>> civil_violence.Agent(hardship=0.99, risk_aversion=0.01)

    """

    #: How aggrieved is the agent by external circumstances.
    hardship: float = 0.5
    #: How legitimate does the agent perceive the current regime.
    legitimacy: float = 0.5
    #: Threshold above which agent starts to openly rebell
    active_threshold: float = 0.1
    #: How likely the agent is to rebel for a fixed greviance level.
    risk_aversion: float = 0.5
    #: How many adjacent squares an agent sees and uses to determine probability of arrest.
    vision: int = 3


@dataclasses.dataclass
class Config:
    # pylint: disable-msg=too-few-public-methods
    """Metaparameters for a single run of the civil violence simulator."""

    #: Vertical grid size
    grid_height: int = 50
    #: Horizontal grid size
    grid_width: int = 50
    #: What fraction of the agents are police
    cop_fraction: float = 0.05
    #: How many adjacent squares a cop can see when determining who to arrest.
    cop_vision: int = 5

    # What's the longest time a citizen can stay in jail?
    max_jail_term: int = 5
    #: How strongly other agent grievances affect each agent in prison.
    prison_interaction: float = 0.1
    #: A fixed parameter to calibrate arrest likelihood.
    arrest_prob_constant: float = 2.3


def count_type_citizens(model, condition, exclude_jailed=True):
    """Count agents as either Quiescent/Active."""
    count = 0
    for agent in model.schedule.agents:
        if agent.breed == "cop":
            continue
        if exclude_jailed and agent.jail_sentence:
            continue
        if agent.condition == condition:
            count += 1
    return count


def count_jailed(model):
    """Count number of jailed agents."""
    count = 0
    for agent in model.schedule.agents:
        if agent.breed == "citizen" and agent.jail_sentence:
            count += 1
    return count


def simulate(agents, config, seed=None, max_steps=1000):
    """Simulate a run of the civil violence model.

    Parameters
    ----------
        agents: list
            List of whynot.simulators.civil_violence.Agent to populate the model
        config: whynot.simulators.civil_violence.Config
            Simulation parameters
        seed: int
            (Optional) Seed for all randomness in model setup and execution.
        max_steps: int
            Maximum number of steps to run the civil_violence model.

    Returns
    -------
        observations: pd.DataFrame
            Pandas dataframe containing the "observations" recorded for each
            agent. Observations are defined in the `agent_reporter` and include
            agent attributes along with:
                "pos" # position on the grid
                "jail_sentence" # agent's jail sentence at model end
                "condition"  # agent's condition (rebelling or acquiesent) at model end
                "arrest_probability" # agent's probability of arrest
                "arrests" # number of time agent has been arrested
                "days_active"  # how long as the agent spent in rebellion

    """
    # Ensure everything will fit on the grid
    num_cells = config.grid_height * config.grid_width
    num_cops = int(np.floor(len(agents) * config.cop_fraction))

    assert len(agents) + num_cops < num_cells

    model = CivilViolenceModel(
        height=config.grid_height,
        width=config.grid_width,
        cop_vision=config.cop_vision,
        max_jail_term=config.max_jail_term,
        prison_interaction=config.prison_interaction,
        arrest_prob_constant=config.arrest_prob_constant,
        max_steps=max_steps,
        seed=seed,
    )
    # Place agents on grid
    for i, agent in enumerate(agents):
        model.add_agent(
            i,
            model.find_empty(),
            agent.hardship,
            agent.legitimacy,
            agent.risk_aversion,
            agent.active_threshold,
            agent.vision,
        )

    for i in range(num_cops):
        model.add_cop(i + len(agents), model.find_empty())

    # Which attributes to report
    agent_reporters = {
        "pos": "pos",
        "breed": "breed",
        "jail_sentence": "jail_sentence",
        "condition": "condition",
        "arrest_probability": "arrest_probability",
        "arrests": "arrests",
        "hardship": "hardship",
        "regime_legitimacy": "regime_legitimacy",
        "days_active": "days_active",
        "risk_aversion": "risk_aversion",
        "threshold": "threshold",
        "arrest_parameter": "arrest_parameter",
        "vision": "vision",
    }

    datacollector = DataCollector(agent_reporters=agent_reporters)
    while model.running:
        model.step()
    datacollector.collect(model)
    dataframe = datacollector.get_agent_vars_dataframe()
    observations = dataframe[dataframe.breed == "citizen"].drop(columns="breed")
    return observations
