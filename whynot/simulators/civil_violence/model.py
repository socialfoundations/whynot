"""Implementation of Epstein's Civil Violence model."""
import math

from mesa import Agent
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import Grid


class Citizen(Agent):
    # pylint: disable-msg=too-many-instance-attributes
    """A member of the general population, may or may not be in active rebellion.

    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes
    ----------
        unique_id: unique int
        x, y: Grid coordinates
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion

    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        hardship,
        regime_legitimacy,
        risk_aversion,
        threshold,
        vision,
    ):
        """Create a new Citizen.

        Parameters
        ----------
            unique_id: unique int
            x, y: Grid coordinates
            hardship: Agent's 'perceived hardship (i.e., physical or economic
                privation).' Exogenous, drawn from U(0,1).
            regime_legitimacy: Agent's perception of regime legitimacy, equal
                across agents.  Exogenous.
            risk_aversion: Exogenous, drawn from U(0,1).
            threshold: if (grievance - (risk_aversion * arrest_probability)) >
                threshold, go/remain Active
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance

        """
        super().__init__(unique_id, model)
        self.breed = "citizen"
        self.pos = pos
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.threshold = threshold
        self.condition = "Quiescent"
        self.vision = vision
        self.jail_sentence = 0
        self.grievance = lambda: self.hardship * (1 - self.regime_legitimacy)
        self.arrest_probability = None
        self.arrest_parameter = None
        self.arrests = 0
        self.prison_interaction = model.prison_interaction
        self.history = [self.grievance()]
        self.days_active = 0

        self.neighborhood = None
        self.neighbors = None
        self.empty_neighbors = None

    def update_arrest_parameter(self):
        """Change probability of arrest based on grievance."""
        self.arrest_parameter = (
            self.grievance() - self.arrest_probability * self.regime_legitimacy
        )

    def step(self):
        """Decide whether to activate, then move if applicable."""
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.

        if self.condition == "Active":
            self.days_active += 1

        self.update_neighbors()
        self.update_estimated_arrest_probability()
        self.update_arrest_parameter()
        net_risk = self.risk_aversion * self.arrest_probability
        if (
            self.condition == "Quiescent"
            and (self.grievance() - net_risk) > self.threshold
        ):
            self.condition = "Active"
        elif (
            self.condition == "Active"
            and (self.grievance() - net_risk) <= self.threshold
        ):
            self.condition = "Quiescent"
        if self.model.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.model.grid.move_agent(self, new_pos)

    def update_neighbors(self):
        """Look around and see who my neighbors are."""
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)
        self.empty_neighbors = [
            c for c in self.neighborhood if self.model.grid.is_cell_empty(c)
        ]

    def update_estimated_arrest_probability(self):
        """Estimate p(Arrest | I go active) based on local ratio of cops to actives."""
        cops_in_vision = len([c for c in self.neighbors if c.breed == "cop"])
        actives_in_vision = 1.0  # citizen counts herself
        for unit in self.neighbors:
            if (
                unit.breed == "citizen"
                and unit.condition == "Active"
                and unit.jail_sentence == 0
            ):
                actives_in_vision += 1
        self.arrest_probability = 1 - math.exp(
            -1 * self.model.arrest_prob_constant * (cops_in_vision / actives_in_vision)
        )

    def update_risk_aversion(self):
        """Change risk aversion for jailed citizens based on cellmates."""
        cellmates_risk_aversion = [
            c.risk_aversion
            for c in self.neighbors
            if ((c.breed == "citizen") and (c.jail_sentence > 0))
        ]
        if cellmates_risk_aversion:
            min_aversion = max(cellmates_risk_aversion)
            diff = self.prison_interaction * (min_aversion - self.risk_aversion)
            self.risk_aversion += diff


class Cop(Agent):
    # pylint: disable-msg=too-many-instance-attributes
    """A cop for life. No defection.

    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes
    ----------
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect

    """

    def __init__(self, unique_id, model, pos, vision):
        """Create a new Cop.

        Parameters
        ----------
            unique_id: unique int
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance

        """
        super().__init__(unique_id, model)
        self.breed = "cop"
        self.pos = pos
        self.vision = vision

        # Useless variables so BatchRunner wouldn't complain.
        self.jail_sentence = None
        self.condition = None
        self.arrest_probability = None
        self.arrests = None
        self.hardship = None
        self.regime_legitimacy = None
        self.risk_aversion = None
        self.threshold = None
        self.grievance = None
        self.arrests = None
        self.days_active = 0
        self.arrest_parameter = None

        self.neighborhood = None
        self.neighbors = None
        self.empty_neighbors = None

    def step(self):
        """Inspect neighborhood and arrest a random active agent. Move if applicable."""
        self.update_neighbors()
        active_neighbors = []
        for agent in self.neighbors:
            if (
                agent.breed == "citizen"
                and agent.condition == "Active"
                and agent.jail_sentence == 0
            ):
                active_neighbors.append(agent)
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            #             sentence = self.random.randint(0, self.model.max_jail_term)
            sentence = self.model.max_jail_term
            arrestee.jail_sentence = sentence
            arrestee.arrests += 1
            # arrestee is made quiescent if placed in Jail
            arrestee.condition = "Quiescent"
            # update perceived regime legitimacy if agent is arrested
            arrestee.update_risk_aversion()

        if self.model.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.model.grid.move_agent(self, new_pos)

    def update_neighbors(self):
        """Look around and see who my neighbors are."""
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)
        self.empty_neighbors = [
            c for c in self.neighborhood if self.model.grid.is_cell_empty(c)
        ]


class RandomActivation(BaseScheduler):
    """A scheduler which activates each agent once per step, in random order.

    The order reshuffled every step.
    This is equivalent to the NetLogo 'ask agents...' and is generally the
    default behavior for an ABM. Assumes that all agents have a step(model) method.
    """

    def step(self):
        """Execute the step of all agents, one at a time, in random order."""
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
        self.steps += 1
        self.time += 1


class CivilViolenceModel(Model):
    """Model encapsulates the entire civil violence simulator and interactions."""

    def __init__(
        self,
        height,
        width,
        cop_vision,
        max_jail_term,
        prison_interaction,
        arrest_prob_constant=2.3,
        movement=True,
        max_steps=1000,
        seed=None,
    ):
        """Seed is used to set randomness in the __new__ function of the Model superclass."""
        # pylint: disable-msg=unused-argument,super-init-not-called
        super().__init__()
        self.height = height
        self.width = width
        self.cop_vision = cop_vision
        self.max_jail_term = max_jail_term

        self.arrest_prob_constant = arrest_prob_constant
        self.movement = True
        self.running = True
        self.max_steps = max_steps
        self.iteration = 0
        self.prison_interaction = prison_interaction
        self.schedule = RandomActivation(self)
        self.grid = Grid(height, width, torus=True)

    def add_agent(
        self,
        unique_id,
        pos,
        hardship,
        legitimacy,
        risk_aversion,
        active_threshold,
        citizen_vision,
    ):
        """Add a new agent to the grid."""
        citizen = Citizen(
            unique_id,
            self,
            pos,
            hardship=hardship,
            regime_legitimacy=legitimacy,
            risk_aversion=risk_aversion,
            threshold=active_threshold,
            vision=citizen_vision,
        )
        x_coord, y_coord = pos
        self.grid[y_coord][x_coord] = citizen
        self.schedule.add(citizen)

    def add_cop(self, unique_id, pos):
        """Add a new copy to the grid."""
        cop = Cop(unique_id, self, pos, vision=self.cop_vision)
        x_coord, y_coord = pos
        self.grid[y_coord][x_coord] = cop
        self.schedule.add(cop)

    def find_empty(self):
        """Find a random empty location in the grid."""
        if self.grid.exists_empty_cells():
            pos = self.random.choice(sorted(self.grid.empties))
            return pos
        return None

    def step(self):
        """Advance the model by one step and collect data."""
        self.schedule.step()
        self.iteration += 1
        if self.iteration > self.max_steps:
            self.running = False
