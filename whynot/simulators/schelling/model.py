"""Implements the classic Schelling model in the Mesa framework."""

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector


class SchellingAgent(Agent):
    """Schelling segregation agent model."""

    def __init__(self, pos, model, agent_type, homophily):
        """Create a new Schelling agent.

        Parameters
        ----------
            unique_id: Unique identifier for the agent.
            x, y: Agent initial location.
            agent_type: Indicator for the agent's type (minority=1, majority=0)

        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.homophily = homophily

    def step(self):
        """Update the agent model after one step."""
        similar = 0
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if similar < self.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1


class Schelling(Model):
    """Model class for the Schelling segregation model."""

    def __init__(
        self,
        height=20,
        width=20,
        density=0.8,
        minority_pc=0.2,
        homophily=3,
        education_boost=0,
        education_pc=0.2,
        seed=None,
    ):
        """Seed is used to set randomness in the __new__ function of the Model superclass."""
        # pylint: disable-msg=unused-argument,super-init-not-called

        self.height = height
        self.width = width
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily
        self.education_boost = education_boost
        self.education_pc = education_pc

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(height, width, torus=True)

        self.happy = 0
        self.datacollector = DataCollector(
            {"happy": "happy"},  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
        )

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for cell in self.grid.coord_iter():
            x_coord = cell[1]
            y_coord = cell[2]
            if self.random.random() < self.density:
                if self.random.random() < self.minority_pc:
                    agent_type = 1
                else:
                    agent_type = 0

                agent_homophily = homophily
                if self.random.random() < self.education_pc:
                    agent_homophily += self.education_boost

                agent = SchellingAgent(
                    (x_coord, y_coord), self, agent_type, agent_homophily
                )
                self.grid.position_agent(agent, (x_coord, y_coord))
                self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """Run one step of the model. If All agents are happy, halt the model."""
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

        if self.happy == self.schedule.get_agent_count():
            self.running = False
