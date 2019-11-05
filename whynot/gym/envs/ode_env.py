"""Base environment class for ODE simulator based environments."""
from whynot.gym import Env
from whynot.gym.utils import seeding


class ODEEnv(Env):
    """Base environment class for ODE simulator based environments."""

    def __init__(self, simulate_fn, config, action_space, observation_space,
                 initial_state, timestep=1.0):
        """Initialize an environment class.

        Parameters
        ----------
            simulate_fn: A function with signature simulate(initial_state,
                config, intervention=None, seed=None) -> whynot.framework.Run
            config: An instance of whynot.simulators.infrastructure.BaseConfig.
            action_space: An instance of whynot.gym.spaces.Space.
            observation_space: An instance of whynot.gym.spaces.Space.
            initial_state: An instance of
                whynot.simulators.infrastructure.BaseState.
            timestep: A float.

        """
        self.config = config
        self.action_space = action_space
        self.observation_space = observation_space

        self.initial_state = initial_state
        self.state = self.initial_state

        self.simulate_fn = simulate_fn

        self.start_time = self.config.start_time
        self.terminal_time = self.config.end_time
        self.timestep = timestep
        self.time = self.start_time

        self.seed()

    def reset(self):
        """Reset the state."""
        self.state = self.initial_state
        self.time = self.start_time
        return self._get_observation(self.state)

    def seed(self, seed=None):
        """Set internal randomness of the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Perform one forward step in the environment.

        Parameters
        ----------
            action: A numpy array reprsenting an action of shape
                [1, action_dim].

        Returns
        -------
            observation: A numpy array of shape [1, obs_dim].
            reward: A numpy array of shape [1, 1].
            done: A numpy array of shape [1, 1]
            info_dict: An empty dict.

        """
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))
        intervention = self._get_intervention(action)
        # Set the start and end time in config to simulate one timestep.
        self.config.start_time = self.time
        self.config.end_time = self.time + self.timestep
        self.time += self.timestep
        # Get the next state from simulation.
        self.state = self.simulate_fn(
            initial_state=self.state, config=self.config,
            intervention=intervention)[self.time]
        done = bool(self.time >= self.terminal_time)
        reward = self._get_reward(intervention, self.state)
        return self._get_observation(self.state), reward, done, {}

    def render(self, mode='human'):
        """Render the environment, unused."""

    @staticmethod
    def _get_observation(state):
        """Convert a state to a numpy array observation.

        By default, returns the fully observed state in order listed in the
        State class.

        Parameters
        ----------
            state: An instance of whynot.simulators.infrastructure.BaseState.

        Returns
        -------
            A numpy array of shape [1, obs_dim].

        """
        return state.values()

    def _get_intervention(self, action):
        """Convert a numpy array action to an intervention.

        Parameters
        ----------
            action: A numpy array reprsenting an action of shape
                [1, action_dim].

        Returns
        -------
            An instance of whynot.simulators.infrastructure.BaseIntervention.

        """
        raise NotImplementedError

    def _get_reward(self, intervention, state):
        """Calculate reward from intervention and the next state.

        Parameters
        ----------
            intervention: An instance of
                whynot.simulators.infrastructure.BaseIntervention.
            state: An instance of whynot.simulators.infrastructure.BaseState.

        Returns
        -------
            A numpy array of shape [1, 1].

        """
        raise NotImplementedError
