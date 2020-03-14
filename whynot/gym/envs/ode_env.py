"""Environment builder for simulators based on dynamical systems."""
import inspect

from whynot.gym import Env
from whynot.gym.utils import seeding


class ODEEnvBuilder(Env):
    """Environment builder for simulators derived from dynamical systems."""

    def __init__(
        self,
        simulate_fn,
        config,
        action_space,
        observation_space,
        initial_state,
        intervention_fn,
        reward_fn,
        observation_fn=None,
        timestep=1.0,
    ):
        """Initialize an environment class.

        Parameters
        ----------
            simulate_fn:  Callable
                A function with signature
                simulate(initial_state, config, intervention=None, seed=None)
                -> whynot.dynamics.Run
            config: whynot.dynamics.BaseConfig
                The base simulator configuration
            action_space: whynot.gym.spaces.Space
                The action space for the reinforcement learner
            observation_space: whynot.gym.spaces.Space
                The space of observations for the agent
            initial_state: whynot.dynamics.BaseState
                The initial state of the simulator
            intervention_fn: Callable
                A function that maps actions to simulator interventions with signature
                get_intervention(action, time) -> whynot.dynamics.BaseState
            reward_fn: Callable
                A function that computes the cost/reward of taking
                an intervention in a particular state state with signature
                get_reward(intervention, state) -> float
            observation_fn: Callable
                (Optional) A function that computes the observed state for the
                state of the simulator with signature
                observation_fn(state) -> np.ndarray.
                If ommitted, the entire simulator state is returned.
            timestep: float
                Time between successive observations in the dynamical system.

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

        self.intervention_fn = intervention_fn
        self.reward_fn = reward_fn

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
        intervention = self.intervention_fn(action, self.time)
        # Set the start and end time in config to simulate one timestep.
        self.config.start_time = self.time
        self.config.end_time = self.time + self.timestep
        self.time += self.timestep
        # Get the next state from simulation.
        self.state = self.simulate_fn(
            initial_state=self.state, config=self.config, intervention=intervention
        )[self.time]
        done = bool(self.time >= self.terminal_time)
        reward = self._get_reward(intervention, self.state)
        return self._get_observation(self.state), reward, done, {}

    def render(self, mode="human"):
        """Render the environment, unused."""

    @staticmethod
    def _get_observation(state):
        """Convert a state to a numpy array observation.

        By default, returns the fully observed state in order listed in the
        State class.

        Parameters
        ----------
            state: An instance of whynot.dynamics.BaseState.

        Returns
        -------
            A numpy array of shape [1, obs_dim].

        """
        return state.values()

    @staticmethod
    def _get_args(func):
        """Return the arguments to the function."""
        return inspect.signature(func).parameters

    def _get_reward(self, intervention, state):
        """Return the reward obtained by intervening in the given state."""
        reward_args = self._get_args(self.reward_fn)
        kwargs = {}
        if "config" in reward_args:
            kwargs["config"] = self.config
        if "time" in reward_args:
            kwargs["time"] = self.time
        return self.reward_fn(intervention=intervention, state=state, **kwargs)

    def __call__(self):
        """Return the class, as if this function were calling the constructor."""
        self.reset()
        return self
