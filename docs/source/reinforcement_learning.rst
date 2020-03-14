.. _reinforcement-learning:

Sequential Decision Making
==========================
WhyNot is also an excellent test bed for sequential decision making and
reinforcement learning in diverse dynamic environments. WhyNot offers RL
environments compatible with the OpenAI Gym API style, so that existing code for
OpenAI Gym can be adapted for WhyNot with minimal changes.

Using Existing WhyNot Environments
----------------------------------
To see all available environments, 

.. code:: python
    
    import whynot.gym as gym
    for env in gym.envs.registry.all():
        print(env.id)

To create an environment, set the random seed, and get an initial observation,

.. code:: python

    env = gym.make('HIV-v0')
    env.seed(1)
    observation = env.reset()

To sample a random action and perform the random action, use the ``step`` 
function. The step function returns the reward, the next observation, whether 
the environment achieves a terminal state, and a dict of additional debugging 
info.

.. code:: python

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

The actions, observations, and rewards in the WhyNot Gym environment are all
represented as numpy arrays. The environment works with algorithms implemented
in any Python numerical computation library, such as PyTorch or TensorFlow.

See `this notebook <https://github.com/zykls/whynot/blob/master/examples/reinforcement_learning/hiv_simulator.ipynb>`_
for an example of training policies on the HIV environment.

Defining a New Custom Environment
---------------------------------
To define a new custom environment on top of a WhyNot simulator, implement 1)
the reward function, 2) a mapping from numerical actions to system
interventions, and, optionally, 3) a mapping from state to observation. The
class :class:`~whynot.gym.envs.ODEEnvBuilder` then wraps an arbitrary dynamical
system simulator into a Gym environment for reinforcement learning.

For example, we defined the HIV environment by 

.. code:: python

    from whynot.gym.envs import ODEEnvBuilder
    from whynot.simulators.hiv import Config, Intervention, State
    from whynot.simulators.hiv import simulate

    def reward_fn(intervention, state):
        reward = ...
        return reward

    def intervention_fn(action, time):
        action_to_intervention_map = ...
        return action_to_intervention_map[action]

    HivEnv = ODEEnvBuilder(
        # Specify the dynamical system
        simulate_fn=simulate,
        # Simulator configuration
        config=Config(),
        # Initial state to begin simulator
        initial_state=State(),
        # Define the action space
        action_space=spaces.Discrete(...)
        # Define the observation space
        observation_space=spaces.Box(...)
        # Convert numerical actions to simulator interventions
        intervention_fn=intervention_fn,
        # Define the reward function
        reward_fn=reward_fn,
    )
