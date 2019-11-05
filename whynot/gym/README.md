# WhyNot Gym

WhyNot Gym is an [OpenAI Gym](https://github.com/openai/gym) style API for
benchmarking RL methods on environments based on WhyNot simulators.

When the user imports
```
import whynot.gym as gym
```
instead of 
```
import gym
```
from the OpenAI Gym package, most of the functionalites should follow exactly
the same.

## Environment
Each environment in WhyNot has the following internal attributes.
* config: A Config object with the parameters in the environment.
* time: The current time in the environment.
* state: The current state in the environment.
* timestep: Size of each time step.
* initial\_state: The initial state.
* action\_space: The action space, either discrete (Discrete) or continuous (Box). The Discrete space allows a fixed range of non-negative numbers. The Box space represents an n-dimensional real vector in a bounded range.
* observation\_space: The observation\_space, similarly can be either discrete or continuous.

## Action
An action corresponds to an intervention setting the values of intervenable
variables. When the variables are continuous, the action space is Box type with
dimension = number of intervenable parameters in the config. When the variables
are discrete, the action space is discrete with number of possible actions =
(# options for variable 1) * … * (# options for variable k).

Each environment has an internal function to convert an action (represented by a
numpy array) to an intervention object in the WhyNot simulator framework that
specifies how to update the simulation config.

## Observation
The environments can be partially observed or fully observed. Similar to the
action space, the observation space can also be discrete or continous. Each
environment has an internal function to convert the current state into an
observation.

## Initialize an environment
All environments are registered to a registry by name. To initialize an
environment, the gym.make function creates an environent, and env.reset
initializes the environment to initial observation and sets time to 0.
```
env = gym.make('ODE-HIV').
initial_obs = env.reset()
```

## Step
The step function takes an action, transitions to the next state according to
system dynamics, and returns the observation and reward.
```
obs, reward, done, info = env.step(action)
```
Internally, the step function converts the action into an intervention, and then
calls the simulator's simulate function with initial\_state=self.state,
config=self.config, intervention=self._get_intervention(action), the simulate
function computes the next state at current time + self.timestep.

The outputs of the step function are
* observation (object): numpy array representing the (full or partially
observed) next state or dict of numpy arrays.
* reward (float): amount of reward achieved by the previous action. 
* done (boolean): whether current time has achieved the terminal time. 
* info (dict): diagnostic information useful for debugging, leave empty for now.

## Reward
TODO: We need to define a reward function for each environment, possibly defined
as reward(state) - cost(intervention). For example, in the ODE-based HIV
environment, the reward at each step is defined as S * immune response - Q *
free virus ammount - R1 * dosage1 - R2 * dosage2, following existing literature
on the HIV simulator.

Another possibility is to set it up as a “goal-based environment”. In this
framing, the objective for the RL agent would be to achieve a desired goal
state. The OpenAI Gym defines a goal-based environment as a special case of the
general environment. Its observation space is a dict with keys ‘observation’,
‘desired\_goal’, and ‘achieved\_goal’. 

## Render
TODO: Renders a visualization of the environment, e.g. as a matplotlib plot of the state variables.

