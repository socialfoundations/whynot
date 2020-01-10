.. _develop:

Extending Whynot
================
Adding more simulators and estimators to WhyNot is straightforward and allows
one to take advantage of WhyNot's framework for rapidly creating and running new
causal inference experiments, as well as benchmarking new methods on a common set
of tasks.

.. _adding-a-simulator:

Adding A Simulator
------------------

Adding a simulator to the package is straightforward. WhyNot is agnostic to the
underlying implementation of the simulator (programming language, runtime, etc),
provided it can be called from Python. 

All of the simulators can be found in ``whynot/simulators``. To create a new simulator
called ``new_simulator``, make a folder in ``whynot/simulators`` and create a file
called ``simulator.py``.

.. code:: bash
    
    mkdir whynot/simulators/new_simulator
    cd whynot/simulators/new_simulator
    touch simulator.py

The ``simulator.py`` file should contain the Python code needed to invoke the
simulator. This can either be the simulation code itself, as in the Lotka
Volterra simulator, or a Python wrapper to an underlying simulator.
At minimum, ``simulator.py`` must include both a ``Config`` class and a
``simulate`` function.

The ``Config`` class encapsulates the hyperparameters needed to run the
simulator.  A ``Config`` includes both things like the number of timesteps to
run the simulator as well as the values for key model parameters, e.g. the
coefficient of friction in a physics simulator. Thus, varying the ``Config`` values
gives different instances of the simulator. In WhyNot, a ``Config`` is
specified as Python ``dataclass``, which also allows for specifying default
values.

.. code:: python

    @dataclasses.dataclass
    class Config:
        # Parameter name: type = default_value
        gravity: float = 9.8
        timesteps: int = 400

In addition to a ``Config`` class, each simulator should implement an
``Intervention`` class that allows the user to specify interventions in the
simulator. For instance, in the previous example, an intervention might
correspond to changing ``gravity`` from `9.8` to `1.62` at timestep `200`.
The intervention class exposes all possible interventions the simulator supports
for the user.

.. code:: python
    
    @dataclasses.dataset
    class Intervention:
        # What timestep to execute the intervention
        timestep: int = 200
        # New gravity value after intervention. If None, then no intervention 
        # is performed.
        gravity: float = None


In the most general case, the ``simulate`` function takes as input a ``Config``,
optionally ``Intervention``, and a random ``seed`` and returns the results of
executing the simulator. This is deliberately vague to allow for a multiplicity
of different simulator types.  We instantiate this concept for dynamical system
simulators below.  For reproducibility, all simulators in WhyNot are required to
be deterministic given the random ``seed``.

.. code:: python

    def simulate(config, seed, intervention=None):
        # Seed the simulator randomness using seed
        # Execute simulator! 


Adding a dynamical system simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
WhyNot provides powerful support for dynamical system simulators.  In discrete
time, a dynamical system consists of a state :math:`x`, a set of parameters
:math:`\theta`, and a state-transition function :math:`f`, which evolves the
state at time :math:`t` according to

.. math::
    x_{t+1} = f(x_{t}; \theta).

As before, the parameters :math:`\theta` are encapsulated in a ``Config`` class.
Now, however, implementing a dynamical system simulator also requires
implementing a ``State`` class representing the variables :math:`x` that
change over time. In WhyNot, the state class is a Python ``dataclass``, and the
default values of the ``dataclass`` fields correspond to the default initial
state of the model.

.. code:: python

    @dataclasses.dataclass
    class State:
        # Each state variable is a separate field.
        state1: float = 0.0
        state2: float = 1.0
        state3: float = 2.0

Simulating a dynamical system requires specifying both the parameters
:math:`\theta` and the initial state :math:`x_0`. Therefore, for dynamical
systems, the ``simulate`` function takes an initial ``State`` object, a
``Config`` object, a random seed, and an optional ``Intervention`` object. The
function simulates the trajectory and returns a ``Run`` of the dynamical system.
A ``Run`` consists of the sequence of `states` :math:`x_{t_1}, x_{t_2}, x_{t_3},
\dots` visited by the system, and the sequence of sampled times :math:`t_1, t_2,
t_3, \dots` The code snippet gives an example implementation.

.. code:: python

    def simulate(initial_state, config, seed, intervention=None):
        # Seed randomness
        rng = np.random.RandomState(seed)

        # Run simulator from initial state with parameters `config`
        timesteps = list(range(0, 100))
        states = [initial_state]
        state = initial_state
        for time in timesteps:
            state = f(time, state, config, intervention, rng)
            states.append(state)
        return wn.dynamics.Run(states=states, times=timesteps)


.. _adding-estimators:

Adding An Estimator
-------------------

By design, Whynot supports adding new estimators to the framework, independent
of the language of implementation. Estimators with a Python interface can be
directly added to the package. This procedure is detailed in
:ref:`adding-python-estimators`. 

Estimators written in ``R`` or without a Python interface can be added to the
companion package ``whynot_estimators``. As estimators are added to Whynot, we
hope this will form the core of a common set of benchmark algorithms for causal
inference tasks.

.. _adding-python-estimators:

Adding Python Estimators
^^^^^^^^^^^^^^^^^^^^^^^^

Causal estimators with a Python interface are located in ``whynot/algorithms``.
To add an estimator, first create a file ``estimator_name.py`` in
``whynot/algorithms``.  

For estimators performing average and heterogeneous treatment effect estimation,
the main function to implement is ``estimate_treatment_effects``, which should
take as input ``covariates``, ``treatment``, and ``outcome``, and return a
:class:`~whynot.framework.InferenceResult` object.

.. code:: python

    from time import perf_counter

    def estimate_treatment_effect(covariates, treatment, outcome, *args, **kwargs):
		""" Estimate average (and possible heterogeneous) treatment effects.

         Parameters
            ----------
                covariates: `np.ndarray`
                    Array of shape [num_samples, num_features] of features.
                treatment:  `np.ndarray`
                    Array of shape [num_samples]  indicating treatment status for each sample.
                outcome:  `np.ndarray`
                    Array of shape [num_samples] containing the observed outcome for each sample.

            Returns
            -------
                result: `whynot.framework.InferenceResult`
                    InferenceResult object for this procedure

        """
        start_time = perf_counter()
        # Perform inference!
        stop_time = perf_counter()
        
        return InferenceResult(ate=average_treatment_effect, 
                               stderr=standard_error,
                               ci=(lower_bound, upper_bound),
                               individual_effects=heterogeneous_treatment_effects,
                               elapsed_time=stop_time - start_time)

To add the estimator to the :func:`~whynot.causal_suite`, add it to the function
``causal_suite`` in ``whynot.causal_suite.py`` 
