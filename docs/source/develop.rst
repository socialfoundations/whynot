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
called ``simulator.py``. For an simple example, see the `lotka volterra
simulator
<https://github.com/zykls/whynot/blob/master/whynot/simulators/lotka_volterra/simulator.py>`_.

Implementing a simulator requires implementing (1) a ``Config`` class to specify
parameters of the simulator, (2) an ``Intervention`` class to specify changes to
the simulator during execution, and (3) a ``simulate`` function to execute the
simulator.

The ``Config`` class encapsulates the hyperparameters needed to run the
simulator.  A ``Config`` includes both things like the number of steps to
run the simulator as well as the values for key model parameters, e.g. the
coefficient of friction in a physics simulator. Thus, varying the ``Config`` values
gives different instances of the simulator. In WhyNot, a ``Config`` is
specified as Python ``dataclass``, which also allows for specifying default
values.

.. code:: python

    @dataclasses.dataclass
    class Config(BaseConfig):
        """Parameterization of Lotka-Volterra dynamics.

        Examples
        --------
        >>> # Configure the simulator so each caught rabbit creates 2 foxes
        >>> lotka_volterra.Config(fox_growth=0.5)

        """

        # Dynamics parameters
        #: Natural growth rate of rabbits, when there's no foxes.
        rabbit_growth: float = 1.0
        #: Natural death rate of rabbits, due to predation.
        rabbit_death: float = 0.1
        #: Natural death rate of fox, when there's no rabbits.
        fox_death: float = 1.5
        #: Factor describing how many caught rabbits create a new fox.
        fox_growth: float = 0.75

        # Simulator book-keeping
        #: Start time of the simulator (in years).
        start_time: float = 0
        #: End time of the simulator (in years).
        end_time: float = 100
        #: Spacing of the evaluation grid
        delta_t: float = 1.0
    

In addition to a ``Config`` class, each simulator should implement an
``Intervention`` class that allows the user to specify interventions in the
simulator. For instance, in the previous example, an intervention might
correspond to changing ``fox_growth`` from `1.0` to `2.0` at time `20`.
The intervention class exposes all possible interventions the simulator supports
for the user.

.. code:: python

    class Intervention:
        """Parameterization of an intervention to the lotka-volterra simulator"""

        def __init__(self, time, fox_growth=None, rabbit_growth=None):
            """Specify an intervention in the dynamical system.

            Parameters
            ----------
                time: int
                    Time of the intervention.
                fox_growth: float
                    New value of fox_growth after intervention. If None, no
                    change.

            """
            self.time = time
            # Parameters to update after intervention
            self.updates = {}
            if fox_growth:
                self.updates["fox_growth"] = fox_growth


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

The parameters :math:`\theta` are encapsulated in a ``Config`` class. In
addition to a ``Config`` and ``Intervetion``,  implementing a dynamical system
simulator also requires implementing a ``State`` class representing the
variables :math:`x` that change over time. In WhyNot, the state class is a
Python ``dataclass``, and the default values of the ``dataclass`` fields
correspond to the default initial state of the model. The ``State``, ``Config``,
and ``Intervention`` should inherit from :class:`~whynot.dynamics.BaseState`,
:class:`~whynot.dynamics.BaseConfig`, and
:class:`~whynot.dynamics.BaseIntervention`, respectively.

.. code:: python

        @dataclasses.dataclass
        class State(BaseState):
            """State of the Lotka-Volterra model."""

            #: Number of rabbits.
            rabbits: float = 10.0
            #: Number of foxes.
            foxes: float = 5.0

The ``simulate`` function takes an initial ``State`` object, a
``Config`` object, a random seed, and an optional ``Intervention`` object. The
function simulates the trajectory and returns a ``Run`` of the dynamical system.
A :class:`~whynot.dynamics.Run` consists of the sequence of `states`
:math:`x_{t_1}, x_{t_2}, x_{t_3}, \dots` visited by the system, and the sequence
of sampled times :math:`t_1, t_2, t_3, \dots` The code snippet gives an example
implementation.

.. code:: python

    def dynamics(time, state, config, intervention, rng):
        """Single time step of the dynamics."""
        
        # Intervene on simulator parameters
        if intervention and time >= intervention.time:
            config.update(intervention)
        
        new_state = ...
        return new_state

    def simulate(initial_state, config, seed, intervention=None):
        """Run a complete trajectory for the simulator."""
        # Seed randomness
        rng = np.random.RandomState(seed)

        # Run simulator from initial state with parameters `config`
        timesteps = list(range(0, 100))
        states = [initial_state]
        state = initial_state
        for time in timesteps:
            state = dynamics(time, state, config, intervention, rng)
            states.append(state)
        return wn.dynamics.Run(states=states, times=timesteps)


.. _adding-estimators:

Adding An Estimator
-------------------
WhyNot ships with a small number of causal estimators, with a larger number
available through the companion package ``whynot_estimators``. Most users will
either use these estimators or implement their own to run experiments on top of
data generated by WhyNot. However, Whynot also supports adding new estimators to
the package, which can then be accessed and experimented with by other users.

Estimators with a Python interface can be directly added to the package. This
procedure is detailed below.  Estimators written in other languages like ``R``
or without a Python interface can be added to the companion package
``whynot_estimators``. As estimators are added to Whynot, we hope this will form
the core of a common set of benchmark algorithms for causal inference tasks.

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
