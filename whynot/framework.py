"""Base experimental class."""
import copy
import dataclasses
import inspect
import itertools
import random
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from whynot import utils
from whynot import causal_graphs


PARAM_COLLECTION = "__parameter_collection__"


@dataclasses.dataclass
class Dataset:
    # pylint:disable-msg=too-few-public-methods
    """Observational dataset and grouth truth unit-level effects.

    Attributes
    ----------
        covariates: `np.ndarray`
            Array of shape [num_samples, num_features] of covariates for each unit.
        treatments:  `np.ndarray`
            Array of shape [num_samples] indicating treatment status for each unit.
        outcomes:  `np.ndarray`
            Array of shape [num_samples] containing the observed outcome for each unit.
        true_effects: `np.ndarray`
            Array of shape [num_samples] containing the unit-level treatment
            effects, :math:`Y_i(1) - Y_i(0)` for each :math:`i`.
        sate: float
            Sample average treatment effect based on ground truth unit-level effects.
        causal_graph: networkx.DiGraph
            If supported by the simulator and experiment, the causal graph associated with the data.

    """

    covariates: np.ndarray
    treatments: np.ndarray
    outcomes: np.ndarray
    true_effects: np.ndarray
    causal_graph: Any = None

    @property
    def sate(self):
        """Return the sample average treatment effect."""
        return np.mean(self.true_effects)

    def bootstrap_sample_ate_ci(self, num_bootstrap_samples=2000):
        """Bootstrap a 95% confidence interval for the sample ate."""
        means = []
        for _ in range(num_bootstrap_samples):
            sample = np.random.choice(
                self.true_effects, size=len(self.true_effects), replace=True)
            means.append(np.mean(sample))
        return (np.quantile(means, 0.025), np.quantile(means, 0.975))


@dataclasses.dataclass
class Run:
    r"""Encapsulate a trajectory from a dynamical system simulator.

    Attributes
    ----------
        states:
            Sequence of states :math:`x_{t_1}, x_{t_2}, \dots` produced by the
            simulator.
        times:
            Sequence of sampled times :math:`{t_1},{t_2}, \dots` at which states
            are recorded.

    Examples
    --------
    >>> state_at_year_2015 = run[2015]

    """

    states: list
    times: list

    def __getitem__(self, time):
        """Return the state at the given time."""
        time_index = np.argmin(np.abs(np.array(self.times) - time))
        return self.states[time_index]

    @property
    def initial_state(self):
        """Return initial state of the run."""
        return self.states[0]


@dataclasses.dataclass
class InferenceResult:
    # pylint:disable-msg=too-few-public-methods
    """Object to store results of causal inference method.

    Attributes
    ----------
        ate:
            Estimated average treatment effect
        stderr:
            Reported standard error of the ATE estimate. Only available
            if supported by the method, otherwise None.
        ci:
            Reported 95% confidence interval for the ATE (lower_bound, upper_bound).
            Only available if supported by the method, otherwise None.
        individual_effects:
            Heterogeneous treatment effect for each unit.
            Only available if supported by the method, otherwise None.
        elapsed_time:
            How long in second (wall-clock time) it took to produce the estimate.

    """

    ate: float = None
    stderr: float = None
    ci: tuple = (None, None)  # pylint: disable-msg=invalid-name
    individual_effects: np.ndarray = None
    elapsed_time: float = None


@dataclasses.dataclass
class ExperimentParameter:
    # pylint:disable-msg=too-few-public-methods
    """Container for a parameter to vary in an experiment.

    Attributes
    ----------
        name:
            Name of the parameter
        description:
            Parameter description
        default:
            Default (uninitialized) value of the parameter
        values:
            Iterator of parameter values that supports sampling (for random search).

    """

    name: str
    default: Any
    values: Any = None
    description: str = ""


def parameter(name, default, values=None, description=""):
    """Decorate functions in an experiment with parameters.

    Usage:
        @parameter(name="treatment-bias", default=0.2, values=[0.1, 0.2, 0.5, 0.9],
                   description="How much is treatment biased for control group.")
        def propensity_score(control_run, treatment_bias)
            ....
    """
    exp_param = ExperimentParameter(name, default, values, description)

    def parameter_decorator(func):
        """Attach parameter collection object to the function."""
        # Ensure the parameter is an argument to the function
        method_params = inspect.signature(func).parameters
        if name not in method_params:
            raise ValueError(f"{name} is not an argument to {func.__name__}")

        if hasattr(func, PARAM_COLLECTION):
            getattr(func, PARAM_COLLECTION).add_parameter(exp_param)
        else:
            setattr(func, PARAM_COLLECTION, ParameterCollection([exp_param]))

        return func

    return parameter_decorator


def extract_params(func, standard_args):
    """Return WhyNot parameters for user-defined function.

    Performs error-checking to ensure parameters are disjoint from
    standard arguments and all arguments to the function are either
    standard arguments or parameters.

    Parameters
    ----------
        func: function
            Possibly parameterized function.

        standard_args: list
            A list of possible arguments provided by the calling class itself
            and shouldn't be treated as parameters for the function.

    Returns
    -------
        params: `whynot.framework.ParameterCollection`
            A collection of parameters for the func.

    """
    if not callable(func):
        msg = f"Trying to extract parameters from {func.__name__}, but not callable."
        raise ValueError(msg)

    # Extract parameters specified by the user via the @parameter decorator.
    specified_params = ParameterCollection([])
    if hasattr(func, PARAM_COLLECTION):
        specified_params = getattr(func, PARAM_COLLECTION)

    # Ensure standard_args is disjoint from the specified params.
    for arg in standard_args:
        if arg in specified_params:
            msg = (f"{arg} is both a parameter and a standard argument to {func.__name__}.")
            raise ValueError(msg)

    # By construction, every element in specified_params
    # must appear in the function signature, i.e. method_params.
    method_args = inspect.signature(func).parameters
    for arg in method_args:
        if arg not in standard_args and arg not in specified_params:
            raise ValueError(f"{arg} in signature, but is not a parameter!")

    return specified_params


class ParameterCollection():
    """Lightweight wrapper class around a set of parameters.

    Provides utility functions to support sampling and assigning subsets of the
    parameters.

    Enforces name uniqueness. Every parameter should have a unique name.

    """

    def __init__(self, params=None):
        """Params is a list of Parameter objects."""
        self.params = {}
        if params is not None:
            for param in params:
                if param.name in self.params:
                    raise ValueError(f"Duplicate name {param.name}")
                self.params[param.name] = param

    def add_parameter(self, param):
        """Add a new parameter to the existing collection."""
        if param.name in self.params:
            raise ValueError(f"Adding parameter with duplicate name {param.name}")
        self.params[param.name] = param

    def default(self):
        """Return the default parameter setting for each parameter."""
        return dict((name, p.default) for name, p in self.params.items())

    def sample(self):
        """Return a random parameter setting for each parameter."""
        sampled_params = {}
        for name, param in self.params.items():
            if param.values is None:
                sampled_params[name] = param.default
            else:
                sampled_params[name] = random.choice(param.values)
        return sampled_params

    def project(self, specified_params):
        """Return fully instantiated parameters using defaults for unspecified params.

        specified_params is a dict of the arguments passed to the run method.
        Any unspecified params are set to their default values.

        Returns a dictonary params[param_name] -> param_value
        """
        for param_name in specified_params:
            if param_name not in self.params:
                raise ValueError(f"Parameter {param_name} specified, but not used!.")

        params = {}
        for name, param in self.params.items():
            if name in specified_params:
                params[name] = specified_params[name]
            else:
                params[name] = param.default

        return params

    def __contains__(self, name):
        """Check if param name is specified."""
        return name in self.params

    def __getitem__(self, name):
        """Return the parameter object corresponding to name."""
        if name in self.params:
            return self.params[name]
        raise ValueError(f"{name} not found")

    def __add__(self, collection):
        """Add two parameter collections together."""
        original_params = list(self.params.values())
        new_params = list(collection.params.values())
        return ParameterCollection(original_params + new_params)

    def __iter__(self):
        """Iterate over the collection."""
        for param in self.params.values():
            yield param

    def __repr__(self):
        """See ___str___."""
        return self.__str__()

    def __str__(self):
        """Display the collection in a human readable format.

        For instance:
        Params:
            Name:		    hidden_dim
            Description:	hidden dimension of 2-layer ReLu network response.
            Default:	    32
            Values:		    [8, 16, 32, 64, 128, 256, 512]
        """
        class_display = "Params:"
        param_display = "\tName:\t\t{}\n\tDescription:\t{}\n\tDefault:\t{}\n\tValues:\t\t{}\n"
        for param in self.params.values():
            param_values = [] if param.values is None else param.values
            class_display += "\n" + param_display.format(
                param.name, param.description, param.default,
                param_values)
        return class_display

    def grid(self):
        """Return a parameter grid."""
        grid = []
        for values in itertools.product(*[p.values for p in self.params.values()]):
            settings = {}
            for param_, value in zip(self.params, values):
                settings[param_.name] = value
            grid.append(settings)
        return grid


####################
# Generic Simulator
####################
class GenericExperiment():
    # pylint:disable-msg=too-few-public-methods
    """Encapsulate a causal simulation experiment."""

    def __init__(self, name, description, run_method):
        """Initialize a generic experiment class."""
        self.name = name
        self.description = description
        self.run_method = run_method

        run_args = ["num_samples", "seed", "parallelize", "show_progress"]
        self.params = extract_params(run_method, run_args)

    def get_parameters(self):
        """Return parameters of the experiment."""
        return self.params

    def run(self, num_samples, seed, parallelize=True, show_progress=False, **parameter_args):
        """Run the experiment and return a causal dataset."""
        run_parameters = self.params.project(parameter_args)
        results = self.run_method(num_samples=num_samples, seed=seed,
                                  show_progress=show_progress,
                                  parallelize=parallelize, **run_parameters)

        (covariates, treatment, outcome), ground_truth = results
        return Dataset(covariates=covariates, treatments=treatment,
                       outcomes=outcome, true_effects=ground_truth)


#############################
# Dynamical system simulators
#############################
class DynamicsExperiment():
    """Encapsulate a causal experiment on a system dynamics simulator."""

    # pylint:disable-msg=too-many-arguments
    def __init__(self, name, description, simulator_config, intervention,
                 state_sampler, simulator, propensity_scorer,
                 outcome_extractor, covariate_builder):
        """Create an experiment based on system dynamics simulator.

        Parameters
        ----------
            name: str
                Name of the experiment
            description: str
                Short description of the experiment
            simulator:
                WhyNot systems dynamics simulator
            simulator_config: `Config`
                Instantiated simulator `Config` for all runs
            intervention: `Intervention`
                Instantiated simulator `Intervention` for treatment runs
            state_sampler:
                Function to sample initial simulator `State` (s)
            propensity_score:
                Either float or function to generate propensity scores for a set of rollouts
            outcome_extractor:
                Function to extract outcome measurement from rollouts
            covariate_builder:
                Function to build covariates from rollouts.

        All of ``simulator_config``, ``intervention``, ``state_sampler``,
        ``propensity_scorer``, ``outcome_extractor``, and ``covariate_builder``
        support a rich set of patterns. We describe the expected API below.

        Note: If you wish to use causal graph construction and want to use numpy
        in any of the user-defined functions, please import the thinly wrapped version
        of numpy `whynot.traceable_numpy` and use this for all of the numpy
        calls, e.g.

        .. code-block:: python

            import whynot.traceable_numpy as np

            def propensity_scorer(run):
                return run[0].covariate_1 / np.max(run[0].values())


        Examples
        --------
        First, ``simulator_config`` can either be a constant ``Config`` object, or a,
        potentially parameterized, function that returns a ``Config``. For
        instance:

        .. code-block:: python

            # simulator_config can be either a constant or a function that returns a
            DynamicsExperiment(
                config=Config(123),...)

            @parameter(name="p", default=0.2)
            def config(p):
                return Config(1234, parameter=p)

        Similarly, ``intervention`` can either be a constant ``Intervention``
        object, or a, potentially parameterized, function that returns an
        ``Intervention``.

        .. code-block:: python

                DynamicsExperiment(
                    intervention=Intervention(year=92),...)

                @parameter(name="p", default=0.2)
                def intervention(p):
                    return Intervention(year=92, parameter=p)

        The ``state_sampler`` generates a sequence of initial states. If the
        sampler uses randomness, it must include ``rng`` in the signature and use
        it as the sole source of randomness. If ``num_samples`` is in the
        signature, then the ``state_sampler`` must returns a sequence of
        ``num_samples`` states.  Otherwise, it must returns a single state.

        .. code-block:: python

            def state_sample(rng):
                # Return a single random sample, use randomness in rng
                return State(rng.rand())

            def state_sampler(rng, num_samples):
                # Return a list of initial states
                return [State(rng.rand()) for _ in range(num_samples)]

        The ``propensity_scorer`` produces, for each run, the probability of
        being assigned to treatment in the observational dataset. The
        ``propensity_scorer`` can be a constant, in which case the probability
        of treatment is uniform for all units. Otherwise, if
        ``propensity_scorer`` is a function, it either produces a propensity
        score for each run separately or all together (for correlated treatment
        assignment).

        For individual runs, the function may include one or more
        of ``run``, ``control_run``, or ``treatment_run`` in the signature.  For
        a whole population, use ``control_runs``, ``runs``, or
        ``treatment_runs`` in the function signature. The ``run`` argument
        defaults to ``control_run`` and similarly for ``runs``. Both ``config`` and
        ``intervention`` can be passed as argument to access the
        ``simulator_config`` and ``intervention`` for the given experiment.

        .. code-block:: python

            # self.propensity_scorer can be a constant
            DynamicsExperiment(
                propensity_scorer=0.8),
                ...)

            ##
            # Compute treatment assignment probability for each run separately.
            ##
            def propensity_scorer(control_run):
                # Return propensity score for a single run, based on control run
                return 0.5 if control_run[20].values()[0] > 0.2 else 0.1

            def propensity_scorer(run):
                # Argument run defaults to the control_run
                return 0.5 if run[20].values()[0] > 0.2 else 0.1

            def propensity_scorer(run, config, intervention):
                # You can pass the config and intervention
                return 0.5 if run[intervention.year].values()[0] > 0.2 else 0.1

            def propensity_scorer(treatment_run):
                # Return propensity score for a single run, based on treatment run
                return 0.5 if treatment_run[20].values()[0] > 0.2 else 0.1

            def propensity_scorer(treatment_run, control_run, config):
                # Assign treatment based on runs with both settings.
                return 0.2

            ##
            # Compute treatment assignment probability for all runs
            # simultaneously. This allows for correlated propensity scores.
            ##
            def propensity_scorer(control_runs):
                # Return propensity score for all runs, based on control run
                return [0.1  for _ in control_runs]

            def propensity_scorer(runs):
                # runs defaults to control_runs
                return [0.1  for _ in runs]

            etc..

        The ``outcome_extractor`` returns, for a given run, the outcome
        :math:`Y`. Both ``config`` and ``intervention`` can be passed as
        arguments.  For instance,

        .. code-block:: python

            def outcome_extractor(run):
                # Return the first state coordinate at time step 100
                return run[100].values()[0]

            def outcome_extractor(run, config, intervention):
                # You can also specify one (or both) of config/intervention
                # First state coordinate 20 steps after intervention.
                return run[intervention.time + 20].values()[0]

        Finally, the ``covariate_builder`` returns, for a given run, the
        observed covariates :math:`X`. If the signature includes ``runs``, the
        method returns covariates for the entire sequence of runs. Otherwise, it
        produces covariates for a single time step. Both ``config`` and
        ``intervention`` can be passed.

        .. code-block:: python

            def covariate_builder(run):
                # Functions with the argument `run` return covariates for a
                # single rollout.
                return run.initial_state.values()

            def covariate_builder(runs):
                # Functions with the argument `runs` return covariates for the
                # entire sequence of runs.
                return np.array([run.initial_state.value() for run in runs])

            def covariate_builder(config, intervention, run):
                # Can optionally specify `config` or `intervention` in both cases.
                return run[intervention.year].values()

        """
        self.name = name
        self.description = description
        self.simulator = simulator

        if not callable(simulator_config):
            self.simulator_config = lambda: simulator_config
        else:
            self.simulator_config = simulator_config

        if not callable(intervention):
            self.intervention = lambda: intervention
        else:
            self.intervention = intervention

        self.state_sampler = state_sampler

        if not callable(propensity_scorer):
            if isinstance(propensity_scorer, float) and 0. <= propensity_scorer <= 1.:
                self.propensity_scorer = lambda: propensity_scorer
            else:
                msg = "If propensity scorer is not callable, it must be a float in [0., 1.]"
                raise ValueError(msg)
        else:
            self.propensity_scorer = propensity_scorer

        self.outcome_extractor = outcome_extractor
        self.covariate_builder = covariate_builder
        self.parameter_collection = self.get_parameters()

    def get_parameters(self):
        """Inspect provided methods and gather parameters.

        Returns
        -------
            params: `whynot.framework.ParameterCollection`
                Collection of all of the parameters specified in the experiment.

        """
        params = ParameterCollection()

        config_args = []
        params += extract_params(self.simulator_config, config_args)
        params += extract_params(self.intervention, config_args)

        initial_state_args = ["rng", "num_samples"]
        params += extract_params(self.state_sampler, initial_state_args)

        propensity_args = ["config", "intervention", "control_run",
                           "control_runs", "treatment_run", "treatment_runs",
                           "run", "runs"]
        params += extract_params(self.propensity_scorer, propensity_args)

        outcome_args = ["config", "intervention", "run"]
        params += extract_params(self.outcome_extractor, outcome_args)

        covariate_args = ["config", "intervention", "run",
                          "runs", "control_run", "control_runs",
                          "treatment_run", "treatment_runs"]
        params += extract_params(self.covariate_builder, covariate_args)

        return params

    @staticmethod
    def _get_args(func):
        """Return the arguments to the function."""
        return inspect.signature(func).parameters

    @staticmethod
    def run_dynamics_simulator(simulate, config, intervention, initial_state, seed):
        """Run the simulation with a common initial state for both treatment and control.

        Use the same initial seed for each run, and return the outcomes of both runs.
        This function is a staticmethod since self is not pickable when the
        class has lambda functions as attributes, e.g. the propensity scorer.

        Parameters
        ----------
            simulate: function
                Entry point to the dynamical systems simulator.
            config:   `Config`
                Configuration object for the simulator, used for both treatment and control.
            intervention: `Intervention`
                Intervention object for the simulator, used for the treatment group.
            initial_state: `State`
                State object used to initial both treatment and control runs of the simulator.
            seed: int
                Random seed used to initialize both runs of the simulator.

        Returns
        -------
            (control_run, treatment_run): `whynot.framework.Run`
                Tuple containing runs of the simulator with and without treatment.

        """
        # Work with copies of the initial state to avoid issues where the
        # simulator had side effects on the state.
        control_run = simulate(copy.deepcopy(initial_state), config=config,
                               intervention=None, seed=seed)
        treatment_run = simulate(copy.deepcopy(initial_state), config=config,
                                 intervention=intervention, seed=seed)
        return control_run, treatment_run

    def _run_all_simulations(self, initial_states, config,
                             intervention, rng, parallelize, show_progress):
        """Run simulation for treatment and control groups for each initial state.

        Parameters
        ----------
            initial_state: list
                List of initial state objects
            config: `Config`
                Configuration object for all runs of the simulator.
            intervention: `Intervention`
                Intervention object for treatment runs of the simulator.
            rng: `np.random.RandomState`
                Source of randomness. Used to choose seeds for different simulator runs.
            parallelize: bool
                Whether or not to execute runs of the simulator in parallel or sequentially.
            show_progress: bool
                Whether or not to display a progress bar for simulator execution.

        Returns
        -------
            control_runs: `list[whynot.framework.Run]`
                List of simulator runs in control group, one for each initial state.
            treatment_runs: `list[whynot.framework.Run]`
                List of simulator runs in treatment group, one for each initial state.

        """
        parallel_args = []
        for state in initial_states:
            seed = rng.randint(0, 9999)
            parallel_args.append((self.simulator.simulate, config, intervention, state, seed))

        if parallelize:
            runs = utils.parallelize(self.run_dynamics_simulator, parallel_args,
                                     show_progress=show_progress)
        elif show_progress:
            runs = [self.run_dynamics_simulator(*args) for args in tqdm(parallel_args)]
        else:
            runs = [self.run_dynamics_simulator(*args) for args in parallel_args]

        control_runs, treatment_runs = list(zip(*runs))

        return control_runs, treatment_runs

    def construct_covariates(self, config, intervention, control_runs,
                             treatment_runs, treatments, params):
        """Build the observational dataset after running simulator.

        Parameters
        ----------
            config: `Config`
                Config object used when running the simulator.
            intervention: `Intervention`
                Intervention object used when running the simulator with treatment.
            control_runs: `list[whynot.framework.Run]`
                List of all of the simulator runs in control condition.
            treatment_runs: `list[whynot.framework.Run]`
                List of all of the simulator runs with treatment applied.
            treatments: `list[bool]`
                Whether or not a particular run was assigned to treatment or control.
            params: dict
                Dictionary mapping parameter names to assigned values.

        Returns
        -------
            covariates: `np.ndarray`
                Numpy array of shape [num_samples, num_features] containing the
                covariates for the observational dataset.

        """
        observed_runs = []
        for treatment, control_run, treatment_run in zip(treatments, control_runs, treatment_runs):
            run = treatment_run if treatment else control_run
            observed_runs.append(run)

        covariate_builder_args = self._get_args(self.covariate_builder)
        kwargs = {}
        for arg in covariate_builder_args:
            if arg in params:
                kwargs[arg] = params[arg]

        if "config" in covariate_builder_args:
            kwargs["config"] = config

        if "intervention" in covariate_builder_args:
            kwargs["intervention"] = intervention

        # Generate all of the covariates in batch
        if "runs" in covariate_builder_args:
            dataset = self.covariate_builder(runs=observed_runs, **kwargs)
        else:
            # Generate the covariates one by one

            dataset = np.array([self.covariate_builder(run=observed_run, **kwargs)
                                for observed_run in observed_runs])

        if dataset.ndim == 1:
            dataset = np.expand_dims(dataset, axis=1)
        return dataset

    def get_outcomes(self, config, intervention, control_runs, treatment_runs, params):
        """Return outcomes for both treatment and control runs.

        Parameters
        ----------
            config: `Config`
                Simulator configuration object used to generate the runs.
            intervention: `Intervention`
                Simulator intervention used to specify treatment runs.
            control_runs:   `list[whynot.framework.Run]`
                List of simulator runs without intervention.
            treatments_runs: `list[whynot.framework.Run]`
                List of simulator runs with intervention.
            params:
                Dictionary mapping parameter names to arguments.

        Returns
        -------
            control_outcomes: `np.ndarray`
                Array of shape [num_samples] with control outcome :math:`Y_i(0)` for each i.
            treatment_outcomes: `np.ndarray`
                Array of shape [num_samples] with treatment outcome :math:`Y_i(1)` for each i.

        """
        extractor_args = self._get_args(self.outcome_extractor)
        kwargs = {}
        for arg in extractor_args:
            if arg in params:
                kwargs[arg] = params[arg]
        if "config" in extractor_args:
            kwargs["config"] = config
        if "intervention" in extractor_args:
            kwargs["intervention"] = intervention

        def get_outcome(run):
            return self.outcome_extractor(run=run, **kwargs)

        control_outcomes = np.array([get_outcome(run) for run in control_runs])
        treatment_outcomes = np.array([get_outcome(run) for run in treatment_runs])

        return control_outcomes, treatment_outcomes

    def get_propensity_scores(self, config, intervention, control_runs, treatment_runs, params):
        """Return probability of treatment for a set of runs.

        Parameters
        ----------
            config: `Config`
                Simulator configuration object used to generate the runs.
            intervention: `Intervention`
                Simulator intervention used to specify treatment runs.
            control_runs:   `list[whynot.framework.Run]`
                List of simulator runs without intervention.
            treatments_runs: `list[whynot.framework.Run]`
                List of simulator runs with intervention.
            params:
                Dictionary mapping parameter names to arguments.

        Returns
        -------
            propensity_score: `np.ndarray`
                Array of shape [num_samples] with probability of treatment for each unit i.

        """
        # pylint:disable-msg=too-many-branches
        if not callable(self.propensity_scorer):
            if isinstance(self.propensity_scorer, float):
                uniform_propensity = self.propensity_scorer
                if 0. <= uniform_propensity <= 1.:
                    return uniform_propensity * np.ones(len(control_runs))

            raise ValueError("If propensity scorer is not callable, it must be a float in [0., 1.]")

        scorer_args = self._get_args(self.propensity_scorer)

        # Build kwargs for scorer based on passed parameters
        kwargs = {}
        for arg in scorer_args:
            if arg in params:
                kwargs[arg] = params[arg]

        if "config" in scorer_args:
            kwargs["config"] = config

        if "intervention" in scorer_args:
            kwargs["intervention"] = intervention

        # Check whether or not the propensity scorer generates all scores at once
        one_shot = False
        if "treatment_runs" in scorer_args:
            one_shot = True
            kwargs["treatment_runs"] = treatment_runs
        if "control_runs" in scorer_args:
            one_shot = True
            kwargs["control_runs"] = control_runs
        if "runs" in scorer_args:
            one_shot = True
            kwargs["runs"] = control_runs

        if one_shot:
            return self.propensity_scorer(**kwargs)

        # If not, then we need to score each run separately.
        propensity_scores = []
        for control_run, treatment_run in zip(control_runs, treatment_runs):
            if "control_run" in scorer_args and "treatment_run" in scorer_args:
                score = self.propensity_scorer(control_run=control_run,
                                               treatment_run=treatment_run,
                                               **kwargs)
            elif "control_run" in scorer_args:
                score = self.propensity_scorer(control_run=control_run, **kwargs)
            elif "treatment_run" in scorer_args:
                score = self.propensity_scorer(treatment_run=treatment_run, **kwargs)
            elif "run" in scorer_args:
                score = self.propensity_scorer(run=control_run, **kwargs)
            else:
                score = self.propensity_scorer(**kwargs)
            propensity_scores.append(score)

        return np.array(propensity_scores)

    def sample_initial_states(self, rng, num_samples, params):
        """Sample initial states for simulator.

        Parameters
        ----------
            rng: `np.random.RandomState`
                Random number generator used for all experiment randomness
            num_samples: int
                Number of samples to use for the experiment.
            params:
                Dictionary mapping parameter names to assigned values.

        Returns
        -------
            initial_states: `list[States]`
                List of length `num_samples` of simulator initial `States.

        """
        state_sampler_args = self._get_args(self.state_sampler)

        # Build kwargs
        kwargs = {}
        for arg in state_sampler_args:
            if arg in params:
                kwargs[arg] = params[arg]

        if "rng" in state_sampler_args:
            kwargs["rng"] = rng

        if "num_samples" in state_sampler_args:
            return self.state_sampler(num_samples=num_samples, **kwargs)

        return [self.state_sampler(**kwargs) for _ in range(num_samples)]

    def get_config(self, params):
        """Return the simulator config.

        Parameters
        ----------
            params: dict
                Dictionary mapping experiment parameter names to assigned values

        Returns
        -------
            config: `Config`
                Simulator configuration object used for this experiment.

        """
        if not callable(self.simulator_config):
            return self.simulator_config

        control_args = self._get_args(self.simulator_config)
        kwargs = {}
        for arg in control_args:
            if arg in params:
                kwargs[arg] = params[arg]

        return self.simulator_config(**kwargs)

    def get_intervention(self, params):
        """Return the intervention correspond to treatment.

        Parameters
        ----------
            params: dict
                Dictionary mapping experiment parameter names to assigned
                values.

        Returns
        -------
            intervention: `Intervention`
                Simulator Intervention object used for treatment group.

        """
        if not callable(self.intervention):
            return self.intervention

        treatment_args = self._get_args(self.intervention)
        kwargs = {}
        for arg in treatment_args:
            if arg in params:
                kwargs[arg] = params[arg]

        return self.intervention(**kwargs)

    def construct_causal_graph(self, control_runs, treatment_runs,
                               config, intervention, run_parameters):
        """Construct the causal graph associated with the experiment.

        This feature is still experimental and will likely have rough
        edges.

        Parameters
        -----------
            control_runs: list
                List of untreated whynot.framework.Run objects generated by the experiment.
            treatment_runs: list
                List of treated whynot.framework.Run objects generated by the experiment.
            config: whynot.simulators.infrastructre.BaseConfig
                Fully instantiated simulator config used for the experiment.
            intervention: whynot.simulators.infrastructre.BaseConfig
                Fully instantiated simulator intervention used for the experiment.
            run_parameters: dict
                User-specified parameters associated with the experiment.

        Returns
        -------
            ate_graph: networkx.DiGraph
                Causal graph corresponding to the experiment.

        """
        # Extract dependencies for each of the ATE components.
        with causal_graphs.trace_stack.new_trace() as trace:
            # All of the supported simulators dont have data dependent
            # control flow, so for now, tracing the simulator and user
            # functions only requires a single run.
            run = control_runs[0]

            # Wrap the run object in a box
            run, node_map = causal_graphs.run_to_box(run, trace)
            control_runs, treatment_runs = [run], [run]

            # Trace the propensity scorer
            propensities = self.get_propensity_scores(
                config, intervention, control_runs, treatment_runs, run_parameters)
            treatment_deps = causal_graphs.backtrack(propensities[0], node_map)

            # Trace the covariate extractor.
            covariates = self.construct_covariates(
                config, intervention, control_runs, treatment_runs, np.array([0.]), run_parameters)
            covariate_deps = causal_graphs.backtrack(covariates[0], node_map)

            # Trace the outcome extractor
            control_outcomes, _ = self.get_outcomes(
                config, intervention, control_runs, treatment_runs, run_parameters)
            outcome_deps = causal_graphs.backtrack(control_outcomes[0], node_map)

        # Unpack the dependencies from backtracking
        treatment_deps = [node.name for node in treatment_deps[0]]
        outcome_deps = [node.name for node in outcome_deps[0]]
        covariate_deps = dict((cov, [node.name for node in deps])
                              for cov, deps in covariate_deps.items())

        # Build the causal graph corresponding to the dynamics.
        ate_graph = causal_graphs.ate_graph_builder(
            self.simulator, run, config, intervention, treatment_deps,
            covariate_deps, outcome_deps)

        return ate_graph

    def run(self, num_samples, seed=None, parallelize=True, show_progress=False,
            causal_graph=False, **parameter_args):
        # pylint:disable-msg=too-many-locals
        """Run a basic parameterized experiment on a dynamical system simulator.

        Parameters
        ----------
            num_samples: int
                Number of units to sample and simulate
            show_progress: bool
                Should progress bar be shown?
            parallelize: bool
                If true, the experiment class will run all of the simulations in
                parallel.
            seed: int
                Random number generator seed used to set all internal randomness.
            causal_graph:
                Whether to attempt to build the causal graph. Currently, this is only
                supported on the hiv, lotka_volterra, and opioid simulators.
            **parameter_args:
                If the experiment is parameterized, additional arguments to
                select the parameters of a particular run.

        Returns
        -------
            dataset: whynot.framework.Dataset
                Dataset object encapsulating the covariates, treatment assignments, and
                outcomes observed in the experiment, as well as unit-level ground truth.
                The covariates are an array of size [num_samples, num_features] where
                num_features is determined by the covariate builder.

        """
        if causal_graph and not self.simulator.SUPPORTS_CAUSAL_GRAPHS:
            error_msg = ("This simulator does not currently support causal graph generation."
                         "Rerun the command with causal_graphs=False to continue.")
            raise ValueError(error_msg)

        run_parameters = self.parameter_collection.project(parameter_args)

        rng = np.random.RandomState(seed)

        config = self.get_config(run_parameters)
        intervention = self.get_intervention(run_parameters)

        initial_states = self.sample_initial_states(rng, num_samples, run_parameters)

        # Run the model
        control_runs, treatment_runs = self._run_all_simulations(
            initial_states, config, intervention, rng, parallelize, show_progress)

        # Assign treatment
        propensities = self.get_propensity_scores(
            config, intervention, control_runs, treatment_runs, run_parameters)
        treatment = (rng.uniform(size=propensities.shape) < propensities).astype(np.int64)

        # Build observational dataset
        covariates = self.construct_covariates(
            config, intervention, control_runs, treatment_runs, treatment, run_parameters)
        control_outcomes, treatment_outcomes = self.get_outcomes(
            config, intervention, control_runs, treatment_runs, run_parameters)

        # Assign treatment
        outcomes = np.copy(control_outcomes)
        outcomes[treatment == 1.0] = treatment_outcomes[treatment == 1.0]

        # Ground truth effect is Y(1) - Y(0)
        treatment_effects = treatment_outcomes - control_outcomes

        if causal_graph:
            graph = self.construct_causal_graph(
                control_runs, treatment_runs, config, intervention, run_parameters)
        else:
            graph = None

        return Dataset(covariates=covariates, treatments=treatment,
                       outcomes=outcomes, true_effects=treatment_effects,
                       causal_graph=graph)
