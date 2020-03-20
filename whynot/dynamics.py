"""Infrastructure for dynamical system simulators and experiments."""
import copy
import dataclasses
import inspect

import numpy as np
from tqdm.auto import tqdm

from whynot.framework import Dataset, extract_params, ParameterCollection
from whynot import causal_graphs, utils


class BaseConfig:
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics."""

    # Start and end time.
    start_time: float = 0
    end_time: float = 0
    # Time elapsed between measurements of the system.
    delta_t: float = 1.0

    def update(self, intervention):
        """Generate a new config after applying the intervention."""
        return dataclasses.replace(self, **intervention.updates)

    @classmethod
    def parameter_names(cls):
        """Return the parameters of the simulator, excluding start, end, and delta_t."""
        excluded = ["start_time", "end_time", "delta_t"]
        return [f.name for f in dataclasses.fields(cls) if f.name not in excluded]


@dataclasses.dataclass
class BaseState:
    # pylint: disable-msg=too-few-public-methods
    """State of the simulator."""

    @classmethod
    def num_variables(cls):
        """Count the number of variables."""
        return len(dataclasses.fields(cls))

    @classmethod
    def variable_names(cls):
        """List the variable names."""
        return [f.name for f in dataclasses.fields(cls)]

    def values(self):
        """Return the state as a numpy array."""
        # This ensures the values are shallow copied, unlike
        # dataclasses.astuple, which is necessary for graph tracing.
        return np.array([getattr(self, name) for name in self.variable_names()])


class BaseIntervention:
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention to a config."""

    def __init__(self, config_class, time, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention.
            config_class:
                The Config class, a child class of dataclass.
            kwargs: dict
                Only valid keyword arguments are parameters of the config class.

        """
        self.time = time
        config_args = set(f.name for f in dataclasses.fields(config_class))
        for arg in kwargs:
            if arg not in config_args:
                raise TypeError(f"__init__() got an unexpected keyword argument {arg}!")
        self.updates = kwargs


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
    >>> state_at_year_closest_to_2015 = run[2015]

    """

    states: list
    times: list

    def __post_init__(self):
        """Error checking called after the constructor."""
        if len(self.states) != len(self.times):
            msg = "Input states and times must be the same length!  {} \neq {}"
            raise ValueError(msg.format(len(self.states), len(self.times)))

    def __getitem__(self, time):
        """Return the state closest to the given time."""
        time_index = np.argmin(np.abs(np.array(self.times) - time))
        return self.states[time_index]

    @property
    def initial_state(self):
        """Return initial state of the run."""
        return self.states[0]

    @property
    def initial_time(self):
        """Return the initial time of the run."""
        return self.times[0]

    @property
    def final_state(self):
        """Return the final state of the run."""
        return self.states[-1]

    @property
    def final_time(self):
        """Return the final time of the run."""
        return self.times[-1]


class DynamicsExperiment:
    """Encapsulate a causal experiment on a system dynamics simulator."""

    # pylint:disable-msg=too-many-arguments
    def __init__(
        self,
        name,
        description,
        simulator_config,
        intervention,
        state_sampler,
        simulator,
        propensity_scorer,
        outcome_extractor,
        covariate_builder,
    ):
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
                Instantiated simulator `Intervention` for treated runs
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
        First, ``simulator_config`` can either be a constant ``Config`` object, or a
        parameterized function that returns a ``Config``. For
        instance:

        .. code-block:: python

            # simulator_config can be a constant Config object,
            DynamicsExperiment(
                ...
                simulator_config=Config(123),
                ...)
            
            # Or simulator_config can be a (parameterized) function that
            # returns a config.
            @parameter(name="p", default=0.2)
            def config(p):
                return Config(1234, parameter=p)

            DynamicsExperiment(
                ...
                simulator_config=config,
                ...)

        Similarly, ``intervention`` can either be a constant ``Intervention``
        object, or a (parameterized) function that returns an
        ``Intervention``.

        .. code-block:: python

                DynamicsExperiment(
                    ...
                    intervention=Intervention(year=92),
                    ...)

                @parameter(name="p", default=0.2)
                def intervention(p):
                    return Intervention(year=92, parameter=p)

                DynamicsExperiment(
                    ...
                    intervention=intervention,
                    ...)

        The ``state_sampler`` generates a sequence of initial states. If the
        sampler uses randomness, it must include ``rng`` in the signature and use
        it as the sole source of randomness.  This is to ensure that
        DynamicsExperiments can be deterministic if desired.
        
        If ``num_samples`` is in the signature, then the ``state_sampler`` must
        returns a sequence of ``num_samples`` states.  Otherwise, it must
        returns a single state.

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

        For individual runs, the function may include one or both of
        ``treated_run`` and ``untreated_run`` in the signature.  For a whole
        population, use ``untreated_runs`` or ``treated_runs`` in the function
        signature.  Both ``config`` and ``intervention`` can be passed as
        argument to access the ``simulator_config`` and ``intervention`` for the
        given experiment.

        .. code-block:: python

            # self.propensity_scorer can be a constant
            DynamicsExperiment(
                ...
                propensity_scorer=0.8),
                ...)

            ##
            # Compute treatment assignment probability for each run separately.
            ##
            def propensity_scorer(untreated_run):
                # Return propensity score for a single run, based on the
                # untreated rollout.
                return 0.5 if untreated_run[20].values()[0] > 0.2 else 0.1

            def propensity_scorer(untreated_run, config, intervention):
                # You can pass the config and intervention
                intervention_covariate = untreated_run[intervention.year].values()[0]
                threshold = config.threshold
                return 1.0 if intervention_covariate > threshold else 0.0

            def propensity_scorer(treated_run):
                # Return propensity score for a single run, based on treated run
                return 0.5 if treated_run[20].values()[0] > 0.2 else 0.1

            def propensity_scorer(treated_run, untreated_run, config):
                # Assign treatment based on runs with both settings.
                if treated_run[10].values()[0] > 5 and untreated_run[10].values() < 10:
                    return config.propensity
                return 1.0 - config.propensity

            ##
            # Compute treatment assignment probability for all runs
            # simultaneously. This allows for correlated propensity scores.
            ##
            def propensity_scorer(untreated_runs):
                # Return propensity score for all runs, based on untreated run.
                # Only treat the top 10% of runs.
                covariates = [run.final_state.values()[0] for run in untreated_runs]
                top10 = np.argsort(covariates)[-10:]
                return [0.9 if idx in top10 else 0.1 for idx in range(len(untreated_runs))]

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
        produces covariates for a single run. Both ``config`` and
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
            if isinstance(propensity_scorer, float) and 0.0 <= propensity_scorer <= 1.0:
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

        propensity_args = [
            "config",
            "intervention",
            "untreated_run",
            "untreated_runs",
            "treated_run",
            "treated_runs",
        ]
        params += extract_params(self.propensity_scorer, propensity_args)

        outcome_args = ["config", "intervention", "run"]
        params += extract_params(self.outcome_extractor, outcome_args)

        covariate_args = ["config", "intervention", "run", "runs"]
        params += extract_params(self.covariate_builder, covariate_args)

        return params

    @staticmethod
    def _get_args(func):
        """Return the arguments to the function."""
        return inspect.signature(func).parameters

    @staticmethod
    def _run_dynamics_simulator(simulate, config, intervention, initial_state, seed):
        """Run the simulation with a common initial state for both treated and untreated.

        Use the same initial seed for each run, and return the outcomes of both runs.
        This function is a staticmethod since self is not pickable when the
        class has lambda functions as attributes, e.g. the propensity scorer.

        Parameters
        ----------
            simulate: function
                Entry point to the dynamical systems simulator.
            config:   `Config`
                Configuration object for the simulator, used for both treated and untreated.
            intervention: `Intervention`
                Intervention object for the simulator, used for the treated group.
            initial_state: `State`
                State object used to initial both treated and untreated runs of the simulator.
            seed: int
                Random seed used to initialize both runs of the simulator.

        Returns
        -------
            (untreated_run, treated_run): `Run`
                Tuple containing runs of the simulator with and without treatment.

        """
        # Work with copies of the initial state to avoid issues where the
        # simulator had side effects on the state.
        untreated_run = simulate(
            copy.deepcopy(initial_state), config=config, intervention=None, seed=seed
        )
        treated_run = simulate(
            copy.deepcopy(initial_state),
            config=config,
            intervention=intervention,
            seed=seed,
        )
        return untreated_run, treated_run

    def _run_all_simulations(
        self, initial_states, config, intervention, rng, parallelize, show_progress
    ):
        """Run simulation for treated and untreated groups for each initial state.

        Parameters
        ----------
            initial_state: list
                List of initial state objects
            config: `Config`
                Configuration object for all runs of the simulator.
            intervention: `Intervention`
                Intervention object for treated runs of the simulator.
            rng: `np.random.RandomState`
                Source of randomness. Used to choose seeds for different simulator runs.
            parallelize: bool
                Whether or not to execute runs of the simulator in parallel or sequentially.
            show_progress: bool
                Whether or not to display a progress bar for simulator execution.

        Returns
        -------
            untreated_runs: `list[Run]`
                List of simulator runs in untreated group, one for each initial state.
            treated_runs: `list[Run]`
                List of simulator runs in treated group, one for each initial state.

        """
        parallel_args = []
        for state in initial_states:
            seed = rng.randint(0, 2 ** 32 - 1)
            parallel_args.append(
                (self.simulator.simulate, config, intervention, state, seed)
            )

        if parallelize:
            runs = utils.parallelize(
                self._run_dynamics_simulator, parallel_args, show_progress=show_progress
            )
        elif show_progress:
            runs = [self._run_dynamics_simulator(*args) for args in tqdm(parallel_args)]
        else:
            runs = [self._run_dynamics_simulator(*args) for args in parallel_args]

        untreated_runs, treated_runs = list(zip(*runs))

        return untreated_runs, treated_runs

    def construct_covariates(
        self, config, intervention, untreated_runs, treated_runs, treatments, params
    ):
        """Build the observational dataset after running simulator.

        Parameters
        ----------
            config: `Config`
                Config object used when running the simulator.
            intervention: `Intervention`
                Intervention object used when running the simulator with treatment.
            untreated_runs: `list[Run]`
                List of all of the simulator runs in untreated condition.
            treated_runs: `list[Run]`
                List of all of the simulator runs with treatment applied.
            treatments: `list[bool]`
                Whether or not a particular run was assigned to treatment or untreated.
            params: dict
                Dictionary mapping parameter names to assigned values.

        Returns
        -------
            covariates: `np.ndarray`
                Numpy array of shape [num_samples, num_features] containing the
                covariates for the observational dataset.

        """
        observed_runs = []
        for treatment, untreated_run, treated_run in zip(
            treatments, untreated_runs, treated_runs
        ):
            run = treated_run if treatment else untreated_run
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

            dataset = np.array(
                [
                    self.covariate_builder(run=observed_run, **kwargs)
                    for observed_run in observed_runs
                ]
            )

        if dataset.ndim == 1:
            dataset = np.expand_dims(dataset, axis=1)
        return dataset

    def get_outcomes(self, config, intervention, untreated_runs, treated_runs, params):
        """Return outcomes for both treated and untreated runs.

        Parameters
        ----------
            config: `Config`
                Simulator configuration object used to generate the runs.
            intervention: `Intervention`
                Simulator intervention used to specify treated runs.
            untreated_runs:   `list[Run]`
                List of simulator runs without intervention.
            treated_runs: `list[Run]`
                List of simulator runs with intervention.
            params:
                Dictionary mapping parameter names to arguments.

        Returns
        -------
            untreated_outcomes: `np.ndarray`
                Array of shape [num_samples] with untreated outcome :math:`Y_i(0)` for each i.
            treated_outcomes: `np.ndarray`
                Array of shape [num_samples] with treated outcome :math:`Y_i(1)` for each i.

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

        untreated_outcomes = np.array([get_outcome(run) for run in untreated_runs])
        treated_outcomes = np.array([get_outcome(run) for run in treated_runs])

        return untreated_outcomes, treated_outcomes

    def get_propensity_scores(
        self, config, intervention, untreated_runs, treated_runs, params
    ):
        """Return probability of treatment for a set of runs.

        Parameters
        ----------
            config: `Config`
                Simulator configuration object used to generate the runs.
            intervention: `Intervention`
                Simulator intervention used to specify treated runs.
            untreated_runs:   `list[Run]`
                List of simulator runs without intervention.
            treated_runs: `list[Run]`
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
                if 0.0 <= uniform_propensity <= 1.0:
                    return uniform_propensity * np.ones(len(untreated_runs))

            raise ValueError(
                "If propensity scorer is not callable, it must be a float in [0., 1.]"
            )

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
        if "treated_runs" in scorer_args:
            one_shot = True
            kwargs["treated_runs"] = treated_runs
        if "untreated_runs" in scorer_args:
            one_shot = True
            kwargs["untreated_runs"] = untreated_runs

        if one_shot:
            return self.propensity_scorer(**kwargs)

        # If not, then we need to score each run separately.
        propensity_scores = []
        for untreated_run, treated_run in zip(untreated_runs, treated_runs):
            if "untreated_run" in scorer_args:
                kwargs["untreated_run"] = untreated_run
            if "treated_run" in scorer_args:
                kwargs["treated_run"] = treated_run
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

        untreated_args = self._get_args(self.simulator_config)
        kwargs = {}
        for arg in untreated_args:
            if arg in params:
                kwargs[arg] = params[arg]

        return self.simulator_config(**kwargs)

    def get_intervention(self, params):
        """Return the intervention corresponding to treatment.

        Parameters
        ----------
            params: dict
                Dictionary mapping experiment parameter names to assigned
                values.

        Returns
        -------
            intervention: `Intervention`
                Simulator Intervention object used for treated group.

        """
        if not callable(self.intervention):
            return self.intervention

        treated_args = self._get_args(self.intervention)
        kwargs = {}
        for arg in treated_args:
            if arg in params:
                kwargs[arg] = params[arg]

        return self.intervention(**kwargs)

    def construct_causal_graph(
        self, untreated_runs, treated_runs, config, intervention, run_parameters
    ):
        """Construct the causal graph associated with the experiment.

        This feature is still experimental and will likely have rough
        edges.

        Parameters
        ----------
            untreated_runs: list
                List of untreated Run objects generated by the experiment.
            treated_runs: list
                List of treated Run objects generated by the experiment.
            config: whynot.dynamics.BaseConfig
                Fully instantiated simulator config used for the experiment.
            intervention: whynot.dynamics.BaseIntervention
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
            run = untreated_runs[0]

            # Wrap the run object in a box
            run, node_map = causal_graphs.run_to_box(run, trace)
            untreated_runs, treated_runs = [run], [run]

            # Trace the propensity scorer
            propensities = self.get_propensity_scores(
                config, intervention, untreated_runs, treated_runs, run_parameters
            )
            treatment_deps = causal_graphs.backtrack(propensities[0], node_map)

            # Trace the covariate extractor.
            covariates = self.construct_covariates(
                config,
                intervention,
                untreated_runs,
                treated_runs,
                np.array([0.0]),
                run_parameters,
            )
            covariate_deps = causal_graphs.backtrack(covariates[0], node_map)

            # Trace the outcome extractor
            untreated_outcomes, _ = self.get_outcomes(
                config, intervention, untreated_runs, treated_runs, run_parameters
            )
            outcome_deps = causal_graphs.backtrack(untreated_outcomes[0], node_map)

        # Unpack the dependencies from backtracking
        treatment_deps = [node.name for node in treatment_deps[0]]
        outcome_deps = [node.name for node in outcome_deps[0]]
        covariate_deps = dict(
            (cov, [node.name for node in deps]) for cov, deps in covariate_deps.items()
        )

        # Build the causal graph corresponding to the dynamics.
        ate_graph = causal_graphs.ate_graph_builder(
            self.simulator,
            run,
            config,
            intervention,
            treatment_deps,
            covariate_deps,
            outcome_deps,
        )

        return ate_graph

    def run(
        self,
        num_samples,
        seed=None,
        parallelize=True,
        show_progress=False,
        causal_graph=False,
        **parameter_args,
    ):
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
                parallel. Parallelization is performed with multiprocessing
                using a ProcessPool from the concurrent.futures module. 
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
            error_msg = (
                "This simulator does not currently support causal graph generation."
                "Rerun the command with causal_graphs=False to continue."
            )
            raise ValueError(error_msg)

        run_parameters = self.parameter_collection.project(parameter_args)

        rng = np.random.RandomState(seed)

        config = self.get_config(run_parameters)
        intervention = self.get_intervention(run_parameters)

        initial_states = self.sample_initial_states(rng, num_samples, run_parameters)

        # Run the model
        untreated_runs, treated_runs = self._run_all_simulations(
            initial_states, config, intervention, rng, parallelize, show_progress
        )

        # Assign treatment
        propensities = self.get_propensity_scores(
            config, intervention, untreated_runs, treated_runs, run_parameters
        )
        treatment = rng.uniform(size=propensities.shape) < propensities
        treatment = treatment.astype(np.int64)

        # Build observational dataset
        covariates = self.construct_covariates(
            config,
            intervention,
            untreated_runs,
            treated_runs,
            treatment,
            run_parameters,
        )
        untreated_outcomes, treated_outcomes = self.get_outcomes(
            config, intervention, untreated_runs, treated_runs, run_parameters
        )

        # Assign treatment
        outcomes = np.copy(untreated_outcomes)
        outcomes[treatment == 1] = treated_outcomes[treatment == 1]

        # Ground truth effect is Y(1) - Y(0)
        treatment_effects = treated_outcomes - untreated_outcomes

        if causal_graph:
            graph = self.construct_causal_graph(
                untreated_runs, treated_runs, config, intervention, run_parameters
            )
        else:
            graph = None

        return Dataset(
            covariates=covariates,
            treatments=treatment,
            outcomes=outcomes,
            true_effects=treatment_effects,
            causal_graph=graph,
        )

    def __repr__(self):
        """Display the experiment by it's name."""
        return self.name
