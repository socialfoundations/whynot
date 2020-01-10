"""Base experimental class."""
import dataclasses
import inspect
import itertools
from typing import Any

import numpy as np


PARAM_COLLECTION = "__parameter_collection__"


@dataclasses.dataclass
class Dataset:
    # pylint:disable-msg=too-few-public-methods
    """Observational dataset and grouth truth unit-level effects.

    Attributes
    ----------
        covariates: `np.ndarray`
            Float array of shape [num_samples, num_features] of covariates for each unit.
        treatments:  `np.ndarray`
            Integer 0/1 array of shape [num_samples] indicating treatment
            status for each unit. 1 indicates treated, 0 indicates unit not treated.
        outcomes:  `np.ndarray`
            Float array of shape [num_samples] containing the observed outcome for each unit.
        true_effects: `np.ndarray`
            Float array of shape [num_samples] containing the unit-level treatment
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

    def bootstrap_sample_ate_ci(self, num_bootstrap_samples=2000, alpha=0.05):
        """Bootstrap a (1-alpha)% confidence interval for the sample ate."""
        means = []
        for _ in range(num_bootstrap_samples):
            sample = np.random.choice(
                self.true_effects, size=len(self.true_effects), replace=True
            )
            means.append(np.mean(sample))
        lower_tail, upper_tail = alpha / 2.0, 1.0 - alpha / 2.0
        return (np.quantile(means, lower_tail), np.quantile(means, upper_tail))


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
            TODO: Ideally, we'd support various significance levels.
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
        def propensity_score(untreated_run, treatment_bias)
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
            msg = (
                f"{arg} is both a parameter and a standard argument to {func.__name__}."
            )
            raise ValueError(msg)

    # By construction, every element in specified_params
    # must appear in the function signature, i.e. method_params.
    method_args = inspect.signature(func).parameters
    for arg in method_args:
        if arg not in standard_args and arg not in specified_params:
            msg = (
                f"'{arg}' is in the signature of function {func.__name__}, "
                f"but '{arg}' is not a standard argument or a parameter. "
                f"Standard arguments: {', '.join(standard_args)}."
            )
            raise ValueError(msg)

    return specified_params


class ParameterCollection:
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
        return {name: p.default for name, p in self.params.items()}

    def sample(self, seed=None):
        """Return a random parameter setting for each parameter."""
        rng = np.random.RandomState(seed)
        sampled_params = {}
        for name, param in self.params.items():
            if param.values is None:
                sampled_params[name] = param.default
            else:
                sampled_params[name] = rng.choice(param.values)
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
        param_display = (
            "\tName:\t\t{}\n\tDescription:\t{}\n\tDefault:\t{}\n\tValues:\t\t{}\n"
        )
        for param in self.params.values():
            param_values = [] if param.values is None else param.values
            class_display += "\n" + param_display.format(
                param.name, param.description, param.default, param_values
            )
        return class_display

    def grid(self):
        """Return a parameter grid.
        
        Examples
        --------
        >>> p1 = ExperimentParams(name="a", values=[1, 2])
        >>> p2 = ExperimentParams(name="b", values=[3, 4])
        >>> collection = ParameterCollection(params=[a, b])
        >>> for a, b in collection.grid():
        ...     print(a, b)
        1, 3
        2, 3
        1, 4
        2, 4
 
        """
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
class GenericExperiment:
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

    def run(
        self, num_samples, seed, parallelize=True, show_progress=False, **parameter_args
    ):
        """Run the experiment and return a causal dataset."""
        run_parameters = self.params.project(parameter_args)
        results = self.run_method(
            num_samples=num_samples,
            seed=seed,
            show_progress=show_progress,
            parallelize=parallelize,
            **run_parameters,
        )

        (covariates, treatment, outcome), ground_truth = results
        return Dataset(
            covariates=covariates,
            treatments=treatment,
            outcomes=outcome,
            true_effects=ground_truth,
        )
