"""Functions for constructing and analyzing causal graphs."""
from collections.abc import Iterable
import types

import dataclasses

from autograd.extend import primitive
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.tracer import Box, isbox, new_box, Node, trace_stack
import networkx as nx
import numpy as np

import whynot as wn

# Allow tracing to use ints and bools (since we only care about the forward
# pass and not derivatives).
for type_ in [bool, np.bool, np.bool_, int, np.int32, np.int64]:
    ArrayBox.register(type_)


class FunctionBox(Box):
    """Generic box class to encapsulate functions and record their invocation."""

    @primitive
    def __call__(func, *args, **kwargs):
        """Execute function and record its invocation in the graph."""
        return func(*args, **kwargs)


FunctionBox.register(types.FunctionType)


class TracerNode(Node):
    """Base case for constructing causal graphs via computation graphs.

    A TracerNode is a node in the computation graph and symbolically
    represents the application of a function to its parent nodes to
    produce a child node.

    This class is used internally by autograd, and we likely do not every
    need to call it explicitly.
    """

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        """Construct a node out of a function application."""
        # pylint:disable-msg=super-init-not-called
        self.parents = parents
        self.value = value
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.parent_argnums = parent_argnums
        # Only root names are named
        self.name = None

    def initialize_root(self, arg, name=None):
        # pylint:disable-msg=arguments-differ
        """Construct the root of the computation graph.

        Parameters
        ----------
            arg: the initial value
            name: (Optional) the name of the root node.

        """
        self.parents = []
        self.value = arg
        self.fun = None
        self.args = []
        self.kwargs = {}
        self.parent_argnums = None
        self.name = name


def dataclass_to_box(dataclass, trace, name_suffix=None, skip_args=None):
    """Convert a dataclass object to a traceable box.

    Parameters
    ----------
        dataclass: instance of dataclasses.dataclass
            Either a whynot State or a Config object
        trace: trace_stack context manager
            Every box is created within the context of a trace
        name_suffix: str
            (Optional) Suffix to append to every field name of the dataclass
        skip_args: list
            (Optional) List of arguments/dataclass field names to skip tracing.

    Returns
    -------
        box: instance of dataclasses.dataclass
            The same input object with fields replaced by boxes to allow for
            dependency tracing.
        nodes: dict
            Dictionary mapping a root TracerNode to the corresponding Box.
            The name of the TracerNode is the fieldname and the optional suffix.

    """
    # Wrap each input argument in a "box" with an associated start node to allow tracing
    flattened = dataclasses.astuple(dataclass)
    names = [field.name for field in dataclasses.fields(dataclass)]

    suffix = f"_{name_suffix}" if name_suffix else ""
    replacements, node_map = {}, {}
    for name, value in zip(names, flattened):
        if skip_args and name in skip_args:
            # Don't wrap this argument in a box
            replacements[name] = value
        elif hasattr(value, "get_boxable"):
            # The value itself isn't traceble, but it exports
            # methods that allow it to be traced.
            boxable = value.get_boxable()
            boxable = check_and_cast_args(boxable)

            node = TracerNode.new_root(boxable, name + suffix)
            box = new_box(boxable, trace, node)

            value.set_boxable(box)

            replacements[name] = value
            node_map[node] = box

        else:
            # Try to wrap the value in a box, casting as possible for dependency tracing
            value = check_and_cast_args(value)
            node = TracerNode.new_root(value, name + suffix)
            box = new_box(value, trace, node)

            replacements[name] = box
            node_map[node] = box

    # Generate a new dataclass, replacing each value with the corresponding box
    dataclass = dataclasses.replace(dataclass, **replacements)

    return dataclass, node_map


def run_to_box(run, trace_ctx, skip_args=None):
    """Convert a rollout of a dynamical system simulator to a collection of traceable elements.

    Parameters
    ----------
        run: whynot.dynamics.Run
            The run for the user to trace.
        trace_ctx:  trace_stack context manager
            Every box is created within the context of this trace
        skip_args: list
            (Optional) List of fields in the state to avoid tracing.
            These fields are not wrapped in box objects.

    Returns
    -------
        boxed_run: whynot.dynamics.Run
            Run object where each State has all of its fields (except those
            listed in skip_args) replaced with a Box object for tracing.

        global_nodemap: dict
            Dictionary mapping root TracerNode to the corresponding box
            objects. The name of each TracerNode is statename_time.
            During dependency tracing, if we encounter a node in this
            map, then we have found an "ancestor" of the output node
            as a base element of the run.

    """
    boxed_states, global_nodemap = [], {}
    for state, time in zip(run.states, run.times):
        boxed_state, nodemap = dataclass_to_box(
            state, trace_ctx, name_suffix=str(time), skip_args=skip_args
        )
        boxed_states.append(boxed_state)
        global_nodemap.update(nodemap)
    boxed_run = wn.dynamics.Run(states=boxed_states, times=run.times)
    return boxed_run, global_nodemap


def check_and_cast_args(args):
    """Check and cast arguments for the dependency tracer.

    Since autograd is expecting to compute derivatives, ints are not
    allowable as arguments. However, for our purposes, casting ints
    to floats will not affect dependency tracing.
    """
    # Currently this function only casts int into floats and raises
    # if the element is a string. This can be expanded to be more robust.
    def check_and_cast(arg):
        """Attempt to cast the arg to something traceable. Otherwise, error."""
        if isinstance(arg, str):
            raise ValueError(f"Cannot build causal graph for arg {arg} of type str")
        if isinstance(arg, int):
            arg = float(arg)
        return arg

    if isinstance(args, Iterable):
        return [check_and_cast(arg) for arg in args]
    return check_and_cast(args)


def is_dead_node(node):
    """Check if this node adds a spurious dependency, e.g. 0 * constant."""
    if hasattr(node.fun, "fun") and node.fun.fun == np.multiply:
        # If we multiplied a parent by a constant 0, skip.
        for idx, arg in enumerate(node.args):
            if idx not in node.parent_argnums and arg == 0.0:
                return True
    return False


def is_array_node(node):
    """Check if this node creates a array."""
    if hasattr(node.fun, "fun"):
        return node.fun.fun.__name__ == wn.traceable_numpy.array_from_args.__name__
    return False


def is_getitem_node(node):
    """Check if the node extracts an item from an array."""
    if hasattr(node.fun, "fun"):
        return node.fun.fun.__name__ == "__getitem__"
    return False


def backtrack(output_boxes, input_node_map):
    """Trace the computation graph from output boxes to ancestors in the input.

    Given the output of a function called on boxed inputs, backtrack finds
    which, if any, of the nodes in the input_node_map each output element
    depends on.

    Parameters
    ----------
        output_boxes: list
            List of outputs from a traced function.
        input_node_map: dict
            Dictionary of TracerNode -> box elements representing the potential root
            nodes of the computation graph.

    Returns
    -------
        output_dependencies: dict
            Dictionary mapping output_idx to a (deduplicated) list of nodes
            in the input that output idx depends on. For instance, if we have

            def f(x, y, z):
                return x + y, y + z

            Then, we get output_dependencies {0: [Node_x, Node_y], 1: [Node_y, Node_z]}.

    """
    if not isinstance(output_boxes, Iterable):
        output_boxes = [output_boxes]

    # For each output, figure out which (if any) of the inputs it depends on
    # by solving a graph search problem on the computation graph.
    output_dependencies = dict(
        (output_idx, set()) for output_idx in range(len(output_boxes))
    )
    for idx, output_box in enumerate(output_boxes):
        # If the output isn't a box, then it's independent of all the inputs.
        if isbox(output_box):
            # Use breadth-first search to find parents
            # pylint: disable-msg=protected-access
            queue = [output_box._node]
            while queue:
                node = queue.pop(0)

                # Check if ancestor is an input node
                if node in input_node_map:
                    input_box = input_node_map[node]
                    if output_box._trace == input_box._trace:
                        output_dependencies[idx].add(node)

                # If the node corresponds to an irrelevant dependency,
                # e.g. 0 * constant, we skip it.
                if is_dead_node(node):
                    continue

                # Add the parents to the queue
                for parent in node.parents:
                    # This is a hack to handle a common pattern where the
                    # user returns z = np.array([x1, x2]) from a function and
                    # then we inspect one item, e.g. z[0]. By default, both x1 and x2
                    # are parents of z, but in this case, we can detect that
                    # z[0] only depends on x1. In general, however, if after
                    # constructing z, we apply another function to the array,
                    # e.g. f(z) we cannot detect the subset of x that the result depends on.
                    # This is one of the limitations of using boxes based on autograd.
                    # Note this limitation means that we might add extra
                    # edges to the causal graph. It will never mean we miss an
                    # edge, so the returned graphs are still valid.
                    if is_getitem_node(node) and is_array_node(parent):
                        index = node.args[1]
                        queue.append(parent.parents[index])
                    else:
                        queue.append(parent)

    return dict((k, list(v)) for k, v in output_dependencies.items())


def trace_dependencies(func, args):
    """Trace dependencies of the outputs of a function.

    Parameters
    ----------
        func: function
            The function to trace
        args: List
            List of numerical arguments to trace dependencies for.

    Returns
    -------
        output_dependencies: map
            Dictionary mapping from output_idx to the set of input_idx that the
            output depends on.

    Example
    -------
    .. code-block:: python

        >>> def func(x, y, z):
        ...     return x + y, y * z

        >>> trace_dependencies(func, [4, 2, 1])
        {0: [0, 1], 1: {1, 2}

    """
    if not isinstance(args, Iterable):
        args = [args]

    args = check_and_cast_args(args)

    with trace_stack.new_trace() as trace:

        # Wrap each input argument in a "box" with an associated start node to allow tracing
        input_nodes = [
            TracerNode.new_root(arg, name=idx) for idx, arg in enumerate(args)
        ]
        input_boxes = [
            new_box(arg, trace, start_node)
            for arg, start_node in zip(args, input_nodes)
        ]

        # Execute the function to build the computation graph
        output_boxes = func(*input_boxes)

        input_node_map = dict(zip(input_nodes, input_boxes))
        raw_dependencies = backtrack(output_boxes, input_node_map)

        # convert to output-> [list input_idxs]
        dependencies = {}
        for output_idx, input_nodes in raw_dependencies.items():
            dependencies[output_idx] = [node.name for node in input_nodes]
        return dependencies


def trace_dynamics(dynamics):
    """Generate a function to compute the dependency graph of the dynamics.

    Given the dynamics map for a dynamical system simulator, we construct
    another function that, given the state, time, and config, computes
    how the output states depend on the input states and the configuration
    parameters. We cannot statically trace the dynamics because, in general,
    they may depend on time or the value of the variables in the state or
    config.

    Parameters
    ----------
        dynamics: function
            A dynamics map that has the signature def dynamics(state, time,
            config), where state is a tuple, and config is a dataclass. The
            function should return the derivative of the state as an iterable.

    Returns
    -------
        dynamics_tracer: function
            A function which, given an instantiated state, time, and config
            for the simulator, computes the dependency graph as a dictionary
                {output_idx:
                        {"states": [input_state_dependencies],
                        "configs": [config_dependencies]}}

    Example
    -------
    .. code-block:: python

        >>> def dynamics(state, time, config):
        ...    x_1, x_2 = state
        ...    if time < 10:
        ...        return x1, config.param
        ...    return x1 + x2, x2

        >>> tracer = dynamics_tracer(dynamics)
        >>> tracer(state, 0, config)
        {0: {"states": ["x1"], config: []},
         1: {"states": [], config: ["param"]},
        >>> tracer(state, 20, config)
        {0: {"states": ["x1", "x2"], config: []},
         1: {"states": ["x2"], config: []},

    """

    def tracer(state, time, config):
        """Compute output dependencies of the dynamics."""
        with trace_stack.new_trace() as trace:
            state_box, state_node_map = wn.causal_graphs.dataclass_to_box(state, trace)
            config_box, config_node_map = wn.causal_graphs.dataclass_to_box(
                config, trace, skip_args=["start_time", "end_time", "delta_t"]
            )

            # Aggregrate the state and config maps
            node_map = state_node_map.copy()
            node_map.update(config_node_map)

            # Run the dynamics forward to construct the computation graph
            outbox = dynamics(state=state_box.values(), time=time, config=config_box)

            # Trace the output dependencies to nodes in the input
            dependencies = wn.causal_graphs.backtrack(outbox, node_map)

            # Generate a map from output name to input and config dependency names
            names = [f.name for f in dataclasses.fields(state)]
            named_dependencies = {}
            for output_idx, input_nodes in dependencies.items():
                state_deps, config_deps = [], []
                for node in input_nodes:
                    if node in state_node_map:
                        state_deps.append(node.name)
                    else:
                        config_deps.append(node.name)
                named_dependencies[names[output_idx]] = {
                    "states": state_deps,
                    "configs": config_deps,
                }

            return named_dependencies

    return tracer


def build_dynamics_graph(simulator, runs, config, config_nodes=False):
    """Build a graph of the dynamics for a collection of runs.

    This feature is still experiment is currently only supported on a
    handful of simulators, namely HIV, lotka_volterra, and opioid.

    Parameters
    ----------
        simulator: whynot simulator module
            Simulator dynamics to trace, e.g. whynot.hiv.
        runs: list
            List of whynot.dynamics.Runs objects generated by the simulator
        config: whynot.dynamics.BaseConfig
            Config object use for all of the simulator runs
        config_nodes: bool
            Whether or not to add nodes and edges to the graphs for
            configuration variables or to leave the explicit.

    Returns
    -------
        graph: networkx.DiGraph
            Causal graph representing the dynamics for the collection of runs.

    Example
    -------
    .. code-block:: python

        >>> import whynot as wn
        >>> config = wn.hiv.Config(delta=1., end_time=1.)
        >>> runs = [wn.hiv.simulate(wn.hiv.State(),  config)]
        >>> graph = wn.causal_graphs.build_dynamics_graph(wn.hiv, runs, config)
        >>> print(list(graph.nodes))
        uninfected_T1_0.0, infected_T1_0.0, ..., immune_response1.0

    """
    if not simulator.SUPPORTS_CAUSAL_GRAPHS:
        raise ValueError(
            "Simulator does not currently support causal graph construction."
        )

    dynamics_tracer = trace_dynamics(simulator.dynamics)

    # TODO: Eventually we should take the union over all of the discovered
    # edges.  For now, just take the edges from the first run.
    run = runs[0]

    graph = nx.DiGraph()

    state_names = run.initial_state.variable_names()
    config_name = config.parameter_names()

    # Add a node for every state/parameter at every timestep
    for time in run.times:
        graph.add_nodes_from([f"{name}_{time}" for name in state_names])
        if config_nodes:
            graph.add_nodes_from([f"PARAM:{name}_{time}" for name in config_name])

    # Add edges by tracing the dynamics. Assumes the connectivity pattern
    # is constant between each time step (but can generally vary with time).
    for idx, start_state in enumerate(run.states[:-1]):
        start_time, end_time = run.times[idx], run.times[idx + 1]
        dependency_map = dynamics_tracer(start_state, start_time, config)
        for output, dependencies in dependency_map.items():
            output_name = f"{output}_{end_time}"
            graph.add_edges_from(
                [
                    (f"{state}_{start_time}", output_name)
                    for state in dependencies["states"]
                ]
            )
            if config_nodes:
                graph.add_edges_from(
                    [
                        (f"PARAM:{conf_name}_{start_time}", output_name)
                        for conf_name in dependencies["configs"]
                    ]
                )

    return graph


def ate_graph_builder(
    simulator,
    run,
    config,
    intervention,
    treatment_dependencies,
    covariate_dependencies,
    outcome_dependencies,
):
    """Build a causal graph for an average treatment effect estimation experiment.

    Parameters
    ----------
        simulator: whynot.simulator module
            The simulator used to conduct the experiment.
        run: whynot.dynamics.Run
            A single run object generated from the simulator under the config
            and intervention.
        config: whynot.dynamics.BaseConfig
            Configuration object used for the experiment.
        intervention: whynot.dynamics.BaseIntervention
            Intervention object used for the experiment.
        treatment_dependencies: list
            List of node names treatment depends on.
        covariate_dependencies: dict
            Dictionary from covariate_idx to a list of nodes the covariate
            depends on.
        outcome_dependencies: list
            List of node names the outcome variable depends on.

    Returns
    -------
        ate_graph: networkx.DiGraph
            Directed graph corresponding to the causal inference experiment.
            The nodes in the graph include:
                1) State variables from the unrolled dynamics
                2) Configuration parameters at each time step
                3) The treatment node, and
                4) The outcome node.
            Edges in the graph include:
                1) Edges between state nodes from the dynamics
                2) Edges between config and state nodes if a state variable
                   depends on a parameter
                3) Edges between state nodes and the treatment
                4) Edges between the treatment and all of the parameters/nodes
                   that are intervened upon.
                5) Edges between the state nodes and the outcome.

            Finally, the graph has an attribute graph["covariate_names"]
            that record which covariate corresponds to which state nodes.

    """
    graph = build_dynamics_graph(simulator, [run], config, config_nodes=True)

    # All nodes are unobserved by default
    for node in graph.nodes:
        graph.nodes[node]["observed"] = "no"

    # Treatment nodes
    graph.add_node("Treatment", observed="yes")
    # Incoming edges
    graph.add_edges_from([(dep, "Treatment") for dep in treatment_dependencies])
    # Outgoing edges
    intervention_nodes = []
    for time in run.times:
        if time >= intervention.time:
            intervention_nodes.extend(
                [f"PARAM:{param}_{time}" for param in intervention.updates]
            )
    graph.add_edges_from([("Treatment", node) for node in intervention_nodes])

    # Outcome nodes
    graph.add_node("Outcome", observed="yes")
    graph.add_edges_from([(node, "Outcome") for node in outcome_dependencies])

    covariate_names = ["" for _ in range(len(covariate_dependencies))]
    for idx, dependencies in covariate_dependencies.items():
        if len(dependencies) > 1:
            error_msg = (
                "Each covariate should depend on a single node, but covariate "
                f"{idx} depends on {len(dependencies)} nodes!"
            )
            raise ValueError(error_msg)
        name = dependencies[0]
        graph.nodes[name]["observed"] = "yes"
        covariate_names[idx] = name

    graph.graph["covariate_names"] = covariate_names

    return graph
