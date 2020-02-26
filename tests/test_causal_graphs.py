"""Test causal graph building functionality."""
import copy
import dataclasses
import itertools
import math

import numpy as np
import pytest

import whynot as wn
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState

#################################
# Basic dependency tracing tests.
##################################
def test_dependency_tracing_basic():
    """Basic tests for tracing dependencies."""

    def func(x1, x2):
        delta_x1 = 45 * x1 + 18 * x2
        delta_x2 = 2 * x2
        return [delta_x1, delta_x2, 23]

    dependencies = wn.causal_graphs.trace_dependencies(func, (1.0, 0.2))
    assert set(dependencies[0]) == set([0, 1])
    assert set(dependencies[1]) == set([1])
    assert set(dependencies[2]) == set()

    # Ensure things work for univariate output
    def univar(x1, x2):
        return x1 + x2

    dependencies = wn.causal_graphs.trace_dependencies(univar, (1.0, 0.2))
    assert set(dependencies[0]) == set([0, 1])

    # univariate input
    def univar_input(x1):
        return 2 * x1 ** 2

    dependencies = wn.causal_graphs.trace_dependencies(univar_input, 4)
    assert set(dependencies[0]) == set([0])


def test_noop():
    """Test is the causal graph builder correctly handles noops."""

    def noop(x1):
        return 0 * x1

    dependencies = wn.causal_graphs.trace_dependencies(noop, 1.0)
    assert set(dependencies[0]) == set()

    def noop1(x, y, z):
        a = 2 * x + y - z
        b = x * y * z * 0.0
        c = 0 * a ** 2 - b
        return c

    dependencies = wn.causal_graphs.trace_dependencies(noop1, (1, 2, 3))
    assert set(dependencies[0]) == set()

    dependencies = wn.causal_graphs.trace_dependencies(noop1, (0.0, 2, 3))
    assert set(dependencies[0]) == set()

    # Avoid spurious hit
    def false_noop(x, y):
        return x + y

    dependencies = wn.causal_graphs.trace_dependencies(false_noop, (0.0, 1.0))
    assert set(dependencies[0]) == set([0, 1])

    dependencies = wn.causal_graphs.trace_dependencies(false_noop, (0.0, 0.0))
    assert set(dependencies[0]) == set([0, 1])


def test_dependency_tracing_numpy():
    """Test for tracing dependencies with numpy operations."""

    def func(a, b, c):
        """Simple matrix multiplication test."""
        vec = np.array([a, b, c])
        mat = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0, 1.0]])

        output = np.dot(mat, vec)
        return output

    dependencies = wn.causal_graphs.trace_dependencies(func, (3, 5, 7))
    assert set(dependencies[0]) == set([0])
    assert set(dependencies[1]) == set([0, 1, 2])
    assert set(dependencies[2]) == set([0, 2])


def test_error_handling():
    """Test dependency tracing correctly handles multiple datatypes."""

    def func(a, b):
        return a + b

    # Floats should succeed
    deps = wn.causal_graphs.trace_dependencies(func, (1.0, 0.2))
    assert set(deps[0]) == set([0, 1])

    # Int should be cast and work
    deps = wn.causal_graphs.trace_dependencies(func, (1.0, 2))
    assert set(deps[0]) == set([0, 1])

    a = np.random.rand(3, 3).astype(np.float32)
    b = np.random.rand(3, 3).astype(np.int32)
    deps = wn.causal_graphs.trace_dependencies(func, (a, b))
    assert set(deps[0]) == set([0, 1])

    a = np.random.rand(3, 3).astype(np.int32)
    b = np.random.rand(3, 3).astype(np.int32)
    deps = wn.causal_graphs.trace_dependencies(func, (a, b))
    assert set(deps[0]) == set([0, 1])

    # Should raise for string arguments
    with pytest.raises(ValueError):
        wn.causal_graphs.trace_dependencies(func, (1.0, "str"))


###########################
# Test tracing the dynamics
###########################
@dataclasses.dataclass
class SimpleState(BaseState):
    """State for simple dynamics simulator."""

    x1: float = 1.0
    x2: float = 2.0
    x3: float = 3.0


@dataclasses.dataclass
class SimpleConfig(BaseConfig):
    """Config for simple dynamics simulator."""

    param: float = 2.0
    start_time: float = 0.0
    end_time: float = 5.0


class SimpleIntervention(BaseIntervention):
    """Change param at the given time."""

    def __init__(self, time=5, **kwargs):
        super(SimpleIntervention, self).__init__(SimpleConfig, time, **kwargs)


def simple_dynamics(state, time, config):
    """Toy dynamical system simulator for unit tests.

    The simulator cycles through three connectivity patterns, depending on the
    time.
    """
    x1, x2, x3 = state
    connectivity = math.floor(time) % 3
    if connectivity == 0:
        # Fully connected
        dx1 = x1 + x2 + x3
        dx2 = config.param * x1 + x2 + x3
        dx3 = x1 + x2 + x3
    elif connectivity == 1:
        # Diagonal
        dx1 = x1
        dx2 = x2
        dx3 = x3
    else:
        # Only x2 depends on input
        dx1 = config.param
        dx2 = x1 + x2 + x3
        dx3 = 0.0

    return [dx1, dx2, dx3]


class SimpleSimulator:
    """Mock of a module."""

    def __init__(self):
        self.dynamics = simple_dynamics
        self.SUPPORTS_CAUSAL_GRAPHS = True

    @staticmethod
    def simulate(initial_state, config, intervention=None, seed=None):
        """Simulate function for simple dynamics."""
        states = [initial_state]
        state = initial_state.values()
        times = [0]
        for time in range(int(config.start_time), int(config.end_time)):
            if intervention and time >= intervention.time:
                cfg = config.update(intervention)
            else:
                cfg = config
            delta = np.array(simple_dynamics(state, time, cfg))
            state += delta
            states.append(SimpleState(*np.copy(state)))
            times.append(time + 1)
        return wn.dynamics.Run(states=states, times=times)


def test_dynamics_tracer():
    """Sanity check for dynamics tracer."""
    tracer = wn.causal_graphs.trace_dynamics(simple_dynamics)
    for time in range(10):
        state = SimpleState()
        config = SimpleConfig()
        dependencies = tracer(state, time, config)

        if time % 3 == 0:
            assert set(dependencies["x1"]["states"]) == set(["x1", "x2", "x3"])
            assert set(dependencies["x1"]["configs"]) == set([])
            assert set(dependencies["x2"]["states"]) == set(["x1", "x2", "x3"])
            assert set(dependencies["x2"]["configs"]) == set(["param"])
            assert set(dependencies["x3"]["states"]) == set(["x1", "x2", "x3"])
            assert set(dependencies["x3"]["configs"]) == set([])
        elif time % 3 == 1:
            assert set(dependencies["x1"]["states"]) == set(["x1"])
            assert set(dependencies["x1"]["configs"]) == set([])
            assert set(dependencies["x2"]["states"]) == set(["x2"])
            assert set(dependencies["x2"]["configs"]) == set([])
            assert set(dependencies["x3"]["states"]) == set(["x3"])
            assert set(dependencies["x3"]["configs"]) == set([])
        else:
            assert set(dependencies["x1"]["states"]) == set([])
            assert set(dependencies["x1"]["configs"]) == set(["param"])
            assert set(dependencies["x2"]["states"]) == set(["x1", "x2", "x3"])
            assert set(dependencies["x2"]["configs"]) == set([])
            assert set(dependencies["x3"]["states"]) == set([])
            assert set(dependencies["x3"]["configs"]) == set([])


def test_ate_causal_graph_builder():
    """Test the causal_graphs=True argument of dynamics experiments."""
    # We need to import numpy like this to enable tracing!
    import whynot.traceable_numpy as wnp

    def covariate_builder(run):
        # Get x1, x2, and x3 from different time steps.
        return wnp.array([run[0].x1, run[2].x2, run[3].x3])

    def outcome_extractor(run):
        # Run the sum of states in the last time step
        return wnp.sum(run.states[-1].values())

    def soft_threshold(x, tau, r=200):
        return 1.0 / (wnp.exp(tau * r - r * x) + 1)

    def propensity_scores(untreated_run, intervention):
        # Treat the run if x1 + x2 above some "soft-threshold"
        # at the intervention time
        run = untreated_run
        return 1.0 - 0.9 * soft_threshold(run[0].x1 + run[4].x2, tau=4)

    intervention = SimpleIntervention(time=3, param=1.0)
    experiment = wn.DynamicsExperiment(
        name="test_experiment.",
        description="testing_ate",
        simulator=SimpleSimulator(),
        simulator_config=SimpleConfig(param=2.0, end_time=6),
        intervention=intervention,
        state_sampler=lambda: SimpleState(),
        propensity_scorer=propensity_scores,
        outcome_extractor=outcome_extractor,
        covariate_builder=covariate_builder,
    )

    dset = experiment.run(num_samples=10, causal_graph=True)
    graph = dset.causal_graph
    times = list(range(0, 7))
    state_names = SimpleState.variable_names()
    config_names = SimpleConfig.parameter_names()

    # First check the graph contains exactly the nodes we expect
    nodes = copy.deepcopy(list(graph.nodes))
    nodes.remove("Treatment")
    nodes.remove("Outcome")
    for time in times:
        for state in state_names:
            nodes.remove(f"{state}_{time}")
        for config_name in config_names:
            nodes.remove(f"PARAM:{config_name}_{time}")
    assert len(nodes) == 0

    # Check the covariates are correct
    assert set(graph.graph["covariate_names"]) == set(["x1_0", "x2_2", "x3_3"])

    # Now, check the graph contains the edges we expect
    edges = copy.deepcopy(list(graph.edges))
    # Check outcome depends on all states at the last time
    for outcome_dep in [f"{name}_{times[-1]}" for name in state_names]:
        edges.remove((outcome_dep, "Outcome"))
    # Check incoming treatment edges are correct
    edges.remove(("x1_0", "Treatment"))
    edges.remove(("x2_4", "Treatment"))
    # Check outgoing treatment edges
    for time in times:
        if time >= intervention.time:
            edges.remove(("Treatment", f"PARAM:param_{time}"))

    # Now check edges match the simulator dynamics.
    for time in times[:-1]:
        if time % 3 == 0:
            start_nodes = [f"{name}_{time}" for name in state_names]
            end_nodes = [f"{name}_{time+1}" for name in state_names]
            for edge in itertools.product(start_nodes, end_nodes):
                edges.remove(edge)
            edges.remove((f"PARAM:param_{time}", f"x2_{time+1}"))
        elif time % 3 == 1:
            for name in state_names:
                edges.remove((f"{name}_{time}", f"{name}_{time+1}"))
        else:
            edges.remove((f"PARAM:param_{time}", f"x1_{time+1}"))
            for name in state_names:
                edges.remove((f"{name}_{time}", f"x2_{time+1}"))

    assert len(edges) == 0
