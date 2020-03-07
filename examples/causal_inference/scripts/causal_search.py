"""An implementation of the IC* (Inductive Causation with latent variables) algorithm.

This implementation is taken nearly wholesale from the causality package:
    https://github.com/akelleh/causality/blob/master/causality/inference/search/__init__.py

Updated are primarily minor syntax changes to work with the most recent version of networkx.
"""

import itertools
import networkx as nx
import statsmodels.api as sm

try:
    xrange
except NameError:
    xrange = range


class RobustRegressionTest:
    def __init__(self, y, x, z, data, alpha):
        self.regression = sm.RLM(data[y], data[x + z])
        self.result = self.regression.fit()
        self.coefficient = self.result.params[x][0]
        confidence_interval = self.result.conf_int(alpha=alpha / 2.0)
        self.upper = confidence_interval[1][x][0]
        self.lower = confidence_interval[0][x][0]

    def independent(self):
        if self.coefficient > 0.0:
            if self.lower > 0.0:
                return False
            else:
                return True
        else:
            if self.upper < 0.0:
                return False
            else:
                return True


class SearchException(Exception):
    pass


class IC:
    def __init__(self, independence_test, alpha=0.05, k=None):
        self.independence_test = independence_test
        self.alpha = alpha
        self.separating_sets = None
        self._g = None
        self.max_k = k

    def search(self, data, variable_types):
        self._build_g(variable_types)
        self._find_skeleton(data, variable_types)
        self._orient_colliders()

        added_arrows = True
        while added_arrows:
            R1_added_arrows = self._apply_recursion_rule_1()
            R2_added_arrows = self._apply_recursion_rule_2()
            added_arrows = R1_added_arrows or R2_added_arrows

        return self._g

    def _build_g(self, variable_types):
        """
        This initializes a complete graph over the variables.  We'll run
        independence tests on the complete graph to cut edges by trying to
        find separating sets.
        """
        self._g = nx.Graph()
        self._g.add_nodes_from(variable_types.keys())
        for var, var_type in variable_types.items():
            self._g.nodes[var]["type"] = var_type
        edges_to_add = []
        for (node_a, node_b) in itertools.combinations(self._g.nodes.keys(), 2):
            edges_to_add.append((node_a, node_b))
        self._g.add_edges_from(edges_to_add, marked=False)

    def _apply_recursion_rule_1(self):
        added_arrows = False
        for c in self._g.nodes():
            for (a, b) in itertools.combinations(self._g.neighbors(c), 2):
                if not self._g.has_edge(a, b):
                    if (
                        c in self._g[a][c]["arrows"]
                        and c not in self._g[b][c]["arrows"]
                        and not (
                            b in self._g[b][c]["arrows"] and self._g[b][c]["marked"]
                        )
                    ):
                        self._g[b][c]["arrows"].append(b)
                        self._g[b][c]["marked"] = True
                        added_arrows = True
                    if (
                        c in self._g[b][c]["arrows"]
                        and c not in self._g[a][c]["arrows"]
                        and not (
                            a in self._g[a][c]["arrows"] and self._g[a][c]["marked"]
                        )
                    ):
                        self._g[a][c]["arrows"].append(a)
                        self._g[a][c]["marked"] = True
                        added_arrows = True
        return added_arrows

    def _apply_recursion_rule_2(self):
        added_arrows = False
        for (a, b) in self._g.edges():
            if b not in self._g[a][b]["arrows"]:
                if self._marked_directed_path(a, b):
                    self._g[a][b]["arrows"].append(b)
                    added_arrows = True
        return added_arrows

    def _marked_directed_path(self, a, b):
        seen = [a]
        neighbors = [(a, neighbor) for neighbor in self._g.neighbors(a)]
        while neighbors:
            (parent, child) = neighbors.pop()
            if (
                child in self._g[parent][child]["arrows"]
                and self._g[parent][child]["marked"]
            ):
                if child == b:
                    return True
                if child not in seen:
                    neighbors += [
                        (child, neighbor) for neighbor in self._g.neighbors(child)
                    ]
                seen.append(child)
        return False

    def _orient_colliders(self):
        for v_i, v_j in self._g.edges():
            self._g[v_i][v_j]["arrows"] = []
        for v_c in self._g.nodes():
            for (v_a, v_b) in itertools.combinations(self._g.neighbors(v_c), 2):
                if not self._g.has_edge(v_a, v_b):
                    if v_c not in self.separating_set(v_a, v_b):
                        self._g[v_a][v_c]["arrows"].append(v_c)
                        self._g[v_b][v_c]["arrows"].append(v_c)

    def separating_set(self, xi, xj, data=None, variable_types=None):
        if not self.separating_sets and data and variable_types:
            if not self._g:
                self._build_g(variable_types)
            self._find_skeleton(data, variable_types)
        elif not self.separating_sets and not (data and variable_types):
            raise SearchException(
                "Can't measure separating sets: Need data and var types."
            )
        if (xi, xj) in self.separating_sets:
            return self.separating_sets[(xi, xj)]
        elif (xj, xi) in self.separating_sets:
            return self.separating_sets[(xj, xi)]
        else:
            return False

    def _find_skeleton(self, data, variable_types):
        """
        For each pair of nodes, run a conditional independence test over
        larger and larger conditioning sets to try to find a set that
        d-separates the pair.  If such a set exists, cut the edge between
        the nodes.  If not, keep the edge.
        """
        self.separating_sets = {}
        if not self.max_k:
            self.max_k = len(self._g.nodes) + 1
        for N in range(self.max_k + 1):
            for (x, y) in list(self._g.edges()):
                x_neighbors = list(self._g.neighbors(x))
                y_neighbors = list(self._g.neighbors(y))
                z_candidates = list(set(x_neighbors + y_neighbors) - set([x, y]))
                for z in itertools.combinations(z_candidates, N):
                    test = self.independence_test([y], [x], list(z), data, self.alpha)
                    if test.independent():
                        self._g.remove_edge(x, y)
                        self.separating_sets[(x, y)] = z
                        break


def f1(true_edges, estimated_edges):
    """Given a list of true and estimed edges, return f1."""
    tp = sum(1.0 for edge in estimated_edges if edge in true_edges)
    fp = len(estimated_edges) - tp
    fn = sum(1.0 for edge in true_edges if edge not in estimated_edges)
    return 2 * tp / (2 * tp + fp + fn)


def undirected_f1(true_graph, estimated_graph):
    """F1 score for discovering the skeleton."""
    # Sorting the edges ensures direction doesn't matter
    true_edges = set(tuple(sorted(edge)) for edge in list(true_graph.edges))
    estimated_edges = set(tuple(sorted(edge)) for edge in list(estimated_graph.edges))
    return f1(true_edges, estimated_edges)


def directed_f1(true_graph, estimated_graph):
    """F1 score for discovering the direction of all the edges."""
    true_edges = set(edge for edge in list(true_graph.edges))
    estimated_edges = set()
    for edge, data in estimated_graph.edges.items():
        if data["arrows"]:
            source, sink = edge
            if sink in data["arrows"]:
                estimated_edges.add((source, sink))
            else:
                estimated_edges.add((sink, source))
    return f1(true_edges, estimated_edges)
