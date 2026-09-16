from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class TSPSolution:
    route: tuple[int, ...]
    objective_distance_km: float
    optimal: bool
    solver: str
    best_bound: float
    gap: float
    runtime_sec: float

    @property
    def distance(self) -> float:
        """Backward-compatible name for the route objective."""
        return self.objective_distance_km


class TSPSolver(Protocol):
    def solve(self, distance_matrix: ArrayLike, depot: int = 0) -> TSPSolution: ...


def _validate_matrix(distance_matrix: ArrayLike, depot: int) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("TSP distance matrix must be square with at least two nodes")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("TSP distances must be finite and non-negative")
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("TSP distance matrix must be symmetric with zero diagonal")
    if not 0 <= depot < len(matrix):
        raise ValueError("invalid TSP depot index")
    return matrix


def _route_distance(route: tuple[int, ...], matrix: np.ndarray) -> float:
    return float(sum(matrix[first, second] for first, second in zip(route, route[1:])))


def _canonical_cycle_orientation(route: tuple[int, ...], depot: int) -> tuple[int, ...]:
    """Choose one stable orientation for a symmetric Hamiltonian cycle."""
    if len(route) < 3 or route[0] != depot or route[-1] != depot:
        raise ValueError("TSP route must start and end at the depot")
    backward = (depot, *reversed(route[1:-1]), depot)
    return min(route, backward)


class NN2OptTSPSolver:
    def solve(self, distance_matrix: ArrayLike, depot: int = 0) -> TSPSolution:
        started = perf_counter()
        matrix = _validate_matrix(distance_matrix, depot)
        unvisited = set(range(len(matrix))) - {depot}
        route = [depot]
        while unvisited:
            current = route[-1]
            next_node = min(unvisited, key=lambda node: (matrix[current, node], node))
            unvisited.remove(next_node)
            route.append(next_node)
        route.append(depot)

        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_pair: tuple[int, int] | None = None
            for first in range(1, len(route) - 2):
                for second in range(first + 1, len(route) - 1):
                    a, b = route[first - 1], route[first]
                    c, d = route[second], route[second + 1]
                    delta = matrix[a, c] + matrix[b, d] - matrix[a, b] - matrix[c, d]
                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_pair = (first, second)
            if best_pair is not None:
                first, second = best_pair
                route[first : second + 1] = reversed(route[first : second + 1])
                improved = True
        result = tuple(route)
        objective = _route_distance(result, matrix)
        return TSPSolution(result, objective, False, "nn_2opt", 0.0, inf, perf_counter() - started)


def _subtours(edges: list[tuple[int, int]], node_count: int) -> list[list[int]]:
    outgoing = {first: second for first, second in edges}
    remaining = set(range(node_count))
    cycles: list[list[int]] = []
    while remaining:
        start = min(remaining)
        cycle: list[int] = []
        current = start
        while current not in cycle:
            cycle.append(current)
            remaining.discard(current)
            current = outgoing[current]
        cycles.append(cycle[cycle.index(current) :])
    return cycles


class GurobiTSPSolver:
    """Exact directed TSP with lazy subtour elimination."""

    def __init__(self, *, time_limit_sec: float | None = None, output: bool = False) -> None:
        self.time_limit_sec = time_limit_sec
        self.output = output

    def solve(self, distance_matrix: ArrayLike, depot: int = 0) -> TSPSolution:
        started = perf_counter()
        matrix = _validate_matrix(distance_matrix, depot)
        try:
            import gurobipy as gp
        except ImportError as error:
            raise RuntimeError("gurobipy is required for GurobiTSPSolver") from error

        node_count = len(matrix)
        model = gp.Model("safe_route_tsp")
        model.Params.OutputFlag = int(self.output)
        model.Params.LazyConstraints = 1
        model.Params.Seed = 0
        model.Params.Threads = 1
        if self.time_limit_sec is not None:
            model.Params.TimeLimit = self.time_limit_sec
        arcs = [(i, j) for i in range(node_count) for j in range(node_count) if i != j]
        x = model.addVars(arcs, vtype=gp.GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(matrix[i, j] * x[i, j] for i, j in arcs), gp.GRB.MINIMIZE)
        model.addConstrs((gp.quicksum(x[i, j] for j in range(node_count) if j != i) == 1 for i in range(node_count)), name="out")
        model.addConstrs((gp.quicksum(x[i, j] for i in range(node_count) if i != j) == 1 for j in range(node_count)), name="in")

        def callback(active_model, where):
            if where != gp.GRB.Callback.MIPSOL:
                return
            selected = [(i, j) for i, j in arcs if active_model.cbGetSolution(x[i, j]) > 0.5]
            for cycle in _subtours(selected, node_count):
                if len(cycle) == node_count:
                    continue
                active_model.cbLazy(gp.quicksum(x[i, j] for i in cycle for j in cycle if i != j) <= len(cycle) - 1)

        warm = NN2OptTSPSolver().solve(matrix, depot)
        for first, second in zip(warm.route, warm.route[1:]):
            x[first, second].Start = 1.0
        model.optimize(callback)
        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi TSP returned no solution (status {model.Status})")
        selected = {(i, j) for i, j in arcs if x[i, j].X > 0.5}
        route = [depot]
        while len(route) <= node_count:
            next_nodes = [j for i, j in selected if i == route[-1]]
            if len(next_nodes) != 1:
                raise RuntimeError("invalid exact TSP successor structure")
            route.append(next_nodes[0])
            if route[-1] == depot:
                break
        if len(route) != node_count + 1 or route[-1] != depot:
            raise RuntimeError("exact TSP solution is not one Hamiltonian cycle")
        canonical_route = _canonical_cycle_orientation(tuple(route), depot)
        optimal = model.Status == gp.GRB.OPTIMAL
        objective = float(model.ObjVal)
        best_bound = float(model.ObjBound)
        gap = float(model.MIPGap) if model.SolCount else inf
        return TSPSolution(
            canonical_route, objective, optimal, "gurobi_lazy",
            best_bound, gap, float(model.Runtime or (perf_counter() - started)),
        )
