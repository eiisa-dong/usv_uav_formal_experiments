from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from usv_uav.algorithms.construction import assign_sorties, construct_initial_solution
from usv_uav.core.models import Instance
from usv_uav.core.solution import Solution
from usv_uav.preprocessing.sortie_index import build_sortie_index
from usv_uav.scheduling.evaluator import EvaluationResult, Evaluator


@dataclass(frozen=True, slots=True)
class SchedulingBoundResult:
    solution: Solution
    evaluation: EvaluationResult
    upper_bound: float
    lower_bound: float
    gap: float
    status: str
    message: str
    selected_sorties: tuple[int, ...]
    model_incumbent: float | None
    model_dual_bound: float | None
    model_mip_gap: float | None
    optimal: bool
    runtime_sec: float


class SchedulingLowerBoundMILP:
    """Set-partitioning scheduling lower bound plus decoder-feasible upper bound.

    The model exactly partitions tasks into sorties and lower-bounds makespan by
    USV travel, maximum selected sortie duration, and average UAV workload. It
    is scalable evidence for E2; M6's schedule-column MILP remains the exact
    tiny-instance equality gate.
    """

    def __init__(self, *, time_limit_sec: float = 3600.0) -> None:
        self.time_limit_sec = time_limit_sec

    def solve(self, instance: Instance, evaluator: Evaluator) -> SchedulingBoundResult:
        started = perf_counter()
        sorties = instance.sortie_pool
        tasks = instance.tasks
        if not sorties:
            raise ValueError("scheduling MILP requires a non-empty sortie pool")
        task_position = {task.id: index for index, task in enumerate(tasks)}
        n_sorties = len(sorties)
        z_index = n_sorties
        row_count = len(tasks) + 1 + n_sorties
        matrix = lil_matrix((row_count, n_sorties + 1), dtype=np.float64)
        lower = np.full(row_count, -np.inf)
        upper = np.full(row_count, np.inf)
        for sortie_index, sortie in enumerate(sorties):
            for task_id in sortie.task_sequence:
                matrix[task_position[task_id], sortie_index] = 1.0
        lower[: len(tasks)] = upper[: len(tasks)] = 1.0

        workload_row = len(tasks)
        for sortie_index, sortie in enumerate(sorties):
            matrix[workload_row, sortie_index] = -sortie.nominal_duration_min / instance.uav_count
        matrix[workload_row, z_index] = 1.0
        lower[workload_row] = 0.0

        for sortie_index, sortie in enumerate(sorties):
            row = len(tasks) + 1 + sortie_index
            matrix[row, sortie_index] = -sortie.nominal_duration_min
            matrix[row, z_index] = 1.0
            lower[row] = 0.0

        objective = np.zeros(n_sorties + 1)
        objective[z_index] = 1.0
        variable_lower = np.zeros(n_sorties + 1)
        variable_upper = np.ones(n_sorties + 1)
        variable_upper[z_index] = np.inf
        result = milp(
            c=objective,
            integrality=np.concatenate((np.ones(n_sorties), np.zeros(1))),
            bounds=Bounds(variable_lower, variable_upper),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"presolve": True, "time_limit": self.time_limit_sec},
        )
        index = build_sortie_index(sorties, instance.usv_route[1:-1])
        solution = construct_initial_solution(instance, index)
        evaluation = evaluator.evaluate(solution)
        if not evaluation.feasible:
            raise RuntimeError("constructive solution is not a decoder-feasible upper bound")

        selected: tuple[int, ...] = ()
        if result.x is not None:
            selected = tuple(
                sorties[position].id
                for position in np.flatnonzero(result.x[:n_sorties] > 0.5)
            )
            try:
                partition_solution = assign_sorties(instance, selected, index)
                partition_evaluation = evaluator.evaluate(partition_solution)
                if (
                    partition_evaluation.feasible
                    and partition_evaluation.makespan_min < evaluation.makespan_min
                ):
                    solution = partition_solution
                    evaluation = partition_evaluation
            except RuntimeError:
                pass

        route_nodes = instance.usv_route[1:-1]
        mission_sail = sum(
            float(instance.safe_distance_matrix[evaluator.distance_index[first], evaluator.distance_index[second]])
            / evaluator.usv_speed_km_min
            for first, second in zip(route_nodes, route_nodes[1:])
        )
        model_incumbent = (
            float(result.fun)
            if result.fun is not None and isfinite(float(result.fun))
            else None
        )
        raw_dual_bound = getattr(result, "mip_dual_bound", None)
        model_dual_bound = (
            float(raw_dual_bound)
            if raw_dual_bound is not None and isfinite(float(raw_dual_bound))
            else None
        )
        optimal = int(getattr(result, "status", -1)) == 0 and bool(getattr(result, "success", False))
        if optimal and model_dual_bound is None:
            model_dual_bound = model_incumbent
        if model_dual_bound is not None and model_incumbent is not None:
            model_dual_bound = min(model_dual_bound, model_incumbent)
        certified_model_bound = max(0.0, model_dual_bound or 0.0)
        lower_bound = max(mission_sail, certified_model_bound)
        upper_bound = evaluation.makespan_min
        if lower_bound > upper_bound + 1e-7:
            raise RuntimeError(
                f"INVALID_LOWER_BOUND: LB={lower_bound}, UB={upper_bound}"
            )
        gap = (
            (upper_bound - lower_bound) / abs(upper_bound)
            if abs(upper_bound) > 1e-9
            else 0.0
        )
        raw_mip_gap = getattr(result, "mip_gap", None)
        model_mip_gap = (
            float(raw_mip_gap)
            if raw_mip_gap is not None and isfinite(float(raw_mip_gap))
            else None
        )
        status_names = {
            0: "optimal",
            1: "limit_reached",
            2: "infeasible",
            3: "unbounded",
            4: "solver_error",
        }
        status_code = int(getattr(result, "status", -1))
        return SchedulingBoundResult(
            solution=solution,
            evaluation=evaluation,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            gap=gap,
            status=status_names.get(status_code, f"status_{status_code}"),
            message=str(result.message),
            selected_sorties=selected,
            model_incumbent=model_incumbent,
            model_dual_bound=model_dual_bound,
            model_mip_gap=model_mip_gap,
            optimal=optimal,
            runtime_sec=perf_counter() - started,
        )


# Compatibility import for V1 callers; V2 reporting labels it structural LB.
SchedulingBoundMILP = SchedulingLowerBoundMILP
