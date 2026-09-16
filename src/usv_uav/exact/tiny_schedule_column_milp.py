from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from usv_uav.core.models import Instance
from usv_uav.core.solution import Solution
from usv_uav.exact.oracle import enumerate_feasible_schedules
from usv_uav.scheduling.evaluator import EvaluationResult, Evaluator


@dataclass(frozen=True, slots=True)
class TinyScheduleColumnResult:
    solution: Solution
    evaluation: EvaluationResult
    status: str
    incumbent: float
    best_bound: float
    mip_gap: float
    cpu_sec: float
    optimal: bool
    schedule_columns: int

    @property
    def best_solution(self) -> Solution:
        return self.solution

    @property
    def best_evaluation(self) -> EvaluationResult:
        return self.evaluation

    @property
    def objective_bound(self) -> float:
        return self.best_bound

    @property
    def gap(self) -> float:
        return self.mip_gap


class TinyScheduleColumnMILP:
    """Exact J<=6 correctness validator based on complete schedule columns.

    This model is intentionally not described as a full scheduling MILP: each
    binary selects a complete schedule already decoded by the authoritative
    evaluator. Its role is the Oracle = TinyColumnMILP = Evaluator gate.
    """

    def __init__(self, *, max_tasks: int = 6, time_limit_sec: float = 3600.0) -> None:
        self.max_tasks = max_tasks
        self.time_limit_sec = time_limit_sec

    def solve(self, instance: Instance, evaluator: Evaluator) -> TinyScheduleColumnResult:
        if len(instance.tasks) > self.max_tasks:
            raise ValueError(
                f"tiny schedule-column MILP supports at most {self.max_tasks} tasks"
            )
        started = perf_counter()
        schedules, _, _ = enumerate_feasible_schedules(
            instance,
            evaluator,
            max_tasks=self.max_tasks,
        )
        if not schedules:
            raise RuntimeError("tiny schedule-column MILP has no feasible schedule columns")
        objective = np.asarray(
            [item.evaluation.makespan_min for item in schedules],
            dtype=np.float64,
        )
        result = milp(
            c=objective,
            integrality=np.ones(len(schedules)),
            bounds=Bounds(np.zeros(len(schedules)), np.ones(len(schedules))),
            constraints=LinearConstraint(np.ones((1, len(schedules))), np.ones(1), np.ones(1)),
            options={"presolve": True, "time_limit": self.time_limit_sec},
        )
        if result.x is None:
            raise RuntimeError(f"tiny schedule-column MILP returned no incumbent: {result.message}")
        selected = np.flatnonzero(result.x > 0.5)
        if len(selected) != 1:
            raise RuntimeError("tiny schedule-column MILP did not select exactly one schedule")
        chosen = schedules[int(selected[0])]
        incumbent = float(result.fun)
        bound = float(getattr(result, "mip_dual_bound", incumbent))
        gap = float(getattr(result, "mip_gap", 0.0))
        return TinyScheduleColumnResult(
            solution=chosen.solution,
            evaluation=chosen.evaluation,
            status=str(result.message),
            incumbent=incumbent,
            best_bound=bound,
            mip_gap=gap,
            cpu_sec=perf_counter() - started,
            optimal=bool(result.success),
            schedule_columns=len(schedules),
        )
