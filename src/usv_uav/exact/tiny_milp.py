from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from usv_uav.core.models import Instance
from usv_uav.core.solution import Solution
from usv_uav.exact.oracle import enumerate_feasible_schedules
from usv_uav.scheduling.evaluator import EvaluationResult, Evaluator


@dataclass(frozen=True, slots=True)
class MILPResult:
    best_solution: Solution
    best_evaluation: EvaluationResult
    objective_bound: float
    gap: float
    status: str
    schedule_columns: int


class TinySchedulingMILP:
    """Exact schedule-column MILP for the independent J<=6 validation gate.

    Every column is a complete decoder-feasible structural schedule. The MILP
    selects exactly one minimum-makespan column. This is intentionally a tiny
    validation model, not the scalable E2 formulation.
    """

    def __init__(self, *, max_tasks: int = 6) -> None:
        self.max_tasks = max_tasks

    def solve(self, instance: Instance, evaluator: Evaluator) -> MILPResult:
        schedules, _, _ = enumerate_feasible_schedules(
            instance,
            evaluator,
            max_tasks=self.max_tasks,
        )
        if not schedules:
            raise RuntimeError("tiny scheduling MILP has no feasible schedule columns")
        objective = np.asarray(
            [schedule.evaluation.makespan_min for schedule in schedules],
            dtype=np.float64,
        )
        result = milp(
            c=objective,
            integrality=np.ones(len(schedules)),
            bounds=Bounds(np.zeros(len(schedules)), np.ones(len(schedules))),
            constraints=LinearConstraint(
                np.ones((1, len(schedules))),
                lb=np.ones(1),
                ub=np.ones(1),
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"tiny scheduling MILP failed: {result.message}")
        selected = np.flatnonzero(result.x > 0.5)
        if len(selected) != 1:
            raise RuntimeError("tiny scheduling MILP did not select exactly one schedule")
        chosen = schedules[int(selected[0])]
        return MILPResult(
            best_solution=chosen.solution,
            best_evaluation=chosen.evaluation,
            objective_bound=float(result.fun),
            gap=0.0,
            status=str(result.message),
            schedule_columns=len(schedules),
        )
