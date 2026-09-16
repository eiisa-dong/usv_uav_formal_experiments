from __future__ import annotations

from dataclasses import dataclass

from usv_uav.core.models import Instance
from usv_uav.exact.oracle import BruteForceOracle, OracleResult
from usv_uav.exact.tiny_schedule_column_milp import TinyScheduleColumnMILP, TinyScheduleColumnResult
from usv_uav.scheduling.evaluator import Evaluator


@dataclass(frozen=True, slots=True)
class ExactValidationResult:
    passed: bool
    oracle: OracleResult
    tiny_exact: TinyScheduleColumnResult
    evaluator_makespan_min: float


def validate_oracle_milp_evaluator(
    instance: Instance,
    evaluator: Evaluator,
    *,
    tolerance: float = 1e-8,
) -> ExactValidationResult:
    oracle = BruteForceOracle().solve(instance, evaluator)
    tiny_exact = TinyScheduleColumnMILP().solve(instance, evaluator)
    decoded = evaluator.evaluate(tiny_exact.solution)
    values = (
        oracle.best_evaluation.makespan_min,
        tiny_exact.evaluation.makespan_min,
        decoded.makespan_min,
    )
    passed = decoded.feasible and max(values) - min(values) <= tolerance
    return ExactValidationResult(passed, oracle, tiny_exact, decoded.makespan_min)
