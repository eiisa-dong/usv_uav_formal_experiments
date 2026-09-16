from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable

from usv_uav.core.models import Instance
from usv_uav.core.solution import Solution
from usv_uav.scheduling.evaluator import EvaluationResult, Evaluator


@dataclass(frozen=True, slots=True)
class SolverBudget:
    wall_time_sec: float
    max_evaluations: int
    max_iterations: int = 1_000_000

    def __post_init__(self) -> None:
        if self.wall_time_sec <= 0 or self.max_evaluations <= 0 or self.max_iterations <= 0:
            raise ValueError("solver budget values must be positive")


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    iteration: int
    evaluations: int
    elapsed_sec: float
    best_objective: float


@dataclass(frozen=True, slots=True)
class AlgorithmResult:
    algorithm: str
    best_solution: Solution
    best_evaluation: EvaluationResult
    wall_time_sec: float
    evaluations: int
    time_to_best_sec: float
    eval_to_best: int
    convergence: tuple[ConvergencePoint, ...]
    operator_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    global_best_events: tuple[dict[str, Any], ...] = ()
    escape_events: tuple[dict[str, Any], ...] = ()


class SearchDeadline:
    """Shared monotonic wall-clock deadline with adaptive work-unit guards.

    The configured budget remains the true deadline.  Observed stage durations
    are used only to avoid starting another expensive atomic work unit when it
    is unlikely to finish before that deadline.
    """

    def __init__(
        self,
        *,
        start_time: float,
        budget_sec: float,
        clock: Callable[[], float] = monotonic,
        safety_margin_sec: float = 0.002,
        prediction_factor: float = 1.25,
    ) -> None:
        if budget_sec <= 0:
            raise ValueError("deadline budget must be positive")
        if safety_margin_sec < 0:
            raise ValueError("deadline safety margin must be non-negative")
        if prediction_factor < 1.0:
            raise ValueError("deadline prediction factor must be at least one")
        self.start_time = float(start_time)
        self.budget_sec = float(budget_sec)
        self.clock = clock
        self.safety_margin_sec = float(safety_margin_sec)
        self.prediction_factor = float(prediction_factor)
        self.deadline_abort_stage: str | None = None
        self._maximum_stage_sec: dict[str, float] = {}

    @property
    def elapsed_sec(self) -> float:
        return max(0.0, float(self.clock()) - self.start_time)

    def remaining_sec(self) -> float:
        return max(0.0, self.budget_sec - self.elapsed_sec)

    def expired(self) -> bool:
        return self.elapsed_sec >= self.budget_sec

    def exclude_elapsed(self, duration_sec: float) -> None:
        if duration_sec < 0:
            raise ValueError("excluded duration must be non-negative")
        self.start_time += duration_sec

    def observe(self, stage: str, duration_sec: float) -> None:
        if not stage:
            raise ValueError("deadline stage must be non-empty")
        if duration_sec < 0:
            raise ValueError("observed duration must be non-negative")
        self._maximum_stage_sec[stage] = max(
            self._maximum_stage_sec.get(stage, 0.0),
            float(duration_sec),
        )
        family = stage.partition(":")[0]
        if family != stage:
            self._maximum_stage_sec[family] = max(
                self._maximum_stage_sec.get(family, 0.0),
                float(duration_sec),
            )

    def mark_abort(self, stage: str) -> None:
        if not stage:
            raise ValueError("deadline abort stage must be non-empty")
        if self.deadline_abort_stage is None:
            self.deadline_abort_stage = stage

    def should_abort(self, stage: str) -> bool:
        """Check the real deadline at a cooperative, state-safe boundary."""
        if self.expired() or self.remaining_sec() <= self.safety_margin_sec:
            self.mark_abort(stage)
            return True
        return False

    def can_start(self, stage: str) -> bool:
        family = stage.partition(":")[0]
        observations = tuple(
            value
            for key in {stage, family}
            if (value := self._maximum_stage_sec.get(key)) is not None
        )
        observed = max(observations, default=None)
        reserve = self.safety_margin_sec
        if observed is not None:
            reserve += observed * self.prediction_factor
        if self.expired() or self.remaining_sec() <= reserve:
            self.mark_abort(stage)
            return False
        return True


class EvaluationBudget:
    def __init__(self, evaluator: Evaluator, budget: SolverBudget) -> None:
        self.evaluator = evaluator
        self.budget = budget
        self.deadline = SearchDeadline(
            start_time=monotonic(),
            budget_sec=budget.wall_time_sec,
        )
        self.start_evaluations = evaluator.evaluation_count

    @property
    def elapsed_sec(self) -> float:
        return self.deadline.elapsed_sec

    @property
    def evaluations(self) -> int:
        return self.evaluator.evaluation_count - self.start_evaluations

    @property
    def remaining_evaluations(self) -> int:
        return max(0, self.budget.max_evaluations - self.evaluations)

    @property
    def remaining_sec(self) -> float:
        return self.deadline.remaining_sec()

    def available(self, iteration: int = 0) -> bool:
        return (
            not self.deadline.expired()
            and self.deadline_abort_stage is None
            and self.evaluations < self.budget.max_evaluations
            and iteration < self.budget.max_iterations
        )

    @property
    def deadline_abort_stage(self) -> str | None:
        return self.deadline.deadline_abort_stage

    def deadline_expired(self, stage: str = "repair") -> bool:
        """Check the real deadline from inside cooperative work."""
        if self.evaluations >= self.budget.max_evaluations:
            return True
        return self.deadline.should_abort(stage)

    def can_start_stage(self, stage: str) -> bool:
        """Guard a non-interruptible stage using its observed duration."""
        if self.evaluations >= self.budget.max_evaluations:
            return False
        return self.deadline.can_start(stage)

    def observe_stage(self, stage: str, duration_sec: float) -> None:
        self.deadline.observe(stage, duration_sec)

    def mark_deadline_abort(self, stage: str) -> None:
        self.deadline.mark_abort(stage)

    def termination_metadata(self, iteration: int) -> dict[str, str | None]:
        if self.deadline_abort_stage is not None or self.deadline.expired():
            reason = "time_budget"
        elif self.evaluations >= self.budget.max_evaluations:
            reason = "max_evaluations"
        elif iteration >= self.budget.max_iterations:
            reason = "max_iterations"
        else:
            reason = "completed"
        return {
            "termination_reason": reason,
            "deadline_abort_stage": self.deadline_abort_stage,
        }

    def exclude_elapsed(self, duration_sec: float) -> None:
        """Exclude diagnostic-only overhead from the solver's logical clock."""
        self.deadline.exclude_elapsed(duration_sec)

    def evaluate(self, solution: Solution) -> EvaluationResult:
        if not self.available() or not self.can_start_stage("full_decoder"):
            raise RuntimeError("evaluation budget exhausted")
        started = monotonic()
        result = self.evaluator.evaluate(solution)
        self.observe_stage("full_decoder", monotonic() - started)
        return result

    def try_evaluate(
        self,
        solution: Solution,
        iteration: int = 0,
    ) -> EvaluationResult | None:
        """Evaluate only when budget remains after candidate construction.

        Candidate construction can cross the wall-clock deadline after a loop's
        initial budget check.  Returning ``None`` makes that normal termination
        rather than recording a solver failure.
        """
        if (
            not self.available(iteration)
            or not self.can_start_stage("full_decoder")
        ):
            return None
        started = monotonic()
        result = self.evaluator.evaluate(solution)
        self.observe_stage("full_decoder", monotonic() - started)
        if self.deadline.expired():
            self.mark_deadline_abort("full_decoder")
            return None
        return result

    def commit_external_evaluations(
        self,
        results: tuple[EvaluationResult, ...],
        *,
        duration_sec: float,
    ) -> bool:
        """Charge a completed worker batch on the controlling thread.

        Returns ``False`` when the batch completed after the hard deadline. The
        calls remain charged because they did execute, but their results must
        not influence the search state.
        """

        if len(results) > self.remaining_evaluations:
            raise RuntimeError("external evaluation batch exceeds remaining budget")
        self.evaluator.commit_external_results(results)
        self.observe_stage("full_decoder", duration_sec)
        if self.deadline.expired():
            self.mark_deadline_abort("full_decoder")
            return False
        return True


class Solver(ABC):
    name: str

    @abstractmethod
    def solve(
        self,
        instance: Instance,
        evaluator: Evaluator,
        seed: int,
        budget: SolverBudget,
    ) -> AlgorithmResult:
        raise NotImplementedError
