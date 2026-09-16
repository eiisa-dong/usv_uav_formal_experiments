"""Execution-only controls for deterministic ALNS candidate evaluation."""
from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from threading import local
from time import perf_counter
from typing import Callable, Literal, TypeVar


CandidateT = TypeVar("CandidateT")
EvaluationT = TypeVar("EvaluationT")


@dataclass(frozen=True, slots=True)
class ParallelRuntimeConfig:
    """Runtime policy kept separate from the frozen ALNS algorithm config."""

    enabled: bool = False
    backend: Literal["serial", "thread"] = "serial"
    max_workers: int = 1
    min_batch_size: int = 2
    deadline_guard_sec: float = 0.20

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("parallel max_workers must be at least one")
        if self.min_batch_size < 2:
            raise ValueError("parallel min_batch_size must be at least two")
        if self.deadline_guard_sec < 0:
            raise ValueError("parallel deadline_guard_sec must be non-negative")

    @property
    def uses_threads(self) -> bool:
        return self.enabled and self.backend == "thread"


@dataclass(frozen=True, slots=True)
class ParallelBatchResult:
    evaluated: tuple[tuple[CandidateT, EvaluationT], ...]
    wall_sec: float
    wait_sec: float


def thread_local_evaluate(
    evaluator_factory: Callable[[], object],
) -> Callable[[CandidateT], EvaluationT]:
    """Create one evaluator lazily per executor worker thread."""

    state = local()

    def evaluate(candidate: CandidateT) -> EvaluationT:
        evaluator = getattr(state, "evaluator", None)
        if evaluator is None:
            evaluator = evaluator_factory()
            state.evaluator = evaluator
        return evaluator.evaluate(candidate)

    return evaluate


def evaluate_candidates_threaded(
    *,
    candidates: tuple[CandidateT, ...],
    evaluate: Callable[[CandidateT], EvaluationT],
    executor: Executor,
) -> ParallelBatchResult:
    """Evaluate a batch concurrently while preserving its input order."""

    candidates = tuple(candidates)
    started = perf_counter()
    wait_started = perf_counter()
    evaluations = tuple(executor.map(evaluate, candidates))
    wait_sec = perf_counter() - wait_started
    wall_sec = perf_counter() - started
    return ParallelBatchResult(
        evaluated=tuple(zip(candidates, evaluations)),
        wall_sec=wall_sec,
        wait_sec=wait_sec,
    )
