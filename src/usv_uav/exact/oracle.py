from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import inf
from typing import Iterator, Sequence

from usv_uav.core.models import Instance
from usv_uav.core.solution import Solution
from usv_uav.preprocessing.sortie_index import SortieIndex, build_sortie_index
from usv_uav.scheduling.evaluator import EvaluationResult, Evaluator


@dataclass(frozen=True, slots=True)
class EnumeratedSchedule:
    solution: Solution
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class OracleResult:
    best_solution: Solution
    best_evaluation: EvaluationResult
    exact_covers: int
    structural_schedules: int
    evaluated_schedules: int


def enumerate_exact_covers(
    instance: Instance,
    index: SortieIndex | None = None,
) -> Iterator[tuple[int, ...]]:
    active_index = index or build_sortie_index(
        instance.sortie_pool, instance.usv_route[1:-1]
    )
    all_tasks = frozenset(task.id for task in instance.tasks)

    def search(covered: frozenset[int], chosen: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if covered == all_tasks:
            yield chosen
            return
        anchor = min(all_tasks - covered)
        for sortie_id in active_index.sorties_by_task.get(anchor, ()):
            sortie = active_index.sortie_by_id[sortie_id]
            task_set = frozenset(sortie.task_sequence)
            if task_set & covered:
                continue
            yield from search(covered | task_set, (*chosen, sortie_id))

    yield from search(frozenset(), ())


def enumerate_uav_schedules(
    sortie_ids: Sequence[int],
    uav_count: int,
) -> Iterator[Solution]:
    """Enumerate each per-UAV ordered schedule exactly once for tiny cases."""
    seen: set[tuple[tuple[int, ...], ...]] = set()
    identifiers = tuple(sortie_ids)
    for assignment in product(range(uav_count), repeat=len(identifiers)):
        groups = [tuple(identifier for identifier, owner in zip(identifiers, assignment) if owner == uav_id) for uav_id in range(uav_count)]
        permutations_by_uav = [tuple(permutations(group)) if group else ((),) for group in groups]
        for ordered_groups in product(*permutations_by_uav):
            canonical = tuple(tuple(group) for group in ordered_groups)
            if canonical in seen:
                continue
            seen.add(canonical)
            yield Solution({uav_id: list(group) for uav_id, group in enumerate(canonical)})


def enumerate_feasible_schedules(
    instance: Instance,
    evaluator: Evaluator,
    *,
    max_tasks: int = 6,
) -> tuple[tuple[EnumeratedSchedule, ...], int, int]:
    if len(instance.tasks) > max_tasks:
        raise ValueError(f"tiny exact enumeration supports at most {max_tasks} tasks")
    index = build_sortie_index(instance.sortie_pool, instance.usv_route[1:-1])
    schedules: list[EnumeratedSchedule] = []
    exact_cover_count = 0
    structural_count = 0
    seen_solutions: set[tuple[tuple[int, ...], ...]] = set()
    for cover in enumerate_exact_covers(instance, index):
        exact_cover_count += 1
        for solution in enumerate_uav_schedules(cover, instance.uav_count):
            canonical = solution.canonical(instance.uav_count)
            if canonical in seen_solutions:
                continue
            seen_solutions.add(canonical)
            structural_count += 1
            evaluation = evaluator.evaluate(solution)
            if evaluation.feasible:
                schedules.append(EnumeratedSchedule(solution, evaluation))
    return tuple(schedules), exact_cover_count, structural_count


class BruteForceOracle:
    def __init__(self, *, max_tasks: int = 6) -> None:
        self.max_tasks = max_tasks

    def solve(self, instance: Instance, evaluator: Evaluator) -> OracleResult:
        schedules, covers, structural = enumerate_feasible_schedules(
            instance,
            evaluator,
            max_tasks=self.max_tasks,
        )
        if not schedules:
            raise RuntimeError("tiny oracle found no complete feasible schedule")
        best = min(
            schedules,
            key=lambda item: (
                item.evaluation.makespan_min,
                item.solution.canonical(instance.uav_count),
            ),
        )
        return OracleResult(
            best_solution=best.solution,
            best_evaluation=best.evaluation,
            exact_covers=covers,
            structural_schedules=structural,
            evaluated_schedules=len(schedules),
        )
