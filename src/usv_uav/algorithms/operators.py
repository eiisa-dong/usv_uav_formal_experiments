from __future__ import annotations

from dataclasses import dataclass
from math import hypot
import random
from typing import Callable

from usv_uav.algorithms.construction import (
    RepairSearchProfile,
    RepairTimeBudgetExceeded,
    destroy_tasks_to_partial,
    repair_candidates,
    repair_solution,
    sequence_preserves_route_precedence,
    solution_sortie_ids,
)
from usv_uav.core.models import Instance
from usv_uav.core.partial_solution import PartialSolution
from usv_uav.core.solution import Solution
from usv_uav.preprocessing.sortie_index import SortieIndex
from usv_uav.scheduling.evaluator import EvaluationResult
from usv_uav.algorithms.route_guided.destroy import D6_route_corridor
from usv_uav.algorithms.route_guided.repair import R4P_progressive_recovery
from usv_uav.algorithms.r4_numba import R4KernelBackend
from usv_uav.algorithms.repair_numba import RepairKernelBackend


DestroyFunction = Callable[..., object]
RepairFunction = Callable[..., tuple[Solution, ...]]
LocalFunction = Callable[[Instance, Solution, SortieIndex, random.Random], Solution | None]


@dataclass(frozen=True, slots=True)
class DestroyOperator:
    name: str
    function: DestroyFunction


@dataclass(frozen=True, slots=True)
class RepairOperator:
    name: str
    function: RepairFunction


@dataclass(frozen=True, slots=True)
class LocalOperator:
    name: str
    function: LocalFunction


def _all_tasks(instance: Instance) -> list[int]:
    return [task.id for task in instance.tasks]


def random_removal(instance, solution, index, evaluation, rng, count) -> PartialSolution:
    del evaluation
    tasks = _all_tasks(instance)
    removed = rng.sample(tasks, min(count, len(tasks)))
    return destroy_tasks_to_partial(solution, removed, index, instance.uav_count)


def worst_removal(instance, solution, index, evaluation, rng, count) -> PartialSolution:
    del evaluation, rng
    score: dict[int, float] = {}
    for sortie_id in solution_sortie_ids(solution, instance.uav_count):
        sortie = index.sortie_by_id[sortie_id]
        contribution = sortie.nominal_duration_min + sortie.nominal_energy_wh / 100.0
        for task_id in sortie.task_sequence:
            score[task_id] = contribution / len(sortie.task_sequence)
    removed = [task for task, _ in sorted(score.items(), key=lambda item: (-item[1], item[0]))[:count]]
    return destroy_tasks_to_partial(solution, removed, index, instance.uav_count)


def synchronization_critical_removal(instance, solution, index, evaluation, rng, count) -> PartialSolution:
    del rng
    score: dict[int, float] = {}
    for execution in evaluation.executions:
        sortie = index.sortie_by_id[execution.sortie_id]
        contribution = execution.hover_min + execution.uav_lateness_to_usv_min
        for task_id in sortie.task_sequence:
            score[task_id] = score.get(task_id, 0.0) + contribution + 1e-6 * sortie.nominal_duration_min
    if not score or max(score.values()) <= 1e-5:
        return worst_removal(instance, solution, index, evaluation, random.Random(0), count)
    removed = [task for task, _ in sorted(score.items(), key=lambda item: (-item[1], item[0]))[:count]]
    return destroy_tasks_to_partial(solution, removed, index, instance.uav_count)


def charging_congestion_removal(instance, solution, index, evaluation, rng, count) -> PartialSolution:
    del rng
    score: dict[int, float] = {}
    for operation in evaluation.charging_operations:
        sortie = index.sortie_by_id[operation.after_sortie_id]
        contribution = operation.waiting_min
        for task_id in sortie.task_sequence:
            score[task_id] = score.get(task_id, 0.0) + contribution
    if not score or max(score.values()) <= 1e-9:
        return worst_removal(instance, solution, index, evaluation, random.Random(0), count)
    removed = [task for task, _ in sorted(score.items(), key=lambda item: (-item[1], item[0]))[:count]]
    return destroy_tasks_to_partial(solution, removed, index, instance.uav_count)


def related_removal(instance, solution, index, evaluation, rng, count) -> PartialSolution:
    del evaluation
    task_by_id = {task.id: task for task in instance.tasks}
    seed = rng.choice(list(task_by_id))
    origin = task_by_id[seed]
    related = sorted(
        task_by_id,
        key=lambda task_id: (
            hypot(task_by_id[task_id].x_km - origin.x_km, task_by_id[task_id].y_km - origin.y_km)
            + 0.25 * abs(task_by_id[task_id].level - origin.level),
            task_id,
        ),
    )[:count]
    return destroy_tasks_to_partial(solution, related, index, instance.uav_count)


def greedy_repair(
    instance,
    partial,
    index,
    *,
    repair_profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    repair_kernel_backend: RepairKernelBackend = "python",
) -> tuple[Solution, ...]:
    return repair_candidates(
        instance,
        partial,
        index,
        strategy="greedy",
        repair_profile=repair_profile,
        deadline_expired=deadline_expired,
        repair_kernel_backend=repair_kernel_backend,
    )


def regret2_repair(
    instance,
    partial,
    index,
    *,
    repair_profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    repair_kernel_backend: RepairKernelBackend = "python",
) -> tuple[Solution, ...]:
    return repair_candidates(
        instance,
        partial,
        index,
        strategy="regret2",
        repair_profile=repair_profile,
        deadline_expired=deadline_expired,
        repair_kernel_backend=repair_kernel_backend,
    )


def augmentation_repair(
    instance,
    partial,
    index,
    *,
    repair_profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    repair_kernel_backend: RepairKernelBackend = "python",
) -> tuple[Solution, ...]:
    return repair_candidates(
        instance,
        partial,
        index,
        strategy="augmentation",
        repair_profile=repair_profile,
        deadline_expired=deadline_expired,
        repair_kernel_backend=repair_kernel_backend,
    )


def recovery_aware_repair(
    instance,
    partial,
    index,
    *,
    recovery_top_m: int = 4,
    recovery_same_quota: int = 2,
    recovery_diff_quota: int = 2,
    repair_profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    r4_kernel_backend: R4KernelBackend = "auto",
) -> tuple[Solution, ...]:
    return repair_candidates(
        instance, partial, index, strategy="recovery_aware",
        recovery_top_m=recovery_top_m,
        recovery_same_quota=recovery_same_quota,
        recovery_diff_quota=recovery_diff_quota,
        repair_profile=repair_profile,
        deadline_expired=deadline_expired,
        r4_kernel_backend=r4_kernel_backend,
    )


def relocate_sortie(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    result = solution.copy()
    sources = [uav for uav, sequence in result.uav_sequences.items() if sequence]
    if not sources or instance.uav_count < 2:
        return None
    source = rng.choice(sources)
    target = rng.choice([uav for uav in range(instance.uav_count) if uav != source])
    position = rng.randrange(len(result.uav_sequences[source]))
    sortie_id = result.uav_sequences[source].pop(position)
    insertion = rng.randrange(len(result.uav_sequences.setdefault(target, [])) + 1)
    result.uav_sequences[target].insert(insertion, sortie_id)
    return result


def task_relocate(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    """Remove one task from its current sortie and generically reinsert it."""
    task_id = rng.choice(_all_tasks(instance))
    partial = destroy_tasks_to_partial(solution, (task_id,), index, instance.uav_count)
    return repair_solution(
        instance,
        partial,
        index,
        strategy="greedy",
        deadline_expired=deadline_expired,
    )


def task_swap(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    """Remove two tasks and use a generic regret repair to exchange grouping."""
    tasks = _all_tasks(instance)
    if len(tasks) < 2:
        return None
    selected = rng.sample(tasks, 2)
    partial = destroy_tasks_to_partial(solution, selected, index, instance.uav_count)
    return repair_solution(
        instance,
        partial,
        index,
        strategy="regret2",
        deadline_expired=deadline_expired,
    )


def swap_sorties(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    del index
    result = solution.copy()
    owners = [uav for uav, sequence in result.uav_sequences.items() if sequence]
    if len(owners) < 2:
        return None
    first, second = rng.sample(owners, 2)
    first_position = rng.randrange(len(result.uav_sequences[first]))
    second_position = rng.randrange(len(result.uav_sequences[second]))
    result.uav_sequences[first][first_position], result.uav_sequences[second][second_position] = (
        result.uav_sequences[second][second_position],
        result.uav_sequences[first][first_position],
    )
    return result


def split_sortie(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    slots = [
        (uav_id, position, identifier)
        for uav_id, sequence in solution.uav_sequences.items()
        for position, identifier in enumerate(sequence)
        if len(index.sortie_by_id[identifier].task_sequence) > 1
    ]
    rng.shuffle(slots)
    for uav_id, position, selected in slots:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        replacements = [
            min(
                (
                    index.sortie_by_id[identifier]
                    for identifier in index.sorties_by_task[task_id]
                    if len(index.sortie_by_id[identifier].task_sequence) == 1
                ),
                key=lambda sortie: (sortie.nominal_duration_min, sortie.id),
            ).id
            for task_id in index.sortie_by_id[selected].task_sequence
        ]
        result = solution.copy()
        result.uav_sequences[uav_id][position:position + 1] = replacements
        if sequence_preserves_route_precedence(result.uav_sequences[uav_id], instance, index):
            return result
    return None


def merge_sorties(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    owners = list(range(instance.uav_count))
    rng.shuffle(owners)
    for uav_id in owners:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        sequence = solution.uav_sequences.get(uav_id, [])
        for first_position in range(len(sequence) - 1):
            second_position = first_position + 1
            first = index.sortie_by_id[sequence[first_position]]
            second = index.sortie_by_id[sequence[second_position]]
            tasks = frozenset((*first.task_sequence, *second.task_sequence))
            if len(tasks) != len(first.task_sequence) + len(second.task_sequence):
                continue
            candidates = [
                index.sortie_by_id[identifier]
                for identifier in index.sorties_by_task_set.get(tasks, ())
            ]
            if candidates:
                merged = min(candidates, key=lambda sortie: (sortie.nominal_duration_min, sortie.id))
                result = solution.copy()
                result.uav_sequences[uav_id][first_position:second_position + 1] = [merged.id]
                if sequence_preserves_route_precedence(result.uav_sequences[uav_id], instance, index):
                    return result
    return None


def split_merge(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    if rng.random() < 0.5:
        return split_sortie(
            instance, solution, index, rng, deadline_expired=deadline_expired
        ) or merge_sorties(
            instance, solution, index, rng, deadline_expired=deadline_expired
        )
    return merge_sorties(
        instance, solution, index, rng, deadline_expired=deadline_expired
    ) or split_sortie(
        instance, solution, index, rng, deadline_expired=deadline_expired
    )


GENERIC_NEIGHBOURS = (
    task_relocate,
    task_swap,
    relocate_sortie,
    swap_sorties,
    split_sortie,
    merge_sorties,
)


def generic_random_neighbor(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution:
    """Sample only mechanism-neutral neighborhoods for fair baselines."""
    functions = list(GENERIC_NEIGHBOURS)
    rng.shuffle(functions)
    for function in functions:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        candidate = function(
            instance,
            solution,
            index,
            rng,
            deadline_expired=deadline_expired,
        )
        if candidate is not None:
            return candidate
    task_id = rng.choice(_all_tasks(instance))
    partial = destroy_tasks_to_partial(solution, (task_id,), index, instance.uav_count)
    return repair_solution(
        instance,
        partial,
        index,
        strategy="greedy",
        deadline_expired=deadline_expired,
    )


def recovery_shift(
    instance, solution, index, rng, *, deadline_expired=None
) -> Solution | None:
    slots = [
        (uav_id, position, selected)
        for uav_id, sequence in solution.uav_sequences.items()
        for position, selected in enumerate(sequence)
    ]
    rng.shuffle(slots)
    for uav_id, position, selected in slots:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        sortie = index.sortie_by_id[selected]
        alternatives = [
            index.sortie_by_id[identifier]
            for identifier in index.sorties_by_origin_sequence.get(
                (sortie.origin_support, sortie.task_sequence),
                (),
            )
            if identifier != selected
            and index.sortie_by_id[identifier].recovery_support != sortie.recovery_support
        ]
        if alternatives:
            for replacement in sorted(alternatives, key=lambda candidate: (candidate.nominal_duration_min, candidate.id)):
                result = solution.copy()
                result.uav_sequences[uav_id][position] = replacement.id
                if sequence_preserves_route_precedence(result.uav_sequences[uav_id], instance, index):
                    return result
    return None


DESTROY_OPERATORS = {
    "D1_random": DestroyOperator("D1_random", random_removal),
    "D2_worst": DestroyOperator("D2_worst", worst_removal),
    "D3_synchronization": DestroyOperator("D3_synchronization", synchronization_critical_removal),
    "D4_charging": DestroyOperator("D4_charging", charging_congestion_removal),
    "D5_related": DestroyOperator("D5_related", related_removal),
    "D6_route_corridor": DestroyOperator("D6_route_corridor", D6_route_corridor),
}

REPAIR_OPERATORS = {
    "R1_greedy": RepairOperator("R1_greedy", greedy_repair),
    "R2_regret2": RepairOperator("R2_regret2", regret2_repair),
    "R3_augmentation": RepairOperator("R3_augmentation", augmentation_repair),
    "R4_recovery_aware": RepairOperator("R4_recovery_aware", recovery_aware_repair),
    "R4P_progressive_recovery": RepairOperator(
        "R4P_progressive_recovery", R4P_progressive_recovery
    ),
}

LOCAL_OPERATORS = {
    "L1_relocate": LocalOperator("L1_relocate", relocate_sortie),
    "L2_swap": LocalOperator("L2_swap", swap_sorties),
    "L3_split_merge": LocalOperator("L3_split_merge", split_merge),
    "L4_recovery_shift": LocalOperator("L4_recovery_shift", recovery_shift),
}
