from __future__ import annotations

from collections.abc import Sequence
from math import inf
from typing import Mapping

from usv_uav.algorithms.route_guided.context import RouteSearchContext
from usv_uav.algorithms.route_guided.models import (
    BlockingAnalysis,
    DwellGuidedAnalysis,
    ForwardFeasibility,
    PhysicalForwardCandidate,
)
from usv_uav.core.models import Sortie
from usv_uav.preprocessing.sortie_index import SortieIndex
from usv_uav.scheduling.evaluator import EvaluationResult, SupportDwellSummary


RouteScore = tuple[float, float, float, float, float]
RouteAffinityGain = tuple[float, float, float]


def support_dwell_after_release(
    summary: SupportDwellSummary,
    released_sortie_ids: set[int] | frozenset[int],
) -> float:
    """Recompute one support's dwell from the derived event profile."""
    latest = max(
        (
            event.completion_time_min
            for event in summary.event_contributors
            if event.sortie_id is None or event.sortie_id not in released_sortie_ids
        ),
        default=summary.arrival_time_min,
    )
    return max(0.0, latest - summary.arrival_time_min)


def critical_release_group(
    summary: SupportDwellSummary,
    released_sortie_ids: set[int] | frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], float]:
    """Return all tied critical whole sorties and their joint dwell gain."""
    remaining = tuple(
        event
        for event in summary.event_contributors
        if event.sortie_id is None or event.sortie_id not in released_sortie_ids
    )
    if not remaining:
        return (), 0.0
    latest = max(event.completion_time_min for event in remaining)
    critical = tuple(
        sorted(
            {
                event.sortie_id
                for event in remaining
                if event.sortie_id is not None
                and abs(event.completion_time_min - latest) <= 1e-9
            }
        )
    )
    if not critical:
        return (), 0.0
    current = support_dwell_after_release(summary, released_sortie_ids)
    residual = support_dwell_after_release(
        summary, set(released_sortie_ids) | set(critical)
    )
    return critical, max(0.0, current - residual)


def analyze_dwell_guided_target(
    *,
    target: PhysicalForwardCandidate,
    critical_support_position: int,
    evaluation: EvaluationResult,
    current_sortie_by_task: Mapping[int, int],
    context: RouteSearchContext,
    release_task_budget: int,
) -> DwellGuidedAnalysis:
    """Estimate the minimum critical-dwell release for a V3.1 target."""
    if release_task_budget <= 0:
        raise ValueError("release task budget must be positive")
    if not target.statically_feasible:
        return DwellGuidedAnalysis(
            target_sortie_id=target.sortie_id,
            critical_support_position=critical_support_position,
            critical_support_id=context.support_order[critical_support_position],
            intermediate_dwell_budget_min=target.intermediate_dwell_budget_min,
            current_intermediate_dwell_min=inf,
            residual_intermediate_dwell_min=inf,
            required_dwell_release_min=inf,
            owner_sortie_ids=(),
            blocking_sortie_ids=(),
            required_removed_task_ids=(),
            released_dwell_gain_min=0.0,
            new_wait_min=inf,
            estimated_dwell_gain_min=-inf,
            dynamic_slack_min=-inf,
            feasibility=ForwardFeasibility.STATIC_IMPOSSIBLE,
        )

    target_sortie = context.sortie_index.sortie_by_id[target.sortie_id]
    owner_sorties = {
        current_sortie_by_task[task_id]
        for task_id in target_sortie.task_sequence
        if task_id in current_sortie_by_task
    }
    if not owner_sorties:
        raise ValueError("guided target tasks are absent from the current solution")
    summaries = {
        context.support_position[summary.support_id]: summary
        for summary in evaluation.support_dwell_summaries
    }
    selected = set(owner_sorties)
    critical_summary = summaries.get(critical_support_position)
    if critical_summary is not None:
        owner_gain = (
            critical_summary.dwell_time_min
            - support_dwell_after_release(critical_summary, selected)
        )
        if owner_gain <= 1e-9:
            tied_group, group_gain = critical_release_group(
                critical_summary, selected
            )
            if group_gain > 1e-9:
                selected.update(tied_group)

    intermediate_positions = tuple(
        range(target.origin_pos + 1, target.recovery_pos)
    )
    current_intermediate = sum(
        summaries[position].dwell_time_min
        for position in intermediate_positions
        if position in summaries
    )

    def residual_intermediate(released: set[int]) -> float:
        return sum(
            support_dwell_after_release(summaries[position], released)
            for position in intermediate_positions
            if position in summaries
        )

    residual = residual_intermediate(selected)
    required_release = max(
        0.0, current_intermediate - target.intermediate_dwell_budget_min
    )
    while residual > target.intermediate_dwell_budget_min + 1e-9:
        groups = []
        for position in intermediate_positions:
            summary = summaries.get(position)
            if summary is None:
                continue
            group, gain = critical_release_group(summary, selected)
            if group and gain > 1e-9:
                groups.append((-gain, len(group), position, group))
        if not groups:
            break
        _, _, _, group = min(groups)
        selected.update(group)
        residual = residual_intermediate(selected)

    required_tasks = tuple(
        sorted(
            {
                task_id
                for sortie_id in selected
                for task_id in context.sortie_index.sortie_by_id[
                    sortie_id
                ].task_sequence
            }
        )
    )
    released_gain = sum(
        summary.dwell_time_min - support_dwell_after_release(summary, selected)
        for position, summary in summaries.items()
        if target.origin_pos <= position <= target.recovery_pos
    )
    new_wait = max(
        0.0,
        target_sortie.nominal_duration_min
        - (target.movement_time_min + residual),
    )
    budget_respected = len(required_tasks) <= release_task_budget
    release_sufficient = residual <= target.intermediate_dwell_budget_min + 1e-9
    estimated_gain = released_gain - new_wait
    if not budget_respected or not release_sufficient:
        estimated_gain = -inf
    feasibility = (
        ForwardFeasibility.CURRENTLY_FEASIBLE
        if current_intermediate <= target.intermediate_dwell_budget_min + 1e-9
        else ForwardFeasibility.DYNAMIC_BLOCKED
    )
    return DwellGuidedAnalysis(
        target_sortie_id=target.sortie_id,
        critical_support_position=critical_support_position,
        critical_support_id=context.support_order[critical_support_position],
        intermediate_dwell_budget_min=target.intermediate_dwell_budget_min,
        current_intermediate_dwell_min=current_intermediate,
        residual_intermediate_dwell_min=residual,
        required_dwell_release_min=required_release,
        owner_sortie_ids=tuple(sorted(owner_sorties)),
        blocking_sortie_ids=tuple(sorted(selected - owner_sorties)),
        required_removed_task_ids=required_tasks,
        released_dwell_gain_min=released_gain,
        new_wait_min=new_wait,
        estimated_dwell_gain_min=estimated_gain,
        dynamic_slack_min=target.intermediate_dwell_budget_min - residual,
        feasibility=feasibility,
    )


def synchronization_candidate_key(
    sortie: Sortie,
    index: SortieIndex,
    route_sail_prefix: Sequence[float],
) -> RouteScore:
    """The frozen S5A1 lexicographic score, exposed as a shared primitive."""
    origin = index.support_position[sortie.origin_support]
    recovery = index.support_position[sortie.recovery_support]
    if recovery < origin:
        raise ValueError(f"backward sortie {sortie.id} has negative recovery span")
    usv_sail_min = route_sail_prefix[recovery] - route_sail_prefix[origin]
    nominal_min = sortie.nominal_duration_min
    return (
        max(0.0, nominal_min - usv_sail_min),
        max(0.0, usv_sail_min - nominal_min),
        sortie.flight_time_min,
        sortie.nominal_energy_wh,
        float(sortie.id),
    )


def sortie_route_score(sortie_id: int, context: RouteSearchContext) -> RouteScore:
    return synchronization_candidate_key(
        context.sortie_index.sortie_by_id[sortie_id],
        context.sortie_index,
        context.route_sail_prefix,
    )


def best_task_score_at_origin(
    task_id: int,
    origin_position: int,
    context: RouteSearchContext,
) -> RouteScore | None:
    sortie_ids = context.sortie_index.by_task_origin(task_id, origin_position)
    if not sortie_ids:
        return None
    return min(sortie_route_score(sortie_id, context) for sortie_id in sortie_ids)


def task_route_affinity_gain(
    task_id: int,
    current_origin_position: int,
    context: RouteSearchContext,
) -> RouteAffinityGain:
    """Return the lexicographic gain from moving a task one or two segments ahead."""
    current = best_task_score_at_origin(task_id, current_origin_position, context)
    if current is None:
        raise ValueError(
            f"task {task_id} has no sortie at current origin {current_origin_position}"
        )
    forward_scores = tuple(
        score
        for position in (current_origin_position + 1, current_origin_position + 2)
        if position < len(context.support_order)
        for score in (best_task_score_at_origin(task_id, position, context),)
        if score is not None
    )
    if not forward_scores:
        return (float("-inf"), float("-inf"), float("-inf"))
    return (
        current[0] - min(score[0] for score in forward_scores),
        current[1] - min(score[1] for score in forward_scores),
        current[2] - min(score[2] for score in forward_scores),
    )


def affinity_descending_key(
    task_id: int,
    gain: RouteAffinityGain,
) -> tuple[float, float, float, int]:
    return (-gain[0], -gain[1], -gain[2], task_id)


def analyze_corridor_blockers(
    *,
    target: PhysicalForwardCandidate,
    evaluation: EvaluationResult,
    current_sortie_by_task: Mapping[int, int],
    context: RouteSearchContext,
) -> BlockingAnalysis:
    """Estimate the minimum current-sortie release needed by one target.

    The authoritative decoder remains unchanged.  This analysis only reads its
    arrival/departure and launch/recovery events.  At each support it attributes
    the observed departure barrier to the current sorties whose events occur
    after USV arrival, then greedily removes the sortie with the largest
    marginal reduction until the target's physical hover slack is respected.
    """
    if not target.statically_feasible:
        return BlockingAnalysis(
            target_sortie_id=target.sortie_id,
            physical_slack_min=target.physical_slack_min,
            blocking_support_positions=(),
            blocking_sortie_ids=(),
            estimated_blocking_time_min=inf,
            residual_blocking_time_min=inf,
            required_removed_task_ids=(),
            feasibility=ForwardFeasibility.STATIC_IMPOSSIBLE,
        )

    target_sortie = context.sortie_index.sortie_by_id[target.sortie_id]
    owner_sorties = {
        current_sortie_by_task[task_id]
        for task_id in target_sortie.task_sequence
        if task_id in current_sortie_by_task
    }
    if len(owner_sorties) == 0:
        raise ValueError("guided target tasks are absent from the current solution")

    dwell_by_support = {
        summary.support_id: summary
        for summary in evaluation.support_dwell_summaries
    }
    execution_by_sortie = {
        execution.sortie_id: execution for execution in evaluation.executions
    }

    # Per support: current sortie -> the barrier it creates relative to arrival.
    delays_by_support: dict[int, dict[int, float]] = {}
    # Origin dwell is absorbed by the V3 synchronized launch rule.  Only dwell
    # after the UAV has actually launched can create additional hover.
    for position in range(target.origin_pos + 1, target.recovery_pos):
        support_id = context.support_order[position]
        dwell = dwell_by_support.get(support_id)
        if dwell is None or dwell.dwell_time_min <= 1e-9:
            continue
        arrival = dwell.arrival_time_min
        departure = dwell.departure_time_min
        sortie_delays: dict[int, float] = {}
        for sortie_id, execution in execution_by_sortie.items():
            if sortie_id in owner_sorties:
                continue
            sortie = context.sortie_index.sortie_by_id[sortie_id]
            barriers: list[float] = []
            if (
                sortie.recovery_support == support_id
                and execution.recovery_time_min > arrival + 1e-9
                and execution.recovery_time_min <= departure + 1e-9
            ):
                barriers.append(execution.recovery_time_min - arrival)
            if (
                sortie.origin_support == support_id
                and sortie.recovery_support != support_id
                and execution.ready_for_launch_time_min > arrival + 1e-9
                and execution.ready_for_launch_time_min <= departure + 1e-9
            ):
                # Use physical readiness rather than the actual synchronized
                # launch time.  Deliberate deck waiting consumes an existing
                # dwell window and is not itself a blocker.
                barriers.append(execution.ready_for_launch_time_min - arrival)
            if barriers:
                sortie_delays[sortie_id] = max(barriers)
        if sortie_delays:
            delays_by_support[position] = sortie_delays

    def residual(selected: set[int]) -> float:
        return sum(
            max(
                (
                    delay
                    for sortie_id, delay in sortie_delays.items()
                    if sortie_id not in selected
                ),
                default=0.0,
            )
            for sortie_delays in delays_by_support.values()
        )

    selected_blockers: set[int] = set()
    estimated_blocking = residual(selected_blockers)
    residual_blocking = estimated_blocking
    all_blockers = {
        sortie_id
        for sortie_delays in delays_by_support.values()
        for sortie_id in sortie_delays
    }
    while (
        residual_blocking > target.physical_slack_min + 1e-9
        and all_blockers - selected_blockers
    ):
        blocker = min(
            all_blockers - selected_blockers,
            key=lambda sortie_id: (
                -(
                    residual_blocking
                    - residual(selected_blockers | {sortie_id})
                ),
                sortie_id,
            ),
        )
        new_residual = residual(selected_blockers | {blocker})
        if new_residual >= residual_blocking - 1e-9:
            break
        selected_blockers.add(blocker)
        residual_blocking = new_residual

    released_sorties = owner_sorties | selected_blockers
    required_tasks = tuple(
        sorted(
            {
                task_id
                for sortie_id in released_sorties
                for task_id in context.sortie_index.sortie_by_id[
                    sortie_id
                ].task_sequence
            }
        )
    )
    blocking_positions = tuple(
        position
        for position, sortie_delays in sorted(delays_by_support.items())
        if set(sortie_delays) & selected_blockers
    )
    feasibility = (
        ForwardFeasibility.CURRENTLY_FEASIBLE
        if estimated_blocking <= target.physical_slack_min + 1e-9
        else ForwardFeasibility.DYNAMIC_BLOCKED
    )
    return BlockingAnalysis(
        target_sortie_id=target.sortie_id,
        physical_slack_min=target.physical_slack_min,
        blocking_support_positions=blocking_positions,
        blocking_sortie_ids=tuple(sorted(selected_blockers)),
        estimated_blocking_time_min=estimated_blocking,
        residual_blocking_time_min=residual_blocking,
        required_removed_task_ids=required_tasks,
        feasibility=feasibility,
    )
