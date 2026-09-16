from __future__ import annotations

import random
from typing import Any, Mapping

from usv_uav.algorithms.construction import destroy_tasks_to_partial
from usv_uav.algorithms.route_guided.context import RouteSearchContext
from usv_uav.algorithms.route_guided.controller import RouteSearchController
from usv_uav.algorithms.route_guided.diagnostics import RouteSearchDiagnostics
from usv_uav.algorithms.route_guided.models import (
    CorridorRepairContext,
    DestroyResult,
    DwellGuidedAnalysis,
    ForwardFeasibility,
    GuidedRepairTarget,
    RepairPolicy,
    RouteCorridor,
)
from usv_uav.algorithms.route_guided.scoped_index import ScopedSortieIndexView
from usv_uav.algorithms.route_guided.scoring import (
    RouteAffinityGain,
    analyze_corridor_blockers,
    analyze_dwell_guided_target,
    affinity_descending_key,
    sortie_route_score,
    task_route_affinity_gain,
)
from usv_uav.core.models import Instance
from usv_uav.core.solution import Solution


DYNAMIC_DWELL_TARGET_LIMIT = 12


def select_critical_support(
    evaluation: Any,
    route_context: RouteSearchContext,
    rng: random.Random,
) -> int | None:
    """Weighted-random selection from the three largest marginal dwell gains."""
    candidates = tuple(
        sorted(
            (
                (
                    summary.joint_critical_dwell_gain_min,
                    route_context.support_position[summary.support_id],
                )
                for summary in evaluation.support_dwell_summaries
                if summary.joint_critical_dwell_gain_min > 1e-9
                and summary.critical_sortie_ids
            ),
            key=lambda item: (-item[0], item[1]),
        )[:3]
    )
    if not candidates:
        return None
    total = sum(gain for gain, _ in candidates)
    draw = rng.random() * total
    cursor = 0.0
    for gain, position in candidates:
        cursor += gain
        if draw <= cursor:
            return position
    return candidates[-1][1]


def task_origins_in_solution(
    solution: Solution,
    route_context: RouteSearchContext,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return task-to-origin and task-to-sortie maps with exact-cover checks."""
    origin_by_task: dict[int, int] = {}
    sortie_by_task: dict[int, int] = {}
    for sequence in solution.uav_sequences.values():
        for sortie_id in sequence:
            origin = route_context.sortie_origin_position[sortie_id]
            for task_id in route_context.sortie_index.sortie_by_id[
                sortie_id
            ].task_sequence:
                if task_id in origin_by_task:
                    raise RuntimeError(f"task {task_id} occurs in multiple solution sorties")
                origin_by_task[task_id] = origin
                sortie_by_task[task_id] = sortie_id
    return origin_by_task, sortie_by_task


def _destroy_result(
    *,
    instance: Instance,
    solution: Solution,
    route_context: RouteSearchContext,
    removed_task_ids,
    corridor=None,
) -> DestroyResult:
    previous, _ = task_origins_in_solution(solution, route_context)
    partial = destroy_tasks_to_partial(
        solution,
        removed_task_ids,
        route_context.sortie_index,
        instance.uav_count,
    )
    actual_removed = tuple(sorted(partial.missing_tasks))
    return DestroyResult(
        partial=partial,
        removed_task_ids=actual_removed,
        previous_origin_positions=tuple(
            (task_id, previous[task_id]) for task_id in actual_removed
        ),
        corridor=corridor,
    )


def D6_route_corridor(
    *,
    instance: Instance,
    solution: Solution,
    evaluation: Any,
    route_context: RouteSearchContext,
    policy: RepairPolicy,
    controller: RouteSearchController,
    rng: random.Random,
    removal_count: int,
    diagnostics: RouteSearchDiagnostics | None = None,
    corridor_bundle_destroy: bool = False,
    forward_partner_required: bool = False,
    physical_feasibility_aware: bool = False,
    guided_candidate_limit: int = 64,
    transactional_b2: bool = False,
    target_transformation_required: bool = False,
    dwell_guided: bool = False,
) -> DestroyResult:
    """Remove whole sorties around the highest route-affinity task."""
    if removal_count <= 0:
        raise ValueError("removal_count must be positive")
    previous, current_sortie = task_origins_in_solution(solution, route_context)
    if not previous:
        raise RuntimeError("route-corridor destroy received an empty solution")
    gains: dict[int, RouteAffinityGain] = {
        task_id: task_route_affinity_gain(task_id, origin, route_context)
        for task_id, origin in previous.items()
    }
    ranked = tuple(
        sorted(previous, key=lambda task_id: affinity_descending_key(task_id, gains[task_id]))
    )
    if dwell_guided:
        guided = _dwell_guided_target_destroy(
            instance=instance,
            solution=solution,
            evaluation=evaluation,
            route_context=route_context,
            controller=controller,
            gains=gains,
            previous=previous,
            current_sortie=current_sortie,
            diagnostics=diagnostics,
            rng=rng,
            release_task_budget=removal_count,
        )
        if guided is not None:
            return guided
        if target_transformation_required:
            return _global_whole_sortie_destroy(
                instance=instance,
                solution=solution,
                route_context=route_context,
                ranked_tasks=ranked,
                gains=gains,
                current_sortie=current_sortie,
                removal_count=removal_count,
                diagnostics=diagnostics,
            )
    if physical_feasibility_aware and not dwell_guided:
        guided = _physical_target_destroy(
            instance=instance,
            solution=solution,
            evaluation=evaluation,
            route_context=route_context,
            policy=policy,
            controller=controller,
            gains=gains,
            previous=previous,
            current_sortie=current_sortie,
            diagnostics=diagnostics,
            guided_candidate_limit=guided_candidate_limit,
            transactional_b2=transactional_b2,
        )
        if guided is not None:
            return guided
        if target_transformation_required:
            return _global_whole_sortie_destroy(
                instance=instance,
                solution=solution,
                route_context=route_context,
                ranked_tasks=ranked,
                gains=gains,
                current_sortie=current_sortie,
                removal_count=removal_count,
                diagnostics=diagnostics,
            )
    if corridor_bundle_destroy:
        return _corridor_bundle_destroy(
            instance=instance,
            solution=solution,
            route_context=route_context,
            policy=policy,
            ranked_tasks=ranked,
            gains=gains,
            previous=previous,
            current_sortie=current_sortie,
            removal_count=removal_count,
            diagnostics=diagnostics,
            forward_partner_required=forward_partner_required,
            physical_feasibility_aware=physical_feasibility_aware,
        )
    anchor = ranked[0]
    corridor = controller.corridor_for(
        previous[anchor], len(route_context.support_order), policy
    )
    inside = tuple(
        task_id for task_id in ranked if corridor.contains(previous[task_id])
    )
    outside = tuple(task_id for task_id in ranked if task_id not in set(inside))
    ordered = (*inside, *outside)

    selected_sorties: set[int] = set()
    selected_tasks: list[int] = []
    selected_task_set: set[int] = set()
    selected_gains: list[RouteAffinityGain] = []
    for task_id in ordered:
        sortie_id = current_sortie[task_id]
        if sortie_id in selected_sorties:
            continue
        selected_sorties.add(sortie_id)
        sortie_tasks = route_context.sortie_index.sortie_by_id[sortie_id].task_sequence
        for sortie_task in sortie_tasks:
            if sortie_task not in selected_task_set:
                selected_tasks.append(sortie_task)
                selected_task_set.add(sortie_task)
                selected_gains.append(gains[sortie_task])
        if len(selected_task_set) >= min(removal_count, len(previous)):
            break
    if not selected_tasks:
        raise RuntimeError("route-corridor destroy could not select a solution sortie")
    result = _destroy_result(
        instance=instance,
        solution=solution,
        route_context=route_context,
        removed_task_ids=selected_tasks,
        corridor=corridor,
    )
    if diagnostics is not None:
        diagnostics.record_d6(tuple(selected_gains), corridor.width)
    return result


def _dwell_guided_target_destroy(
    *,
    instance: Instance,
    solution: Solution,
    evaluation: Any,
    route_context: RouteSearchContext,
    controller: RouteSearchController,
    gains: Mapping[int, RouteAffinityGain],
    previous: Mapping[int, int],
    current_sortie: Mapping[int, int],
    diagnostics: RouteSearchDiagnostics | None,
    rng: random.Random,
    release_task_budget: int,
) -> DestroyResult | None:
    critical_position = select_critical_support(evaluation, route_context, rng)
    if critical_position is None:
        if diagnostics is not None:
            diagnostics.record_dwell_guided_screen(
                static_pool_size=0,
                dynamically_screened=0,
                positive_gain_targets=0,
            )
        return None
    support_id = route_context.support_order[critical_position]
    summary = next(
        value
        for value in evaluation.support_dwell_summaries
        if value.support_id == support_id
    )
    critical_tasks = tuple(
        sorted(
            {
                task_id
                for sortie_id in summary.critical_sortie_ids
                for task_id in route_context.sortie_index.sortie_by_id[
                    sortie_id
                ].task_sequence
            }
        )
    )
    target_ids, static_pool_size = route_context.dwell_guided_targets(
        critical_position,
        critical_tasks,
        limit=DYNAMIC_DWELL_TARGET_LIMIT,
    )
    scheduled_sorties = {
        sortie_id
        for sequence in solution.uav_sequences.values()
        for sortie_id in sequence
    }
    analyses: list[DwellGuidedAnalysis] = []
    dynamically_screened = 0
    for sortie_id in target_ids:
        if sortie_id in scheduled_sorties:
            continue
        sortie = route_context.sortie_index.sortie_by_id[sortie_id]
        if not set(sortie.task_sequence) <= set(previous):
            continue
        dynamically_screened += 1
        analysis = analyze_dwell_guided_target(
            target=route_context.physical_profile_by_sortie[sortie_id],
            critical_support_position=critical_position,
            evaluation=evaluation,
            current_sortie_by_task=current_sortie,
            context=route_context,
            release_task_budget=release_task_budget,
        )
        if analysis.release_sufficient and analysis.positive_gain:
            analyses.append(analysis)
    if diagnostics is not None:
        diagnostics.record_dwell_guided_screen(
            static_pool_size=static_pool_size,
            dynamically_screened=dynamically_screened,
            positive_gain_targets=len(analyses),
        )
    if not analyses:
        return None
    target_analysis = min(analyses, key=lambda analysis: analysis.ranking_key)
    target_profile = route_context.physical_profile_by_sortie[
        target_analysis.target_sortie_id
    ]
    target_sortie = route_context.sortie_index.sortie_by_id[
        target_analysis.target_sortie_id
    ]
    target_tasks = tuple(target_sortie.task_sequence)
    corridor = RouteCorridor(target_profile.origin_pos, target_profile.recovery_pos)
    result = _destroy_result(
        instance=instance,
        solution=solution,
        route_context=route_context,
        removed_task_ids=target_analysis.required_removed_task_ids,
        corridor=corridor,
    )
    release_scale = controller.release_scale(len(result.removed_task_ids))
    guided_target = GuidedRepairTarget(
        target_sortie_id=target_analysis.target_sortie_id,
        required_missing_task_ids=target_tasks,
        corridor=corridor,
        owner_sortie_ids=target_analysis.owner_sortie_ids,
        blocker_sortie_ids=target_analysis.blocking_sortie_ids,
        release_scale=release_scale,
        critical_support_position=critical_position,
        estimated_dwell_gain_min=target_analysis.estimated_dwell_gain_min,
        release_task_budget=release_task_budget,
        strict_no_undo=True,
    )
    repair_context = CorridorRepairContext(
        corridor=corridor,
        previous_origin_by_task=result.previous_origin_by_task,
        anchor_task=target_tasks[0],
        compatible_partner_tasks=target_tasks[1:],
        guided_target=guided_target,
    )
    result = DestroyResult(
        partial=result.partial,
        removed_task_ids=result.removed_task_ids,
        previous_origin_positions=result.previous_origin_positions,
        corridor=result.corridor,
        repair_context=repair_context,
    )
    if diagnostics is not None:
        diagnostics.record_d6(
            tuple(gains[task_id] for task_id in result.removed_task_ids),
            corridor.width,
        )
        diagnostics.record_dwell_guided_target(target_analysis)
        diagnostics.record_b2_target_selected(target_task_count=len(target_tasks))
    return result


def _global_whole_sortie_destroy(
    *,
    instance: Instance,
    solution: Solution,
    route_context: RouteSearchContext,
    ranked_tasks: tuple[int, ...],
    gains: Mapping[int, RouteAffinityGain],
    current_sortie: Mapping[int, int],
    removal_count: int,
    diagnostics: RouteSearchDiagnostics | None,
) -> DestroyResult:
    """Use a full-domain V3 fallback when no physical target exists."""
    selected_sorties: set[int] = set()
    selected_tasks: list[int] = []
    selected_task_set: set[int] = set()
    selected_gains: list[RouteAffinityGain] = []
    for task_id in ranked_tasks:
        sortie_id = current_sortie[task_id]
        if sortie_id in selected_sorties:
            continue
        selected_sorties.add(sortie_id)
        for sortie_task in route_context.sortie_index.sortie_by_id[
            sortie_id
        ].task_sequence:
            if sortie_task in selected_task_set:
                continue
            selected_tasks.append(sortie_task)
            selected_task_set.add(sortie_task)
            selected_gains.append(gains[sortie_task])
        if len(selected_task_set) >= min(removal_count, len(current_sortie)):
            break
    if not selected_tasks:
        raise RuntimeError("V3 global destroy fallback could not select a solution sortie")
    corridor = RouteCorridor(0, len(route_context.support_order) - 1)
    result = _destroy_result(
        instance=instance,
        solution=solution,
        route_context=route_context,
        removed_task_ids=selected_tasks,
        corridor=corridor,
    )
    if diagnostics is not None:
        diagnostics.record_d6(tuple(selected_gains), corridor.width)
        diagnostics.record_d6_targetless_global_fallback()
    return result


def _physical_target_destroy(
    *,
    instance: Instance,
    solution: Solution,
    evaluation: Any,
    route_context: RouteSearchContext,
    policy: RepairPolicy,
    controller: RouteSearchController,
    gains: Mapping[int, RouteAffinityGain],
    previous: Mapping[int, int],
    current_sortie: Mapping[int, int],
    diagnostics: RouteSearchDiagnostics | None,
    guided_candidate_limit: int,
    transactional_b2: bool,
) -> DestroyResult | None:
    """Select one physical forward transformation and release its true blockers."""
    if guided_candidate_limit <= 0:
        raise ValueError("guided candidate limit must be positive")
    scheduled_sorties = {
        sortie_id
        for sequence in solution.uav_sequences.values()
        for sortie_id in sequence
    }
    analyses = []
    considered = 0
    for sortie_id in route_context.physical_forward_sortie_ids:
        if sortie_id in scheduled_sorties:
            continue
        sortie = route_context.sortie_index.sortie_by_id[sortie_id]
        if not set(sortie.task_sequence) <= set(previous):
            continue
        profile = route_context.physical_profile_by_sortie[sortie_id]
        if (
            transactional_b2
            and profile.lower_bound_duration_min
            > sortie.nominal_duration_min + 1e-9
        ):
            # Minimum-required charging has no speculative hover reserve.
            # Keep such physically possible sorties in the global domain, but
            # do not select them as the structure promised by a B2 transaction.
            if diagnostics is not None:
                diagnostics.record_b2_minimum_charge_unsafe_target()
            continue
        # V3 targets the structure to create, then releases every current
        # whole sortie occupying one of its tasks.  It is deliberately not
        # restricted to the former span-2/C3 absorption pattern.
        considered += 1
        analysis = analyze_corridor_blockers(
            target=profile,
            evaluation=evaluation,
            current_sortie_by_task=current_sortie,
            context=route_context,
        )
        if diagnostics is not None:
            diagnostics.record_forward_analysis(analysis)
        if analysis.release_sufficient:
            analyses.append(analysis)
        if considered >= guided_candidate_limit:
            break
    if not analyses:
        return None

    target_analysis = min(
        analyses,
        key=lambda analysis: (
            analysis.feasibility is not ForwardFeasibility.DYNAMIC_BLOCKED,
            -analysis.physical_slack_min,
            len(analysis.required_removed_task_ids),
            sortie_route_score(analysis.target_sortie_id, route_context),
        ),
    )
    target_profile = route_context.physical_profile_by_sortie[
        target_analysis.target_sortie_id
    ]
    target_sortie = route_context.sortie_index.sortie_by_id[
        target_analysis.target_sortie_id
    ]
    target_tasks = tuple(target_sortie.task_sequence)
    owner_sorties = tuple(
        sorted({current_sortie[task_id] for task_id in target_tasks})
    )
    owner_tasks = {
        task_id
        for sortie_id in owner_sorties
        for task_id in route_context.sortie_index.sortie_by_id[
            sortie_id
        ].task_sequence
    }
    release_tasks = tuple(
        sorted(set(target_analysis.required_removed_task_ids) | owner_tasks)
    )
    corridor = RouteCorridor(
        target_profile.origin_pos,
        target_profile.recovery_pos,
    )
    result = _destroy_result(
        instance=instance,
        solution=solution,
        route_context=route_context,
        removed_task_ids=release_tasks,
        corridor=corridor,
    )
    retained_sorties = {
        sortie_id
        for sequence in result.partial.uav_sequences.values()
        for sortie_id in sequence
    }
    if set(owner_sorties) & retained_sorties:
        raise RuntimeError("destroy-target inconsistency: an owner sortie was retained")
    if not set(target_tasks) <= result.partial.missing_tasks:
        raise RuntimeError(
            "destroy-target inconsistency: target tasks were not completely released"
        )
    retained_tasks = {
        task_id
        for sortie_id in retained_sorties
        for task_id in route_context.sortie_index.sortie_by_id[
            sortie_id
        ].task_sequence
    }
    if set(target_tasks) & retained_tasks:
        raise RuntimeError(
            "destroy-target inconsistency: target tasks remain in the partial solution"
        )
    release_scale = controller.release_scale(len(result.removed_task_ids))
    guided_target = GuidedRepairTarget(
        target_sortie_id=target_analysis.target_sortie_id,
        required_missing_task_ids=target_tasks,
        corridor=corridor,
        owner_sortie_ids=owner_sorties,
        blocker_sortie_ids=target_analysis.blocking_sortie_ids,
        release_scale=release_scale,
    )
    repair_context = CorridorRepairContext(
        corridor=corridor,
        previous_origin_by_task=result.previous_origin_by_task,
        anchor_task=target_tasks[0],
        compatible_partner_tasks=target_tasks[1:],
        guided_target=guided_target,
    )
    result = DestroyResult(
        partial=result.partial,
        removed_task_ids=result.removed_task_ids,
        previous_origin_positions=result.previous_origin_positions,
        corridor=result.corridor,
        repair_context=repair_context,
    )
    if diagnostics is not None:
        diagnostics.record_d6(
            tuple(gains[task_id] for task_id in result.removed_task_ids),
            corridor.width,
        )
        diagnostics.record_guided_target(target_analysis)
        diagnostics.record_b2_target_selected(target_task_count=len(target_tasks))
    return result


def _corridor_bundle_destroy(
    *,
    instance: Instance,
    solution: Solution,
    route_context: RouteSearchContext,
    policy: RepairPolicy,
    ranked_tasks: tuple[int, ...],
    gains: Mapping[int, RouteAffinityGain],
    previous: Mapping[int, int],
    current_sortie: Mapping[int, int],
    removal_count: int,
    diagnostics: RouteSearchDiagnostics | None,
    forward_partner_required: bool,
    physical_feasibility_aware: bool = False,
) -> DestroyResult:
    """Release a pool-proven, forward-recombinable bundle of whole sorties."""
    support_count = len(route_context.support_order)
    view = ScopedSortieIndexView(
        context=route_context,
        previous_origin_positions=previous,
        policy=policy,
        physical_feasibility_aware=physical_feasibility_aware,
    )

    def forward_corridor(task_id: int) -> RouteCorridor:
        origin = previous[task_id]
        return RouteCorridor(origin, min(support_count - 1, origin + 2))

    def eligible_partners(task_id: int, corridor: RouteCorridor):
        origin = previous[task_id]
        candidates = view.compatible_forward_partners(
            task_id,
            corridor,
            anchor_origin_position=origin,
            max_span=2,
        )
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.partner_task in previous
            and previous[candidate.partner_task] in {origin + 1, origin + 2}
        )
        return tuple(
            sorted(
                eligible,
                key=lambda candidate: (
                    not any(
                        route_context.sortie_span[sortie_id] == 2
                        for sortie_id in candidate.compatible_sortie_ids
                    ),
                    candidate.best_compatible_route_score,
                    candidate.partner_task,
                ),
            )
        )

    anchor = ranked_tasks[0]
    corridor = forward_corridor(anchor)
    partners = eligible_partners(anchor, corridor)
    if forward_partner_required and not partners:
        for candidate_anchor in ranked_tasks[1:]:
            candidate_corridor = forward_corridor(candidate_anchor)
            candidate_partners = eligible_partners(
                candidate_anchor, candidate_corridor
            )
            if candidate_partners:
                anchor = candidate_anchor
                corridor = candidate_corridor
                partners = candidate_partners
                break

    selected_sorties: set[int] = set()
    selected_tasks: list[int] = []
    selected_task_set: set[int] = set()
    selected_gains: list[RouteAffinityGain] = []
    compatible_partner_tasks: list[int] = []
    partner_sorties_removed = 0

    def select_whole_sortie(sortie_id: int, *, partner_task: int | None = None) -> bool:
        nonlocal partner_sorties_removed
        if sortie_id in selected_sorties:
            return False
        selected_sorties.add(sortie_id)
        if partner_task is not None:
            partner_sorties_removed += 1
            if partner_task not in compatible_partner_tasks:
                compatible_partner_tasks.append(partner_task)
        for task_id in route_context.sortie_index.sortie_by_id[
            sortie_id
        ].task_sequence:
            if task_id not in selected_task_set:
                selected_tasks.append(task_id)
                selected_task_set.add(task_id)
                selected_gains.append(gains[task_id])
        return True

    # Stage 1: the anchor and every task in its current sortie are released.
    select_whole_sortie(current_sortie[anchor])

    # Stage 2: release current whole sorties of pool-proven future partners.
    for partner in partners:
        if len(selected_task_set) >= min(removal_count, len(previous)):
            break
        select_whole_sortie(
            current_sortie[partner.partner_task],
            partner_task=partner.partner_task,
        )

    # Stage 3: expand through any span-1/2 sortie that can recombine with Missing.
    if len(selected_task_set) < min(removal_count, len(previous)):
        related: dict[int, tuple[bool, int, int]] = {}
        for missing_task in tuple(selected_tasks):
            for sortie_id in route_context.sortie_index.by_task(missing_task):
                span = route_context.sortie_span[sortie_id]
                origin = route_context.sortie_origin_position[sortie_id]
                recovery = route_context.sortie_recovery_position[sortie_id]
                if (
                    span not in {1, 2}
                    or not corridor.contains(origin)
                    or not corridor.contains(recovery)
                ):
                    continue
                sortie = route_context.sortie_index.sortie_by_id[sortie_id]
                for task_id in sortie.task_sequence:
                    if task_id in selected_task_set or task_id not in current_sortie:
                        continue
                    key = (
                        span != 2,
                        route_context.sortie_index.sortie_rank_by_id[sortie_id],
                        task_id,
                    )
                    if task_id not in related or key < related[task_id]:
                        related[task_id] = key
        for task_id in sorted(related, key=related.__getitem__):
            if len(selected_task_set) >= min(removal_count, len(previous)):
                break
            select_whole_sortie(current_sortie[task_id], partner_task=task_id)

    # Stage 4: deterministic route-affinity fallback.
    for task_id in ranked_tasks:
        if len(selected_task_set) >= min(removal_count, len(previous)):
            break
        select_whole_sortie(current_sortie[task_id])

    if not selected_tasks:
        raise RuntimeError("corridor bundle destroy could not select a solution sortie")
    result = _destroy_result(
        instance=instance,
        solution=solution,
        route_context=route_context,
        removed_task_ids=selected_tasks,
        corridor=corridor,
    )
    repair_context = CorridorRepairContext(
        corridor=corridor,
        previous_origin_by_task=result.previous_origin_by_task,
        anchor_task=anchor,
        compatible_partner_tasks=tuple(
            task_id
            for task_id in compatible_partner_tasks
            if task_id in result.partial.missing_tasks
        ),
    )
    result = DestroyResult(
        partial=result.partial,
        removed_task_ids=result.removed_task_ids,
        previous_origin_positions=result.previous_origin_positions,
        corridor=result.corridor,
        repair_context=repair_context,
    )
    if diagnostics is not None:
        diagnostics.record_d6(tuple(selected_gains), corridor.width)
        diagnostics.record_corridor_bundle(
            partner_candidates=len(partners),
            partner_sorties_removed=partner_sorties_removed,
        )
    return result


class RouteGuidedDestroyEngine:
    def __init__(
        self,
        *,
        controller: RouteSearchController,
        diagnostics: RouteSearchDiagnostics,
        legacy_destroy_operators: Mapping[str, Any],
        corridor_bundle_destroy: bool = False,
        forward_partner_required: bool = False,
        physical_feasibility_aware: bool = False,
        guided_candidate_limit: int = 64,
        transactional_b2: bool = False,
        target_transformation_required: bool = False,
        dwell_guided: bool = False,
    ) -> None:
        self.controller = controller
        self.diagnostics = diagnostics
        self.legacy_destroy_operators = legacy_destroy_operators
        self.corridor_bundle_destroy = corridor_bundle_destroy
        self.forward_partner_required = forward_partner_required
        self.physical_feasibility_aware = physical_feasibility_aware
        self.guided_candidate_limit = guided_candidate_limit
        self.transactional_b2 = transactional_b2
        self.target_transformation_required = target_transformation_required
        self.dwell_guided = dwell_guided

    def apply(
        self,
        *,
        instance: Instance,
        solution: Solution,
        evaluation: Any,
        route_context: RouteSearchContext,
        policy: RepairPolicy,
        rng: random.Random,
        removal_count: int,
        destroy_name: str,
    ) -> DestroyResult:
        if destroy_name == "D6_route_corridor":
            return D6_route_corridor(
                instance=instance,
                solution=solution,
                evaluation=evaluation,
                route_context=route_context,
                policy=policy,
                controller=self.controller,
                rng=rng,
                removal_count=removal_count,
                diagnostics=self.diagnostics,
                corridor_bundle_destroy=self.corridor_bundle_destroy,
                forward_partner_required=self.forward_partner_required,
                physical_feasibility_aware=self.physical_feasibility_aware,
                guided_candidate_limit=self.guided_candidate_limit,
                transactional_b2=self.transactional_b2,
                target_transformation_required=self.target_transformation_required,
                dwell_guided=self.dwell_guided,
            )
        operator = self.legacy_destroy_operators[destroy_name]
        partial = operator.function(
            instance,
            solution,
            route_context.sortie_index,
            evaluation,
            rng,
            removal_count,
        )
        previous, _ = task_origins_in_solution(solution, route_context)
        removed = tuple(sorted(partial.missing_tasks))
        return DestroyResult(
            partial=partial,
            removed_task_ids=removed,
            previous_origin_positions=tuple(
                (task_id, previous[task_id]) for task_id in removed
            ),
        )
