from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from time import perf_counter
from typing import Callable, Literal, Mapping, TypeAlias

from usv_uav.algorithms.construction import (
    Insertion,
    RepairTimeBudgetExceeded,
    RepairSearchProfile,
    _candidate_insertions,
    _complete_partial,
    _insert,
    destroy_tasks_to_partial,
    find_feasible_insertions,
)
from usv_uav.algorithms.route_guided.context import RouteSearchContext
from usv_uav.algorithms.route_guided.controller import RouteSearchController
from usv_uav.algorithms.route_guided.destroy import task_origins_in_solution
from usv_uav.algorithms.route_guided.diagnostics import RouteSearchDiagnostics
from usv_uav.algorithms.route_guided.models import (
    CandidateClass,
    CorridorRepairContext,
    DestroyResult,
    NeighborhoodScope,
    RepairPolicy,
)
from usv_uav.algorithms.route_guided.scoped_index import ScopedSortieIndexView
from usv_uav.algorithms.route_guided.scoring import critical_release_group
from usv_uav.core.models import Instance, Sortie
from usv_uav.core.solution import Solution
from usv_uav.scheduling.evaluator import EvaluationResult


RepairSeed: TypeAlias = tuple[tuple[float, ...], Sortie, Insertion]


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    solution: Solution
    scope: NeighborhoodScope
    primary_span: int
    candidate_class: CandidateClass | None = None
    forward_absorbed_task_ids: tuple[int, ...] = ()
    forward_relocated_task_ids: tuple[int, ...] = ()
    guided_target_sortie_id: int | None = None
    guided_insertion: Insertion | None = None


@dataclass(frozen=True, slots=True)
class EvaluatedRepairCandidate:
    candidate: RepairCandidate
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class ProgressiveRepairOutcome:
    generated: tuple[RepairCandidate, ...]
    evaluated: tuple[EvaluatedRepairCandidate, ...]
    feasible: tuple[EvaluatedRepairCandidate, ...]
    final_scope: NeighborhoodScope | None
    budget_exhausted: bool
    construction_failures: int
    b2_transaction: B2TransactionResult | None = None


@dataclass(frozen=True, slots=True)
class B2TransactionResult:
    status: Literal[
        "destroy_target_inconsistent",
        "structurally_uninsertable",
        "reconstructed",
    ]
    target_sortie_id: int
    target_task_ids: tuple[int, ...]
    owner_sortie_ids: tuple[int, ...]
    missing_after_destroy: tuple[int, ...]
    legal_insertions: tuple[Insertion, ...]
    attempted_insertions: tuple[Insertion, ...]
    candidates: tuple[RepairCandidate, ...]


@dataclass(frozen=True, slots=True)
class ForwardCandidatePool:
    """The screened repair seeds grouped by mutually exclusive V2.1 structure."""

    by_class: Mapping[CandidateClass, tuple[RepairSeed, ...]]
    unclassified: tuple[RepairSeed, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_class", MappingProxyType(dict(self.by_class)))

    @classmethod
    def build(
        cls,
        screened: list[RepairSeed],
        *,
        repair_context: CorridorRepairContext,
        context: RouteSearchContext,
        physical_feasibility_aware: bool = False,
    ) -> "ForwardCandidatePool":
        grouped: dict[CandidateClass, list[RepairSeed]] = {
            candidate_class: [] for candidate_class in CandidateClass
        }
        unclassified: list[RepairSeed] = []
        for value in screened:
            sortie = value[1]
            tasks = set(sortie.task_sequence)
            if not tasks <= set(repair_context.previous_origin_by_task):
                raise RuntimeError("repair seed contains a task outside Missing")
            origin = context.sortie_origin_position[sortie.id]
            span = context.sortie_span[sortie.id]
            previous = repair_context.previous_origin_by_task
            guided_target = repair_context.guided_target
            is_guided = (
                physical_feasibility_aware
                and guided_target is not None
                and sortie.id == guided_target.target_sortie_id
            )
            absorbed = span >= 2 and any(
                previous[task_id] > origin for task_id in tasks
            )
            relocated = any(origin > previous[task_id] for task_id in tasks)
            if is_guided or (absorbed and not physical_feasibility_aware):
                candidate_class = CandidateClass.FORWARD_ABSORPTION
            elif relocated:
                candidate_class = CandidateClass.FORWARD_RELOCATION
            elif span == 0:
                candidate_class = CandidateClass.SAME_POINT
            elif span == 1:
                candidate_class = CandidateClass.ADJACENT_RECOVERY
            else:
                unclassified.append(value)
                continue
            grouped[candidate_class].append(value)

        anchor = repair_context.anchor_task
        compatible = set(repair_context.compatible_partner_tasks)

        def coupled_key(value: RepairSeed):
            tasks = set(value[1].task_sequence)
            is_coupled = (
                anchor is not None
                and anchor in tasks
                and bool(tasks & compatible)
            )
            return (not is_coupled, value[0])

        grouped[CandidateClass.FORWARD_ABSORPTION].sort(key=coupled_key)
        grouped[CandidateClass.FORWARD_RELOCATION].sort(
            key=lambda value: (
                not (
                    anchor is not None
                    and anchor in value[1].task_sequence
                ),
                value[0],
            )
        )
        return cls(
            by_class={
                candidate_class: tuple(values)
                for candidate_class, values in grouped.items()
            },
            unclassified=tuple(unclassified),
        )

    def select(
        self,
        quotas: Mapping[CandidateClass, int],
        *,
        top_m: int = 4,
    ) -> tuple[tuple[CandidateClass | None, RepairSeed], ...]:
        selected: list[tuple[CandidateClass | None, RepairSeed]] = []
        selected_keys: set[tuple[int, tuple[int, int]]] = set()

        def append(candidate_class: CandidateClass | None, value: RepairSeed) -> None:
            key = (value[1].id, value[2])
            if key not in selected_keys and len(selected) < top_m:
                selected.append((candidate_class, value))
                selected_keys.add(key)

        quota_order = (
            CandidateClass.SAME_POINT,
            CandidateClass.ADJACENT_RECOVERY,
            CandidateClass.FORWARD_RELOCATION,
            CandidateClass.FORWARD_ABSORPTION,
        )
        for candidate_class in quota_order:
            for value in self.by_class[candidate_class][
                : quotas.get(candidate_class, 0)
            ]:
                append(candidate_class, value)

        # Missing quotas are first refilled by the forward structures, then local ones.
        for candidate_class in reversed(quota_order):
            for value in self.by_class[candidate_class]:
                append(candidate_class, value)
        for value in self.unclassified:
            append(None, value)
        return tuple(selected)


def _solution_key(solution: Solution, uav_count: int):
    return solution.canonical(uav_count)


def _primary_inserted_span(
    solution: Solution,
    destroy_result: DestroyResult,
    context: RouteSearchContext,
) -> int:
    retained = {
        sortie_id
        for sequence in destroy_result.partial.uav_sequences.values()
        for sortie_id in sequence
    }
    inserted = [
        context.sortie_span[sortie_id]
        for sequence in solution.uav_sequences.values()
        for sortie_id in sequence
        if sortie_id not in retained
    ]
    return max(inserted, default=0)


def _candidate_structural_tasks(
    solution: Solution,
    destroy_result: DestroyResult,
    context: RouteSearchContext,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    new_origins, new_sorties = task_origins_in_solution(solution, context)
    previous = destroy_result.previous_origin_by_task
    relocated = tuple(
        task_id
        for task_id in destroy_result.removed_task_ids
        if new_origins[task_id] > previous[task_id]
    )
    absorbed = tuple(
        task_id
        for task_id in destroy_result.removed_task_ids
        if context.sortie_span[new_sorties[task_id]] >= 2
        and previous[task_id] > new_origins[task_id]
    )
    return absorbed, relocated


def _global_completion_index(
    index: ScopedSortieIndexView,
    policy: RepairPolicy,
    *,
    protect_b2_corridor: bool = False,
) -> ScopedSortieIndexView:
    global_policy = replace(
        policy,
        scope=NeighborhoodScope.GLOBAL,
        origin_radius=None,
        max_recovery_span=None,
    )
    return replace(
        index,
        policy=global_policy,
        corridor=None,
        guided_target_sortie_id=None,
        excluded_sortie_ids=(
            index.excluded_sortie_ids
            if protect_b2_corridor
            else frozenset()
        ),
        protected_corridor=(
            index.protected_corridor if protect_b2_corridor else None
        ),
        strict_protected_corridor=protect_b2_corridor,
    )


def _reconstruct_b2_target(
    *,
    instance: Instance,
    destroy_result: DestroyResult,
    index: ScopedSortieIndexView,
    context: RouteSearchContext,
    policy: RepairPolicy,
    repair_profile: RepairSearchProfile | None,
    strict_no_undo: bool = False,
) -> B2TransactionResult | None:
    repair_context = destroy_result.repair_context
    guided = repair_context.guided_target if repair_context is not None else None
    if guided is None:
        return None
    partial = destroy_result.partial
    target_task_ids = tuple(guided.required_missing_task_ids)
    common = {
        "target_sortie_id": guided.target_sortie_id,
        "target_task_ids": target_task_ids,
        "owner_sortie_ids": tuple(guided.owner_sortie_ids),
        "missing_after_destroy": tuple(sorted(partial.missing_tasks)),
    }
    if not set(target_task_ids) <= partial.missing_tasks:
        return B2TransactionResult(
            status="destroy_target_inconsistent",
            legal_insertions=(),
            attempted_insertions=(),
            candidates=(),
            **common,
        )

    target = context.sortie_index.sortie_by_id[guided.target_sortie_id]
    legal_insertions = find_feasible_insertions(
        partial,
        target,
        instance,
        index,
        repair_profile,
        True,
    )
    if not legal_insertions:
        return B2TransactionResult(
            status="structurally_uninsertable",
            legal_insertions=(),
            attempted_insertions=(),
            candidates=(),
            **common,
        )

    def insertion_key(insertion: Insertion) -> tuple[float, int, int]:
        uav_id, position = insertion
        load = sum(
            context.sortie_index.sortie_by_id[sortie_id].nominal_duration_min
            for sortie_id in partial.uav_sequences.get(uav_id, ())
        )
        return load, uav_id, position

    attempted_insertions = tuple(
        sorted(legal_insertions, key=insertion_key)[: (1 if strict_no_undo else 3)]
    )
    completion_indexes = (
        (_global_completion_index(index, policy, protect_b2_corridor=True),)
        if strict_no_undo
        else (
            _global_completion_index(index, policy, protect_b2_corridor=True),
            _global_completion_index(index, policy),
        )
    )
    candidates: list[RepairCandidate] = []
    seen: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    for insertion in attempted_insertions:
        completed = None
        for completion_index in completion_indexes:
            try:
                completed = _complete_partial(
                    _insert(partial, target, insertion, repair_profile),
                    instance,
                    completion_index,
                    "augmentation",
                    repair_profile,
                    True,
                    True,
                ).to_solution()
                break
            except RuntimeError:
                continue
        if completed is None:
            continue
        if not any(
            guided.target_sortie_id in sequence
            for sequence in completed.uav_sequences.values()
        ):
            raise RuntimeError("B2 transaction lost its explicitly inserted target")
        key = _solution_key(completed, instance.uav_count)
        if key in seen:
            continue
        seen.add(key)
        absorbed, relocated = _candidate_structural_tasks(
            completed, destroy_result, context
        )
        candidates.append(RepairCandidate(
            solution=completed,
            scope=policy.scope,
            primary_span=context.sortie_span[target.id],
            candidate_class=CandidateClass.GUIDED_PHYSICAL_FORWARD,
            forward_absorbed_task_ids=absorbed,
            forward_relocated_task_ids=relocated,
            guided_target_sortie_id=target.id,
            guided_insertion=insertion,
        ))
    if not candidates:
        return B2TransactionResult(
            status="structurally_uninsertable",
            legal_insertions=tuple(legal_insertions),
            attempted_insertions=attempted_insertions,
            candidates=(),
            **common,
        )
    return B2TransactionResult(
        status="reconstructed",
        legal_insertions=tuple(legal_insertions),
        attempted_insertions=attempted_insertions,
        candidates=tuple(candidates),
        **common,
    )


def _R4P_progressive_recovery_impl(
    *,
    instance: Instance,
    destroy_result: DestroyResult,
    index: ScopedSortieIndexView,
    context: RouteSearchContext,
    policy: RepairPolicy,
    repair_profile: RepairSearchProfile | None = None,
    corridor_coupled: bool = False,
    candidate_quotas: Mapping[CandidateClass, int] | None = None,
    physical_feasibility_aware: bool = False,
    final_candidate_classes: bool = False,
    prebuilt_b2_candidates: tuple[RepairCandidate, ...] = (),
    full_decoder_candidate_limit: int = 4,
) -> tuple[RepairCandidate, ...]:
    """Recovery beam for historical V2 variants and final V3 roles."""
    partial = destroy_result.partial
    if not partial.missing_tasks:
        return (RepairCandidate(partial.to_solution(), policy.scope, 0),)
    screened = _candidate_insertions(
        partial,
        instance,
        index,
        "recovery_aware",
        repair_profile,
        True,
        True,
        True,
    )
    started = perf_counter() if repair_profile is not None else 0.0
    selection_time_sec = 0.0
    selection_started = perf_counter() if repair_profile is not None else 0.0
    by_span = {
        span: [
            value for value in screened
            if context.sortie_span[value[1].id] == span
        ]
        for span in (0, 1, 2)
    }
    if repair_profile is not None:
        selection_time_sec += perf_counter() - selection_started
    if repair_profile is not None:
        repair_profile.recovery_positions_considered += len(screened)
        repair_profile.same_recovery_candidates += len(by_span[0])
        repair_profile.different_recovery_candidates += len(screened) - len(by_span[0])
    completion_index_by_class: dict[
        CandidateClass, tuple[ScopedSortieIndexView, ...]
    ] = {}
    if final_candidate_classes:
        repair_context = destroy_result.repair_context or CorridorRepairContext(
            corridor=destroy_result.corridor,
            previous_origin_by_task=destroy_result.previous_origin_by_task,
            anchor_task=None,
            compatible_partner_tasks=(),
        )
        guided_target_id = None
        if repair_context.guided_target is not None:
            guided_target_id = repair_context.guided_target.target_sortie_id
        global_index = _global_completion_index(index, policy)
        global_screened = _candidate_insertions(
            partial,
            instance,
            global_index,
            "recovery_aware",
            repair_profile,
            True,
            True,
            True,
        )
        selection_started = perf_counter() if repair_profile is not None else 0.0
        previous = repair_context.previous_origin_by_task

        def seed_key(value: RepairSeed) -> tuple[int, Insertion]:
            return value[1].id, value[2]

        def is_b0(value: RepairSeed) -> bool:
            sortie = value[1]
            origin = context.sortie_origin_position[sortie.id]
            return context.sortie_span[sortie.id] <= 1 and all(
                previous[task_id] == origin for task_id in sortie.task_sequence
            )

        b0 = tuple(
            value
            for value in global_screened
            if value[1].id != guided_target_id and is_b0(value)
        )
        b0_keys = {seed_key(value) for value in b0}
        b1 = tuple(
            value
            for value in global_screened
            if value[1].id != guided_target_id
            and seed_key(value) not in b0_keys
            and context.sortie_span[value[1].id] <= 1
            and all(
                abs(
                    context.sortie_origin_position[value[1].id]
                    - previous[task_id]
                ) <= 1
                for task_id in value[1].task_sequence
            )
        )
        reserved_keys = b0_keys | {seed_key(value) for value in b1}
        b3 = tuple(
            value
            for value in global_screened
            if value[1].id != guided_target_id
            and seed_key(value) not in reserved_keys
        )
        seed_groups = (
            (CandidateClass.LEGACY_LOCAL_BEST, b0, False),
            (CandidateClass.ROUTE_GUIDED_LOCAL, b1, False),
            (CandidateClass.GLOBAL_DIVERSIFICATION, b3, False),
        )
        classified_seeds = tuple(
            (candidate_class, value, is_guided)
            for candidate_class, values, is_guided in seed_groups
            for value in values[:16]
        )
        completion_index_by_class = {
            CandidateClass.LEGACY_LOCAL_BEST: (global_index,),
            CandidateClass.ROUTE_GUIDED_LOCAL: (global_index,),
            CandidateClass.GLOBAL_DIVERSIFICATION: (global_index,),
        }
        if repair_profile is not None:
            selection_time_sec += perf_counter() - selection_started
    elif corridor_coupled:
        selection_started = perf_counter() if repair_profile is not None else 0.0
        repair_context = destroy_result.repair_context or CorridorRepairContext(
            corridor=destroy_result.corridor,
            previous_origin_by_task=destroy_result.previous_origin_by_task,
            anchor_task=None,
            compatible_partner_tasks=(),
        )
        guided_target_id = (
            repair_context.guided_target.target_sortie_id
            if physical_feasibility_aware
            and repair_context.guided_target is not None
            else None
        )
        pool = ForwardCandidatePool.build(
            [value for value in screened if value[1].id != guided_target_id],
            repair_context=repair_context,
            context=context,
            physical_feasibility_aware=physical_feasibility_aware,
        )
        active_quotas = candidate_quotas or {
            candidate_class: 1 for candidate_class in CandidateClass
        }
        alternatives = pool.select(active_quotas, top_m=4)
        guided_seeds = tuple(
            (
                CandidateClass.FORWARD_ABSORPTION,
                value,
                True,
            )
            for value in screened
            if value[1].id == guided_target_id
        )[:8]
        classified_seeds = (
            *guided_seeds,
            *((candidate_class, value, False) for candidate_class, value in alternatives),
        )
        if repair_profile is not None:
            selection_time_sec += perf_counter() - selection_started
    else:
        selection_started = perf_counter() if repair_profile is not None else 0.0
        quotas = {
            0: policy.r4_span0_quota,
            1: policy.r4_span1_quota,
            2: policy.r4_span2_quota,
        }
        seeds = [
            value
            for span in (0, 1, 2)
            for value in by_span[span][:quotas[span]]
        ]
        selected_keys = {(value[1].id, value[2]) for value in seeds}
        for value in screened:
            if len(seeds) >= 4:
                break
            key = (value[1].id, value[2])
            if key not in selected_keys:
                seeds.append(value)
                selected_keys.add(key)
        classified_seeds = tuple((None, value, False) for value in seeds[:4])
        if repair_profile is not None:
            selection_time_sec += perf_counter() - selection_started
    if repair_profile is not None:
        repair_profile.top_m_retained += min(
            4,
            (
                len({candidate_class for candidate_class, _, _ in classified_seeds})
                if final_candidate_classes
                else len(classified_seeds)
            ),
        )
        repair_profile.recovery_filter_time_sec += perf_counter() - started
        repair_profile.r4p_sort_or_topk_sec += selection_time_sec
        repair_profile.r4p_shortlisted_candidates += len(classified_seeds)
    if not classified_seeds and not prebuilt_b2_candidates:
        raise RuntimeError(
            f"progressive recovery cannot cover tasks {sorted(partial.missing_tasks)}"
        )

    candidates: list[RepairCandidate] = []
    seen = {
        _solution_key(candidate.solution, instance.uav_count)
        for candidate in prebuilt_b2_candidates
    }
    emitted_classes: set[CandidateClass] = {
        candidate.candidate_class
        for candidate in prebuilt_b2_candidates
        if candidate.candidate_class is not None
    }
    # B2 may need several legal UAV insertion trials, but the entire
    # progressive-repair beam still respects the V3 Top-M=4 decoder budget.
    max_generic_candidates = max(
        0, full_decoder_candidate_limit - len(prebuilt_b2_candidates)
    )
    for candidate_class, (_, sortie, insertion), is_guided in classified_seeds:
        if len(candidates) >= max_generic_candidates:
            break
        if (
            final_candidate_classes
            and candidate_class is not None
            and candidate_class in emitted_classes
        ):
            continue
        if not set(sortie.task_sequence) <= partial.missing_tasks:
            raise RuntimeError("progressive recovery seed violates exact coverage")
        completion_indexes = completion_index_by_class.get(
            candidate_class, (index,)
        )
        if (
            not final_candidate_classes
            and is_guided
            and physical_feasibility_aware
        ):
            # Strict protection is attempted first.  The next two candidates
            # relax only the heuristic completion filters; the physical lower
            # bound remains mandatory and Full Decoder remains authoritative.
            completion_indexes = (
                index,
                replace(index, protected_corridor=None),
                replace(
                    index,
                    protected_corridor=None,
                    excluded_sortie_ids=frozenset(),
                ),
            )
        for completion_index in completion_indexes:
            if len(candidates) >= max_generic_candidates:
                break
            if repair_profile is not None:
                repair_profile.r4p_materialized_candidates += 1
                materialize_started = perf_counter()
            seeded = _insert(partial, sortie, insertion, repair_profile)
            if repair_profile is not None:
                repair_profile.r4p_candidate_materialize_sec += (
                    perf_counter() - materialize_started
                )
                residual_started = perf_counter()
            try:
                completed_partial = _complete_partial(
                    seeded,
                    instance,
                    completion_index,
                    "augmentation",
                    repair_profile,
                    True,
                    True,
                )
            except RuntimeError:
                if repair_profile is not None:
                    repair_profile.r4p_residual_repair_sec += (
                        perf_counter() - residual_started
                    )
                continue
            if repair_profile is not None:
                repair_profile.r4p_residual_repair_sec += (
                    perf_counter() - residual_started
                )
                materialize_started = perf_counter()
            completed = completed_partial.to_solution()
            key = _solution_key(completed, instance.uav_count)
            if key in seen:
                if repair_profile is not None:
                    repair_profile.r4p_candidate_materialize_sec += (
                        perf_counter() - materialize_started
                    )
                continue
            seen.add(key)
            absorbed, relocated = (
                _candidate_structural_tasks(completed, destroy_result, context)
                if corridor_coupled
                else ((), ())
            )
            candidates.append(
                RepairCandidate(
                    solution=completed,
                    scope=policy.scope,
                    primary_span=context.sortie_span[sortie.id],
                    candidate_class=candidate_class,
                    forward_absorbed_task_ids=absorbed,
                    forward_relocated_task_ids=relocated,
                    guided_target_sortie_id=(sortie.id if is_guided else None),
                )
            )
            if repair_profile is not None:
                repair_profile.r4p_candidate_materialize_sec += (
                    perf_counter() - materialize_started
                )
                repair_profile.r4p_successful_insertions += 1
            if candidate_class is not None:
                emitted_classes.add(candidate_class)
            if final_candidate_classes:
                break
    if not candidates and not prebuilt_b2_candidates:
        raise RuntimeError(
            "progressive recovery beam produced no complete route-compatible solution"
        )
    return (*prebuilt_b2_candidates, *candidates)


def R4P_progressive_recovery(
    *,
    instance: Instance,
    destroy_result: DestroyResult,
    index: ScopedSortieIndexView,
    context: RouteSearchContext,
    policy: RepairPolicy,
    repair_profile: RepairSearchProfile | None = None,
    corridor_coupled: bool = False,
    candidate_quotas: Mapping[CandidateClass, int] | None = None,
    physical_feasibility_aware: bool = False,
    final_candidate_classes: bool = False,
    prebuilt_b2_candidates: tuple[RepairCandidate, ...] = (),
    full_decoder_candidate_limit: int = 4,
) -> tuple[RepairCandidate, ...]:
    """Profile one exact R4P call without changing its candidate semantics."""
    if repair_profile is None:
        return _R4P_progressive_recovery_impl(
            instance=instance,
            destroy_result=destroy_result,
            index=index,
            context=context,
            policy=policy,
            corridor_coupled=corridor_coupled,
            candidate_quotas=candidate_quotas,
            physical_feasibility_aware=physical_feasibility_aware,
            final_candidate_classes=final_candidate_classes,
            prebuilt_b2_candidates=prebuilt_b2_candidates,
            full_decoder_candidate_limit=full_decoder_candidate_limit,
        )
    phase_before = (
        repair_profile.r4p_index_lookup_sec,
        repair_profile.r4p_candidate_enumeration_sec,
        repair_profile.r4p_static_filter_sec,
        repair_profile.r4p_route_window_filter_sec,
        repair_profile.r4p_dynamic_filter_sec,
        repair_profile.r4p_proxy_score_sec,
        repair_profile.r4p_sort_or_topk_sec,
        repair_profile.r4p_candidate_materialize_sec,
        repair_profile.r4p_residual_repair_sec,
    )
    call_started = perf_counter()
    repair_profile.r4p_calls += 1
    try:
        return _R4P_progressive_recovery_impl(
            instance=instance,
            destroy_result=destroy_result,
            index=index,
            context=context,
            policy=policy,
            repair_profile=repair_profile,
            corridor_coupled=corridor_coupled,
            candidate_quotas=candidate_quotas,
            physical_feasibility_aware=physical_feasibility_aware,
            final_candidate_classes=final_candidate_classes,
            prebuilt_b2_candidates=prebuilt_b2_candidates,
            full_decoder_candidate_limit=full_decoder_candidate_limit,
        )
    finally:
        elapsed = perf_counter() - call_started
        phase_after = (
            repair_profile.r4p_index_lookup_sec,
            repair_profile.r4p_candidate_enumeration_sec,
            repair_profile.r4p_static_filter_sec,
            repair_profile.r4p_route_window_filter_sec,
            repair_profile.r4p_dynamic_filter_sec,
            repair_profile.r4p_proxy_score_sec,
            repair_profile.r4p_sort_or_topk_sec,
            repair_profile.r4p_candidate_materialize_sec,
            repair_profile.r4p_residual_repair_sec,
        )
        attributed = sum(
            after - before for before, after in zip(phase_before, phase_after)
        )
        repair_profile.r4p_total_sec += elapsed
        repair_profile.r4p_other_sec += max(0.0, elapsed - attributed)


def _decoder_guided_retry_candidate(
    *,
    instance: Instance,
    source: EvaluatedRepairCandidate,
    destroy_result: DestroyResult,
    context: RouteSearchContext,
    policy: RepairPolicy,
    repair_profile: RepairSearchProfile | None,
) -> RepairCandidate | None:
    """Build the single V3.1 feedback repair while keeping its target fixed."""
    guided = (
        destroy_result.repair_context.guided_target
        if destroy_result.repair_context is not None
        else None
    )
    if guided is None or not guided.strict_no_undo:
        return None
    target_codes = {
        detail.code
        for detail in source.evaluation.violation_details
        if detail.sortie_id == guided.target_sortie_id
    }
    if not target_codes & {"HOVER_SORTIE_TIME", "HOVER_SOC_RESERVE"}:
        return None
    scheduled = {
        sortie_id
        for sequence in source.candidate.solution.uav_sequences.values()
        for sortie_id in sequence
    }
    forbidden = set(guided.owner_sortie_ids) | set(guided.blocker_sortie_ids)
    summaries = {
        context.support_position[summary.support_id]: summary
        for summary in source.evaluation.support_dwell_summaries
    }
    groups = []
    for position in range(guided.corridor.start_pos + 1, guided.corridor.end_pos):
        summary = summaries.get(position)
        if summary is None:
            continue
        group, gain = critical_release_group(summary)
        eligible = tuple(
            sortie_id
            for sortie_id in group
            if sortie_id in scheduled
            and sortie_id != guided.target_sortie_id
            and sortie_id not in forbidden
        )
        if eligible and gain > 1e-9:
            groups.append((-gain, position, eligible))
    if not groups:
        return None
    _, _, eligible = min(groups)
    blocker_id = eligible[0]
    blocker = context.sortie_index.sortie_by_id[blocker_id]
    total_release_tasks = set(destroy_result.removed_task_ids) | set(
        blocker.task_sequence
    )
    if (
        guided.release_task_budget is not None
        and len(total_release_tasks) > guided.release_task_budget
    ):
        return None
    previous_origins, _ = task_origins_in_solution(
        source.candidate.solution, context
    )
    partial = destroy_tasks_to_partial(
        source.candidate.solution,
        blocker.task_sequence,
        context.sortie_index,
        instance.uav_count,
    )
    retry_forbidden = frozenset((*forbidden, blocker_id))
    retry_policy = replace(
        policy,
        scope=NeighborhoodScope.GLOBAL,
        origin_radius=None,
        max_recovery_span=None,
    )
    view = ScopedSortieIndexView(
        context=context,
        previous_origin_positions={
            task_id: previous_origins[task_id]
            for task_id in partial.missing_tasks
        },
        policy=retry_policy,
        corridor=None,
        physical_feasibility_aware=True,
        excluded_sortie_ids=retry_forbidden,
        protected_corridor=guided.corridor,
        strict_protected_corridor=True,
    )
    try:
        completed = _complete_partial(
            partial,
            instance,
            view,
            "augmentation",
            repair_profile,
            True,
            True,
        ).to_solution()
    except RuntimeError:
        return None
    if not any(
        guided.target_sortie_id in sequence
        for sequence in completed.uav_sequences.values()
    ):
        raise RuntimeError("decoder-guided retry lost its fixed B2 target")
    return replace(source.candidate, solution=completed)


class ProgressiveRepairEngine:
    def __init__(
        self,
        *,
        controller: RouteSearchController,
        diagnostics: RouteSearchDiagnostics,
        repair_operators,
        corridor_coupled: bool = False,
        candidate_quotas: Mapping[CandidateClass, int] | None = None,
        physical_feasibility_aware: bool = False,
        final_candidate_classes: bool = False,
        dwell_guided: bool = False,
    ) -> None:
        self.controller = controller
        self.diagnostics = diagnostics
        self.repair_operators = repair_operators
        self.corridor_coupled = corridor_coupled
        self.candidate_quotas = candidate_quotas
        self.physical_feasibility_aware = physical_feasibility_aware
        self.final_candidate_classes = final_candidate_classes
        self.dwell_guided = dwell_guided

    def _generate(
        self,
        *,
        instance: Instance,
        destroy_result: DestroyResult,
        route_context: RouteSearchContext,
        policy: RepairPolicy,
        repair_name: str,
        repair_profile: RepairSearchProfile | None,
        attempt_b2_transaction: bool = True,
        deadline_expired: Callable[[], bool] | None = None,
    ) -> tuple[tuple[RepairCandidate, ...], B2TransactionResult | None]:
        guided_target = (
            destroy_result.repair_context.guided_target
            if destroy_result.repair_context is not None
            else None
        )
        view = ScopedSortieIndexView(
            context=route_context,
            previous_origin_positions=destroy_result.previous_origin_by_task,
            policy=policy,
            corridor=destroy_result.corridor,
            physical_feasibility_aware=self.physical_feasibility_aware,
            guided_target_sortie_id=(
                guided_target.target_sortie_id
                if self.physical_feasibility_aware
                and guided_target is not None
                else None
            ),
            excluded_sortie_ids=(
                frozenset(
                    (
                        *(
                            guided_target.owner_sortie_ids
                            if guided_target.strict_no_undo
                            else ()
                        ),
                        *guided_target.blocker_sortie_ids,
                    )
                )
                if self.physical_feasibility_aware
                and guided_target is not None
                else frozenset()
            ),
            protected_corridor=(
                guided_target.corridor
                if self.physical_feasibility_aware
                and guided_target is not None
                else None
            ),
            deadline_expired=deadline_expired,
        )
        if repair_name == "R4P_progressive_recovery":
            b2_transaction = (
                _reconstruct_b2_target(
                    instance=instance,
                    destroy_result=destroy_result,
                    index=view,
                    context=route_context,
                    policy=policy,
                    repair_profile=repair_profile,
                    strict_no_undo=(
                        self.dwell_guided
                        and guided_target is not None
                        and guided_target.strict_no_undo
                    ),
                )
                if self.final_candidate_classes and attempt_b2_transaction
                else None
            )
            try:
                candidates = R4P_progressive_recovery(
                    instance=instance,
                    destroy_result=destroy_result,
                    index=view,
                    context=route_context,
                    policy=policy,
                    repair_profile=repair_profile,
                    corridor_coupled=self.corridor_coupled,
                    candidate_quotas=self.candidate_quotas,
                    physical_feasibility_aware=self.physical_feasibility_aware,
                    final_candidate_classes=self.final_candidate_classes,
                    prebuilt_b2_candidates=(
                        b2_transaction.candidates
                        if b2_transaction is not None
                        else ()
                    ),
                    full_decoder_candidate_limit=(
                        3
                        if self.dwell_guided and guided_target is not None
                        else 4
                    ),
                )
            except RuntimeError:
                if b2_transaction is None:
                    raise
                candidates = b2_transaction.candidates
            return candidates, b2_transaction
        solutions = self.repair_operators[repair_name].function(
            instance,
            destroy_result.partial,
            view,
            repair_profile=repair_profile,
        )
        return tuple(
            RepairCandidate(
                solution=solution,
                scope=policy.scope,
                primary_span=_primary_inserted_span(
                    solution, destroy_result, route_context
                ),
            )
            for solution in solutions
        ), None

    def apply(
        self,
        *,
        instance: Instance,
        destroy_result: DestroyResult,
        route_context: RouteSearchContext,
        policy: RepairPolicy,
        repair_name: str,
        repair_profile: RepairSearchProfile | None = None,
        evaluate_candidate: Callable[[Solution], EvaluationResult | None] | None = None,
        deadline_expired: Callable[[], bool] | None = None,
    ) -> ProgressiveRepairOutcome:
        """Generate, fully decode, and escalate only when the active scope fails."""
        self.diagnostics.repair_calls += 1
        generated: list[RepairCandidate] = []
        evaluated: list[EvaluatedRepairCandidate] = []
        construction_failures = 0
        b2_transaction: B2TransactionResult | None = None
        guided = (
            destroy_result.repair_context.guided_target
            if destroy_result.repair_context is not None
            else None
        )
        if self.physical_feasibility_aware and guided is not None:
            policy = self.controller.policy_for_release_size(
                policy, len(destroy_result.removed_task_ids)
            )
        policies = self.controller.escalation_policies(policy)
        for policy_index, active_policy in enumerate(policies):
            if deadline_expired is not None and deadline_expired():
                return ProgressiveRepairOutcome(
                    generated=tuple(generated),
                    evaluated=tuple(evaluated),
                    feasible=(),
                    final_scope=active_policy.scope,
                    budget_exhausted=True,
                    construction_failures=construction_failures,
                    b2_transaction=b2_transaction,
                )
            self.diagnostics.record_scope(active_policy.scope)
            scope_b2_transaction = None
            try:
                scope_candidates, scope_b2_transaction = self._generate(
                    instance=instance,
                    destroy_result=destroy_result,
                    route_context=route_context,
                    policy=active_policy,
                    repair_name=repair_name,
                    repair_profile=repair_profile,
                    attempt_b2_transaction=policy_index == 0,
                    deadline_expired=deadline_expired,
                )
                if scope_b2_transaction is not None:
                    b2_transaction = scope_b2_transaction
                    self.diagnostics.record_b2_reconstruction(
                        scope_b2_transaction.status
                    )
            except RepairTimeBudgetExceeded:
                return ProgressiveRepairOutcome(
                    generated=tuple(generated),
                    evaluated=tuple(evaluated),
                    feasible=(),
                    final_scope=active_policy.scope,
                    budget_exhausted=True,
                    construction_failures=construction_failures,
                    b2_transaction=b2_transaction,
                )
            except RuntimeError:
                scope_candidates = ()
                construction_failures += 1
            remaining_decoder_calls = 4 - len(evaluated)
            if self.dwell_guided and guided is not None and policy_index == 0:
                remaining_decoder_calls = min(remaining_decoder_calls, 3)
            scope_candidates = scope_candidates[:max(0, remaining_decoder_calls)]
            generated.extend(scope_candidates)
            for candidate in scope_candidates:
                self.diagnostics.record_candidate("generated", candidate.primary_span)
                self.diagnostics.record_candidate_class(
                    "generated", candidate.candidate_class
                )
                if (
                    candidate.guided_target_sortie_id is not None
                    and not self.final_candidate_classes
                ):
                    self.diagnostics.record_guided_reconstructed()
            if scope_candidates and evaluate_candidate is None:
                return ProgressiveRepairOutcome(
                    generated=tuple(generated),
                    evaluated=(),
                    feasible=(),
                    final_scope=active_policy.scope,
                    budget_exhausted=False,
                    construction_failures=construction_failures,
                    b2_transaction=b2_transaction,
                )
            scope_evaluated: list[EvaluatedRepairCandidate] = []
            budget_exhausted = False
            for candidate in scope_candidates:
                assert evaluate_candidate is not None
                if deadline_expired is not None and deadline_expired():
                    budget_exhausted = True
                    break
                retained = {
                    sortie_id
                    for sequence in destroy_result.partial.uav_sequences.values()
                    for sortie_id in sequence
                }
                inserted = {
                    sortie_id
                    for sequence in candidate.solution.uav_sequences.values()
                    for sortie_id in sequence
                    if sortie_id not in retained
                }
                if self.physical_feasibility_aware and any(
                    not route_context.is_physically_feasible(sortie_id)
                    for sortie_id in inserted
                ):
                    self.diagnostics.wasted_static_impossible_decoder_calls += 1
                    continue
                evaluation = evaluate_candidate(candidate.solution)
                if evaluation is None:
                    budget_exhausted = True
                    break
                item = EvaluatedRepairCandidate(candidate, evaluation)
                evaluated.append(item)
                scope_evaluated.append(item)
                if candidate.guided_target_sortie_id is not None:
                    self.diagnostics.record_guided_decoder(
                        evaluation,
                        target_sortie_id=candidate.guided_target_sortie_id,
                    )
            if (
                self.dwell_guided
                and policy_index == 0
                and scope_b2_transaction is not None
                and scope_b2_transaction.status == "reconstructed"
                and len(evaluated) < 4
            ):
                initial_b2 = next(
                    (
                        item
                        for item in scope_evaluated
                        if item.candidate.guided_target_sortie_id
                        == scope_b2_transaction.target_sortie_id
                        and not item.evaluation.feasible
                    ),
                    None,
                )
                retry = (
                    _decoder_guided_retry_candidate(
                        instance=instance,
                        source=initial_b2,
                        destroy_result=destroy_result,
                        context=route_context,
                        policy=active_policy,
                        repair_profile=repair_profile,
                    )
                    if initial_b2 is not None
                    else None
                )
                if retry is not None:
                    generated.append(retry)
                    self.diagnostics.record_candidate(
                        "generated", retry.primary_span
                    )
                    self.diagnostics.record_candidate_class(
                        "generated", retry.candidate_class
                    )
                    retry_evaluation = evaluate_candidate(retry.solution)
                    if retry_evaluation is None:
                        budget_exhausted = True
                    else:
                        retry_item = EvaluatedRepairCandidate(
                            retry, retry_evaluation
                        )
                        evaluated.append(retry_item)
                        scope_evaluated.append(retry_item)
                        self.diagnostics.record_guided_decoder(
                            retry_evaluation,
                            target_sortie_id=retry.guided_target_sortie_id,
                        )
                        self.diagnostics.record_b2_retry(
                            success=retry_evaluation.feasible
                        )
            if (
                scope_b2_transaction is not None
                and scope_b2_transaction.status == "reconstructed"
            ):
                b2_evaluations = tuple(
                    item
                    for item in scope_evaluated
                    if item.candidate.guided_target_sortie_id
                    == scope_b2_transaction.target_sortie_id
                )
                self.diagnostics.record_b2_decoder(
                    b2_evaluations,
                    target_sortie_id=scope_b2_transaction.target_sortie_id,
                )
            feasible = tuple(item for item in scope_evaluated if item.evaluation.feasible)
            for item in feasible:
                self.diagnostics.record_feasible_candidate(
                    candidate_class=item.candidate.candidate_class,
                    span=item.candidate.primary_span,
                )
            if feasible or budget_exhausted:
                if repair_name == "R4P_progressive_recovery":
                    self.diagnostics.record_r4p_outcome(
                        generated_spans=tuple(
                            candidate.primary_span for candidate in generated
                        ),
                        feasible_objectives=tuple(
                            (
                                item.candidate.primary_span,
                                item.evaluation.objective,
                            )
                            for item in feasible
                        ),
                    )
                return ProgressiveRepairOutcome(
                    generated=tuple(generated),
                    evaluated=tuple(evaluated),
                    feasible=feasible,
                    final_scope=active_policy.scope,
                    budget_exhausted=budget_exhausted,
                    construction_failures=construction_failures,
                    b2_transaction=b2_transaction,
                )
            if policy_index + 1 < len(policies):
                self.diagnostics.record_escalation(
                    active_policy.scope, policies[policy_index + 1].scope
                )
        if (
            repair_name == "R4P_progressive_recovery"
            and evaluate_candidate is not None
        ):
            self.diagnostics.record_r4p_outcome(
                generated_spans=tuple(
                    candidate.primary_span for candidate in generated
                ),
                feasible_objectives=(),
            )
        return ProgressiveRepairOutcome(
            generated=tuple(generated),
            evaluated=tuple(evaluated),
            feasible=(),
            final_scope=policies[-1].scope if policies else None,
            budget_exhausted=False,
            construction_failures=construction_failures,
            b2_transaction=b2_transaction,
        )

    @staticmethod
    def relocation_deltas(
        selected: RepairCandidate,
        destroy_result: DestroyResult,
        route_context: RouteSearchContext,
    ) -> tuple[int, ...]:
        new_origins, _ = task_origins_in_solution(selected.solution, route_context)
        return tuple(
            new_origins[task_id] - previous_origin
            for task_id, previous_origin in destroy_result.previous_origin_positions
        )
