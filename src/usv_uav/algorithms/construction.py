from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Iterable, Literal, Mapping, Sequence

from usv_uav.core.models import Instance, Sortie
from usv_uav.core.partial_solution import PartialSolution
from usv_uav.core.solution import Solution
from usv_uav.preprocessing.sortie_index import SortieIndex, build_sortie_index
from usv_uav.algorithms.route_guided.scoring import (
    synchronization_candidate_key as _shared_synchronization_candidate_key,
)
from usv_uav.algorithms.r4_numba import (
    R4KernelBackend,
    R4KernelResult,
    r4_numba_eligible,
    rank_r4_options,
)
from usv_uav.algorithms.repair_numba import (
    RepairKernelBackend,
    RepairKernelResult,
    RepairNumbaWorkspace,
    create_repair_numba_workspace,
    rank_repair_options,
    repair_numba_eligible,
)


RepairStrategy = Literal["greedy", "regret2", "augmentation", "recovery_aware"]


class RepairTimeBudgetExceeded(Exception):
    """Cooperative abort raised only at a repair-safe structural boundary."""


Insertion = tuple[int, int]
RankedInsertion = tuple[tuple[float, ...], Sortie, Insertion]
RoutePositionArrays = tuple[tuple[int, ...], tuple[int, ...], bool]
INDEXED_LOOKUP_REPAIR_STRATEGIES = frozenset(
    {"greedy", "regret2", "augmentation", "recovery_aware"}
)
ROUTE_WINDOW_REPAIR_STRATEGIES = frozenset(
    {"greedy", "regret2", "augmentation", "recovery_aware"}
)


def fixed_route_sail_prefix(
    instance: Instance,
    usv_speed_km_min: float,
) -> tuple[float, ...]:
    """Cumulative USV sailing minutes along the frozen support route."""
    if usv_speed_km_min <= 0:
        raise ValueError("usv_speed_km_min must be positive")
    matrix_index = {
        support_id: index + 1
        for index, support_id in enumerate(instance.selected_supports)
    }
    route = instance.usv_route[1:-1]
    prefix = [0.0]
    for origin, recovery in zip(route, route[1:]):
        distance = float(
            instance.safe_distance_matrix[
                matrix_index[origin], matrix_index[recovery]
            ]
        )
        prefix.append(prefix[-1] + distance / usv_speed_km_min)
    return tuple(prefix)


def synchronization_candidate_key(
    sortie: Sortie,
    index: SortieIndex,
    route_sail_prefix: Sequence[float],
) -> tuple[float, float, float, float, float]:
    """Compatibility facade for the shared frozen S5A1 score."""
    return _shared_synchronization_candidate_key(
        sortie, index, route_sail_prefix
    )


@dataclass(slots=True)
class RepairSearchProfile:
    """Exclusive timers and search-volume counters for one repair operator."""

    sortie_lookup_calls: int = 0
    index_lookup_calls: int = 0
    indexed_sorties_returned: int = 0
    sorties_raw: int = 0
    sorties_scanned: int = 0
    sortie_lookup_time_sec: float = 0.0
    insertion_positions_tested: int = 0
    insertion_feasibility_checks: int = 0
    insertion_time_sec: float = 0.0
    insertion_positions_raw: int = 0
    insertion_positions_legal: int = 0
    insertion_windows_total: int = 0
    insertion_windows_empty: int = 0
    insertion_window_width_sum: int = 0
    insertion_window_width_max: int = 0
    insertion_window_time_sec: float = 0.0
    recovery_positions_considered: int = 0
    same_recovery_candidates: int = 0
    different_recovery_candidates: int = 0
    recovery_filter_time_sec: float = 0.0
    solution_copy_calls: int = 0
    candidate_solution_builds: int = 0
    candidate_build_time_sec: float = 0.0
    ranking_items: int = 0
    sorting_time_sec: float = 0.0
    top_m_retained: int = 0
    # Shared Repair Workload Audit.  The option counters deliberately retain
    # the unit used at each funnel stage (sorties until static filtering,
    # insertion positions thereafter); their definitions are emitted with the
    # experiment report instead of being treated as interchangeable counts.
    repair_raw_options: int = 0
    repair_route_legal_options: int = 0
    repair_static_feasible_options: int = 0
    repair_proxy_scored_options: int = 0
    repair_materialized_options: int = 0
    repair_decoder_options: int = 0
    regret_option_evaluations: int = 0
    regret_first_best_updates: int = 0
    regret_second_best_updates: int = 0
    repair_index_lookup_sec: float = 0.0
    repair_option_enumeration_sec: float = 0.0
    repair_legality_check_sec: float = 0.0
    repair_proxy_score_sec: float = 0.0
    repair_rank_select_sec: float = 0.0
    repair_apply_insertion_sec: float = 0.0
    repair_residual_loop_sec: float = 0.0
    repair_bookkeeping_sec: float = 0.0
    # R4 recovery-aware seven-stage attribution. These timers are exclusive:
    # the ALNS layer assigns the unclassified repair-call remainder to
    # materialization and records full Decoder time separately.
    r4_profile_calls: int = 0
    r4_profile_total_sec: float = 0.0
    r4_candidate_filter_sec: float = 0.0
    r4_route_window_sec: float = 0.0
    r4_option_enumeration_sec: float = 0.0
    r4_proxy_scoring_sec: float = 0.0
    r4_materialization_sec: float = 0.0
    r4_decoder_sec: float = 0.0
    r4_sort_select_sec: float = 0.0
    r4_raw_candidate_count: int = 0
    r4_static_feasible_count: int = 0
    r4_option_count: int = 0
    r4_decoded_candidate_count: int = 0
    r4_feasible_candidate_count: int = 0
    r4_returned_candidate_count: int = 0
    # The serial V1 kernel fuses filtering, legality, scoring, and bounded
    # selection. Calls count every initial and residual kernel invocation.
    r4_numba_calls: int = 0
    r4_numba_input_sec: float = 0.0
    r4_numba_kernel_sec: float = 0.0
    r4_numba_output_sec: float = 0.0
    repair_numba_calls: int = 0
    repair_numba_input_sec: float = 0.0
    repair_numba_kernel_sec: float = 0.0
    repair_numba_output_sec: float = 0.0
    # R4P-only attribution.  These fields remain zero for the other repairs.
    # Candidate counters are a funnel over unique sortie ids at each screening
    # stage; materialized/successful counters refer to complete repair attempts.
    r4p_calls: int = 0
    r4p_total_sec: float = 0.0
    r4p_index_lookup_sec: float = 0.0
    r4p_candidate_enumeration_sec: float = 0.0
    r4p_static_filter_sec: float = 0.0
    r4p_route_window_filter_sec: float = 0.0
    r4p_dynamic_filter_sec: float = 0.0
    r4p_proxy_score_sec: float = 0.0
    r4p_sort_or_topk_sec: float = 0.0
    r4p_candidate_materialize_sec: float = 0.0
    r4p_residual_repair_sec: float = 0.0
    r4p_other_sec: float = 0.0
    r4p_raw_candidates: int = 0
    r4p_after_physical_filter: int = 0
    r4p_after_route_window: int = 0
    r4p_after_static_filter: int = 0
    r4p_after_dynamic_filter: int = 0
    r4p_shortlisted_candidates: int = 0
    r4p_materialized_candidates: int = 0
    r4p_successful_insertions: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


R4RepairProfile = RepairSearchProfile


def solution_sortie_ids(solution: Solution, uav_count: int) -> tuple[int, ...]:
    return tuple(
        sortie_id
        for uav_id in range(uav_count)
        for sortie_id in solution.uav_sequences.get(uav_id, ())
    )


def covered_tasks(sortie_ids: Iterable[int], index: SortieIndex) -> frozenset[int]:
    return frozenset(
        task_id
        for sortie_id in sortie_ids
        for task_id in index.sortie_by_id[sortie_id].task_sequence
    )


def destroy_tasks_to_partial(
    solution: Solution,
    removed_task_ids: Iterable[int],
    index: SortieIndex,
    uav_count: int,
) -> PartialSolution:
    """Remove whole affected sorties while preserving unaffected owners/order."""
    removed = set(removed_task_ids)
    missing = set(removed)
    sequences: dict[int, list[int]] = {}
    for uav_id in range(uav_count):
        retained: list[int] = []
        for sortie_id in solution.uav_sequences.get(uav_id, ()):
            sortie_tasks = set(index.sortie_by_id[sortie_id].task_sequence)
            if sortie_tasks & removed:
                missing.update(sortie_tasks)
            else:
                retained.append(sortie_id)
        sequences[uav_id] = retained
    return PartialSolution(sequences, missing)


def remove_tasks(
    solution: Solution,
    task_ids: Iterable[int],
    index: SortieIndex,
    uav_count: int,
) -> tuple[int, ...]:
    """Compatibility facade returning flattened retained sortie ids."""
    partial = destroy_tasks_to_partial(solution, task_ids, index, uav_count)
    return tuple(
        sortie_id
        for uav_id in range(uav_count)
        for sortie_id in partial.uav_sequences.get(uav_id, ())
    )


def _route_positions(instance: Instance) -> dict[int, int]:
    return {support_id: rank for rank, support_id in enumerate(instance.usv_route[1:-1])}


def sequence_preserves_route_precedence(
    sequence: Sequence[int],
    instance: Instance,
    index: SortieIndex,
) -> bool:
    positions = _route_positions(instance)
    previous_recovery = -1
    for sortie_id in sequence:
        sortie = index.sortie_by_id[sortie_id]
        origin = positions.get(sortie.origin_support)
        recovery = positions.get(sortie.recovery_support)
        if origin is None or recovery is None or origin > recovery or origin < previous_recovery:
            return False
        previous_recovery = recovery
    return True


def route_feasible_insertion_window(
    sequence: Sequence[int],
    candidate: Sortie,
    index: SortieIndex,
    support_position: Mapping[int, int] | None = None,
    prepared_positions: RoutePositionArrays | None = None,
) -> range | tuple[int, ...]:
    """Return exactly the insertion positions allowed by fixed-route precedence."""
    positions = support_position or index.support_position
    candidate_origin = positions[candidate.origin_support]
    candidate_recovery = positions[candidate.recovery_support]
    if prepared_positions is None:
        prepared_positions = _route_position_arrays(sequence, index, positions)
    origin_positions, recovery_positions, monotone = prepared_positions
    if not monotone:
        return tuple(
            position
            for position in range(len(sequence) + 1)
            if (
                position == 0
                or recovery_positions[position - 1] <= candidate_origin
            )
            and (
                position == len(sequence)
                or candidate_recovery <= origin_positions[position]
            )
        )
    lo = bisect_left(origin_positions, candidate_recovery)
    hi = bisect_right(recovery_positions, candidate_origin)
    return range(lo, hi + 1) if lo <= hi else range(0)


def _route_position_arrays(
    sequence: Sequence[int],
    index: SortieIndex,
    support_position: Mapping[int, int],
) -> RoutePositionArrays:
    origin_positions = tuple(
        support_position[index.sortie_by_id[sortie_id].origin_support]
        for sortie_id in sequence
    )
    recovery_positions = tuple(
        support_position[index.sortie_by_id[sortie_id].recovery_support]
        for sortie_id in sequence
    )
    monotone = all(
        origin_positions[position] <= recovery_positions[position]
        and (
            position == 0
            or recovery_positions[position - 1] <= origin_positions[position]
        )
        for position in range(len(sequence))
    )
    return origin_positions, recovery_positions, monotone


def find_feasible_insertions(
    partial: PartialSolution,
    sortie: Sortie,
    instance: Instance,
    index: SortieIndex,
    profile: RepairSearchProfile | None = None,
    use_route_window: bool = False,
    route_positions_by_uav: Mapping[int, RoutePositionArrays] | None = None,
    r4p_detail: bool = False,
    r4_attribution: bool = False,
) -> tuple[Insertion, ...]:
    """Return every (UAV, position) satisfying route precedence."""
    started = perf_counter() if profile is not None else 0.0
    positions = index.support_position or _route_positions(instance)
    origin = positions[sortie.origin_support]
    recovery = positions[sortie.recovery_support]
    if origin > recovery:
        if profile is not None:
            insertion_elapsed = perf_counter() - started
            profile.insertion_time_sec += insertion_elapsed
            profile.repair_legality_check_sec += insertion_elapsed
        return ()
    insertions: list[Insertion] = []
    route_window_survived = False
    dynamic_filter_sec = 0.0
    for uav_id in range(instance.uav_count):
        sequence = partial.uav_sequences.get(uav_id, [])
        if use_route_window:
            window_started = perf_counter() if profile is not None else 0.0
            candidate_positions = route_feasible_insertion_window(
                sequence,
                sortie,
                index,
                positions,
                (
                    route_positions_by_uav.get(uav_id)
                    if route_positions_by_uav is not None
                    else None
                ),
            )
            if profile is not None:
                window_elapsed = perf_counter() - window_started
                profile.insertion_window_time_sec += window_elapsed
                if r4_attribution:
                    profile.r4_route_window_sec += window_elapsed
                if r4p_detail:
                    profile.r4p_route_window_filter_sec += window_elapsed
        else:
            candidate_positions = range(len(sequence) + 1)
        route_window_survived = route_window_survived or bool(candidate_positions)
        if profile is not None:
            raw_width = len(sequence) + 1
            legal_width = len(candidate_positions)
            profile.insertion_positions_raw += raw_width
            profile.insertion_positions_legal += legal_width
            profile.insertion_windows_total += 1
            profile.insertion_windows_empty += int(legal_width == 0)
            profile.insertion_window_width_sum += legal_width
            profile.insertion_window_width_max = max(
                profile.insertion_window_width_max,
                legal_width,
            )
        for position in candidate_positions:
            dynamic_started = (
                perf_counter() if profile is not None and r4p_detail else 0.0
            )
            if profile is not None:
                profile.insertion_positions_tested += 1
                profile.insertion_feasibility_checks += 1
            previous = index.sortie_by_id[sequence[position - 1]] if position > 0 else None
            following = index.sortie_by_id[sequence[position]] if position < len(sequence) else None
            if previous is not None and positions[previous.recovery_support] > origin:
                if profile is not None and r4p_detail:
                    dynamic_filter_sec += perf_counter() - dynamic_started
                continue
            if following is not None and recovery > positions[following.origin_support]:
                if profile is not None and r4p_detail:
                    dynamic_filter_sec += perf_counter() - dynamic_started
                continue
            insertions.append((uav_id, position))
            if profile is not None and r4p_detail:
                dynamic_filter_sec += perf_counter() - dynamic_started
    if profile is not None:
        insertion_elapsed = perf_counter() - started
        profile.insertion_time_sec += insertion_elapsed
        profile.repair_legality_check_sec += insertion_elapsed
        profile.repair_route_legal_options += len(insertions)
        if r4p_detail:
            profile.r4p_dynamic_filter_sec += dynamic_filter_sec
            profile.r4p_after_route_window += int(route_window_survived)
            profile.r4p_after_dynamic_filter += int(bool(insertions))
    return tuple(insertions)


def _candidate_cost(
    partial: PartialSolution,
    sortie: Sortie,
    insertion: Insertion,
    index: SortieIndex,
    strategy: RepairStrategy,
    prepared_load: float | None = None,
) -> tuple[float, ...]:
    uav_id, position = insertion
    load = (
        sum(
            index.sortie_by_id[value].nominal_duration_min
            for value in partial.uav_sequences.get(uav_id, ())
        )
        if prepared_load is None
        else prepared_load
    )
    task_count = len(sortie.task_sequence)
    base = sortie.nominal_duration_min / task_count
    if strategy == "augmentation":
        return (-float(task_count), base, 0.01 * load, sortie.nominal_energy_wh, float(uav_id), float(position), float(sortie.id))
    # Recovery-aware screening intentionally contains no same/different bonus.
    return (base, 0.01 * load, sortie.nominal_energy_wh, -float(task_count), float(uav_id), float(position), float(sortie.id))


def _candidate_insertions(
    partial: PartialSolution,
    instance: Instance,
    index: SortieIndex,
    strategy: RepairStrategy,
    profile: RepairSearchProfile | None = None,
    use_indexed_lookup: bool = False,
    use_route_window: bool = False,
    r4p_detail: bool = False,
    deadline_expired: Callable[[], bool] | None = None,
    r4_attribution: bool = False,
    select_first: bool = False,
    top_two_by_task: dict[int, list[RankedInsertion]] | None = None,
) -> list[RankedInsertion]:
    if select_first and top_two_by_task is not None:
        raise ValueError("select_first and top_two_by_task are mutually exclusive")
    if deadline_expired is None:
        deadline_expired = getattr(index, "deadline_expired", None)
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    r4_started = (
        perf_counter() if profile is not None and r4_attribution else 0.0
    )
    r4_phase_before = (
        (
            profile.r4_candidate_filter_sec,
            profile.r4_route_window_sec,
            profile.r4_proxy_scoring_sec,
            profile.r4_sort_select_sec,
        )
        if profile is not None and r4_attribution
        else ()
    )
    detail_started = (
        perf_counter() if profile is not None and r4p_detail else 0.0
    )
    phase_before = None
    if profile is not None and r4p_detail:
        phase_before = (
            profile.r4p_index_lookup_sec,
            profile.r4p_static_filter_sec,
            profile.r4p_route_window_filter_sec,
            profile.r4p_dynamic_filter_sec,
            profile.r4p_proxy_score_sec,
            profile.r4p_sort_or_topk_sec,
        )
    result: list[RankedInsertion] = []
    best: RankedInsertion | None = None
    ranked_items = 0
    selection_sec = 0.0
    if profile is not None:
        profile.sortie_lookup_calls += 1
        profile.sorties_raw += len(index.sortie_by_id)
    route_positions_by_uav = None
    enumeration_sec = 0.0
    if use_route_window:
        window_started = perf_counter() if profile is not None else 0.0
        positions = index.support_position or _route_positions(instance)
        route_positions_by_uav = {
            uav_id: _route_position_arrays(
                partial.uav_sequences.get(uav_id, ()),
                index,
                positions,
            )
            for uav_id in range(instance.uav_count)
        }
        if profile is not None:
            elapsed_window = perf_counter() - window_started
            enumeration_sec += elapsed_window
            profile.insertion_window_time_sec += elapsed_window
            profile.insertion_time_sec += elapsed_window
            if r4p_detail:
                profile.r4p_route_window_filter_sec += elapsed_window
            if r4_attribution:
                profile.r4_route_window_sec += elapsed_window
    lookup_started = perf_counter() if profile is not None else 0.0
    dense_context = (
        getattr(index, "context", None)
        if getattr(index, "uses_dense_repair_kernel", False)
        else None
    )
    if use_indexed_lookup:
        raw_ids: tuple[int, ...] | None = None
        if profile is not None and r4p_detail:
            raw_index = getattr(index, "base", index)
            raw_ids = raw_index.candidate_sortie_ids(partial.missing_tasks)
        candidate_qids_fn = getattr(index, "candidate_qids", None)
        if candidate_qids_fn is not None and dense_context is not None:
            candidate_qids = candidate_qids_fn(partial.missing_tasks)
            sortie_ids = tuple(
                dense_context.sortie_ids_by_qid[qid] for qid in candidate_qids
            )
            candidate_sorties = (
                (qid, dense_context.sortie_object[qid]) for qid in candidate_qids
            )
        else:
            sortie_ids = index.candidate_sortie_ids(partial.missing_tasks)
            candidate_sorties = (
                (None, index.sortie_by_id[sortie_id]) for sortie_id in sortie_ids
            )
        if profile is not None:
            profile.index_lookup_calls += 1
            profile.indexed_sorties_returned += len(sortie_ids)
            profile.repair_raw_options += len(sortie_ids)
            if r4_attribution:
                profile.r4_raw_candidate_count += len(sortie_ids)
            if r4p_detail:
                profile.r4p_raw_candidates += len(raw_ids or sortie_ids)
                profile.r4p_after_physical_filter += len(sortie_ids)
    else:
        candidate_sorties = ((None, sortie) for sortie in index.sortie_by_id.values())
        if profile is not None:
            profile.repair_raw_options += len(index.sortie_by_id)
            if r4_attribution:
                profile.r4_raw_candidate_count += len(index.sortie_by_id)
        if profile is not None and r4p_detail:
            profile.r4p_raw_candidates += len(index.sortie_by_id)
            profile.r4p_after_physical_filter += len(index.sortie_by_id)
    if profile is not None:
        lookup_elapsed = perf_counter() - lookup_started
        profile.repair_index_lookup_sec += lookup_elapsed
        if r4p_detail:
            profile.r4p_index_lookup_sec += lookup_elapsed
        if r4_attribution:
            profile.r4_candidate_filter_sec += lookup_elapsed
        lookup_started = perf_counter()
    uav_loads: tuple[float, ...] | None = None
    for qid, sortie in candidate_sorties:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        enumeration_started = perf_counter() if profile is not None else 0.0
        if profile is not None:
            profile.sorties_scanned += 1
        static_started = (
            perf_counter()
            if profile is not None and (r4p_detail or r4_attribution)
            else 0.0
        )
        if qid is not None:
            tasks = dense_context.task_signature[qid]
            statically_compatible = all(
                task_id in partial.missing_tasks for task_id in tasks
            )
        else:
            tasks = set(sortie.task_sequence)
            statically_compatible = bool(tasks) and tasks <= partial.missing_tasks
        if profile is not None and r4p_detail:
            profile.r4p_static_filter_sec += perf_counter() - static_started
        if profile is not None and r4_attribution:
            profile.r4_candidate_filter_sec += perf_counter() - static_started
        if not statically_compatible:
            if profile is not None:
                enumeration_sec += perf_counter() - enumeration_started
            continue
        if profile is not None:
            profile.repair_static_feasible_options += 1
            enumeration_sec += perf_counter() - enumeration_started
        if profile is not None and r4p_detail:
            profile.r4p_after_static_filter += 1
        if profile is not None and r4_attribution:
            profile.r4_static_feasible_count += 1
        if profile is not None:
            profile.sortie_lookup_time_sec += perf_counter() - lookup_started
        insertions = find_feasible_insertions(
            partial,
            sortie,
            instance,
            index,
            profile,
            use_route_window,
            route_positions_by_uav,
            r4p_detail,
            r4_attribution,
        )
        if insertions and uav_loads is None:
            proxy_cache_started = perf_counter() if profile is not None else 0.0
            uav_loads = tuple(
                sum(
                    index.sortie_by_id[sortie_id].nominal_duration_min
                    for sortie_id in partial.uav_sequences.get(uav_id, ())
                )
                for uav_id in range(instance.uav_count)
            )
            if profile is not None:
                proxy_cache_elapsed = perf_counter() - proxy_cache_started
                profile.repair_proxy_score_sec += proxy_cache_elapsed
                if r4p_detail:
                    profile.r4p_proxy_score_sec += proxy_cache_elapsed
                if r4_attribution:
                    profile.r4_proxy_scoring_sec += proxy_cache_elapsed
        for insertion in insertions:
            proxy_started = perf_counter() if profile is not None else 0.0
            assert uav_loads is not None
            if qid is None:
                cost = _candidate_cost(
                    partial,
                    sortie,
                    insertion,
                    index,
                    strategy,
                    uav_loads[insertion[0]],
                )
            else:
                uav_id, position = insertion
                task_count = dense_context.task_count[qid]
                base = dense_context.proxy_base[qid]
                load = uav_loads[uav_id]
                energy = dense_context.nominal_energy[qid]
                sortie_id = dense_context.sortie_ids_by_qid[qid]
                if strategy == "augmentation":
                    cost = (
                        -float(task_count),
                        base,
                        0.01 * load,
                        energy,
                        float(uav_id),
                        float(position),
                        float(sortie_id),
                    )
                else:
                    cost = (
                        base,
                        0.01 * load,
                        energy,
                        -float(task_count),
                        float(uav_id),
                        float(position),
                        float(sortie_id),
                    )
            if profile is not None:
                proxy_elapsed = perf_counter() - proxy_started
                profile.repair_proxy_score_sec += proxy_elapsed
                profile.repair_proxy_scored_options += 1
                if r4p_detail:
                    profile.r4p_proxy_score_sec += proxy_elapsed
                if r4_attribution:
                    profile.r4_proxy_scoring_sec += proxy_elapsed
            candidate = (cost, sortie, insertion)
            ranked_items += 1
            if top_two_by_task is not None:
                selection_started = perf_counter() if profile is not None else 0.0
                for task_id in sortie.task_sequence:
                    retained = top_two_by_task.get(task_id)
                    if retained is None:
                        continue
                    if profile is not None:
                        profile.regret_option_evaluations += 1
                    if not retained or cost < retained[0][0]:
                        retained.insert(0, candidate)
                    elif len(retained) < 2 or cost < retained[1][0]:
                        retained.insert(1, candidate)
                    if len(retained) > 2:
                        retained.pop()
                if profile is not None:
                    selection_sec += perf_counter() - selection_started
            elif select_first:
                selection_started = perf_counter() if profile is not None else 0.0
                if best is None or cost < best[0]:
                    best = candidate
                if profile is not None:
                    selection_sec += perf_counter() - selection_started
            else:
                result.append(candidate)
        if profile is not None:
            lookup_started = perf_counter()
    if profile is not None:
        profile.sortie_lookup_time_sec += perf_counter() - lookup_started
        profile.repair_option_enumeration_sec += enumeration_sec
        profile.ranking_items += ranked_items
        sorting_started = (
            perf_counter()
            if not select_first and top_two_by_task is None
            else 0.0
        )
    if top_two_by_task is not None:
        result = []
    elif select_first:
        result = [best] if best is not None else []
    else:
        result.sort(key=lambda value: value[0])
    if profile is not None:
        sorting_elapsed = (
            selection_sec
            if select_first or top_two_by_task is not None
            else perf_counter() - sorting_started
        )
        profile.sorting_time_sec += sorting_elapsed
        profile.repair_rank_select_sec += sorting_elapsed
        if r4p_detail:
            profile.r4p_sort_or_topk_sec += sorting_elapsed
            assert phase_before is not None
            phase_after = (
                profile.r4p_index_lookup_sec,
                profile.r4p_static_filter_sec,
                profile.r4p_route_window_filter_sec,
                profile.r4p_dynamic_filter_sec,
                profile.r4p_proxy_score_sec,
                profile.r4p_sort_or_topk_sec,
            )
            attributed = sum(
                after - before
                for before, after in zip(phase_before, phase_after)
            )
            profile.r4p_candidate_enumeration_sec += max(
                0.0, perf_counter() - detail_started - attributed
            )
        if r4_attribution:
            profile.r4_sort_select_sec += sorting_elapsed
            profile.r4_option_count += len(result)
            r4_phase_after = (
                profile.r4_candidate_filter_sec,
                profile.r4_route_window_sec,
                profile.r4_proxy_scoring_sec,
                profile.r4_sort_select_sec,
            )
            r4_attributed = sum(
                after - before
                for before, after in zip(r4_phase_before, r4_phase_after)
            )
            profile.r4_option_enumeration_sec += max(
                0.0, perf_counter() - r4_started - r4_attributed
            )
    return result


def _regret2_candidate_top_two(
    partial: PartialSolution,
    instance: Instance,
    index: SortieIndex,
    profile: RepairSearchProfile | None = None,
    use_indexed_lookup: bool = False,
    use_route_window: bool = False,
    deadline_expired: Callable[[], bool] | None = None,
) -> dict[int, tuple[RankedInsertion, ...]]:
    """Retain the stable first two R2 candidates for each missing task."""
    retained: dict[int, list[RankedInsertion]] = {
        task_id: [] for task_id in partial.missing_tasks
    }
    _candidate_insertions(
        partial,
        instance,
        index,
        "regret2",
        profile,
        use_indexed_lookup,
        use_route_window,
        deadline_expired=deadline_expired,
        top_two_by_task=retained,
    )
    return {
        task_id: tuple(candidates)
        for task_id, candidates in retained.items()
    }


def _use_r4_numba_kernel(
    index: SortieIndex,
    backend: R4KernelBackend,
) -> bool:
    if backend not in {"auto", "python", "numba"}:
        raise ValueError(f"unknown R4 kernel backend {backend}")
    if backend == "python":
        return False
    eligible = r4_numba_eligible(index)
    if backend == "numba" and not eligible:
        raise ValueError(
            "R4 Numba V1 requires the canonical index built with support order"
        )
    return eligible


def _use_repair_numba_kernel(
    index: SortieIndex,
    backend: RepairKernelBackend,
) -> bool:
    if backend not in {"auto", "python", "numba"}:
        raise ValueError(f"unknown shared repair kernel backend {backend}")
    if backend == "python":
        return False
    eligible = repair_numba_eligible(index)
    if backend == "numba" and not eligible:
        raise ValueError(
            "shared Numba repair requires the canonical index built with support order"
        )
    return eligible


def _record_numba_r4_work(
    profile: RepairSearchProfile,
    index: SortieIndex,
    result: R4KernelResult,
    output_sec: float,
) -> None:
    profile.sortie_lookup_calls += 1
    profile.index_lookup_calls += 1
    profile.sorties_raw += len(index.sortie_by_id)
    profile.indexed_sorties_returned += result.raw_candidates
    profile.sorties_scanned += result.raw_candidates
    profile.sortie_lookup_time_sec += result.input_candidate_sec
    profile.repair_raw_options += result.raw_candidates
    profile.repair_static_feasible_options += result.static_feasible
    profile.repair_route_legal_options += result.options
    profile.repair_proxy_scored_options += result.options
    profile.ranking_items += result.options

    profile.insertion_positions_raw += result.insertion_positions_raw
    profile.insertion_positions_legal += result.options
    profile.insertion_positions_tested += result.options
    profile.insertion_feasibility_checks += result.options
    profile.insertion_windows_total += result.insertion_windows_total
    profile.insertion_windows_empty += result.insertion_windows_empty
    profile.insertion_window_width_sum += result.options
    profile.insertion_window_width_max = max(
        profile.insertion_window_width_max,
        result.insertion_window_width_max,
    )
    profile.insertion_time_sec += result.input_route_sec + result.kernel_sec
    profile.insertion_window_time_sec += result.input_route_sec
    profile.sorting_time_sec += output_sec

    # The V1 kernel is deliberately fused. Exact wall time therefore lives in
    # option-enumeration while the dedicated Numba fields expose the new phase
    # boundary; assigning synthetic sub-timings would make the profile lie.
    profile.repair_index_lookup_sec += result.input_candidate_sec
    profile.repair_legality_check_sec += result.input_route_sec
    profile.repair_option_enumeration_sec += result.kernel_sec
    profile.repair_rank_select_sec += output_sec
    profile.r4_candidate_filter_sec += result.input_candidate_sec
    profile.r4_route_window_sec += result.input_route_sec
    profile.r4_option_enumeration_sec += result.kernel_sec
    profile.r4_sort_select_sec += output_sec
    profile.r4_raw_candidate_count += result.raw_candidates
    profile.r4_static_feasible_count += result.static_feasible
    profile.r4_option_count += result.options
    profile.r4_numba_calls += 1
    profile.r4_numba_input_sec += (
        result.input_candidate_sec + result.input_route_sec
    )
    profile.r4_numba_kernel_sec += result.kernel_sec
    profile.r4_numba_output_sec += output_sec


def _record_numba_repair_work(
    profile: RepairSearchProfile,
    index: SortieIndex,
    result: RepairKernelResult,
    output_sec: float,
) -> None:
    profile.sortie_lookup_calls += 1
    profile.index_lookup_calls += 1
    profile.sorties_raw += len(index.sortie_by_id)
    profile.indexed_sorties_returned += result.raw_candidates
    profile.sorties_scanned += result.raw_candidates
    profile.sortie_lookup_time_sec += result.input_candidate_sec
    profile.repair_raw_options += result.raw_candidates
    profile.repair_static_feasible_options += result.static_feasible
    profile.repair_route_legal_options += result.options
    profile.repair_proxy_scored_options += result.options
    profile.ranking_items += result.options
    profile.regret_option_evaluations += result.regret_memberships

    profile.insertion_positions_raw += result.insertion_positions_raw
    profile.insertion_positions_legal += result.options
    profile.insertion_positions_tested += result.options
    profile.insertion_feasibility_checks += result.options
    profile.insertion_windows_total += result.insertion_windows_total
    profile.insertion_windows_empty += result.insertion_windows_empty
    profile.insertion_window_width_sum += result.options
    profile.insertion_window_width_max = max(
        profile.insertion_window_width_max,
        result.insertion_window_width_max,
    )
    profile.insertion_time_sec += result.input_route_sec + result.kernel_sec
    profile.insertion_window_time_sec += result.input_route_sec
    profile.sorting_time_sec += output_sec

    # Filtering, legality, scoring, and bounded selection are fused. Keep the
    # measured kernel exclusive instead of inventing synthetic sub-phase time.
    profile.repair_index_lookup_sec += result.input_candidate_sec
    profile.repair_legality_check_sec += result.input_route_sec
    profile.repair_option_enumeration_sec += result.kernel_sec
    profile.repair_rank_select_sec += output_sec
    profile.repair_numba_calls += 1
    profile.repair_numba_input_sec += (
        result.input_candidate_sec + result.input_route_sec
    )
    profile.repair_numba_kernel_sec += result.kernel_sec
    profile.repair_numba_output_sec += output_sec


def _numba_repair_best(
    partial: PartialSolution,
    instance: Instance,
    index: SortieIndex,
    strategy: Literal["greedy", "augmentation"],
    profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    workspace: RepairNumbaWorkspace | None = None,
) -> tuple[list[RankedInsertion], RepairKernelResult]:
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    result = rank_repair_options(
        partial,
        index,
        instance.uav_count,
        strategy=strategy,
        workspace=workspace,
    )
    output_started = perf_counter()
    numeric = index.numeric_repair_index
    candidates = [
        (
            tuple(float(value) for value in result.best_scores[row]),
            index.sortie_by_id[
                int(numeric.sortie_ids[int(result.best_qids[row])])
            ],
            (int(result.best_uav_ids[row]), int(result.best_positions[row])),
        )
        for row in range(len(result.best_qids))
    ]
    output_sec = perf_counter() - output_started
    if profile is not None:
        _record_numba_repair_work(profile, index, result, output_sec)
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    return candidates, result


def _numba_regret2_candidate_top_two(
    partial: PartialSolution,
    instance: Instance,
    index: SortieIndex,
    profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    workspace: RepairNumbaWorkspace | None = None,
) -> tuple[dict[int, tuple[RankedInsertion, ...]], RepairKernelResult]:
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    result = rank_repair_options(
        partial,
        index,
        instance.uav_count,
        strategy="regret2",
        workspace=workspace,
    )
    output_started = perf_counter()
    numeric = index.numeric_repair_index
    retained: dict[int, tuple[RankedInsertion, ...]] = {}
    for row, task_slot in enumerate(result.missing_slots):
        task_id = int(numeric.task_ids[int(task_slot)])
        candidates: list[RankedInsertion] = []
        for rank in range(2):
            qid = int(result.task_qids[row, rank])
            if qid < 0:
                continue
            candidates.append(
                (
                    tuple(
                        float(value)
                        for value in result.task_scores[row, rank]
                    ),
                    index.sortie_by_id[int(numeric.sortie_ids[qid])],
                    (
                        int(result.task_uav_ids[row, rank]),
                        int(result.task_positions[row, rank]),
                    ),
                )
            )
        retained[task_id] = tuple(candidates)
    output_sec = perf_counter() - output_started
    if profile is not None:
        _record_numba_repair_work(profile, index, result, output_sec)
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    return retained, result


def _numba_r4_candidates(
    partial: PartialSolution,
    instance: Instance,
    index: SortieIndex,
    *,
    augmentation: bool,
    top_m: int,
    same_quota: int = 0,
    different_quota: int = 0,
    profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
) -> tuple[
    list[tuple[tuple[float, ...], Sortie, Insertion]],
    R4KernelResult,
]:
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    result = rank_r4_options(
        partial,
        index,
        instance.uav_count,
        augmentation=augmentation,
        top_m=top_m,
        same_quota=same_quota,
        different_quota=different_quota,
    )
    output_started = perf_counter()
    numeric = index.numeric_repair_index
    candidates = [
        (
            tuple(float(value) for value in result.scores[row]),
            index.sortie_by_id[int(numeric.sortie_ids[int(result.qids[row])])],
            (int(result.uav_ids[row]), int(result.positions[row])),
        )
        for row in range(len(result.qids))
    ]
    output_sec = perf_counter() - output_started
    if profile is not None:
        _record_numba_r4_work(profile, index, result, output_sec)
    if deadline_expired is not None and deadline_expired():
        raise RepairTimeBudgetExceeded
    return candidates, result


def _insert(
    partial: PartialSolution,
    sortie: Sortie,
    insertion: Insertion,
    profile: RepairSearchProfile | None = None,
) -> PartialSolution:
    started = perf_counter() if profile is not None else 0.0
    result = partial.copy()
    if profile is not None:
        profile.solution_copy_calls += 1
        profile.candidate_solution_builds += 1
    uav_id, position = insertion
    result.uav_sequences.setdefault(uav_id, []).insert(position, sortie.id)
    result.missing_tasks.difference_update(sortie.task_sequence)
    if profile is not None:
        elapsed = perf_counter() - started
        profile.candidate_build_time_sec += elapsed
        profile.repair_apply_insertion_sec += elapsed
    return result


def _complete_partial(
    partial: PartialSolution,
    instance: Instance,
    index: SortieIndex,
    strategy: RepairStrategy,
    profile: RepairSearchProfile | None = None,
    use_indexed_lookup: bool = False,
    use_route_window: bool = False,
    deadline_expired: Callable[[], bool] | None = None,
    r4_attribution: bool = False,
    r4_kernel_backend: R4KernelBackend = "auto",
    repair_kernel_backend: RepairKernelBackend = "python",
) -> PartialSolution:
    started = perf_counter() if profile is not None else 0.0
    phase_before = (
        (
            profile.repair_index_lookup_sec,
            profile.repair_option_enumeration_sec,
            profile.repair_legality_check_sec,
            profile.repair_proxy_score_sec,
            profile.repair_rank_select_sec,
            profile.repair_apply_insertion_sec,
        )
        if profile is not None
        else ()
    )
    state = partial.copy()
    if profile is not None:
        profile.solution_copy_calls += 1
        profile.candidate_build_time_sec += perf_counter() - started
    shared_numba_enabled = (
        not r4_attribution
        and strategy in {"greedy", "regret2", "augmentation"}
        and _use_repair_numba_kernel(index, repair_kernel_backend)
    )
    repair_workspace = (
        create_repair_numba_workspace(index, instance.uav_count)
        if shared_numba_enabled
        else None
    )
    while state.missing_tasks:
        active_deadline = deadline_expired or getattr(index, "deadline_expired", None)
        if active_deadline is not None and active_deadline():
            raise RepairTimeBudgetExceeded
        regret_top_two: dict[int, tuple[RankedInsertion, ...]] | None = None
        if (
            r4_attribution
            and strategy == "augmentation"
            and _use_r4_numba_kernel(index, r4_kernel_backend)
        ):
            candidates, _ = _numba_r4_candidates(
                state,
                instance,
                index,
                augmentation=True,
                top_m=1,
                profile=profile,
                deadline_expired=active_deadline,
            )
        elif strategy == "regret2" and shared_numba_enabled:
            regret_top_two, _ = _numba_regret2_candidate_top_two(
                state,
                instance,
                index,
                profile,
                active_deadline,
                repair_workspace,
            )
            candidates = []
        elif strategy == "regret2":
            regret_top_two = _regret2_candidate_top_two(
                state,
                instance,
                index,
                profile,
                use_indexed_lookup,
                use_route_window,
                active_deadline,
            )
            candidates = []
        elif shared_numba_enabled:
            candidates, _ = _numba_repair_best(
                state,
                instance,
                index,
                strategy,
                profile,
                active_deadline,
                repair_workspace,
            )
        else:
            candidates = _candidate_insertions(
                state,
                instance,
                index,
                strategy,
                profile,
                use_indexed_lookup,
                use_route_window,
                deadline_expired=active_deadline,
                r4_attribution=r4_attribution,
                select_first=(
                    strategy == "greedy"
                    # R4's augmentation residual path stays frozen.
                    or (strategy == "augmentation" and not r4_attribution)
                ),
            )
        if regret_top_two is None and not candidates:
            raise RuntimeError(f"repair cannot cover remaining tasks {sorted(state.missing_tasks)}")
        if strategy == "regret2":
            assert regret_top_two is not None
            regret_started = perf_counter() if profile is not None else 0.0
            choices: list[tuple[float, tuple[float, ...], Sortie, Insertion]] = []
            for task_id in sorted(state.missing_tasks):
                task_candidates = regret_top_two[task_id]
                if not task_candidates:
                    continue
                best = task_candidates[0]
                if profile is not None:
                    profile.regret_first_best_updates += 1
                first_cost = best[0][0]
                second_cost = task_candidates[1][0][0] if len(task_candidates) > 1 else first_cost + 1e6
                if profile is not None and len(task_candidates) > 1:
                    profile.regret_second_best_updates += 1
                choices.append((second_cost - first_cost, best[0], best[1], best[2]))
            if not choices:
                raise RuntimeError(f"regret repair cannot cover remaining tasks {sorted(state.missing_tasks)}")
            _, _, sortie, insertion = max(
                choices,
                key=lambda value: (value[0], tuple(-item for item in value[1])),
            )
            if profile is not None:
                profile.repair_rank_select_sec += perf_counter() - regret_started
        else:
            _, sortie, insertion = candidates[0]
        state = _insert(state, sortie, insertion, profile)
    if profile is not None:
        phase_after = (
            profile.repair_index_lookup_sec,
            profile.repair_option_enumeration_sec,
            profile.repair_legality_check_sec,
            profile.repair_proxy_score_sec,
            profile.repair_rank_select_sec,
            profile.repair_apply_insertion_sec,
        )
        attributed = sum(
            after - before for before, after in zip(phase_before, phase_after)
        )
        profile.repair_residual_loop_sec += max(
            0.0, perf_counter() - started - attributed
        )
    return state


def repair_candidates(
    instance: Instance,
    partial: PartialSolution,
    index: SortieIndex | None = None,
    *,
    strategy: RepairStrategy = "greedy",
    recovery_top_m: int = 4,
    recovery_same_quota: int = 2,
    recovery_diff_quota: int = 2,
    repair_profile: RepairSearchProfile | None = None,
    r4_profile: RepairSearchProfile | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    r4_kernel_backend: R4KernelBackend = "auto",
    repair_kernel_backend: RepairKernelBackend = "python",
) -> tuple[Solution, ...]:
    if repair_profile is not None and r4_profile is not None:
        raise ValueError("provide only one repair profile")
    profile = repair_profile if repair_profile is not None else r4_profile
    active_index = index or build_sortie_index(
        instance.sortie_pool, instance.usv_route[1:-1]
    )
    if strategy != "recovery_aware":
        return (
            _complete_partial(
                partial,
                instance,
                active_index,
                strategy,
                profile,
                strategy in INDEXED_LOOKUP_REPAIR_STRATEGIES,
                strategy in ROUTE_WINDOW_REPAIR_STRATEGIES,
                deadline_expired,
                repair_kernel_backend=repair_kernel_backend,
            ).to_solution(),
        )
    if not partial.missing_tasks:
        return (partial.to_solution(),)

    use_numba = _use_r4_numba_kernel(active_index, r4_kernel_backend)
    if use_numba:
        seeds, kernel_result = _numba_r4_candidates(
            partial,
            instance,
            active_index,
            augmentation=False,
            top_m=recovery_top_m,
            same_quota=recovery_same_quota,
            different_quota=recovery_diff_quota,
            profile=profile,
            deadline_expired=deadline_expired,
        )
        if profile is not None:
            profile.recovery_positions_considered += kernel_result.options
            profile.same_recovery_candidates += kernel_result.same_options
            profile.different_recovery_candidates += kernel_result.different_options
            profile.top_m_retained += len(seeds)
            profile.recovery_filter_time_sec += kernel_result.kernel_sec
    else:
        screened = _candidate_insertions(
            partial,
            instance,
            active_index,
            "recovery_aware",
            profile,
            True,
            True,
            deadline_expired=deadline_expired,
            r4_attribution=True,
        )
        recovery_started = perf_counter() if profile is not None else 0.0
        same = [value for value in screened if value[1].origin_support == value[1].recovery_support]
        different = [value for value in screened if value[1].origin_support != value[1].recovery_support]
        if profile is not None:
            profile.recovery_positions_considered += len(screened)
            profile.same_recovery_candidates += len(same)
            profile.different_recovery_candidates += len(different)
        seeds = [*same[:recovery_same_quota], *different[:recovery_diff_quota]]
        selected_keys = {(value[1].id, value[2]) for value in seeds}
        for value in screened:
            if len(seeds) >= recovery_top_m:
                break
            key = (value[1].id, value[2])
            if key not in selected_keys:
                seeds.append(value)
                selected_keys.add(key)
        if profile is not None:
            profile.top_m_retained += len(seeds[:recovery_top_m])
            recovery_elapsed = perf_counter() - recovery_started
            profile.recovery_filter_time_sec += recovery_elapsed
            profile.r4_sort_select_sec += recovery_elapsed
    if not seeds:
        raise RuntimeError(f"recovery-aware repair cannot cover tasks {sorted(partial.missing_tasks)}")

    solutions: list[Solution] = []
    seen: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    for _, sortie, insertion in seeds[:recovery_top_m]:
        active_deadline = deadline_expired or getattr(
            active_index, "deadline_expired", None
        )
        if active_deadline is not None and active_deadline():
            raise RepairTimeBudgetExceeded
        try:
            completed = _complete_partial(
                _insert(partial, sortie, insertion, profile),
                instance,
                active_index,
                "augmentation",
                profile,
                True,
                True,
                active_deadline,
                r4_attribution=True,
                r4_kernel_backend=r4_kernel_backend,
            ).to_solution()
        except RuntimeError:
            continue
        key = tuple(
            (uav_id, tuple(completed.uav_sequences.get(uav_id, ())))
            for uav_id in range(instance.uav_count)
        )
        if key not in seen:
            seen.add(key)
            solutions.append(completed)
    if not solutions:
        raise RuntimeError("recovery-aware beam produced no complete route-compatible solution")
    return tuple(solutions)


def assign_sorties(
    instance: Instance,
    sortie_ids: Sequence[int],
    index: SortieIndex,
) -> Solution:
    """Deterministic global assignment used only for initial/exact construction."""
    position = _route_positions(instance)
    ordered = sorted(
        sortie_ids,
        key=lambda identifier: (
            position[index.sortie_by_id[identifier].origin_support],
            position[index.sortie_by_id[identifier].recovery_support],
            identifier,
        ),
    )
    sequences = {uav_id: [] for uav_id in range(instance.uav_count)}
    last_recovery = [-1] * instance.uav_count
    load = [0.0] * instance.uav_count
    for sortie_id in ordered:
        sortie = index.sortie_by_id[sortie_id]
        origin_position = position[sortie.origin_support]
        recovery_position = position[sortie.recovery_support]
        eligible = [uav_id for uav_id in range(instance.uav_count) if last_recovery[uav_id] <= origin_position]
        if not eligible:
            raise RuntimeError(f"no UAV can accept sortie {sortie_id} without violating route precedence")
        owner = min(eligible, key=lambda uav_id: (load[uav_id], len(sequences[uav_id]), uav_id))
        sequences[owner].append(sortie_id)
        last_recovery[owner] = recovery_position
        load[owner] += sortie.nominal_duration_min
    return Solution(sequences)


def repair_solution(
    instance: Instance,
    partial_or_retained: PartialSolution | Sequence[int],
    index: SortieIndex | None = None,
    *,
    strategy: RepairStrategy = "greedy",
    deadline_expired: Callable[[], bool] | None = None,
) -> Solution:
    """Compatibility single-candidate facade over local insertion repair."""
    active_index = index or build_sortie_index(
        instance.sortie_pool, instance.usv_route[1:-1]
    )
    if isinstance(partial_or_retained, PartialSolution):
        partial = partial_or_retained
    else:
        retained_solution = assign_sorties(instance, tuple(partial_or_retained), active_index)
        occupied = covered_tasks(partial_or_retained, active_index)
        partial = PartialSolution(
            {uav_id: list(retained_solution.uav_sequences.get(uav_id, ())) for uav_id in range(instance.uav_count)},
            {task.id for task in instance.tasks} - set(occupied),
        )
    return repair_candidates(
        instance,
        partial,
        active_index,
        strategy=strategy,
        deadline_expired=deadline_expired,
    )[0]


def construct_initial_solution(
    instance: Instance,
    index: SortieIndex | None = None,
    *,
    max_recovery_span: int | None = None,
) -> Solution:
    active_index = index or build_sortie_index(
        instance.sortie_pool, instance.usv_route[1:-1]
    )
    task_ids = tuple(task.id for task in instance.tasks)
    if max_recovery_span is None:
        # Preserve the frozen S4C constructor byte-for-byte along its legacy path.
        same_recovery_sorties = tuple(
            sortie
            for sortie in active_index.sortie_by_id.values()
            if sortie.origin_support == sortie.recovery_support
        )
        if same_recovery_sorties:
            same_recovery_index = build_sortie_index(
                same_recovery_sorties, instance.usv_route[1:-1]
            )
            if not same_recovery_index.uncovered_tasks(task_ids):
                active_index = same_recovery_index
    else:
        base_index = active_index.base if hasattr(active_index, "base") else active_index
        active_index = base_index.view(max_span=max_recovery_span)
        same_recovery_index = base_index.view(max_span=0)
        if not same_recovery_index.uncovered_tasks(task_ids):
            active_index = same_recovery_index
    empty = PartialSolution(
        {uav_id: [] for uav_id in range(instance.uav_count)},
        set(task_ids),
    )
    return repair_candidates(instance, empty, active_index, strategy="augmentation")[0]
