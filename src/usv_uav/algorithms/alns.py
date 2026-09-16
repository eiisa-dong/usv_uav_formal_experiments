from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil, exp
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Literal

from numpy.random import SeedSequence

from usv_uav.algorithms.alns_config import (
    ALNSConfig,
    RouteGuidedInitializationConfig,
    load_alns_config,
)
from usv_uav.algorithms.base import (
    AlgorithmResult,
    ConvergencePoint,
    EvaluationBudget,
    Solver,
    SolverBudget,
)
from usv_uav.algorithms.construction import (
    RepairSearchProfile,
    RepairTimeBudgetExceeded,
    construct_initial_solution,
    destroy_tasks_to_partial,
    find_feasible_insertions,
    fixed_route_sail_prefix,
    solution_sortie_ids,
    synchronization_candidate_key,
)
from usv_uav.algorithms.operators import DESTROY_OPERATORS, LOCAL_OPERATORS, REPAIR_OPERATORS
from usv_uav.algorithms.parallel_runtime import (
    ParallelRuntimeConfig,
    evaluate_candidates_threaded,
    thread_local_evaluate,
)
from usv_uav.algorithms.r4_numba import (
    R4KernelBackend,
    compiled_signature_count,
    warmup_r4_numba_kernel,
)
from usv_uav.algorithms.repair_numba import (
    RepairKernelBackend,
    compiled_signature_count as repair_compiled_signature_count,
    warmup_repair_numba_kernel,
)
from usv_uav.algorithms.route_guided.context import RouteSearchContext
from usv_uav.algorithms.route_guided.controller import (
    DescentMomentumEscapeController,
    RouteSearchController,
)
from usv_uav.algorithms.route_guided.destroy import RouteGuidedDestroyEngine
from usv_uav.algorithms.route_guided.diagnostics import (
    EscapeSearchDiagnostics,
    NormalOnlySearchProfile,
    RouteSearchDiagnostics,
)
from usv_uav.algorithms.route_guided.initialization import (
    initialize_route_guided_solution,
)
from usv_uav.algorithms.route_guided.models import (
    CandidateClass,
    EscapeDecision,
    SearchAction,
)
from usv_uav.algorithms.route_guided.repair import ProgressiveRepairEngine
from usv_uav.core.models import Instance, Sortie
from usv_uav.core.solution import Solution
from usv_uav.preprocessing.sortie_index import SortieIndex, build_sortie_index
from usv_uav.scheduling.evaluator import EvaluationResult, Evaluator


MISSING_SIZE_BINS = ("1-3", "4-6", "7-12", "13-24", "25+")
REPAIR_WORKLOAD_COUNTER_FIELDS = (
    "repair_raw_options",
    "repair_route_legal_options",
    "repair_static_feasible_options",
    "repair_proxy_scored_options",
    "repair_materialized_options",
    "repair_decoder_options",
    "regret_option_evaluations",
    "regret_first_best_updates",
    "regret_second_best_updates",
)
REPAIR_WORKLOAD_TIMER_FIELDS = (
    "repair_index_lookup_sec",
    "repair_option_enumeration_sec",
    "repair_legality_check_sec",
    "repair_proxy_score_sec",
    "repair_rank_select_sec",
    "repair_apply_insertion_sec",
    "repair_residual_loop_sec",
    "repair_bookkeeping_sec",
)
R4_PROFILE_GENERATION_FIELDS = (
    "r4_candidate_filter_sec",
    "r4_route_window_sec",
    "r4_option_enumeration_sec",
    "r4_proxy_scoring_sec",
    "r4_sort_select_sec",
)
RepairProfileMode = Literal["summary", "full"]


def _missing_size_bin(size: int) -> str:
    if size <= 3:
        return "1-3"
    if size <= 6:
        return "4-6"
    if size <= 12:
        return "7-12"
    if size <= 24:
        return "13-24"
    return "25+"


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, ceil(percentile * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


def _integer_distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": max(values, default=0),
    }


def _record_r4_generation_profile(
    profile: RepairSearchProfile,
    before: dict[str, float],
    elapsed_sec: float,
) -> None:
    """Reconcile one R4 repair call into exclusive generation phases."""

    attributed = sum(
        float(getattr(profile, field)) - before[field]
        for field in R4_PROFILE_GENERATION_FIELDS
    )
    profile.r4_profile_calls += 1
    profile.r4_profile_total_sec += elapsed_sec
    profile.r4_materialization_sec += max(0.0, elapsed_sec - attributed)


def _default_config(name: str) -> ALNSConfig:
    root = Path(__file__).resolve().parents[3]
    return load_alns_config(root / "configs" / "algorithms" / name)


def _weighted_choice(names: tuple[str, ...], weights: dict[str, float], rng: random.Random) -> str:
    total = sum(max(0.01, weights[name]) for name in names)
    draw = rng.random() * total
    cursor = 0.0
    for name in names:
        cursor += max(0.01, weights[name])
        if draw <= cursor:
            return name
    return names[-1]


def update_segment_weights(
    weights: dict[str, float],
    calls: dict[str, int],
    rewards: dict[str, float],
    reaction_factor: float,
) -> None:
    """Apply one ALNS segment update; unused operators retain their weight."""
    for name in weights:
        if calls.get(name, 0) <= 0:
            continue
        score = rewards.get(name, 0.0) / calls[name]
        weights[name] = (1.0 - reaction_factor) * weights[name] + reaction_factor * score


def _accept_search_candidate(
    *,
    action: SearchAction,
    delta: float,
    temperature: float,
    rng: random.Random,
) -> tuple[bool, float, float | None]:
    """Apply frozen SA except for V3.2's forced global relocation."""
    if action is SearchAction.GLOBAL_ESCAPE:
        return True, 1.0, None
    if delta <= 0:
        return True, 1.0, None
    probability = exp(-delta / max(temperature, 1e-12))
    draw = rng.random()
    return draw < probability, probability, draw


def _sample_legacy_removal_count(
    *,
    n_tasks: int,
    config: ALNSConfig,
    rng: random.Random,
) -> int:
    removal_fraction = rng.uniform(config.removal_fraction_min, config.removal_fraction_max)
    return max(1, min(n_tasks, ceil(n_tasks * removal_fraction)))


def _choose_destroy_target(
    *,
    n_tasks: int,
    stagnation: int,
    config: ALNSConfig,
    rng: random.Random,
    use_multiscale: bool,
) -> tuple[int, str]:
    if not use_multiscale:
        return (
            _sample_legacy_removal_count(n_tasks=n_tasks, config=config, rng=rng),
            "legacy",
        )

    multiscale = config.multiscale_destroy
    if n_tasks <= multiscale.preserve_legacy_up_to_j:
        return (
            _sample_legacy_removal_count(n_tasks=n_tasks, config=config, rng=rng),
            "legacy",
        )

    if stagnation > 0 and stagnation % multiscale.macro_every_stagnation == 0:
        return (
            _sample_legacy_removal_count(n_tasks=n_tasks, config=config, rng=rng),
            "macro",
        )

    if stagnation >= multiscale.medium_after_stagnation:
        lo = min(multiscale.medium_min_tasks, n_tasks)
        hi = min(multiscale.medium_max_tasks, n_tasks)
        return rng.randint(lo, hi), "medium"

    lo = min(multiscale.micro_min_tasks, n_tasks)
    hi = min(multiscale.micro_max_tasks, n_tasks)
    return rng.randint(lo, hi), "micro"


LEGACY_DESTROY_NAMES = tuple(
    name for name in DESTROY_OPERATORS if name != "D6_route_corridor"
)
LEGACY_REPAIR_NAMES = tuple(
    name for name in REPAIR_OPERATORS if name != "R4P_progressive_recovery"
)


VARIANT_PORTFOLIOS = {
    "A0": (("D1_random", "D2_worst"), ("R1_greedy", "R2_regret2"), ()),
    "A1": (("D1_random", "D2_worst", "D3_synchronization"), ("R1_greedy", "R2_regret2"), ()),
    "A2": (("D1_random", "D2_worst", "D3_synchronization", "D4_charging"), ("R1_greedy", "R2_regret2"), ()),
    "A3": (("D1_random", "D2_worst", "D3_synchronization", "D4_charging"), ("R1_greedy", "R2_regret2", "R3_augmentation"), ()),
    "A4": (("D1_random", "D2_worst", "D3_synchronization", "D4_charging"), ("R1_greedy", "R2_regret2", "R3_augmentation", "R4_recovery_aware"), ()),
    "A5": (("D1_random", "D2_worst", "D3_synchronization", "D4_charging"), ("R1_greedy", "R2_regret2", "R3_augmentation", "R4_recovery_aware"), ("L4_recovery_shift",)),
    "FULL": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "MULTISCALE_FULL": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "MS1_TIME_SA": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "MS1_TIME_SA_LS": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "S5A_ROUTE_INIT": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "S5A_ROUTE_SCORE": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "RG_MS_ALNS_V2": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V2_1": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V2_2": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V3": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V3_1": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V3_2": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V3_2_1": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V3_2_1_NORMAL_ONLY": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P": (
        tuple(DESTROY_OPERATORS),
        tuple(name for name in REPAIR_OPERATORS if name != "R4_recovery_aware"),
        tuple(LOCAL_OPERATORS),
    ),
    "VANILLA": (("D1_random", "D2_worst"), ("R1_greedy", "R2_regret2"), ()),
    "NO_SYNC": (tuple(name for name in LEGACY_DESTROY_NAMES if name != "D3_synchronization"), LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "NO_CHARGE_CONGESTION": (tuple(name for name in LEGACY_DESTROY_NAMES if name != "D4_charging"), LEGACY_REPAIR_NAMES, tuple(LOCAL_OPERATORS)),
    "NO_AUGMENTATION_REPAIR": (LEGACY_DESTROY_NAMES, tuple(name for name in LEGACY_REPAIR_NAMES if name != "R3_augmentation"), tuple(LOCAL_OPERATORS)),
    "NO_RECOVERY_AWARE": (LEGACY_DESTROY_NAMES, tuple(name for name in LEGACY_REPAIR_NAMES if name != "R4_recovery_aware"), tuple(LOCAL_OPERATORS)),
    "NO_RECOVERY_SHIFT": (LEGACY_DESTROY_NAMES, LEGACY_REPAIR_NAMES, tuple(name for name in LOCAL_OPERATORS if name != "L4_recovery_shift")),
}


PSALNS_VARIANT = "PSALNS"
PSALNS_PILOT_BASELINE_VARIANT = "MS1_TIME_SA_LS"
ALNS_VARIANT_ALIASES = {
    PSALNS_VARIANT: PSALNS_PILOT_BASELINE_VARIANT,
}


MS1_TIME_TEMPERATURE_VARIANTS = frozenset({"MS1_TIME_SA", "MS1_TIME_SA_LS"})
MS1_TIME_LOCAL_SEARCH_VARIANTS = frozenset({"MS1_TIME_SA_LS"})
S5A_ROUTE_INITIALIZATION_VARIANTS = frozenset(
    {"S5A_ROUTE_INIT", "S5A_ROUTE_SCORE"}
)
SYNCHRONIZATION_CANDIDATES_PER_ORIGIN = 32
SYNCHRONIZATION_INSERTIONS_PER_CANDIDATE = 6
TIME_TEMPERATURE_VARIANTS = (
    MS1_TIME_TEMPERATURE_VARIANTS
    | S5A_ROUTE_INITIALIZATION_VARIANTS
    | {
        "RG_MS_ALNS_V2",
        "RG_MS_ALNS_V2_1",
        "RG_MS_ALNS_V2_2",
        "RG_MS_ALNS_V3",
        "RG_MS_ALNS_V3_1",
        "RG_MS_ALNS_V3_2",
        "RG_MS_ALNS_V3_2_1",
        "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
        "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
    }
)
TIME_LOCAL_SEARCH_VARIANTS = (
    MS1_TIME_LOCAL_SEARCH_VARIANTS
    | S5A_ROUTE_INITIALIZATION_VARIANTS
    | {
        "RG_MS_ALNS_V2",
        "RG_MS_ALNS_V2_1",
        "RG_MS_ALNS_V2_2",
        "RG_MS_ALNS_V3",
        "RG_MS_ALNS_V3_1",
        "RG_MS_ALNS_V3_2",
        "RG_MS_ALNS_V3_2_1",
        "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
        "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
    }
)


@dataclass(frozen=True, slots=True)
class InitializationOutcome:
    solution: Solution
    evaluation: EvaluationResult
    metadata: dict[str, object]
    full_evaluation_time_sec: float
    feasible_evaluations: int
    infeasible_evaluations: int
    infeasible_codes: dict[str, int]


def _synchronization_replacement_solutions(
    instance: Instance,
    current: Solution,
    sortie: Sortie,
    index: SortieIndex,
) -> tuple[Solution, ...]:
    """Replace an exact union of current sorties with one scored forward sortie."""
    candidate_tasks = set(sortie.task_sequence)
    selected_ids = solution_sortie_ids(current, instance.uav_count)
    affected_ids = tuple(
        sortie_id
        for sortie_id in selected_ids
        if set(index.sortie_by_id[sortie_id].task_sequence) & candidate_tasks
    )
    if not affected_ids or any(
        index.support_position[index.sortie_by_id[sortie_id].recovery_support]
        > index.support_position[index.sortie_by_id[sortie_id].origin_support]
        for sortie_id in affected_ids
    ):
        return ()
    affected_tasks = {
        task_id
        for sortie_id in affected_ids
        for task_id in index.sortie_by_id[sortie_id].task_sequence
    }
    if affected_tasks != candidate_tasks:
        return ()

    partial = destroy_tasks_to_partial(
        current,
        candidate_tasks,
        index,
        instance.uav_count,
    )
    insertions = find_feasible_insertions(partial, sortie, instance, index)

    def insertion_key(insertion: tuple[int, int]) -> tuple[float, int, int]:
        uav_id, position = insertion
        prefix_load = sum(
            index.sortie_by_id[sortie_id].nominal_duration_min
            for sortie_id in partial.uav_sequences.get(uav_id, ())[:position]
        )
        return (-prefix_load, uav_id, -position)

    solutions: list[Solution] = []
    for uav_id, position in sorted(insertions, key=insertion_key)[
        :SYNCHRONIZATION_INSERTIONS_PER_CANDIDATE
    ]:
        replaced = partial.copy()
        replaced.uav_sequences.setdefault(uav_id, []).insert(position, sortie.id)
        replaced.missing_tasks.difference_update(sortie.task_sequence)
        solutions.append(replaced.to_solution())
    return tuple(solutions)


def _initial_solution_span_counts(
    instance: Instance,
    solution: Solution,
    index: SortieIndex,
) -> dict[str, int]:
    positions = index.support_position
    counts = {"span0": 0, "span1": 0, "span2": 0, "span_gt2": 0}
    for sequence in solution.uav_sequences.values():
        for sortie_id in sequence:
            sortie = index.sortie_by_id[sortie_id]
            span = positions[sortie.recovery_support] - positions[sortie.origin_support]
            if span in (0, 1, 2):
                counts[f"span{span}"] += 1
            elif span > 2:
                counts["span_gt2"] += 1
    return counts


def _initialize_synchronization_cost(
    instance: Instance,
    index: SortieIndex,
    limiter: EvaluationBudget,
    config: RouteGuidedInitializationConfig,
) -> InitializationOutcome:
    """Build a feasible S5A1 warm start through scored exact-cover replacements."""
    started_initialization = perf_counter()
    started_evaluation = perf_counter()
    current_solution = construct_initial_solution(instance, index)
    current_evaluation = limiter.evaluate(current_solution)
    full_evaluation_time_sec = perf_counter() - started_evaluation
    if not current_evaluation.feasible:
        raise RuntimeError(
            f"legacy fallback initial solution is infeasible: {current_evaluation.violations}"
        )

    feasible_evaluations = 1
    infeasible_evaluations = 0
    infeasible_codes: dict[str, int] = {}
    candidate_solutions_evaluated = 0
    candidates_considered = 0
    replacements_accepted = 0
    route_sail_prefix = fixed_route_sail_prefix(
        instance, limiter.evaluator.usv_speed_km_min
    )
    positions = index.support_position
    candidates_by_origin: dict[int, list[Sortie]] = {}
    for sortie in index.sortie_by_id.values():
        origin = positions[sortie.origin_support]
        recovery = positions[sortie.recovery_support]
        if 0 < recovery - origin <= config.local_max_span:
            candidates_by_origin.setdefault(origin, []).append(sortie)

    for origin in sorted(candidates_by_origin):
        eligible_candidates = sorted(
            candidates_by_origin[origin],
            key=lambda sortie: synchronization_candidate_key(
                sortie, index, route_sail_prefix
            ),
        )
        exact_candidates_at_origin = 0
        accepted_at_origin = False
        for sortie in eligible_candidates:
            replacement_solutions = _synchronization_replacement_solutions(
                instance,
                current_solution,
                sortie,
                index,
            )
            if not replacement_solutions:
                continue
            exact_candidates_at_origin += 1
            candidates_considered += 1
            if exact_candidates_at_origin > SYNCHRONIZATION_CANDIDATES_PER_ORIGIN:
                break
            for candidate_solution in replacement_solutions:
                if not limiter.available():
                    break
                started_evaluation = perf_counter()
                candidate_evaluation = limiter.try_evaluate(candidate_solution)
                full_evaluation_time_sec += perf_counter() - started_evaluation
                if candidate_evaluation is None:
                    break
                candidate_solutions_evaluated += 1
                if candidate_evaluation.feasible:
                    feasible_evaluations += 1
                    if (
                        candidate_evaluation.objective
                        < current_evaluation.objective - 1e-9
                    ):
                        current_solution = candidate_solution
                        current_evaluation = candidate_evaluation
                        replacements_accepted += 1
                        accepted_at_origin = True
                        break
                else:
                    infeasible_evaluations += 1
                    for detail in candidate_evaluation.violation_details:
                        infeasible_codes[detail.code] = (
                            infeasible_codes.get(detail.code, 0) + 1
                        )
            if accepted_at_origin or not limiter.available():
                break
        if not limiter.available():
            break

    counts = _initial_solution_span_counts(instance, current_solution, index)
    synchronization_keys = tuple(
        synchronization_candidate_key(
            index.sortie_by_id[sortie_id], index, route_sail_prefix
        )
        for sequence in current_solution.uav_sequences.values()
        for sortie_id in sequence
    )
    sortie_count = current_evaluation.sortie_count
    used_scored_initialization = replacements_accepted > 0
    metadata: dict[str, object] = {
        "initialization_mode": "route_guided_cost_progressive",
        "initialization_candidate_scoring": config.candidate_scoring,
        "initialization_scope_candidate_scoring": (
            config.candidate_scoring if used_scored_initialization else "scope_only"
        ),
        "initialization_attempts": 1 + candidate_solutions_evaluated,
        "initialization_scope_used": (
            "local" if used_scored_initialization else "global"
        ),
        "initialization_time_sec": perf_counter() - started_initialization,
        "initialization_evaluations": 1 + candidate_solutions_evaluated,
        "initialization_scored_candidates_considered": candidates_considered,
        "initialization_scored_candidate_solutions_evaluated": candidate_solutions_evaluated,
        "initialization_scored_replacements_accepted": replacements_accepted,
        "initialization_scored_retries": 0,
        "initialization_excluded_scored_sorties": 0,
        "initial_objective": current_evaluation.objective,
        "initial_field_makespan_min": current_evaluation.field_makespan_min,
        "initial_total_mission_time_min": current_evaluation.total_mission_time_min,
        "initial_sortie_count": sortie_count,
        "initial_span0_sorties": counts["span0"],
        "initial_span1_sorties": counts["span1"],
        "initial_span2_sorties": counts["span2"],
        "initial_span_gt2_sorties": counts["span_gt2"],
        "initial_same_recovery_ratio": (
            counts["span0"] / sortie_count if sortie_count else 0.0
        ),
        "initial_different_recovery_ratio": current_evaluation.different_recovery_ratio,
        "initial_proxy_usv_wait_min": sum(key[0] for key in synchronization_keys),
        "initial_proxy_hover_min": sum(key[1] for key in synchronization_keys),
        "initial_hover_min": current_evaluation.uav_hover_min,
        "initial_usv_dwell_min": current_evaluation.usv_dwell_min,
        "initial_charging_wait_min": current_evaluation.charging_wait_min,
        "local_construction_success": True,
        "local_decoder_feasible": used_scored_initialization,
        "extended_construction_success": None,
        "extended_decoder_feasible": None,
        "global_fallback_used": not used_scored_initialization,
    }
    return InitializationOutcome(
        solution=current_solution,
        evaluation=current_evaluation,
        metadata=metadata,
        full_evaluation_time_sec=full_evaluation_time_sec,
        feasible_evaluations=feasible_evaluations,
        infeasible_evaluations=infeasible_evaluations,
        infeasible_codes=infeasible_codes,
    )


def initialize_solution(
    instance: Instance,
    index: SortieIndex,
    limiter: EvaluationBudget,
    config: RouteGuidedInitializationConfig,
) -> InitializationOutcome:
    """Construct and fully decode a feasibility-preserving progressive warm start."""
    if config.candidate_scoring == "synchronization_lexicographic":
        return _initialize_synchronization_cost(instance, index, limiter, config)
    started_initialization = perf_counter()
    stage_status: dict[str, dict[str, bool | None]] = {
        "local": {"construction": None, "decoder": None},
        "extended": {"construction": None, "decoder": None},
    }
    attempts = (
        (
            ("local", config.local_max_span),
            ("extended", config.extended_max_span),
            ("global", None),
        )
        if config.enabled
        else (("global", None),)
    )
    full_evaluation_time_sec = 0.0
    initialization_evaluations = 0
    infeasible_codes: dict[str, int] = {}
    construction_errors: list[str] = []

    for scope, max_span in attempts:
        try:
            solution = construct_initial_solution(
                instance,
                index,
                max_recovery_span=max_span,
            )
        except RuntimeError as exc:
            construction_errors.append(f"{scope}:{exc}")
            if scope in stage_status:
                stage_status[scope]["construction"] = False
            continue
        if scope in stage_status:
            stage_status[scope]["construction"] = True

        started_evaluation = perf_counter()
        evaluation = limiter.evaluate(solution)
        full_evaluation_time_sec += perf_counter() - started_evaluation
        initialization_evaluations += 1
        if scope in stage_status:
            stage_status[scope]["decoder"] = evaluation.feasible
        if not evaluation.feasible:
            for detail in evaluation.violation_details:
                infeasible_codes[detail.code] = infeasible_codes.get(detail.code, 0) + 1
            continue

        counts = _initial_solution_span_counts(instance, solution, index)
        route_sail_prefix = fixed_route_sail_prefix(
            instance, limiter.evaluator.usv_speed_km_min
        )
        synchronization_keys = tuple(
            synchronization_candidate_key(
                index.sortie_by_id[sortie_id], index, route_sail_prefix
            )
            for sequence in solution.uav_sequences.values()
            for sortie_id in sequence
        )
        sortie_count = evaluation.sortie_count
        initialization_time_sec = perf_counter() - started_initialization
        metadata: dict[str, object] = {
            "initialization_mode": (
                "route_guided_cost_progressive"
                if config.candidate_scoring == "synchronization_lexicographic"
                else "route_guided_progressive"
                if config.enabled
                else "legacy_global"
            ),
            "initialization_candidate_scoring": config.candidate_scoring,
            "initialization_scope_candidate_scoring": config.candidate_scoring,
            "initialization_scored_candidates_considered": 0,
            "initialization_scored_candidate_solutions_evaluated": 0,
            "initialization_scored_replacements_accepted": 0,
            "initialization_scored_retries": 0,
            "initialization_excluded_scored_sorties": 0,
            "initialization_attempts": len(construction_errors) + initialization_evaluations,
            "initialization_scope_used": scope,
            "initialization_time_sec": initialization_time_sec,
            "initialization_evaluations": initialization_evaluations,
            "initial_objective": evaluation.objective,
            "initial_field_makespan_min": evaluation.field_makespan_min,
            "initial_total_mission_time_min": evaluation.total_mission_time_min,
            "initial_sortie_count": sortie_count,
            "initial_span0_sorties": counts["span0"],
            "initial_span1_sorties": counts["span1"],
            "initial_span2_sorties": counts["span2"],
            "initial_span_gt2_sorties": counts["span_gt2"],
            "initial_same_recovery_ratio": (
                counts["span0"] / sortie_count if sortie_count else 0.0
            ),
            "initial_different_recovery_ratio": evaluation.different_recovery_ratio,
            "initial_proxy_usv_wait_min": sum(key[0] for key in synchronization_keys),
            "initial_proxy_hover_min": sum(key[1] for key in synchronization_keys),
            "initial_hover_min": evaluation.uav_hover_min,
            "initial_usv_dwell_min": evaluation.usv_dwell_min,
            "initial_charging_wait_min": evaluation.charging_wait_min,
            "local_construction_success": stage_status["local"]["construction"],
            "local_decoder_feasible": stage_status["local"]["decoder"],
            "extended_construction_success": stage_status["extended"]["construction"],
            "extended_decoder_feasible": stage_status["extended"]["decoder"],
            "global_fallback_used": config.enabled and scope == "global",
        }
        return InitializationOutcome(
            solution=solution,
            evaluation=evaluation,
            metadata=metadata,
            full_evaluation_time_sec=full_evaluation_time_sec,
            feasible_evaluations=1,
            infeasible_evaluations=initialization_evaluations - 1,
            infeasible_codes=infeasible_codes,
        )

    detail = "; ".join(construction_errors)
    if not config.enabled and initialization_evaluations == 1:
        raise RuntimeError(f"initial solution is infeasible: {evaluation.violations}")
    raise RuntimeError(
        "route-guided initialization exhausted local, extended, and global scopes"
        + (f": {detail}" if detail else "")
    )


def _cadence_references(config: ALNSConfig, task_count: int) -> tuple[float, float]:
    """Return the MS0-derived references required by an MS1 variant.

    MS1 was calibrated only on J=90, 150, and 200.  Refusing an uncalibrated
    task count avoids silently applying a cadence selected for another scale.
    """
    cadence = config.search_cadence
    try:
        return (
            cadence.temperature_ratio_by_task_count[task_count],
            cadence.local_search_interval_sec_by_task_count[task_count],
        )
    except KeyError as exc:
        raise ValueError(
            f"MS1 cadence has no calibrated references for J={task_count}"
        ) from exc


def _temperature_at_elapsed_time(
    initial_temperature: float,
    reference_ratio: float,
    reference_horizon_sec: float,
    elapsed_sec: float,
) -> float:
    """Map an MS0 end-temperature ratio onto elapsed wall-clock time."""
    return initial_temperature * reference_ratio ** (elapsed_sec / reference_horizon_sec)


def _next_time_cadence_due(
    current_due_sec: float,
    interval_sec: float,
    elapsed_sec: float,
) -> float:
    """Advance a periodic deadline past an executed time-based event."""
    next_due_sec = current_due_sec
    while next_due_sec <= elapsed_sec:
        next_due_sec += interval_sec
    return next_due_sec


class ALNSSolver(Solver):
    def __init__(
        self,
        *,
        config: ALNSConfig,
        variant: str,
        diagnostic_hook: Callable[[str, dict[str, Any]], None] | None = None,
        parallel_runtime: ParallelRuntimeConfig | None = None,
        r4_kernel_backend: R4KernelBackend = "auto",
        repair_profile_mode: RepairProfileMode = "full",
        repair_kernel_backend: RepairKernelBackend = "python",
    ) -> None:
        requested_variant = variant
        variant = ALNS_VARIANT_ALIASES.get(variant, variant)
        if variant not in VARIANT_PORTFOLIOS:
            raise ValueError(f"unknown ALNS variant {requested_variant}")
        expected_scoring = {
            "S5A_ROUTE_INIT": "scope_only",
            "S5A_ROUTE_SCORE": "synchronization_lexicographic",
            "RG_MS_ALNS_V2": "synchronization_lexicographic",
            "RG_MS_ALNS_V2_1": "synchronization_lexicographic",
            "RG_MS_ALNS_V2_2": "synchronization_lexicographic",
            "RG_MS_ALNS_V3": "synchronization_lexicographic",
            "RG_MS_ALNS_V3_1": "synchronization_lexicographic",
            "RG_MS_ALNS_V3_2": "synchronization_lexicographic",
            "RG_MS_ALNS_V3_2_1": "synchronization_lexicographic",
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY": (
                "synchronization_lexicographic"
            ),
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P": (
                "synchronization_lexicographic"
            ),
        }.get(variant)
        route_initialization = config.route_guided_initialization
        if expected_scoring is None and route_initialization.enabled:
            raise ValueError(
                "route-guided initialization must be selected with an S5A variant"
            )
        if expected_scoring is not None and (
            not route_initialization.enabled
            or route_initialization.candidate_scoring != expected_scoring
        ):
            raise ValueError(
                f"{variant} requires route-guided candidate scoring {expected_scoring}"
            )
        expected_route_engine = {
            "RG_MS_ALNS_V2": "route_guided_v2",
            "RG_MS_ALNS_V2_1": "route_guided_v2_1",
            "RG_MS_ALNS_V2_2": "route_guided_v2_2",
            "RG_MS_ALNS_V3": "route_guided_v3",
            "RG_MS_ALNS_V3_1": "route_guided_v3_1",
            "RG_MS_ALNS_V3_2": "route_guided_v3_2",
            "RG_MS_ALNS_V3_2_1": "route_guided_v3_2_1",
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY": "route_guided_v3_2_1",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P": "route_guided_v3_2_1",
        }.get(variant)
        uses_route_v2 = expected_route_engine is not None
        if uses_route_v2 != config.route_guided_v2.enabled or (
            expected_route_engine is not None
            and config.search_engine != expected_route_engine
        ):
            raise ValueError(
                "route-guided variant and search-engine configuration must match"
            )
        self.config = config
        self.requested_variant = requested_variant
        self.variant = variant
        self.name = (
            "psalns"
            if requested_variant == PSALNS_VARIANT
            else "vanilla_alns"
            if variant in {"A0", "VANILLA"}
            else "proposed_alns"
        )
        self.destroy_names, self.repair_names, self.local_names = VARIANT_PORTFOLIOS[variant]
        normal_profile_variants = {
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        if config.normal_profile.disable_r4p:
            if variant not in normal_profile_variants:
                raise ValueError(
                    "normal_profile.disable_r4p is restricted to NORMAL-only variants"
                )
            self.repair_names = tuple(
                name
                for name in self.repair_names
                if name != "R4P_progressive_recovery"
            )
        if (
            variant == "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P"
            and not config.normal_profile.disable_r4p
        ):
            raise ValueError("NORMAL-NoR4P variant requires disable_r4p=true")
        self.diagnostic_hook = diagnostic_hook
        self.parallel_runtime = parallel_runtime or ParallelRuntimeConfig()
        if r4_kernel_backend not in {"auto", "python", "numba"}:
            raise ValueError(f"unknown R4 kernel backend {r4_kernel_backend}")
        self.r4_kernel_backend = r4_kernel_backend
        if repair_kernel_backend not in {"auto", "python", "numba"}:
            raise ValueError(
                f"unknown shared repair kernel backend {repair_kernel_backend}"
            )
        self.repair_kernel_backend = repair_kernel_backend
        if repair_profile_mode not in {"summary", "full"}:
            raise ValueError(
                f"unknown repair profile mode {repair_profile_mode}"
            )
        self.repair_profile_mode = repair_profile_mode
        expected_controller_mode = (
            "normal_only_profile"
            if variant in normal_profile_variants
            else "escape_rearm"
            if variant == "RG_MS_ALNS_V3_2_1"
            else None
        )
        if expected_controller_mode is not None and (
            config.route_guided_v3_2.controller_mode
            != expected_controller_mode
        ):
            raise ValueError(
                f"{variant} requires controller_mode={expected_controller_mode}"
            )

    def solve(self, instance, evaluator, seed, budget) -> AlgorithmResult:
        executor_context = (
            ThreadPoolExecutor(
                max_workers=self.parallel_runtime.max_workers,
                thread_name_prefix="psalns-decoder",
            )
            if self.parallel_runtime.uses_threads
            else nullcontext(None)
        )
        with executor_context as parallel_executor:
            threaded_evaluate = (
                thread_local_evaluate(evaluator.fork)
                if parallel_executor is not None
                else None
            )
            return self._solve(
                instance,
                evaluator,
                seed,
                budget,
                parallel_executor=parallel_executor,
                threaded_evaluate=threaded_evaluate,
            )

    def _solve(
        self,
        instance,
        evaluator,
        seed,
        budget,
        *,
        parallel_executor: ThreadPoolExecutor | None,
        threaded_evaluate: Callable[[Solution], EvaluationResult] | None,
    ) -> AlgorithmResult:
        if self.variant in {
            "RG_MS_ALNS_V3", "RG_MS_ALNS_V3_1", "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1", "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        } and (
            evaluator.model_semantics_version != "3.0"
            or evaluator.launch_policy != "dwell_window_synchronized"
        ):
            raise ValueError(
                "RG_MS_ALNS_V3 requires model semantics 3.0 with "
                "dwell-window synchronized launch"
            )
        v32_controller_enabled = self.variant in {
            "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1",
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        normal_only_profile_enabled = (
            v32_controller_enabled
            and self.config.route_guided_v3_2.normal_only_profile
        )
        search_rng_seed = seed
        controller_rng_seed: int | None = None
        controller_rng: random.Random | None = None
        if v32_controller_enabled:
            search_sequence, controller_sequence = SeedSequence(seed).spawn(2)
            search_rng_seed = int(search_sequence.generate_state(1, dtype="uint64")[0])
            controller_rng_seed = int(
                controller_sequence.generate_state(1, dtype="uint64")[0]
            )
            controller_rng = random.Random(controller_rng_seed)
        rng = random.Random(search_rng_seed)
        r4_numba_enabled = (
            "R4_recovery_aware" in self.repair_names
            and self.r4_kernel_backend != "python"
        )
        r4_numba_compile_sec = (
            warmup_r4_numba_kernel() if r4_numba_enabled else 0.0
        )
        repair_numba_enabled = (
            bool(
                {"R1_greedy", "R2_regret2", "R3_augmentation"}
                & set(self.repair_names)
            )
            and self.repair_kernel_backend != "python"
        )
        repair_numba_compile_sec = (
            warmup_repair_numba_kernel() if repair_numba_enabled else 0.0
        )
        limiter = EvaluationBudget(evaluator, budget)
        diagnostic_trace_overhead_sec = 0.0
        normal_only_profile = (
            NormalOnlySearchProfile() if normal_only_profile_enabled else None
        )

        def emit_diagnostic(event: str, **payload: Any) -> None:
            nonlocal diagnostic_trace_overhead_sec
            if self.diagnostic_hook is None:
                return
            started_trace = perf_counter()
            self.diagnostic_hook(event, payload)
            trace_elapsed = perf_counter() - started_trace
            limiter.exclude_elapsed(trace_elapsed)
            diagnostic_trace_overhead_sec += trace_elapsed
            if normal_only_profile is not None:
                normal_only_profile.normal_diagnostics_sec += trace_elapsed

        destroy_time_sec = 0.0
        repair_time_sec = 0.0
        repair_internal_profiles = {
            name: RepairSearchProfile() for name in self.repair_names
        }
        detailed_repair_profile = self.repair_profile_mode == "full"
        repair_missing_sizes = {name: [] for name in self.repair_names}
        repair_removed_sorties = {name: [] for name in self.repair_names}
        repair_workload_bins = {
            name: {
                label: {
                    "calls": 0,
                    "repair_sec": 0.0,
                    **{field: 0 for field in REPAIR_WORKLOAD_COUNTER_FIELDS},
                    **{field: 0.0 for field in REPAIR_WORKLOAD_TIMER_FIELDS},
                }
                for label in MISSING_SIZE_BINS
            }
            for name in self.repair_names
        }
        destroy_task_sizes: list[int] = []
        destroy_sortie_sizes: list[int] = []
        destroy_operator_calls: dict[str, int] = {
            name: 0 for name in self.destroy_names
        }

        def record_repair_workload(
            *,
            repair_name: str,
            before: dict[str, int | float],
            missing_tasks: int,
            removed_sorties: int,
            exclusive_repair_sec: float,
            materialized_candidates: int,
            decoder_calls: int,
        ) -> None:
            profile = repair_internal_profiles[repair_name]
            profile.repair_materialized_options += materialized_candidates
            profile.repair_decoder_options += decoder_calls
            attributed = sum(
                float(getattr(profile, field)) - float(before[field])
                for field in REPAIR_WORKLOAD_TIMER_FIELDS
                if field != "repair_bookkeeping_sec"
            )
            profile.repair_bookkeeping_sec += max(
                0.0, exclusive_repair_sec - attributed
            )
            bucket = repair_workload_bins[repair_name][
                _missing_size_bin(missing_tasks)
            ]
            bucket["calls"] += 1
            bucket["repair_sec"] += exclusive_repair_sec
            for field in (
                *REPAIR_WORKLOAD_COUNTER_FIELDS,
                *REPAIR_WORKLOAD_TIMER_FIELDS,
            ):
                bucket[field] += getattr(profile, field) - before[field]
            repair_missing_sizes[repair_name].append(missing_tasks)
            repair_removed_sorties[repair_name].append(removed_sorties)

        full_evaluation_time_sec = 0.0
        local_search_time_sec = 0.0
        candidate_sets = 0
        candidate_solutions_generated = 0
        candidate_solutions_full_evaluated = 0
        max_candidate_set_size = 0
        repair_deadline_aborts = 0
        parallel_batch_calls = 0
        parallel_candidate_count = 0
        parallel_max_batch_size = 0
        parallel_wall_sec = 0.0
        parallel_wait_sec = 0.0
        parallel_batch_max_sec = 0.0
        serial_candidate_evals = 0
        threaded_candidate_evals = 0
        index = build_sortie_index(instance.sortie_pool, instance.usv_route[1:-1])
        candidate_index = index
        uses_route_v2 = self.variant in {
            "RG_MS_ALNS_V2",
            "RG_MS_ALNS_V2_1",
            "RG_MS_ALNS_V2_2",
            "RG_MS_ALNS_V3",
            "RG_MS_ALNS_V3_1",
            "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1",
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        corridor_coupled = self.variant in {
            "RG_MS_ALNS_V2_1",
            "RG_MS_ALNS_V2_2",
            "RG_MS_ALNS_V3",
            "RG_MS_ALNS_V3_1",
            "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1",
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        physical_feasibility_aware = self.variant in {
            "RG_MS_ALNS_V2_2",
            "RG_MS_ALNS_V3",
            "RG_MS_ALNS_V3_1",
            "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1",
            "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        final_candidate_classes = self.variant in {
            "RG_MS_ALNS_V3", "RG_MS_ALNS_V3_1", "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1", "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        dwell_guided = self.variant in {
            "RG_MS_ALNS_V3_1", "RG_MS_ALNS_V3_2",
            "RG_MS_ALNS_V3_2_1", "RG_MS_ALNS_V3_2_1_NORMAL_ONLY",
            "RG_MS_ALNS_V3_2_1_NORMAL_NO_R4P",
        }
        route_context = None
        route_controller = None
        route_diagnostics = None
        destroy_engine = None
        repair_engine = None
        if uses_route_v2:
            route_context = RouteSearchContext.build(
                instance,
                index,
                evaluator.usv_speed_km_min,
                evaluator.uav if physical_feasibility_aware else None,
            )
            if final_candidate_classes:
                candidate_index = build_sortie_index(
                    tuple(
                        sortie
                        for sortie in instance.sortie_pool
                        if route_context.is_physically_feasible(sortie.id)
                    ),
                    instance.usv_route[1:-1],
                )
            route_controller = RouteSearchController(self.config.route_guided_v2)
            route_diagnostics = RouteSearchDiagnostics(
                corridor_coupled=corridor_coupled,
                physical_feasibility_aware=physical_feasibility_aware,
                final_candidate_classes=final_candidate_classes,
                dwell_guided=dwell_guided,
            )
            v21 = self.config.route_guided_v2_1
            v22 = self.config.route_guided_v2_2
            v3 = self.config.route_guided_v3
            if physical_feasibility_aware:
                route_diagnostics.record_physical_pool(
                    feasible_forward=len(route_context.physical_forward_sortie_ids),
                    static_impossible_forward=len(
                        route_context.static_impossible_forward_sortie_ids
                    ),
                )
            destroy_engine = RouteGuidedDestroyEngine(
                controller=route_controller,
                diagnostics=route_diagnostics,
                legacy_destroy_operators=DESTROY_OPERATORS,
                corridor_bundle_destroy=(
                    corridor_coupled
                    and (
                        physical_feasibility_aware
                        or v21.corridor_bundle_destroy
                    )
                ),
                forward_partner_required=(
                    corridor_coupled and v21.forward_partner_required
                ),
                physical_feasibility_aware=physical_feasibility_aware,
                guided_candidate_limit=(
                    v3.guided_candidate_limit
                    if final_candidate_classes
                    else v22.guided_candidate_limit
                ),
                transactional_b2=final_candidate_classes,
                target_transformation_required=final_candidate_classes,
                dwell_guided=dwell_guided,
            )
            quota = (
                v22.candidate_quota
                if physical_feasibility_aware
                else v21.candidate_quota
            )
            repair_engine = ProgressiveRepairEngine(
                controller=route_controller,
                diagnostics=route_diagnostics,
                repair_operators=REPAIR_OPERATORS,
                corridor_coupled=corridor_coupled,
                physical_feasibility_aware=physical_feasibility_aware,
                final_candidate_classes=final_candidate_classes,
                dwell_guided=dwell_guided,
                candidate_quotas=(
                    {
                        CandidateClass.SAME_POINT: quota.same,
                        CandidateClass.ADJACENT_RECOVERY: quota.adjacent,
                        CandidateClass.FORWARD_RELOCATION: (
                            quota.forward_relocation
                        ),
                        CandidateClass.FORWARD_ABSORPTION: (
                            quota.forward_absorption
                        ),
                    }
                    if corridor_coupled
                    else None
                ),
            )
            initialization = initialize_route_guided_solution(
                instance=instance,
                index=candidate_index,
                limiter=limiter,
                config=self.config.route_guided_initialization,
                frozen_s5a1_initializer=initialize_solution,
            )
        else:
            initialization = initialize_solution(
                instance,
                index,
                limiter,
                self.config.route_guided_initialization,
            )
        initial = initialization.solution
        current_solution = initial
        current_evaluation = initialization.evaluation
        full_evaluation_time_sec += initialization.full_evaluation_time_sec
        if normal_only_profile is not None:
            normal_only_profile.full_decoder_calls = limiter.evaluations
        best_solution = current_solution.copy()
        best_evaluation = current_evaluation
        time_to_best = limiter.elapsed_sec
        eval_to_best = limiter.evaluations
        convergence = [ConvergencePoint(0, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective)]
        basin_best_solution = current_solution.copy()
        basin_best_evaluation = current_evaluation
        escape_controller: DescentMomentumEscapeController | None = None
        escape_diagnostics: EscapeSearchDiagnostics | None = None
        if v32_controller_enabled:
            assert controller_rng is not None
            escape_controller = DescentMomentumEscapeController(
                config=self.config.route_guided_v3_2,
                budget_sec=budget.wall_time_sec,
                initial_objective=current_evaluation.objective,
                rng=controller_rng,
                start_time=limiter.elapsed_sec,
            )
            escape_diagnostics = EscapeSearchDiagnostics()
        emit_diagnostic(
            "initial",
            iteration=0,
            elapsed_sec=limiter.elapsed_sec,
            current_solution=current_solution,
            current_evaluation=current_evaluation,
            best_solution=best_solution,
            best_evaluation=best_evaluation,
            index=index,
        )
        operator_names = (*self.destroy_names, *self.repair_names, *self.local_names)
        weights = {name: 1.0 for name in operator_names}
        segment_calls = {name: 0 for name in operator_names}
        segment_rewards = {name: 0.0 for name in operator_names}
        stats = {name: {"calls": 0.0, "accepted": 0.0, "improvements": 0.0, "reward": 0.0} for name in operator_names}
        for name in self.repair_names:
            stats[name].update({
                "selected": 0,
                "total_time_sec": 0.0,
                "wall_sec": 0.0,
                "candidate_sets": 0,
                "candidates_generated": 0,
                "runtime_errors": 0,
                "missing_tasks_sum": 0,
                "missing_tasks_max": 0,
                "current_improvements": 0,
                "basin_best_updates": 0,
                "global_best_updates": 0,
                "total_current_gain": 0.0,
                "total_global_best_gain": 0.0,
            })
        temperature = self.config.initial_temperature_fraction * current_evaluation.objective
        initial_temperature = temperature
        iteration = 0
        stagnation = 0
        task_count = len(instance.tasks)
        uses_time_temperature = self.variant in TIME_TEMPERATURE_VARIANTS
        uses_time_local_search = self.variant in TIME_LOCAL_SEARCH_VARIANTS
        use_multiscale = self.variant in {
            "MULTISCALE_FULL",
            *TIME_TEMPERATURE_VARIANTS,
        }
        cadence_temperature_ratio: float | None = None
        cadence_local_search_interval_sec: float | None = None
        next_local_search_due_sec: float | None = None
        if uses_time_temperature:
            (
                cadence_temperature_ratio,
                cadence_local_search_interval_sec,
            ) = _cadence_references(self.config, task_count)
            if uses_time_local_search:
                next_local_search_due_sec = cadence_local_search_interval_sec
        feasible_evaluations = initialization.feasible_evaluations
        infeasible_evaluations = initialization.infeasible_evaluations
        infeasible_codes = initialization.infeasible_codes.copy()
        destroy_scale_calls = {
            "legacy": 0,
            "micro": 0,
            "medium": 0,
            "macro": 0,
        }
        destroy_target_tasks_sum = 0
        destroy_actual_tasks_sum = 0
        destroy_actual_tasks_max = 0

        def audit_point(active_iteration, evaluation):
            different = round(evaluation.different_recovery_ratio * evaluation.sortie_count)
            return {
                "iteration": active_iteration,
                "evaluation": limiter.evaluations,
                "objective": evaluation.objective,
                "same_sorties": evaluation.sortie_count - different,
                "different_sorties": different,
                "different_recovery_ratio": evaluation.different_recovery_ratio,
                "hover_min": evaluation.uav_hover_min,
                "usv_dwell_min": evaluation.usv_dwell_min,
                "charging_wait_min": evaluation.charging_wait_min,
            }

        search_audit = [audit_point(0, best_evaluation)]

        def evaluate_v2_candidate(candidate: Solution) -> EvaluationResult | None:
            nonlocal full_evaluation_time_sec, feasible_evaluations, infeasible_evaluations
            started_evaluation = perf_counter()
            evaluation = limiter.try_evaluate(candidate, iteration)
            evaluation_elapsed = perf_counter() - started_evaluation
            full_evaluation_time_sec += evaluation_elapsed
            if normal_only_profile is not None and evaluation is not None:
                normal_only_profile.normal_decoder_sec += evaluation_elapsed
                normal_only_profile.full_decoder_calls += 1
            if evaluation is None:
                return None
            if evaluation.feasible:
                feasible_evaluations += 1
            else:
                infeasible_evaluations += 1
                for detail in evaluation.violation_details:
                    infeasible_codes[detail.code] = (
                        infeasible_codes.get(detail.code, 0) + 1
                    )
            return evaluation

        def b2_transaction_payload(repair_outcome) -> dict[str, Any] | None:
            transaction = repair_outcome.b2_transaction
            if transaction is None:
                return None
            assert route_context is not None
            target = route_context.sortie_index.sortie_by_id[
                transaction.target_sortie_id
            ]
            decoded = []
            for item in repair_outcome.evaluated:
                if (
                    item.candidate.guided_target_sortie_id
                    != transaction.target_sortie_id
                ):
                    continue
                execution = next(
                    (
                        value
                        for value in item.evaluation.executions
                        if value.sortie_id == transaction.target_sortie_id
                    ),
                    None,
                )
                decoded.append({
                    "uav_id": (
                        item.candidate.guided_insertion[0]
                        if item.candidate.guided_insertion is not None
                        else None
                    ),
                    "position": (
                        item.candidate.guided_insertion[1]
                        if item.candidate.guided_insertion is not None
                        else None
                    ),
                    "launch_time": (
                        execution.launch_time_min if execution is not None else None
                    ),
                    "recovery_time": (
                        execution.recovery_time_min if execution is not None else None
                    ),
                    "hover": execution.hover_min if execution is not None else None,
                    "decoder_feasible": item.evaluation.feasible,
                    "violation_codes": sorted(
                        {detail.code for detail in item.evaluation.violation_details}
                    ),
                })
            decoder_feasible = any(value["decoder_feasible"] for value in decoded)
            if transaction.status != "reconstructed":
                outcome = transaction.status
            elif decoder_feasible:
                outcome = "full_feasible"
            elif decoded:
                outcome = "reconstructed_but_dynamically_infeasible"
            else:
                outcome = "not_decoded"
            chosen = (
                transaction.attempted_insertions[0]
                if transaction.attempted_insertions
                else None
            )
            return {
                "target_sortie_id": transaction.target_sortie_id,
                "target_tasks": list(transaction.target_task_ids),
                "origin": target.origin_support,
                "recovery": target.recovery_support,
                "span": route_context.sortie_span[target.id],
                "owner_sorties_before_destroy": list(
                    transaction.owner_sortie_ids
                ),
                "missing_after_destroy": list(
                    transaction.missing_after_destroy
                ),
                "candidate_uavs": sorted(
                    {uav_id for uav_id, _ in transaction.legal_insertions}
                ),
                "legal_insertion_positions": [
                    [uav_id, position]
                    for uav_id, position in transaction.legal_insertions
                ],
                "chosen_uav": chosen[0] if chosen is not None else None,
                "chosen_position": chosen[1] if chosen is not None else None,
                "chosen_insertions": [
                    [uav_id, position]
                    for uav_id, position in transaction.attempted_insertions
                ],
                "transaction_status": transaction.status,
                "outcome": outcome,
                "decoded_candidates": decoded,
            }

        def guided_work_snapshot() -> tuple[int, int, int]:
            if route_diagnostics is None:
                return (0, 0, 0)
            return (
                route_diagnostics.d6_calls,
                route_diagnostics.b2_reconstructed,
                route_diagnostics.dwell_guided_d6_calls,
            )

        def finish_v32_iteration(
            *,
            decision: EscapeDecision | None,
            branch_started: float,
            current_objective_before: float,
            basin_best_objective_before: float,
            global_best_objective_before: float,
            destroy_operator: str,
            repair_operator: str,
            candidate_class: CandidateClass | None,
            accepted: bool,
            current_improvement: bool,
            basin_best_updated: bool,
            global_best_updated: bool,
            global_best_gain: float,
            guided_before: tuple[int, int, int],
        ) -> None:
            if decision is None:
                return
            assert escape_controller is not None
            assert escape_diagnostics is not None
            guided_after = guided_work_snapshot()
            normal_b2_calls = (
                guided_after[1] - guided_before[1]
                if decision.action is SearchAction.NORMAL
                else 0
            )
            if decision.action is SearchAction.NORMAL:
                assert guided_after == guided_before, (
                    "NORMAL must not execute D6, B2 reconstruction, or guided screening"
                )
            if decision.action is SearchAction.GLOBAL_ESCAPE:
                assert guided_after == guided_before, (
                    "GLOBAL_ESCAPE must use the non-guided global path"
                )
            now = limiter.elapsed_sec
            controller_observe_started = perf_counter()
            escape_controller.observe(
                action=decision.action,
                elapsed_sec=now,
                accepted=accepted,
                current_objective=current_evaluation.objective,
                basin_best_updated=basin_best_updated,
                global_best_updated=global_best_updated,
                global_best_objective=best_evaluation.objective,
            )
            if normal_only_profile is not None:
                normal_only_profile.controller_sec += (
                    perf_counter() - controller_observe_started
                )
            internal_diagnostics_started = perf_counter()
            if best_evaluation.objective > basin_best_evaluation.objective + 1e-9:
                raise AssertionError("global best must not be worse than basin best")
            escape_diagnostics.record_iteration(
                iteration=iteration,
                elapsed_sec=now,
                decision=decision,
                current_objective=current_objective_before,
                basin_best_objective=basin_best_objective_before,
                global_best_objective=global_best_objective_before,
                wall_sec=perf_counter() - branch_started,
                destroy_operator=destroy_operator,
                repair_operator=repair_operator,
                candidate_class=candidate_class,
                accepted=accepted,
                current_improvement=current_improvement,
                basin_best_updated=basin_best_updated,
                global_best_updated=global_best_updated,
                global_best_gain=global_best_gain,
                objective_after=current_evaluation.objective,
                normal_b2_calls=normal_b2_calls,
            )
            if global_best_updated:
                escape_diagnostics.record_global_best(
                    elapsed_sec=now,
                    iteration=iteration,
                    old_global_best=global_best_objective_before,
                    new_global_best=best_evaluation.objective,
                    decision=decision,
                    destroy_operator=destroy_operator,
                    repair_operator=repair_operator,
                    candidate_class=candidate_class,
                )
            if (
                route_diagnostics is not None
                and candidate_class is CandidateClass.GUIDED_PHYSICAL_FORWARD
            ):
                route_diagnostics.record_b2_search_outcome(
                    current_improvement=current_improvement,
                    basin_best_updated=basin_best_updated,
                    global_best_updated=global_best_updated,
                    global_best_gain=global_best_gain,
                )
            if normal_only_profile is not None:
                normal_only_profile.normal_diagnostics_sec += (
                    perf_counter() - internal_diagnostics_started
                )

        while limiter.available(iteration):
            iteration += 1
            branch_started = perf_counter()
            if uses_time_temperature:
                assert cadence_temperature_ratio is not None
                temperature = _temperature_at_elapsed_time(
                    initial_temperature,
                    cadence_temperature_ratio,
                    self.config.search_cadence.reference_horizon_sec,
                    limiter.elapsed_sec,
                )
            best_before_iteration = best_evaluation.objective
            current_objective_before = current_evaluation.objective
            basin_best_objective_before = basin_best_evaluation.objective
            global_best_objective_before = best_evaluation.objective
            guided_before = guided_work_snapshot()
            escape_decision: EscapeDecision | None = None
            if escape_controller is not None:
                controller_started = perf_counter()
                escape_decision = escape_controller.decide(
                    elapsed_sec=limiter.elapsed_sec
                )
                if normal_only_profile is not None:
                    normal_only_profile.controller_sec += (
                        perf_counter() - controller_started
                    )
            search_action = (
                escape_decision.action
                if escape_decision is not None
                else SearchAction.NORMAL
            )
            operator_selection_started = perf_counter()
            if search_action is SearchAction.GUIDED_ESCAPE:
                destroy_name = "D6_route_corridor"
            elif search_action is SearchAction.GLOBAL_ESCAPE:
                destroy_name = rng.choice(
                    self.config.route_guided_v3_2.global_escape.destroy_pool
                )
            else:
                normal_destroy_names = (
                    tuple(
                        name
                        for name in self.destroy_names
                        if name != "D6_route_corridor"
                    )
                    if v32_controller_enabled
                    else self.destroy_names
                )
                destroy_name = _weighted_choice(normal_destroy_names, weights, rng)
            repair_name = _weighted_choice(self.repair_names, weights, rng)
            if search_action is SearchAction.GUIDED_ESCAPE or (
                not v32_controller_enabled
                and physical_feasibility_aware
                and destroy_name == "D6_route_corridor"
            ):
                # V2.2's target transformation and guided reconstruction are
                # one compound neighborhood operator.
                repair_name = "R4P_progressive_recovery"
            if search_action is SearchAction.GLOBAL_ESCAPE:
                removal_count = _sample_legacy_removal_count(
                    n_tasks=task_count,
                    config=self.config,
                    rng=rng,
                )
                destroy_scale = "macro"
            else:
                removal_count, destroy_scale = _choose_destroy_target(
                    n_tasks=task_count,
                    stagnation=stagnation,
                    config=self.config,
                    rng=rng,
                    use_multiscale=use_multiscale,
                )
            stats[destroy_name]["calls"] += 1
            stats[repair_name]["calls"] += 1
            stats[repair_name]["selected"] += 1
            segment_calls[destroy_name] += 1
            segment_calls[repair_name] += 1
            if normal_only_profile is not None:
                normal_only_profile.normal_operator_selection_sec += (
                    perf_counter() - operator_selection_started
                )
            started_destroy = perf_counter()
            destroy_result = None
            if uses_route_v2:
                assert route_controller is not None
                assert destroy_engine is not None
                assert route_context is not None
                policy = route_controller.make_policy(
                    n_tasks=task_count,
                    destroy_scale=destroy_scale,
                    stagnation=stagnation,
                    repair_name=repair_name,
                )
                destroy_result = destroy_engine.apply(
                    instance=instance,
                    solution=current_solution,
                    evaluation=current_evaluation,
                    route_context=route_context,
                    policy=policy,
                    rng=rng,
                    removal_count=removal_count,
                    destroy_name=destroy_name,
                )
                partial = destroy_result.partial
            else:
                partial = DESTROY_OPERATORS[destroy_name].function(
                    instance,
                    current_solution,
                    index,
                    current_evaluation,
                    rng,
                    removal_count,
                )
            destroy_elapsed = perf_counter() - started_destroy
            destroy_time_sec += destroy_elapsed
            if normal_only_profile is not None:
                normal_only_profile.normal_destroy_sec += destroy_elapsed
            actual_removed = len(partial.missing_tasks)
            retained_sorties = {
                sortie_id
                for sequence in partial.uav_sequences.values()
                for sortie_id in sequence
            }
            removed_sortie_count = len(
                set(solution_sortie_ids(current_solution, instance.uav_count))
                - retained_sorties
            )
            destroy_task_sizes.append(actual_removed)
            destroy_sortie_sizes.append(removed_sortie_count)
            destroy_operator_calls[destroy_name] += 1
            if (
                final_candidate_classes
                and destroy_result is not None
                and destroy_result.repair_context is not None
                and destroy_result.repair_context.guided_target is not None
            ):
                # In V3 the target transformation determines both the release
                # size and its Micro/Medium/Macro interpretation.
                destroy_scale = (
                    destroy_result.repair_context.guided_target.release_scale
                )
                removal_count = actual_removed
            emit_diagnostic(
                "destroy",
                iteration=iteration,
                elapsed_sec=limiter.elapsed_sec,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                destroy_scale=destroy_scale,
                removal_count=removal_count,
                current_solution=current_solution,
                current_evaluation=current_evaluation,
                best_evaluation=best_evaluation,
                partial_solution=partial,
                index=index,
            )
            destroy_scale_calls[destroy_scale] += 1
            destroy_target_tasks_sum += removal_count
            destroy_actual_tasks_sum += actual_removed
            destroy_actual_tasks_max = max(
                destroy_actual_tasks_max,
                actual_removed,
            )
            repair_profile = stats[repair_name]
            repair_profile["missing_tasks_sum"] += actual_removed
            repair_profile["missing_tasks_max"] = max(
                repair_profile["missing_tasks_max"], actual_removed
            )
            decoder_before_repair = full_evaluation_time_sec
            decoder_calls_before_repair = limiter.evaluations
            workload_profile = repair_internal_profiles[repair_name]
            workload_before = {
                field: getattr(workload_profile, field)
                for field in (
                    *REPAIR_WORKLOAD_COUNTER_FIELDS,
                    *REPAIR_WORKLOAD_TIMER_FIELDS,
                )
            }
            r4_profile_before = (
                {
                    field: float(getattr(workload_profile, field))
                    for field in R4_PROFILE_GENERATION_FIELDS
                }
                if detailed_repair_profile
                and repair_name == "R4_recovery_aware"
                else None
            )
            started_repair = perf_counter()
            selected_route_candidate = None
            repair_budget_exhausted = False
            repair_stage = f"repair:{repair_name}"
            repair_deadline = lambda: limiter.deadline_expired(repair_stage)
            try:
                if uses_route_v2:
                    assert destroy_result is not None
                    assert repair_engine is not None
                    assert route_context is not None
                    repair_outcome = repair_engine.apply(
                        instance=instance,
                        destroy_result=destroy_result,
                        route_context=route_context,
                        policy=policy,
                        repair_name=repair_name,
                        repair_profile=(
                            repair_internal_profiles[repair_name]
                            if detailed_repair_profile
                            else None
                        ),
                        evaluate_candidate=evaluate_v2_candidate,
                        deadline_expired=repair_deadline,
                    )
                    repair_budget_exhausted = repair_outcome.budget_exhausted
                    b2_payload = b2_transaction_payload(repair_outcome)
                    if b2_payload is not None:
                        emit_diagnostic(
                            "b2_transaction",
                            iteration=iteration,
                            elapsed_sec=limiter.elapsed_sec,
                            **b2_payload,
                        )
                    candidate_solutions = tuple(
                        candidate.solution for candidate in repair_outcome.generated
                    )
                elif repair_name == "R4_recovery_aware":
                    candidate_solutions = REPAIR_OPERATORS[repair_name].function(
                        instance, partial, index,
                        recovery_top_m=self.config.recovery_top_m,
                        recovery_same_quota=self.config.recovery_same_quota,
                        recovery_diff_quota=self.config.recovery_diff_quota,
                        repair_profile=(
                            repair_internal_profiles[repair_name]
                            if detailed_repair_profile
                            else None
                        ),
                        deadline_expired=repair_deadline,
                        r4_kernel_backend=self.r4_kernel_backend,
                    )
                else:
                    candidate_solutions = REPAIR_OPERATORS[repair_name].function(
                        instance,
                        partial,
                        index,
                        repair_profile=(
                            repair_internal_profiles[repair_name]
                            if detailed_repair_profile
                            else None
                        ),
                        deadline_expired=repair_deadline,
                        repair_kernel_backend=self.repair_kernel_backend,
                    )
            except RepairTimeBudgetExceeded:
                candidate_solutions = ()
                repair_budget_exhausted = True
            except RuntimeError:
                elapsed_repair = perf_counter() - started_repair
                if r4_profile_before is not None:
                    _record_r4_generation_profile(
                        workload_profile,
                        r4_profile_before,
                        elapsed_repair,
                    )
                repair_time_sec += elapsed_repair
                repair_profile["total_time_sec"] += elapsed_repair
                decoder_in_repair = (
                    full_evaluation_time_sec - decoder_before_repair
                )
                exclusive_repair_elapsed = max(
                    0.0, elapsed_repair - decoder_in_repair
                )
                repair_profile["wall_sec"] += exclusive_repair_elapsed
                if normal_only_profile is not None:
                    normal_only_profile.normal_repair_sec += (
                        exclusive_repair_elapsed
                    )
                record_repair_workload(
                    repair_name=repair_name,
                    before=workload_before,
                    missing_tasks=actual_removed,
                    removed_sorties=removed_sortie_count,
                    exclusive_repair_sec=exclusive_repair_elapsed,
                    materialized_candidates=0,
                    decoder_calls=limiter.evaluations - decoder_calls_before_repair,
                )
                repair_profile["runtime_errors"] += 1
                stagnation += 1
                if not uses_time_temperature:
                    temperature *= self.config.cooling_rate
                emit_diagnostic(
                    "iteration_end",
                    iteration=iteration,
                    elapsed_sec=limiter.elapsed_sec,
                    status="repair_runtime_error",
                    current_solution=current_solution,
                    current_evaluation=current_evaluation,
                    best_solution=best_solution,
                    best_evaluation=best_evaluation,
                    index=index,
                )
                finish_v32_iteration(
                    decision=escape_decision,
                    branch_started=branch_started,
                    current_objective_before=current_objective_before,
                    basin_best_objective_before=basin_best_objective_before,
                    global_best_objective_before=global_best_objective_before,
                    destroy_operator=destroy_name,
                    repair_operator=repair_name,
                    candidate_class=None,
                    accepted=False,
                    current_improvement=False,
                    basin_best_updated=False,
                    global_best_updated=False,
                    global_best_gain=0.0,
                    guided_before=guided_before,
                )
                continue
            elapsed_repair = perf_counter() - started_repair
            if r4_profile_before is not None:
                _record_r4_generation_profile(
                    workload_profile,
                    r4_profile_before,
                    elapsed_repair,
                )
            limiter.observe_stage(repair_stage, elapsed_repair)
            if limiter.deadline.expired():
                limiter.mark_deadline_abort(repair_stage)
                candidate_solutions = ()
                repair_budget_exhausted = True
            repair_time_sec += elapsed_repair
            repair_profile["total_time_sec"] += elapsed_repair
            decoder_in_repair = full_evaluation_time_sec - decoder_before_repair
            exclusive_repair_elapsed = max(
                0.0, elapsed_repair - decoder_in_repair
            )
            repair_profile["wall_sec"] += exclusive_repair_elapsed
            if normal_only_profile is not None:
                normal_only_profile.normal_repair_sec += (
                    exclusive_repair_elapsed
                )
            candidate_set_size = len(candidate_solutions)
            repair_profile["candidate_sets"] += 1
            repair_profile["candidates_generated"] += candidate_set_size
            candidate_sets += 1
            candidate_solutions_generated += candidate_set_size
            max_candidate_set_size = max(max_candidate_set_size, candidate_set_size)
            if uses_route_v2:
                candidate_solutions_full_evaluated += len(repair_outcome.evaluated)
                evaluated_candidates = [
                    (item.candidate.solution, item.evaluation, item.candidate)
                    for item in repair_outcome.feasible
                ]
            else:
                evaluated_candidates = []
                evaluated_raw: list[tuple[Solution, EvaluationResult]] = []
                use_threaded_batch = (
                    parallel_executor is not None
                    and threaded_evaluate is not None
                    and min(
                        candidate_set_size, limiter.remaining_evaluations
                    ) >= self.parallel_runtime.min_batch_size
                    and limiter.remaining_sec
                    > self.parallel_runtime.deadline_guard_sec
                )
                if use_threaded_batch:
                    batch_candidates = tuple(
                        candidate_solutions[: limiter.remaining_evaluations]
                    )
                    batch = evaluate_candidates_threaded(
                        candidates=batch_candidates,
                        evaluate=threaded_evaluate,
                        executor=parallel_executor,
                    )
                    batch_evaluations = tuple(
                        evaluation for _, evaluation in batch.evaluated
                    )
                    committed_before_deadline = (
                        limiter.commit_external_evaluations(
                            batch_evaluations,
                            duration_sec=batch.wall_sec,
                        )
                    )
                    full_evaluation_time_sec += batch.wall_sec
                    parallel_batch_calls += 1
                    parallel_candidate_count += len(batch.evaluated)
                    parallel_max_batch_size = max(
                        parallel_max_batch_size, len(batch.evaluated)
                    )
                    parallel_wall_sec += batch.wall_sec
                    parallel_wait_sec += batch.wait_sec
                    parallel_batch_max_sec = max(
                        parallel_batch_max_sec, batch.wall_sec
                    )
                    threaded_candidate_evals += len(batch.evaluated)
                    candidate_solutions_full_evaluated += len(batch.evaluated)
                    if committed_before_deadline:
                        evaluated_raw.extend(batch.evaluated)
                        for evaluation in batch_evaluations:
                            if evaluation.feasible:
                                feasible_evaluations += 1
                            else:
                                infeasible_evaluations += 1
                                for detail in evaluation.violation_details:
                                    infeasible_codes[detail.code] = (
                                        infeasible_codes.get(detail.code, 0) + 1
                                    )
                    else:
                        repair_budget_exhausted = True
                else:
                    for candidate in candidate_solutions:
                        evaluation = evaluate_v2_candidate(candidate)
                        if evaluation is None:
                            repair_budget_exhausted = True
                            break
                        candidate_solutions_full_evaluated += 1
                        serial_candidate_evals += 1
                        evaluated_raw.append((candidate, evaluation))
                emit_diagnostic(
                    "candidate_batch",
                    iteration=iteration,
                    elapsed_sec=limiter.elapsed_sec,
                    backend="thread" if use_threaded_batch else "serial",
                    candidates=tuple(candidate for candidate, _ in evaluated_raw),
                    evaluations=tuple(evaluation for _, evaluation in evaluated_raw),
                )
                evaluated_candidates.extend(
                    (candidate, evaluation, None)
                    for candidate, evaluation in evaluated_raw
                    if evaluation.feasible
                )
            if r4_profile_before is not None:
                r4_decoder_elapsed = max(
                    0.0, full_evaluation_time_sec - decoder_before_repair
                )
                r4_decoded = max(
                    0, limiter.evaluations - decoder_calls_before_repair
                )
                workload_profile.r4_decoder_sec += r4_decoder_elapsed
                workload_profile.r4_profile_total_sec += r4_decoder_elapsed
                workload_profile.r4_decoded_candidate_count += r4_decoded
                workload_profile.r4_feasible_candidate_count += len(
                    evaluated_candidates
                )
                workload_profile.r4_returned_candidate_count += candidate_set_size
            record_repair_workload(
                repair_name=repair_name,
                before=workload_before,
                missing_tasks=actual_removed,
                removed_sorties=removed_sortie_count,
                exclusive_repair_sec=exclusive_repair_elapsed,
                materialized_candidates=candidate_set_size,
                decoder_calls=limiter.evaluations - decoder_calls_before_repair,
            )
            if repair_budget_exhausted:
                repair_deadline_aborts += 1
                emit_diagnostic(
                    "iteration_end",
                    iteration=iteration,
                    elapsed_sec=limiter.elapsed_sec,
                    status="time_budget",
                    current_solution=current_solution,
                    current_evaluation=current_evaluation,
                    best_solution=best_solution,
                    best_evaluation=best_evaluation,
                    index=index,
                )
                finish_v32_iteration(
                    decision=escape_decision,
                    branch_started=branch_started,
                    current_objective_before=current_objective_before,
                    basin_best_objective_before=basin_best_objective_before,
                    global_best_objective_before=global_best_objective_before,
                    destroy_operator=destroy_name,
                    repair_operator=repair_name,
                    candidate_class=None,
                    accepted=False,
                    current_improvement=False,
                    basin_best_updated=False,
                    global_best_updated=False,
                    global_best_gain=0.0,
                    guided_before=guided_before,
                )
                break
            if not evaluated_candidates:
                stagnation += 1
                if not uses_time_temperature:
                    temperature *= self.config.cooling_rate
                emit_diagnostic(
                    "iteration_end",
                    iteration=iteration,
                    elapsed_sec=limiter.elapsed_sec,
                    status="no_feasible_candidate",
                    current_solution=current_solution,
                    current_evaluation=current_evaluation,
                    best_solution=best_solution,
                    best_evaluation=best_evaluation,
                    index=index,
                )
                finish_v32_iteration(
                    decision=escape_decision,
                    branch_started=branch_started,
                    current_objective_before=current_objective_before,
                    basin_best_objective_before=basin_best_objective_before,
                    global_best_objective_before=global_best_objective_before,
                    destroy_operator=destroy_name,
                    repair_operator=repair_name,
                    candidate_class=None,
                    accepted=False,
                    current_improvement=False,
                    basin_best_updated=False,
                    global_best_updated=False,
                    global_best_gain=0.0,
                    guided_before=guided_before,
                )
                continue
            candidate_solution, candidate_evaluation, selected_route_candidate = min(
                evaluated_candidates,
                key=lambda pair: pair[1].objective,
            )
            if selected_route_candidate is not None:
                assert route_diagnostics is not None
                assert repair_engine is not None
                assert destroy_result is not None
                assert route_context is not None
                route_diagnostics.record_candidate(
                    "selected", selected_route_candidate.primary_span
                )
                route_diagnostics.record_candidate_class(
                    "selected", selected_route_candidate.candidate_class
                )
                route_diagnostics.record_structural_tasks(
                    absorbed_task_ids=(
                        selected_route_candidate.forward_absorbed_task_ids
                    ),
                    relocated_task_ids=(
                        selected_route_candidate.forward_relocated_task_ids
                    ),
                )
                route_diagnostics.record_relocations(
                    repair_engine.relocation_deltas(
                        selected_route_candidate, destroy_result, route_context
                    ),
                    from_d6=destroy_name == "D6_route_corridor",
                )
                if selected_route_candidate.guided_target_sortie_id is not None:
                    route_diagnostics.record_guided_lifecycle("iteration_best")
            emit_diagnostic(
                "repair",
                iteration=iteration,
                elapsed_sec=limiter.elapsed_sec,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                repaired_solution=candidate_solution,
                repaired_evaluation=candidate_evaluation,
                index=index,
            )

            used_local: str | None = None
            local_budget_exhausted = False
            if (
                candidate_evaluation.feasible
                and search_action is not SearchAction.GLOBAL_ESCAPE
                and self.local_names
                and self.config.local_search_interval > 0
                and (
                    (
                        not uses_time_local_search
                        and iteration % self.config.local_search_interval == 0
                    )
                    or (
                        uses_time_local_search
                        and next_local_search_due_sec is not None
                        and limiter.elapsed_sec >= next_local_search_due_sec
                    )
                )
                and limiter.available(iteration)
            ):
                used_local = rng.choice(self.local_names)
                stats[used_local]["calls"] += 1
                segment_calls[used_local] += 1
                local_stage = f"local_search:{used_local}"
                local_deadline = lambda: limiter.deadline_expired(local_stage)
                started_local_search = perf_counter()
                if local_deadline():
                    local_solution = None
                    local_budget_exhausted = True
                else:
                    try:
                        local_solution = LOCAL_OPERATORS[used_local].function(
                            instance,
                            candidate_solution,
                            candidate_index,
                            rng,
                            deadline_expired=local_deadline,
                        )
                    except RepairTimeBudgetExceeded:
                        local_solution = None
                        local_budget_exhausted = True
                local_search_elapsed = perf_counter() - started_local_search
                limiter.observe_stage(local_stage, local_search_elapsed)
                if limiter.deadline.expired():
                    limiter.mark_deadline_abort(local_stage)
                    local_solution = None
                    local_budget_exhausted = True
                local_search_time_sec += local_search_elapsed
                if normal_only_profile is not None:
                    normal_only_profile.normal_local_search_sec += (
                        local_search_elapsed
                    )
                if local_solution is None:
                    local_evaluation = None
                else:
                    started_evaluation = perf_counter()
                    local_evaluation = limiter.try_evaluate(local_solution, iteration)
                    local_decoder_elapsed = perf_counter() - started_evaluation
                    full_evaluation_time_sec += local_decoder_elapsed
                    if (
                        normal_only_profile is not None
                        and local_evaluation is not None
                    ):
                        normal_only_profile.normal_decoder_sec += (
                            local_decoder_elapsed
                        )
                        normal_only_profile.full_decoder_calls += 1
                    if local_evaluation is None and limiter.deadline_abort_stage:
                        local_budget_exhausted = True
                if local_evaluation is not None:
                    if local_evaluation.feasible:
                        feasible_evaluations += 1
                    else:
                        infeasible_evaluations += 1
                        for detail in local_evaluation.violation_details:
                            infeasible_codes[detail.code] = infeasible_codes.get(detail.code, 0) + 1
                    if local_evaluation.feasible and local_evaluation.objective < candidate_evaluation.objective - 1e-9:
                        candidate_solution, candidate_evaluation = local_solution, local_evaluation
                if uses_time_local_search:
                    assert cadence_local_search_interval_sec is not None
                    assert next_local_search_due_sec is not None
                    next_local_search_due_sec = _next_time_cadence_due(
                        next_local_search_due_sec,
                        cadence_local_search_interval_sec,
                        limiter.elapsed_sec,
                    )

            if local_budget_exhausted:
                emit_diagnostic(
                    "iteration_end",
                    iteration=iteration,
                    elapsed_sec=limiter.elapsed_sec,
                    status="time_budget",
                    current_solution=current_solution,
                    current_evaluation=current_evaluation,
                    best_solution=best_solution,
                    best_evaluation=best_evaluation,
                    index=index,
                )
                finish_v32_iteration(
                    decision=escape_decision,
                    branch_started=branch_started,
                    current_objective_before=current_objective_before,
                    basin_best_objective_before=basin_best_objective_before,
                    global_best_objective_before=global_best_objective_before,
                    destroy_operator=destroy_name,
                    repair_operator=repair_name,
                    candidate_class=None,
                    accepted=False,
                    current_improvement=False,
                    basin_best_updated=False,
                    global_best_updated=False,
                    global_best_gain=0.0,
                    guided_before=guided_before,
                )
                break

            reward = 0.0
            accepted = False
            random_draw: float | None = None
            acceptance_probability = 0.0
            delta = candidate_evaluation.objective - current_evaluation.objective
            is_improvement = False
            is_global_best = False
            current_before_solution = current_solution
            current_before_evaluation = current_evaluation
            if candidate_evaluation.feasible:
                is_improvement = (
                    candidate_evaluation.objective
                    < current_evaluation.objective - 1e-9
                )
                is_global_best = (
                    candidate_evaluation.objective
                    < best_evaluation.objective - 1e-9
                )
                if v32_controller_enabled:
                    (
                        accepted,
                        acceptance_probability,
                        random_draw,
                    ) = _accept_search_candidate(
                        action=search_action,
                        delta=delta,
                        temperature=temperature,
                        rng=rng,
                    )
                elif delta <= 0:
                    acceptance_probability = 1.0
                    accepted = True
                else:
                    acceptance_probability = exp(
                        -delta / max(temperature, 1e-12)
                    )
                    random_draw = rng.random()
                    accepted = random_draw < acceptance_probability
                if is_global_best:
                    best_solution = candidate_solution.copy()
                    best_evaluation = candidate_evaluation
                    time_to_best = limiter.elapsed_sec
                    eval_to_best = limiter.evaluations
                    convergence.append(ConvergencePoint(iteration, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective))
                    search_audit.append(audit_point(iteration, best_evaluation))
                    reward = self.config.reward_global_best
                elif is_improvement:
                    reward = self.config.reward_improved
                elif accepted:
                    reward = self.config.reward_accepted
            if best_evaluation.objective < best_before_iteration - 1e-9:
                stagnation = 0
            else:
                stagnation += 1
            if accepted:
                current_solution = candidate_solution
                current_evaluation = candidate_evaluation
                stats[destroy_name]["accepted"] += 1
                stats[repair_name]["accepted"] += 1
                if used_local:
                    stats[used_local]["accepted"] += 1
            basin_best_updated = False
            if v32_controller_enabled and accepted:
                if search_action in {
                    SearchAction.GUIDED_ESCAPE,
                    SearchAction.GLOBAL_ESCAPE,
                }:
                    basin_best_solution = current_solution.copy()
                    basin_best_evaluation = current_evaluation
                    basin_best_updated = True
                elif (
                    current_evaluation.objective
                    < basin_best_evaluation.objective - 1e-9
                ):
                    basin_best_solution = current_solution.copy()
                    basin_best_evaluation = current_evaluation
                    basin_best_updated = True
            global_best_gain = (
                global_best_objective_before - best_evaluation.objective
                if is_global_best
                else 0.0
            )
            repair_quality = stats[repair_name]
            repair_quality["current_improvements"] += int(is_improvement)
            repair_quality["basin_best_updates"] += int(basin_best_updated)
            repair_quality["global_best_updates"] += int(is_global_best)
            repair_quality["total_current_gain"] += max(
                0.0,
                current_objective_before - candidate_evaluation.objective,
            )
            repair_quality["total_global_best_gain"] += max(
                0.0, global_best_gain
            )
            if reward >= self.config.reward_improved:
                stats[destroy_name]["improvements"] += 1
                stats[repair_name]["improvements"] += 1
                if used_local:
                    stats[used_local]["improvements"] += 1
            if selected_route_candidate is not None:
                assert route_diagnostics is not None
                if accepted:
                    route_diagnostics.record_candidate(
                        "accepted", selected_route_candidate.primary_span
                    )
                    route_diagnostics.record_candidate_class(
                        "accepted", selected_route_candidate.candidate_class
                    )
                    if selected_route_candidate.guided_target_sortie_id is not None:
                        route_diagnostics.record_guided_lifecycle("accepted")
                        route_diagnostics.record_b2_actual_dwell_reduction(
                            current_before_evaluation.usv_dwell_min
                            - candidate_evaluation.usv_dwell_min
                        )
                if is_improvement:
                    route_diagnostics.record_candidate(
                        "improved", selected_route_candidate.primary_span
                    )
                    route_diagnostics.record_candidate_class(
                        "improved", selected_route_candidate.candidate_class
                    )
                    if selected_route_candidate.guided_target_sortie_id is not None:
                        route_diagnostics.record_guided_lifecycle("improved")
            emit_diagnostic(
                "proposal",
                iteration=iteration,
                elapsed_sec=limiter.elapsed_sec,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                local_operator=used_local,
                current_before_solution=current_before_solution,
                current_before_evaluation=current_before_evaluation,
                candidate_solution=candidate_solution,
                candidate_evaluation=candidate_evaluation,
                current_after_solution=current_solution,
                current_after_evaluation=current_evaluation,
                best_after_solution=best_solution,
                best_after_evaluation=best_evaluation,
                delta=delta,
                temperature=temperature,
                acceptance_probability=acceptance_probability,
                random_draw=random_draw,
                accepted=accepted,
                is_improvement=is_improvement,
                is_global_best=is_global_best,
                search_action=search_action.value,
                p_jump=(escape_decision.p_jump if escape_decision else 0.0),
                descent_momentum=(
                    escape_decision.descent_momentum if escape_decision else 0.0
                ),
                stagnation_pressure=(
                    escape_decision.stagnation_pressure if escape_decision else 0.0
                ),
                index=index,
            )
            finish_v32_iteration(
                decision=escape_decision,
                branch_started=branch_started,
                current_objective_before=current_objective_before,
                basin_best_objective_before=basin_best_objective_before,
                global_best_objective_before=global_best_objective_before,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                candidate_class=(
                    selected_route_candidate.candidate_class
                    if selected_route_candidate is not None
                    else None
                ),
                accepted=accepted,
                current_improvement=is_improvement,
                basin_best_updated=basin_best_updated,
                global_best_updated=is_global_best,
                global_best_gain=global_best_gain,
                guided_before=guided_before,
            )
            for name in (destroy_name, repair_name, *((used_local,) if used_local else ())):
                stats[name]["reward"] += reward
                segment_rewards[name] += reward
            if iteration % self.config.segment_length == 0:
                update_segment_weights(weights, segment_calls, segment_rewards, self.config.reaction_factor)
                segment_calls = {name: 0 for name in operator_names}
                segment_rewards = {name: 0.0 for name in operator_names}
            if not uses_time_temperature:
                temperature *= self.config.cooling_rate
            emit_diagnostic(
                "iteration_end",
                iteration=iteration,
                elapsed_sec=limiter.elapsed_sec,
                status="proposal_evaluated",
                current_solution=current_solution,
                current_evaluation=current_evaluation,
                best_solution=best_solution,
                best_evaluation=best_evaluation,
                index=index,
            )

        update_segment_weights(weights, segment_calls, segment_rewards, self.config.reaction_factor)
        if uses_time_temperature:
            assert cadence_temperature_ratio is not None
            temperature = _temperature_at_elapsed_time(
                initial_temperature,
                cadence_temperature_ratio,
                self.config.search_cadence.reference_horizon_sec,
                limiter.elapsed_sec,
            )
        for name in stats:
            stats[name]["final_weight"] = weights[name]
        for name in self.repair_names:
            profile = stats[name]
            calls = profile["calls"]
            profile["mean_time_per_call_sec"] = profile["total_time_sec"] / calls if calls else 0.0
            profile["mean_wall_sec_per_call"] = profile["wall_sec"] / calls if calls else 0.0
            profile["mean_candidates_per_call"] = profile["candidates_generated"] / calls if calls else 0.0
            profile["mean_missing_tasks"] = profile["missing_tasks_sum"] / calls if calls else 0.0
            profile["repair_time_share"] = (
                profile["total_time_sec"] / repair_time_sec if repair_time_sec else 0.0
            )
            profile["gb_gain_per_sec"] = (
                profile["total_global_best_gain"] / profile["wall_sec"]
                if profile["wall_sec"] else 0.0
            )
        for name, profile in repair_internal_profiles.items():
            stats[name].update(profile.to_dict())
        destroy_call_count = sum(destroy_scale_calls.values())
        mean_target_removed_tasks = (
            destroy_target_tasks_sum / destroy_call_count
            if destroy_call_count > 0
            else 0.0
        )
        mean_actual_removed_tasks = (
            destroy_actual_tasks_sum / destroy_call_count
            if destroy_call_count > 0
            else 0.0
        )
        mean_candidate_set_size = (
            candidate_solutions_generated / candidate_sets
            if candidate_sets > 0
            else 0.0
        )
        destroy_overshoot_ratio = (
            destroy_actual_tasks_sum / destroy_target_tasks_sum
            if destroy_target_tasks_sum > 0
            else 0.0
        )
        local_search_calls = int(
            sum(stats[name]["calls"] for name in self.local_names)
        )
        global_best_updates = max(0, len(convergence) - 1)
        final_elapsed_sec = limiter.elapsed_sec
        repair_call_count = int(
            sum(stats[name]["calls"] for name in self.repair_names)
        )
        denominator = final_elapsed_sec if final_elapsed_sec > 0 else 1.0
        repair_workload_summary = {
            name: {
                "calls": len(repair_missing_sizes[name]),
                "missing_tasks": _integer_distribution(
                    repair_missing_sizes[name]
                ),
                "removed_sorties": _integer_distribution(
                    repair_removed_sorties[name]
                ),
            }
            for name in self.repair_names
        }
        repair_workload_by_missing_bin = tuple(
            {
                "repair": name,
                "missing_bin": label,
                **repair_workload_bins[name][label],
            }
            for name in self.repair_names
            for label in MISSING_SIZE_BINS
        )
        destroy_workload_summary = {
            "calls": len(destroy_task_sizes),
            "destroyed_tasks": _integer_distribution(destroy_task_sizes),
            "destroyed_sorties": _integer_distribution(destroy_sortie_sizes),
            "mode_calls": dict(destroy_scale_calls),
            "operator_calls": destroy_operator_calls,
        }
        r4p_stats = stats.get("R4P_progressive_recovery", {})
        emit_diagnostic(
            "final",
            iteration=iteration,
            elapsed_sec=limiter.elapsed_sec,
            current_solution=current_solution,
            current_evaluation=current_evaluation,
            best_solution=best_solution,
            best_evaluation=best_evaluation,
            index=index,
        )
        normal_only_metadata: dict[str, int | float | str] = {}
        if normal_only_profile is not None:
            assert escape_diagnostics is not None
            assert route_diagnostics is not None
            assert escape_diagnostics.action_calls[SearchAction.GUIDED_ESCAPE] == 0
            assert escape_diagnostics.action_calls[SearchAction.GLOBAL_ESCAPE] == 0
            assert route_diagnostics.d6_calls == 0
            assert route_diagnostics.b2_reconstructed == 0
            assert route_diagnostics.b2_full_decoder_calls == 0
            assert escape_diagnostics.normal_b2_calls == 0
            normal_only_profile.normal_candidate_build_sec = sum(
                profile.candidate_build_time_sec
                for profile in repair_internal_profiles.values()
            )
            normal_only_metadata = normal_only_profile.to_dict(
                wall_sec=final_elapsed_sec,
                iterations=iteration,
                generated_candidates=candidate_solutions_generated,
                effective_evaluations=limiter.evaluations,
            )
        return AlgorithmResult(
            algorithm=self.name,
            best_solution=best_solution,
            best_evaluation=best_evaluation,
            wall_time_sec=final_elapsed_sec,
            evaluations=limiter.evaluations,
            time_to_best_sec=time_to_best,
            eval_to_best=eval_to_best,
            convergence=tuple(convergence),
            operator_stats=stats,
            global_best_events=(
                tuple(escape_diagnostics.global_best_events)
                if escape_diagnostics is not None
                else ()
            ),
            escape_events=(
                tuple(escape_diagnostics.escape_events)
                if escape_diagnostics is not None
                else ()
            ),
            metadata={
                "variant": self.variant,
                **limiter.termination_metadata(iteration),
                **initialization.metadata,
                "destroy_operators": self.destroy_names,
                "repair_operators": self.repair_names,
                "repair_profile_mode": self.repair_profile_mode,
                "normal_profile_disable_r4p": (
                    self.config.normal_profile.disable_r4p
                ),
                "local_operators": self.local_names,
                "seed": seed,
                "iterations": iteration,
                "destroy_time_sec": destroy_time_sec,
                "repair_time_sec": repair_time_sec,
                "repair_runtime_errors": int(sum(
                    stats[name]["runtime_errors"]
                    for name in self.repair_names
                )),
                "repair_deadline_aborts": repair_deadline_aborts,
                "repair_operator_profiles": {name: stats[name].copy() for name in self.repair_names},
                "r4p_selected": int(r4p_stats.get("selected", 0)),
                "r4p_accepted": int(r4p_stats.get("accepted", 0)),
                "r4p_current_improvements": int(
                    r4p_stats.get("current_improvements", 0)
                ),
                "r4p_basin_best_updates": int(
                    r4p_stats.get("basin_best_updates", 0)
                ),
                "r4p_global_best_updates": int(
                    r4p_stats.get("global_best_updates", 0)
                ),
                "r4p_total_current_gain": float(
                    r4p_stats.get("total_current_gain", 0.0)
                ),
                "r4p_total_global_best_gain": float(
                    r4p_stats.get("total_global_best_gain", 0.0)
                ),
                "r4p_gb_gain_per_sec": float(
                    r4p_stats.get("gb_gain_per_sec", 0.0)
                ),
                "repair_internal_profiles": {
                    name: profile.to_dict()
                    for name, profile in repair_internal_profiles.items()
                },
                "r4_repair_internal_profile": repair_internal_profiles.get(
                    "R4_recovery_aware", RepairSearchProfile()
                ).to_dict(),
                "r4p_repair_internal_profile": repair_internal_profiles.get(
                    "R4P_progressive_recovery", RepairSearchProfile()
                ).to_dict(),
                "full_evaluation_time_sec": full_evaluation_time_sec,
                "local_search_time_sec": local_search_time_sec,
                "candidate_sets": candidate_sets,
                "candidate_set_count": candidate_sets,
                "candidate_solutions_generated": candidate_solutions_generated,
                "candidate_set_size_sum": candidate_solutions_generated,
                "candidate_solutions_full_evaluated": candidate_solutions_full_evaluated,
                "repair_calls": repair_call_count,
                "materialized_candidates": candidate_solutions_generated,
                "full_decoder_calls": limiter.evaluations,
                "effective_evaluations": limiter.evaluations,
                "iterations_per_sec": iteration / denominator,
                "repair_calls_per_sec": repair_call_count / denominator,
                "repair_calls_per_iteration": (
                    repair_call_count / iteration if iteration else 0.0
                ),
                "materialized_candidates_per_sec": (
                    candidate_solutions_generated / denominator
                ),
                "materialized_candidates_per_iteration": (
                    candidate_solutions_generated / iteration if iteration else 0.0
                ),
                "full_decoder_calls_per_sec": limiter.evaluations / denominator,
                "decoder_calls_per_iteration": (
                    limiter.evaluations / iteration if iteration else 0.0
                ),
                "effective_evaluation_definition": (
                    "Every Evaluator.evaluate call charged to EvaluationBudget, "
                    "including initialization, repair-candidate, and local-search decodes."
                ),
                "iteration_definition": (
                    "One pass through the ALNS destroy-repair proposal loop."
                ),
                "materialized_candidate_definition": (
                    "One complete repair Solution returned by the repair engine "
                    "before full decoding."
                ),
                "full_decoder_call_definition": (
                    "One Evaluator.evaluate invocation; identical to one effective evaluation."
                ),
                "repair_workload_summary": repair_workload_summary,
                "repair_workload_by_missing_bin": repair_workload_by_missing_bin,
                "destroy_workload_summary": destroy_workload_summary,
                "mean_candidate_set_size": mean_candidate_set_size,
                "max_candidate_set_size": max_candidate_set_size,
                "candidate_set_size_max": max_candidate_set_size,
                "parallel_backend": (
                    "thread" if self.parallel_runtime.uses_threads else "serial"
                ),
                "r4_kernel_backend": (
                    "numba" if r4_numba_enabled else "python"
                ),
                "r4_numba_enabled": r4_numba_enabled,
                "numba_compile_sec": r4_numba_compile_sec,
                "r4_numba_compiled_signature_count": (
                    compiled_signature_count() if r4_numba_enabled else 0
                ),
                "repair_kernel_backend": (
                    "numba" if repair_numba_enabled else "python"
                ),
                "repair_numba_enabled": repair_numba_enabled,
                "repair_numba_compile_sec": repair_numba_compile_sec,
                "repair_numba_compiled_signature_count": (
                    repair_compiled_signature_count()
                    if repair_numba_enabled
                    else 0
                ),
                "parallel_workers": (
                    self.parallel_runtime.max_workers
                    if self.parallel_runtime.uses_threads
                    else 1
                ),
                "parallel_batch_calls": parallel_batch_calls,
                "parallel_candidate_count": parallel_candidate_count,
                "parallel_mean_batch_size": (
                    parallel_candidate_count / parallel_batch_calls
                    if parallel_batch_calls else 0.0
                ),
                "parallel_max_batch_size": parallel_max_batch_size,
                "parallel_batch_max_sec": parallel_batch_max_sec,
                "parallel_wall_sec": parallel_wall_sec,
                "parallel_wait_sec": parallel_wait_sec,
                "serial_candidate_evals": serial_candidate_evals,
                "threaded_candidate_evals": threaded_candidate_evals,
                "initial_temperature": initial_temperature,
                "final_temperature": temperature,
                "temperature_cadence_mode": (
                    "elapsed_time" if uses_time_temperature else "iteration"
                ),
                "local_search_cadence_mode": (
                    "elapsed_time" if uses_time_local_search else "iteration"
                ),
                "temperature_reference_ratio": cadence_temperature_ratio,
                "temperature_reference_horizon_sec": (
                    self.config.search_cadence.reference_horizon_sec
                    if uses_time_temperature
                    else None
                ),
                "local_search_cadence_sec": cadence_local_search_interval_sec,
                "final_stagnation": stagnation,
                "segment_length": self.config.segment_length,
                "feasible_evaluations": feasible_evaluations,
                "infeasible_evaluations": infeasible_evaluations,
                "infeasible_codes": infeasible_codes,
                "destroy_call_count": destroy_call_count,
                "destroy_legacy_calls": destroy_scale_calls["legacy"],
                "destroy_micro_calls": destroy_scale_calls["micro"],
                "destroy_medium_calls": destroy_scale_calls["medium"],
                "destroy_macro_calls": destroy_scale_calls["macro"],
                "destroy_target_tasks_sum": destroy_target_tasks_sum,
                "destroy_actual_tasks_sum": destroy_actual_tasks_sum,
                "destroy_actual_tasks_max": destroy_actual_tasks_max,
                "mean_target_removed_tasks": mean_target_removed_tasks,
                "mean_actual_removed_tasks": mean_actual_removed_tasks,
                "destroy_overshoot_ratio": destroy_overshoot_ratio,
                "local_search_calls": local_search_calls,
                "global_best_updates": global_best_updates,
                "diagnostic_trace_enabled": self.diagnostic_hook is not None,
                "diagnostic_trace_overhead_sec": diagnostic_trace_overhead_sec,
                "search_audit": search_audit,
                "search_engine": self.config.search_engine,
                "search_engine_version": self.config.search_engine_version,
                "controller_mode": (
                    self.config.route_guided_v3_2.controller_mode
                    if v32_controller_enabled else "none"
                ),
                **normal_only_metadata,
                **(
                    {
                        "search_rng_seed": search_rng_seed,
                        "controller_rng_seed": controller_rng_seed,
                        "controller_semantics_version": (
                            self.config.controller_semantics_version
                        ),
                        "time_budget_sec": budget.wall_time_sec,
                        "medium_escape_ratio": (
                            self.config.route_guided_v3_2.escape.medium_budget_ratio
                        ),
                        "global_escape_ratio": (
                            self.config.route_guided_v3_2.escape.global_budget_ratio
                        ),
                        "medium_escape_threshold_sec": (
                            escape_controller.medium_threshold_sec
                        ),
                        "global_escape_threshold_sec": (
                            escape_controller.global_threshold_sec
                        ),
                        "final_basin_best_objective": (
                            basin_best_evaluation.objective
                        ),
                        **escape_diagnostics.to_dict(),
                    }
                    if escape_diagnostics is not None
                    and escape_controller is not None
                    else {}
                ),
                **(
                    route_diagnostics.to_dict()
                    if route_diagnostics is not None
                    else {}
                ),
            },
        )


class VanillaALNS(ALNSSolver):
    def __init__(
        self,
        config: ALNSConfig | None = None,
        *,
        diagnostic_hook: Callable[[str, dict[str, Any]], None] | None = None,
        parallel_runtime: ParallelRuntimeConfig | None = None,
        r4_kernel_backend: R4KernelBackend = "auto",
        repair_profile_mode: RepairProfileMode = "full",
        repair_kernel_backend: RepairKernelBackend = "python",
    ) -> None:
        super().__init__(
            config=config or _default_config("vanilla_alns.yaml"),
            variant="A0",
            diagnostic_hook=diagnostic_hook,
            parallel_runtime=parallel_runtime,
            r4_kernel_backend=r4_kernel_backend,
            repair_profile_mode=repair_profile_mode,
            repair_kernel_backend=repair_kernel_backend,
        )


class ProposedALNS(ALNSSolver):
    def __init__(
        self,
        config: ALNSConfig | None = None,
        *,
        variant: str = "FULL",
        diagnostic_hook: Callable[[str, dict[str, Any]], None] | None = None,
        parallel_runtime: ParallelRuntimeConfig | None = None,
        r4_kernel_backend: R4KernelBackend = "auto",
        repair_profile_mode: RepairProfileMode | None = None,
        repair_kernel_backend: RepairKernelBackend | None = None,
    ) -> None:
        if variant == PSALNS_VARIANT:
            frozen_config = _default_config("alns_ms1.yaml")
            if config is not None and config != frozen_config:
                raise ValueError(
                    "PSALNS Formal V1 parameters are frozen to alns_ms1.yaml"
                )
            config = frozen_config
        super().__init__(
            config=config
            or _default_config(
                "alns_ms1.yaml" if variant == PSALNS_VARIANT else "alns.yaml"
            ),
            variant=variant,
            diagnostic_hook=diagnostic_hook,
            parallel_runtime=parallel_runtime,
            r4_kernel_backend=r4_kernel_backend,
            repair_profile_mode=(
                repair_profile_mode
                if repair_profile_mode is not None
                else "summary" if variant == PSALNS_VARIANT else "full"
            ),
            repair_kernel_backend=(
                repair_kernel_backend
                if repair_kernel_backend is not None
                else "auto" if variant == PSALNS_VARIANT else "python"
            ),
        )


class PSALNS(ProposedALNS):
    """Formal V1 name for the frozen V3.2.1-Pilot Baseline.

    The public name is new; the implementation is deliberately normalized to
    ``MS1_TIME_SA_LS`` so historical Pilot runs and new PS-ALNS runs execute
    the same S4C search path with the same ``alns_ms1.yaml`` parameters.
    """

    def __init__(
        self,
        config: ALNSConfig | None = None,
        *,
        diagnostic_hook: Callable[[str, dict[str, Any]], None] | None = None,
        parallel_runtime: ParallelRuntimeConfig | None = None,
        r4_kernel_backend: R4KernelBackend = "auto",
        repair_profile_mode: RepairProfileMode = "summary",
        repair_kernel_backend: RepairKernelBackend = "auto",
    ) -> None:
        super().__init__(
            config=config,
            variant=PSALNS_VARIANT,
            diagnostic_hook=diagnostic_hook,
            parallel_runtime=parallel_runtime,
            r4_kernel_backend=r4_kernel_backend,
            repair_profile_mode=repair_profile_mode,
            repair_kernel_backend=repair_kernel_backend,
        )
