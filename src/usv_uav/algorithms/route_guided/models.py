from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping

from usv_uav.core.partial_solution import PartialSolution


class NeighborhoodScope(str, Enum):
    LOCAL = "local"
    FORWARD = "forward"
    GLOBAL = "global"


class SearchAction(str, Enum):
    NORMAL = "normal"
    GUIDED_ESCAPE = "guided_escape"
    GLOBAL_ESCAPE = "global_escape"


@dataclass(frozen=True, slots=True)
class EscapeDecision:
    action: SearchAction
    descent_rate: float
    reference_descent_rate: float
    descent_momentum: float
    basin_stagnation_sec: float
    global_stagnation_sec: float
    stagnation_pressure: float
    hazard_rate: float
    delta_t: float
    p_jump: float
    random_draw: float | None
    guided_rearmed: bool
    global_rearmed: bool
    guided_attempted_in_current_basin: bool

    @property
    def probability_eligible(self) -> bool:
        return self.random_draw is not None


class CandidateClass(str, Enum):
    # Historical V2.1/V2.2 diagnostics.
    SAME_POINT = "c0"
    ADJACENT_RECOVERY = "c1"
    FORWARD_RELOCATION = "c2"
    FORWARD_ABSORPTION = "c3"

    # Final V3 reconstruction roles.  These describe why a candidate is sent
    # to the decoder rather than imposing a recovery-span quota.
    LEGACY_LOCAL_BEST = "b0"
    ROUTE_GUIDED_LOCAL = "b1"
    GUIDED_PHYSICAL_FORWARD = "b2"
    GLOBAL_DIVERSIFICATION = "b3"


class ForwardFeasibility(str, Enum):
    STATIC_IMPOSSIBLE = "static_impossible"
    DYNAMIC_BLOCKED = "physically_possible_but_blocked"
    CURRENTLY_FEASIBLE = "currently_feasible"


@dataclass(frozen=True, slots=True)
class RouteCorridor:
    start_pos: int
    end_pos: int

    def __post_init__(self) -> None:
        if self.start_pos < 0 or self.end_pos < self.start_pos:
            raise ValueError("route corridor bounds must satisfy 0 <= start <= end")

    @property
    def width(self) -> int:
        return self.end_pos - self.start_pos + 1

    def contains(self, position: int) -> bool:
        return self.start_pos <= position <= self.end_pos


@dataclass(frozen=True, slots=True)
class PhysicalForwardCandidate:
    sortie_id: int
    origin_pos: int
    recovery_pos: int
    movement_time_min: float
    lower_bound_duration_min: float
    lower_bound_energy_wh: float
    time_slack_min: float
    energy_slack_min: float
    intermediate_dwell_budget_min: float = float("inf")

    @property
    def span(self) -> int:
        return self.recovery_pos - self.origin_pos

    @property
    def physical_slack_min(self) -> float:
        return min(self.time_slack_min, self.energy_slack_min)

    @property
    def statically_feasible(self) -> bool:
        return self.physical_slack_min >= -1e-9


@dataclass(frozen=True, slots=True)
class BlockingAnalysis:
    target_sortie_id: int
    physical_slack_min: float
    blocking_support_positions: tuple[int, ...]
    blocking_sortie_ids: tuple[int, ...]
    estimated_blocking_time_min: float
    residual_blocking_time_min: float
    required_removed_task_ids: tuple[int, ...]
    feasibility: ForwardFeasibility

    @property
    def release_sufficient(self) -> bool:
        return self.residual_blocking_time_min <= self.physical_slack_min + 1e-9


@dataclass(frozen=True, slots=True)
class DwellGuidedAnalysis:
    target_sortie_id: int
    critical_support_position: int
    critical_support_id: int
    intermediate_dwell_budget_min: float
    current_intermediate_dwell_min: float
    residual_intermediate_dwell_min: float
    required_dwell_release_min: float
    owner_sortie_ids: tuple[int, ...]
    blocking_sortie_ids: tuple[int, ...]
    required_removed_task_ids: tuple[int, ...]
    released_dwell_gain_min: float
    new_wait_min: float
    estimated_dwell_gain_min: float
    dynamic_slack_min: float
    feasibility: ForwardFeasibility

    @property
    def release_sufficient(self) -> bool:
        return (
            self.residual_intermediate_dwell_min
            <= self.intermediate_dwell_budget_min + 1e-9
        )

    @property
    def positive_gain(self) -> bool:
        return self.estimated_dwell_gain_min > 1e-9

    @property
    def ranking_key(self) -> tuple[float, int, float, int]:
        return (
            -self.estimated_dwell_gain_min,
            len(self.required_removed_task_ids),
            -self.dynamic_slack_min,
            self.target_sortie_id,
        )


@dataclass(frozen=True, slots=True)
class GuidedRepairTarget:
    target_sortie_id: int
    required_missing_task_ids: tuple[int, ...]
    corridor: RouteCorridor
    owner_sortie_ids: tuple[int, ...]
    blocker_sortie_ids: tuple[int, ...]
    release_scale: Literal["micro", "medium", "macro"]
    critical_support_position: int | None = None
    estimated_dwell_gain_min: float = 0.0
    release_task_budget: int | None = None
    strict_no_undo: bool = False

    def __post_init__(self) -> None:
        if not self.required_missing_task_ids:
            raise ValueError("guided target must release at least one task")
        if len(set(self.required_missing_task_ids)) != len(
            self.required_missing_task_ids
        ):
            raise ValueError("guided target missing tasks must be unique")
        if len(set(self.blocker_sortie_ids)) != len(self.blocker_sortie_ids):
            raise ValueError("guided target blocker sorties must be unique")
        if not self.owner_sortie_ids:
            raise ValueError("guided target must have at least one current owner sortie")
        if len(set(self.owner_sortie_ids)) != len(self.owner_sortie_ids):
            raise ValueError("guided target owner sorties must be unique")


@dataclass(frozen=True, slots=True)
class CorridorRepairContext:
    corridor: RouteCorridor | None
    previous_origin_by_task: Mapping[int, int]
    anchor_task: int | None
    compatible_partner_tasks: tuple[int, ...]
    guided_target: GuidedRepairTarget | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_origin_by_task",
            MappingProxyType(dict(self.previous_origin_by_task)),
        )
        if len(set(self.compatible_partner_tasks)) != len(
            self.compatible_partner_tasks
        ):
            raise ValueError("compatible partner tasks must be unique")
        if self.guided_target is not None:
            if self.corridor != self.guided_target.corridor:
                raise ValueError("guided target corridor must match repair corridor")
            if not set(self.guided_target.required_missing_task_ids) <= set(
                self.previous_origin_by_task
            ):
                raise ValueError("guided target tasks must belong to Missing")


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    scope: NeighborhoodScope
    origin_radius: int | None
    max_recovery_span: int | None
    allow_global_fallback: bool
    destroy_scale: Literal["legacy", "micro", "medium", "macro"]
    corridor_backward: int
    corridor_forward: int
    r4_span0_quota: int = 1
    r4_span1_quota: int = 1
    r4_span2_quota: int = 2

    def __post_init__(self) -> None:
        if self.origin_radius is not None and self.origin_radius < 0:
            raise ValueError("origin_radius must be non-negative or None")
        if self.max_recovery_span is not None and self.max_recovery_span < 0:
            raise ValueError("max_recovery_span must be non-negative or None")
        if min(self.corridor_backward, self.corridor_forward) < 0:
            raise ValueError("corridor widths must be non-negative")
        if min(self.r4_span0_quota, self.r4_span1_quota, self.r4_span2_quota) < 0:
            raise ValueError("R4P span quotas must be non-negative")
        if self.r4_span0_quota + self.r4_span1_quota + self.r4_span2_quota != 4:
            raise ValueError("R4P span quotas must preserve the Top-M=4 budget")


@dataclass(frozen=True, slots=True)
class DestroyResult:
    partial: PartialSolution
    removed_task_ids: tuple[int, ...]
    previous_origin_positions: tuple[tuple[int, int], ...]
    corridor: RouteCorridor | None = None
    repair_context: CorridorRepairContext | None = None

    def __post_init__(self) -> None:
        removed = set(self.removed_task_ids)
        if removed != self.partial.missing_tasks:
            raise ValueError("removed_task_ids must exactly match partial.missing_tasks")
        previous = dict(self.previous_origin_positions)
        if set(previous) != removed:
            raise ValueError("every removed task must have one previous origin position")
        if self.repair_context is not None:
            if dict(self.repair_context.previous_origin_by_task) != previous:
                raise ValueError("repair context origins must match destroyed tasks")
            if self.repair_context.corridor != self.corridor:
                raise ValueError("repair context corridor must match destroy corridor")
            if not set(self.repair_context.compatible_partner_tasks) <= removed:
                raise ValueError("repair context partners must belong to missing tasks")

    @property
    def previous_origin_by_task(self) -> dict[int, int]:
        return dict(self.previous_origin_positions)
