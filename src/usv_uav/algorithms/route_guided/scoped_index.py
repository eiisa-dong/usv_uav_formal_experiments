from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping

from usv_uav.algorithms.route_guided.context import RouteSearchContext
from usv_uav.algorithms.route_guided.models import (
    NeighborhoodScope,
    RepairPolicy,
    RouteCorridor,
)
from usv_uav.algorithms.route_guided.scoring import RouteScore, sortie_route_score
from usv_uav.preprocessing.sortie_index import SortieIndex


@dataclass(frozen=True, slots=True)
class ForwardPartnerCandidate:
    partner_task: int
    compatible_sortie_ids: tuple[int, ...]
    best_compatible_route_score: RouteScore


@dataclass(frozen=True, slots=True)
class ScopedSortieIndexView:
    """A read-only candidate filter over the canonical S4C sortie index."""

    context: RouteSearchContext
    previous_origin_positions: Mapping[int, int]
    policy: RepairPolicy
    corridor: RouteCorridor | None = None
    physical_feasibility_aware: bool = False
    guided_target_sortie_id: int | None = None
    excluded_sortie_ids: frozenset[int] = frozenset()
    protected_corridor: RouteCorridor | None = None
    strict_protected_corridor: bool = False
    deadline_expired: Callable[[], bool] | None = field(
        default=None, repr=False, compare=False
    )
    _seen_generation: list[int] = field(init=False, repr=False, compare=False)
    _generation: int = field(init=False, repr=False, compare=False)
    _candidate_qids: list[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Scratch storage belongs to this repair-local view, so repeated missing-task
        # lookups avoid allocating and hashing a new set without sharing mutable state
        # across repairs or solver threads.
        object.__setattr__(self, "_seen_generation", [0] * len(self.context.sortie_object))
        object.__setattr__(self, "_generation", 0)
        object.__setattr__(self, "_candidate_qids", [])

    @property
    def base(self) -> SortieIndex:
        return self.context.sortie_index

    @property
    def uses_dense_repair_kernel(self) -> bool:
        return True

    @property
    def sortie_by_id(self):
        # Retained solution sequences may contain sorties outside the candidate scope.
        return self.base.sortie_by_id

    @property
    def support_position(self):
        return self.base.support_position

    @property
    def sortie_rank_by_id(self):
        return self.base.sortie_rank_by_id

    @property
    def sortie_ids_in_original_order(self):
        return self.base.sortie_ids_in_original_order

    def _iter_qids_for_task(
        self,
        task_id: int,
        previous_origin_pos: int,
        scope: NeighborhoodScope,
        corridor: RouteCorridor | None = None,
    ) -> Iterator[int]:
        active_corridor = self.corridor if corridor is None else corridor
        if not 0 <= previous_origin_pos < len(self.context.support_order):
            raise ValueError("previous origin is outside the frozen support route")
        context = self.context
        sortie_ids = context.sortie_ids_by_qid
        origin_pos = context.origin_pos
        recovery_pos = context.recovery_pos
        nominal_time = context.nominal_time
        physical_possible = context.physical_possible
        route_sail_prefix = context.route_sail_prefix
        physical_recovery_horizon = context.physical_recovery_horizon
        excluded_sortie_ids = self.excluded_sortie_ids
        guided_target_sortie_id = self.guided_target_sortie_id
        physical_feasibility_aware = self.physical_feasibility_aware
        protected_corridor = self.protected_corridor
        strict_protected_corridor = self.strict_protected_corridor
        for qid in context.task_to_qids.get(task_id, ()):
            sortie_id = sortie_ids[qid]
            origin = origin_pos[qid]
            recovery = recovery_pos[qid]
            is_guided_target = sortie_id == self.guided_target_sortie_id
            if sortie_id in excluded_sortie_ids and not is_guided_target:
                continue
            if (
                active_corridor is not None
                and not (
                    physical_feasibility_aware
                    and guided_target_sortie_id is not None
                    and not is_guided_target
                )
                and (
                not active_corridor.contains(origin)
                or (
                    physical_feasibility_aware
                    and not active_corridor.contains(recovery)
                )
                )
            ):
                continue
            if physical_feasibility_aware and not physical_possible[qid]:
                continue
            if (
                strict_protected_corridor
                and protected_corridor is not None
                and (
                    protected_corridor.start_pos
                    < origin
                    < protected_corridor.end_pos
                    or protected_corridor.start_pos
                    < recovery
                    < protected_corridor.end_pos
                )
            ):
                # A residual sortie with an event inside the target corridor can
                # recreate dwell after B2 has already been fixed and launched.
                continue
            if is_guided_target:
                yield qid
                continue
            if (
                protected_corridor is not None
                and protected_corridor.contains(origin)
                and recovery < protected_corridor.end_pos
            ):
                if (
                    nominal_time[qid]
                    > route_sail_prefix[recovery] - route_sail_prefix[origin] + 1e-9
                ):
                    continue
            if scope is NeighborhoodScope.GLOBAL:
                yield qid
                continue
            radius = self.policy.origin_radius
            max_span = self.policy.max_recovery_span
            if radius is None or max_span is None:
                raise ValueError("bounded scopes require radius and span limits")
            recovery_allowed = (
                recovery in physical_recovery_horizon[origin]
                if physical_feasibility_aware
                else recovery - origin <= max_span
            )
            if abs(origin - previous_origin_pos) <= radius and recovery_allowed:
                yield qid

    def query_for_task(
        self,
        task_id: int,
        previous_origin_pos: int,
        scope: NeighborhoodScope,
        corridor: RouteCorridor | None = None,
    ) -> tuple[int, ...]:
        sortie_ids = self.context.sortie_ids_by_qid
        return tuple(
            sortie_ids[qid]
            for qid in self._iter_qids_for_task(
                task_id, previous_origin_pos, scope, corridor
            )
        )

    def by_task(self, task_id: int) -> tuple[int, ...]:
        previous = self.previous_origin_positions.get(task_id)
        if previous is None:
            return ()
        return self.query_for_task(task_id, previous, self.policy.scope)

    def by_task_origin(self, task_id: int, origin_position: int) -> tuple[int, ...]:
        qid_by_sortie_id = self.context.qid_by_sortie_id
        origin_pos = self.context.origin_pos
        return tuple(
            sortie_id
            for sortie_id in self.by_task(task_id)
            if origin_pos[qid_by_sortie_id[sortie_id]] == origin_position
        )

    def by_task_origin_span(
        self,
        task_id: int,
        origin_position: int,
        span: int,
    ) -> tuple[int, ...]:
        qid_by_sortie_id = self.context.qid_by_sortie_id
        origin_pos = self.context.origin_pos
        recovery_pos = self.context.recovery_pos
        return tuple(
            sortie_id
            for sortie_id in self.by_task_origin(task_id, origin_position)
            if (
                recovery_pos[qid_by_sortie_id[sortie_id]]
                - origin_pos[qid_by_sortie_id[sortie_id]]
                == span
            )
        )

    def uncovered_tasks(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        return tuple(task_id for task_id in task_ids if not self.by_task(task_id))

    def candidate_qids(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        generation = self._generation + 1
        object.__setattr__(self, "_generation", generation)
        seen_generation = self._seen_generation
        qids = self._candidate_qids
        qids.clear()
        context = self.context
        task_to_qids = context.task_to_qids
        previous_origins = self.previous_origin_positions
        scope = self.policy.scope
        # NORMAL's common path has no guided/protected exclusions.  Keep the
        # branch outside the candidate loop so dense-array access is the only
        # per-qid data lookup before generation-stamp deduplication.
        if (
            self.corridor is None
            and not self.excluded_sortie_ids
            and self.guided_target_sortie_id is None
            and self.protected_corridor is None
            and not self.strict_protected_corridor
        ):
            physical = self.physical_feasibility_aware
            physical_possible = context.physical_possible
            origin_pos = context.origin_pos
            recovery_pos = context.recovery_pos
            if scope is NeighborhoodScope.GLOBAL:
                for task_id in task_ids:
                    if task_id not in previous_origins:
                        continue
                    for qid in task_to_qids.get(task_id, ()):
                        if physical and not physical_possible[qid]:
                            continue
                        if seen_generation[qid] != generation:
                            seen_generation[qid] = generation
                            qids.append(qid)
            else:
                radius = self.policy.origin_radius
                max_span = self.policy.max_recovery_span
                if radius is None or max_span is None:
                    raise ValueError("bounded scopes require radius and span limits")
                for task_id in task_ids:
                    previous = previous_origins.get(task_id)
                    if previous is None:
                        continue
                    for qid in task_to_qids.get(task_id, ()):
                        if physical and not physical_possible[qid]:
                            continue
                        origin = origin_pos[qid]
                        if (
                            abs(origin - previous) <= radius
                            and (
                                physical
                                or recovery_pos[qid] - origin <= max_span
                            )
                            and seen_generation[qid] != generation
                        ):
                            seen_generation[qid] = generation
                            qids.append(qid)
        else:
            for task_id in task_ids:
                previous = previous_origins.get(task_id)
                if previous is None:
                    continue
                for qid in self._iter_qids_for_task(task_id, previous, scope):
                    if seen_generation[qid] != generation:
                        seen_generation[qid] = generation
                        qids.append(qid)
        qids.sort()
        return tuple(qids)

    def candidate_sortie_ids(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        sortie_ids = self.context.sortie_ids_by_qid
        return tuple(sortie_ids[qid] for qid in self.candidate_qids(task_ids))

    def compatible_forward_partners(
        self,
        anchor_task: int,
        corridor: RouteCorridor,
        *,
        anchor_origin_position: int | None = None,
        max_span: int = 2,
    ) -> tuple[ForwardPartnerCandidate, ...]:
        """Return pool-proven partners without constructing a quadratic pair index."""
        if max_span < 0:
            raise ValueError("max_span must be non-negative")
        anchor_position = (
            corridor.start_pos
            if anchor_origin_position is None
            else anchor_origin_position
        )
        if not corridor.contains(anchor_position):
            raise ValueError("anchor origin must lie inside the corridor")

        ids_by_partner: dict[int, list[int]] = {}
        for sortie_id in self.base.by_task(anchor_task):
            if (
                self.physical_feasibility_aware
                and not self.context.is_physically_feasible(sortie_id)
            ):
                continue
            origin = self.context.sortie_origin_position[sortie_id]
            recovery = self.context.sortie_recovery_position[sortie_id]
            span = self.context.sortie_span[sortie_id]
            if (
                span > max_span
                or origin < anchor_position
                or origin > anchor_position + 1
                or recovery > anchor_position + 2
                or not corridor.contains(origin)
                or not corridor.contains(recovery)
            ):
                continue
            for task_id in self.base.sortie_by_id[sortie_id].task_sequence:
                if task_id != anchor_task:
                    ids_by_partner.setdefault(task_id, []).append(sortie_id)

        candidates: list[ForwardPartnerCandidate] = []
        for partner_task in sorted(ids_by_partner):
            compatible_ids = tuple(ids_by_partner[partner_task])
            candidates.append(
                ForwardPartnerCandidate(
                    partner_task=partner_task,
                    compatible_sortie_ids=compatible_ids,
                    best_compatible_route_score=min(
                        sortie_route_score(sortie_id, self.context)
                        for sortie_id in compatible_ids
                    ),
                )
            )
        return tuple(candidates)


class ScopedIndexReference(ScopedSortieIndexView):
    """Frozen mapping/set implementation retained as the equivalence oracle."""

    @property
    def uses_dense_repair_kernel(self) -> bool:
        return False

    def query_for_task(
        self,
        task_id: int,
        previous_origin_pos: int,
        scope: NeighborhoodScope,
        corridor: RouteCorridor | None = None,
    ) -> tuple[int, ...]:
        active_corridor = self.corridor if corridor is None else corridor
        if not 0 <= previous_origin_pos < len(self.context.support_order):
            raise ValueError("previous origin is outside the frozen support route")
        result: list[int] = []
        for sortie_id in self.base.by_task(task_id):
            origin = self.context.sortie_origin_position[sortie_id]
            recovery = self.context.sortie_recovery_position[sortie_id]
            span = self.context.sortie_span[sortie_id]
            is_guided_target = sortie_id == self.guided_target_sortie_id
            if sortie_id in self.excluded_sortie_ids and not is_guided_target:
                continue
            if (
                active_corridor is not None
                and not (
                    self.physical_feasibility_aware
                    and self.guided_target_sortie_id is not None
                    and not is_guided_target
                )
                and (
                    not active_corridor.contains(origin)
                    or (
                        self.physical_feasibility_aware
                        and not active_corridor.contains(recovery)
                    )
                )
            ):
                continue
            if (
                self.physical_feasibility_aware
                and not self.context.is_physically_feasible(sortie_id)
            ):
                continue
            if (
                self.strict_protected_corridor
                and self.protected_corridor is not None
                and (
                    self.protected_corridor.start_pos
                    < origin
                    < self.protected_corridor.end_pos
                    or self.protected_corridor.start_pos
                    < recovery
                    < self.protected_corridor.end_pos
                )
            ):
                continue
            if is_guided_target:
                result.append(sortie_id)
                continue
            if (
                self.protected_corridor is not None
                and self.protected_corridor.contains(origin)
                and recovery < self.protected_corridor.end_pos
            ):
                sortie = self.base.sortie_by_id[sortie_id]
                if (
                    sortie.nominal_duration_min
                    > self.context.usv_time(origin, recovery) + 1e-9
                ):
                    continue
            if scope is NeighborhoodScope.GLOBAL:
                result.append(sortie_id)
                continue
            radius = self.policy.origin_radius
            max_span = self.policy.max_recovery_span
            if radius is None or max_span is None:
                raise ValueError("bounded scopes require radius and span limits")
            recovery_allowed = (
                recovery in self.context.physical_recovery_horizon[origin]
                if self.physical_feasibility_aware
                else span <= max_span
            )
            if abs(origin - previous_origin_pos) <= radius and recovery_allowed:
                result.append(sortie_id)
        return tuple(result)

    def candidate_qids(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        ranks: set[int] = set()
        for task_id in task_ids:
            ranks.update(
                self.base.sortie_rank_by_id[sortie_id]
                for sortie_id in self.by_task(task_id)
            )
        return tuple(sorted(ranks))

    def candidate_sortie_ids(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        sortie_ids = self.base.sortie_ids_in_original_order
        return tuple(sortie_ids[qid] for qid in self.candidate_qids(task_ids))
