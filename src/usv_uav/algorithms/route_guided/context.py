from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from usv_uav.algorithms.route_guided.models import PhysicalForwardCandidate
from usv_uav.config import UAVConfig
from usv_uav.core.models import Instance, Sortie
from usv_uav.preprocessing.sortie_index import SortieIndex


@dataclass(frozen=True, slots=True)
class RouteSearchContext:
    """Immutable caches derived from the frozen first-layer support route."""

    support_order: tuple[int, ...]
    support_position: Mapping[int, int]
    route_sail_prefix: tuple[float, ...]
    # Dense, original-pool-order repair-kernel storage.  A qid is the position
    # in these tuples and is therefore also the canonical sortie pool rank.
    sortie_ids_by_qid: tuple[int, ...]
    qid_by_sortie_id: Mapping[int, int]
    origin_pos: tuple[int, ...]
    recovery_pos: tuple[int, ...]
    nominal_time: tuple[float, ...]
    nominal_energy: tuple[float, ...]
    proxy_base: tuple[float, ...]
    task_signature: tuple[tuple[int, ...], ...]
    task_count: tuple[int, ...]
    physical_possible: tuple[bool, ...]
    sortie_object: tuple[Sortie, ...]
    task_to_qids: Mapping[int, tuple[int, ...]]
    sortie_origin_position: Mapping[int, int]
    sortie_recovery_position: Mapping[int, int]
    sortie_span: Mapping[int, int]
    physical_profile_by_sortie: Mapping[int, PhysicalForwardCandidate]
    physical_recovery_horizon: Mapping[int, tuple[int, ...]]
    physically_feasible_sortie_ids: frozenset[int]
    physical_forward_sortie_ids: tuple[int, ...]
    physical_forward_rank_by_sortie: Mapping[int, int]
    physical_forward_by_support_task: Mapping[
        tuple[int, int], tuple[int, ...]
    ]
    static_impossible_forward_sortie_ids: tuple[int, ...]
    sortie_index: SortieIndex

    @classmethod
    def build(
        cls,
        instance: Instance,
        sortie_index: SortieIndex,
        usv_speed_km_min: float,
        uav: UAVConfig | None = None,
    ) -> "RouteSearchContext":
        support_order = tuple(instance.usv_route[1:-1])
        support_position = dict(sortie_index.support_position)
        if tuple(sorted(support_position, key=support_position.get)) != support_order:
            raise ValueError("sortie index support order does not match the frozen USV route")
        if usv_speed_km_min <= 0:
            raise ValueError("usv_speed_km_min must be positive")
        matrix_index = {
            support_id: index + 1
            for index, support_id in enumerate(instance.selected_supports)
        }
        prefix_values = [0.0]
        for origin_support, recovery_support in zip(
            support_order, support_order[1:]
        ):
            distance = float(
                instance.safe_distance_matrix[
                    matrix_index[origin_support], matrix_index[recovery_support]
                ]
            )
            prefix_values.append(prefix_values[-1] + distance / usv_speed_km_min)
        prefix = tuple(prefix_values)
        if len(prefix) != len(support_order):
            raise ValueError("USV route prefix length does not match support order")
        origin: dict[int, int] = {}
        recovery: dict[int, int] = {}
        span: dict[int, int] = {}
        profiles: dict[int, PhysicalForwardCandidate] = {}
        recovery_horizon: dict[int, set[int]] = {
            position: set() for position in range(len(support_order))
        }
        feasible_ids: set[int] = set()
        physical_forward_ids: list[int] = []
        static_impossible_forward_ids: list[int] = []
        support_task_ids: dict[tuple[int, int], list[int]] = {}
        for sortie_id, sortie in sortie_index.sortie_by_id.items():
            origin_position = support_position[sortie.origin_support]
            recovery_position = support_position[sortie.recovery_support]
            if recovery_position < origin_position:
                raise ValueError(f"sortie {sortie_id} has a negative route span")
            origin[sortie_id] = origin_position
            recovery[sortie_id] = recovery_position
            active_span = recovery_position - origin_position
            span[sortie_id] = active_span
            movement_time = prefix[recovery_position] - prefix[origin_position]
            lower_bound_hover = max(
                0.0, movement_time - sortie.nominal_duration_min
            )
            lower_bound_duration = max(
                sortie.nominal_duration_min, movement_time
            )
            if uav is None:
                lower_bound_energy = sortie.nominal_energy_wh
                time_slack = float("inf")
                energy_slack = float("inf")
            else:
                lower_bound_energy = (
                    sortie.nominal_energy_wh
                    + uav.hovering_energy_wh_min * lower_bound_hover
                )
                time_slack = (
                    uav.max_sortie_duration_min - lower_bound_duration
                )
                energy_slack = (
                    uav.battery_wh
                    - uav.safety_soc_wh
                    - lower_bound_energy
                ) / uav.hovering_energy_wh_min
            profile = PhysicalForwardCandidate(
                sortie_id=sortie_id,
                origin_pos=origin_position,
                recovery_pos=recovery_position,
                movement_time_min=movement_time,
                lower_bound_duration_min=lower_bound_duration,
                lower_bound_energy_wh=lower_bound_energy,
                time_slack_min=time_slack,
                energy_slack_min=energy_slack,
                intermediate_dwell_budget_min=(
                    min(
                        uav.max_sortie_duration_min - movement_time,
                        sortie.nominal_duration_min
                        + (
                            uav.battery_wh
                            - uav.safety_soc_wh
                            - sortie.nominal_energy_wh
                        )
                        / uav.hovering_energy_wh_min
                        - movement_time,
                    )
                    if uav is not None
                    else float("inf")
                ),
            )
            profiles[sortie_id] = profile
            if profile.statically_feasible:
                feasible_ids.add(sortie_id)
                recovery_horizon[origin_position].add(recovery_position)
                if active_span >= 1:
                    physical_forward_ids.append(sortie_id)
                    for position in range(origin_position, recovery_position + 1):
                        for task_id in sortie.task_sequence:
                            support_task_ids.setdefault(
                                (position, task_id), []
                            ).append(sortie_id)
            elif active_span >= 1:
                static_impossible_forward_ids.append(sortie_id)

        def physical_score(sortie_id: int):
            sortie = sortie_index.sortie_by_id[sortie_id]
            profile = profiles[sortie_id]
            return (
                max(0.0, sortie.nominal_duration_min - profile.movement_time_min),
                max(0.0, profile.movement_time_min - sortie.nominal_duration_min),
                sortie.flight_time_min,
                sortie.nominal_energy_wh,
                sortie.id,
            )

        physical_forward_ids.sort(key=physical_score)
        for ids in support_task_ids.values():
            ids.sort(key=physical_score)
        physical_forward_rank = {
            sortie_id: rank
            for rank, sortie_id in enumerate(physical_forward_ids)
        }
        sortie_ids_by_qid = sortie_index.sortie_ids_in_original_order
        sortie_object = tuple(
            sortie_index.sortie_by_id[sortie_id]
            for sortie_id in sortie_ids_by_qid
        )
        qid_by_sortie_id = {
            sortie_id: qid for qid, sortie_id in enumerate(sortie_ids_by_qid)
        }
        return cls(
            support_order=support_order,
            support_position=MappingProxyType(support_position),
            route_sail_prefix=prefix,
            sortie_ids_by_qid=sortie_ids_by_qid,
            qid_by_sortie_id=MappingProxyType(qid_by_sortie_id),
            origin_pos=tuple(origin[sortie_id] for sortie_id in sortie_ids_by_qid),
            recovery_pos=tuple(
                recovery[sortie_id] for sortie_id in sortie_ids_by_qid
            ),
            nominal_time=tuple(
                sortie.nominal_duration_min for sortie in sortie_object
            ),
            nominal_energy=tuple(
                sortie.nominal_energy_wh for sortie in sortie_object
            ),
            proxy_base=tuple(
                sortie.nominal_duration_min / len(sortie.task_sequence)
                for sortie in sortie_object
            ),
            task_signature=tuple(
                sortie.task_sequence for sortie in sortie_object
            ),
            task_count=tuple(
                len(sortie.task_sequence) for sortie in sortie_object
            ),
            physical_possible=tuple(
                sortie_id in feasible_ids for sortie_id in sortie_ids_by_qid
            ),
            sortie_object=sortie_object,
            task_to_qids=MappingProxyType(
                {
                    task_id: tuple(ranks)
                    for task_id, ranks in sortie_index.sortie_ranks_by_task.items()
                }
            ),
            sortie_origin_position=MappingProxyType(origin),
            sortie_recovery_position=MappingProxyType(recovery),
            sortie_span=MappingProxyType(span),
            physical_profile_by_sortie=MappingProxyType(profiles),
            physical_recovery_horizon=MappingProxyType(
                {
                    position: tuple(sorted(recoveries))
                    for position, recoveries in recovery_horizon.items()
                }
            ),
            physically_feasible_sortie_ids=frozenset(feasible_ids),
            physical_forward_sortie_ids=tuple(physical_forward_ids),
            physical_forward_rank_by_sortie=MappingProxyType(
                physical_forward_rank
            ),
            physical_forward_by_support_task=MappingProxyType(
                {key: tuple(ids) for key, ids in support_task_ids.items()}
            ),
            static_impossible_forward_sortie_ids=tuple(
                static_impossible_forward_ids
            ),
            sortie_index=sortie_index,
        )

    def usv_time(self, origin_position: int, recovery_position: int) -> float:
        if not 0 <= origin_position <= recovery_position < len(self.route_sail_prefix):
            raise ValueError("USV route positions are outside the frozen support route")
        return (
            self.route_sail_prefix[recovery_position]
            - self.route_sail_prefix[origin_position]
        )

    def is_physically_feasible(self, sortie_id: int) -> bool:
        return sortie_id in self.physically_feasible_sortie_ids

    def dwell_guided_targets(
        self,
        support_position: int,
        task_ids: tuple[int, ...],
        *,
        limit: int = 12,
    ) -> tuple[tuple[int, ...], int]:
        """Return a bounded static pool local to one support and its blockers."""
        if limit <= 0:
            raise ValueError("dwell-guided target limit must be positive")
        if not 0 <= support_position < len(self.support_order):
            raise ValueError("support position is outside the frozen route")
        candidates = {
            sortie_id
            for task_id in task_ids
            for sortie_id in self.physical_forward_by_support_task.get(
                (support_position, task_id), ()
            )
        }
        ordered = tuple(
            sorted(
                candidates,
                key=self.physical_forward_rank_by_sortie.__getitem__,
            )
        )
        return ordered[:limit], len(ordered)
