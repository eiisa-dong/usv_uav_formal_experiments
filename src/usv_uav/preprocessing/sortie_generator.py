from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

from usv_uav.config import UAVConfig
from usv_uav.core.models import Sortie, SupportPoint, Task


@dataclass(frozen=True, slots=True)
class SortieGenerationStats:
    route_support_pairs: int
    dfs_extensions: int
    lower_bound_pruned: int
    max_depth_stops: int
    feasible_sorties: int


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def generate_sortie_pool(
    tasks: Sequence[Task],
    selected_supports: Sequence[SupportPoint],
    usv_route: Sequence[int],
    uav: UAVConfig,
    *,
    max_tasks_per_sortie: int,
) -> tuple[tuple[Sortie, ...], SortieGenerationStats]:
    """Enumerate every nominally feasible ordered 1..Q task sortie.

    A branch is removed only when completing its current prefix directly to
    the recovery support already violates a monotone time or energy bound.
    Therefore no feasible extension is lost.
    """
    if not tasks or not selected_supports:
        raise ValueError("tasks and selected supports must be non-empty")
    if max_tasks_per_sortie <= 0:
        raise ValueError("max_tasks_per_sortie must be positive")
    support_by_id = {support.id: support for support in selected_supports}
    if len(support_by_id) != len(selected_supports):
        raise ValueError("selected support ids must be unique")
    route_middle = tuple(usv_route[1:-1])
    if tuple(usv_route[:1]) != (0,) or tuple(usv_route[-1:]) != (0,):
        raise ValueError("USV route must start and end at port id 0")
    if len(route_middle) != len(selected_supports) or set(route_middle) != set(support_by_id):
        raise ValueError("USV route must visit every selected support exactly once")

    task_coordinates = {task.id: task.coordinate for task in tasks}
    task_by_id = {task.id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("task ids must be unique")
    task_ids = tuple(sorted(task_by_id))
    route_position = {support_id: position for position, support_id in enumerate(route_middle)}
    sorties: list[Sortie] = []
    dfs_extensions = 0
    pruned = 0
    depth_stops = 0
    pair_count = 0

    for origin_id in route_middle:
        origin = support_by_id[origin_id]
        for recovery_id in route_middle:
            if route_position[origin_id] > route_position[recovery_id]:
                continue
            pair_count += 1
            recovery = support_by_id[recovery_id]

            def extend(
                sequence: tuple[int, ...],
                used: frozenset[int],
                last_coordinate: tuple[float, float],
                travelled_without_recovery_km: float,
                inspection_min: float,
            ) -> None:
                nonlocal dfs_extensions, pruned, depth_stops
                for task_id in task_ids:
                    if task_id in used:
                        continue
                    dfs_extensions += 1
                    task = task_by_id[task_id]
                    leg_km = _distance(last_coordinate, task_coordinates[task_id])
                    partial_distance_km = travelled_without_recovery_km + leg_km
                    next_inspection_min = inspection_min + task.service_min
                    lower_distance_km = partial_distance_km + _distance(task_coordinates[task_id], recovery.coordinate)
                    flight_min = lower_distance_km / uav.speed_km_min
                    total_min = flight_min + next_inspection_min
                    energy_wh = (
                        flight_min * uav.flight_energy_wh_min
                        + next_inspection_min * uav.inspection_energy_wh_min
                    )
                    if (
                        total_min > uav.max_sortie_duration_min + 1e-9
                        or energy_wh + uav.safety_soc_wh > uav.battery_wh + 1e-9
                    ):
                        pruned += 1
                        continue

                    next_sequence = (*sequence, task_id)
                    sorties.append(Sortie(
                        id=len(sorties),
                        origin_support=origin_id,
                        task_sequence=next_sequence,
                        recovery_support=recovery_id,
                        flight_distance_km=lower_distance_km,
                        flight_time_min=flight_min,
                        inspection_time_min=next_inspection_min,
                        nominal_energy_wh=energy_wh,
                    ))
                    if len(next_sequence) >= max_tasks_per_sortie:
                        depth_stops += 1
                        continue
                    extend(
                        next_sequence,
                        used | {task_id},
                        task_coordinates[task_id],
                        partial_distance_km,
                        next_inspection_min,
                    )

            extend((), frozenset(), origin.coordinate, 0.0, 0.0)

    stats = SortieGenerationStats(
        route_support_pairs=pair_count,
        dfs_extensions=dfs_extensions,
        lower_bound_pruned=pruned,
        max_depth_stops=depth_stops,
        feasible_sorties=len(sorties),
    )
    return tuple(sorties), stats
