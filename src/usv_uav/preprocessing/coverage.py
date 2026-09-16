from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from usv_uav.config import UAVConfig
from usv_uav.core.models import SupportPoint, Task


@dataclass(frozen=True, slots=True, eq=False)
class CoverageMatrix:
    support_ids: tuple[int, ...]
    task_ids: tuple[int, ...]
    feasible: NDArray[np.bool_]
    roundtrip_distance_km: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected = (len(self.support_ids), len(self.task_ids))
        feasible = np.array(self.feasible, dtype=np.bool_, copy=True)
        distance = np.array(self.roundtrip_distance_km, dtype=np.float64, copy=True)
        if feasible.shape != expected or distance.shape != expected:
            raise ValueError(f"coverage arrays must have shape {expected}")
        if not np.all(np.isfinite(distance)) or np.any(distance < 0):
            raise ValueError("coverage distances must be finite and non-negative")
        feasible.setflags(write=False)
        distance.setflags(write=False)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(self, "roundtrip_distance_km", distance)

    @property
    def uncovered_task_ids(self) -> tuple[int, ...]:
        covered = np.any(self.feasible, axis=0)
        return tuple(task_id for task_id, is_covered in zip(self.task_ids, covered) if not is_covered)


def nominal_single_task_feasible(
    support: SupportPoint,
    task: Task,
    uav: UAVConfig,
) -> tuple[bool, float]:
    one_way_km = hypot(support.x_km - task.x_km, support.y_km - task.y_km)
    roundtrip_km = 2.0 * one_way_km
    flight_min = roundtrip_km / uav.speed_km_min
    total_min = flight_min + task.service_min
    nominal_energy_wh = (
        flight_min * uav.flight_energy_wh_min
        + task.service_min * uav.inspection_energy_wh_min
    )
    feasible = (
        total_min <= uav.max_sortie_duration_min + 1e-9
        and nominal_energy_wh + uav.safety_soc_wh <= uav.battery_wh + 1e-9
    )
    return feasible, roundtrip_km


def build_coverage_matrix(
    supports: Sequence[SupportPoint],
    tasks: Sequence[Task],
    uav: UAVConfig,
) -> CoverageMatrix:
    if not supports or not tasks:
        raise ValueError("supports and tasks must be non-empty")
    feasible = np.zeros((len(supports), len(tasks)), dtype=np.bool_)
    distances = np.zeros((len(supports), len(tasks)), dtype=np.float64)
    for support_index, support in enumerate(supports):
        for task_index, task in enumerate(tasks):
            is_feasible, roundtrip_km = nominal_single_task_feasible(support, task, uav)
            feasible[support_index, task_index] = is_feasible
            distances[support_index, task_index] = roundtrip_km
    return CoverageMatrix(
        support_ids=tuple(support.id for support in supports),
        task_ids=tuple(task.id for task in tasks),
        feasible=feasible,
        roundtrip_distance_km=distances,
    )


def subset_coverage_matrix(
    coverage: CoverageMatrix,
    active_task_ids: Sequence[int],
) -> CoverageMatrix:
    """Take master-coverage columns in scenario order without recomputation."""
    requested = tuple(int(value) for value in active_task_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("active_task_ids must be non-empty and unique")
    positions = {task_id: index for index, task_id in enumerate(coverage.task_ids)}
    missing = tuple(task_id for task_id in requested if task_id not in positions)
    if missing:
        raise ValueError(f"active tasks absent from master coverage: {missing}")
    columns = [positions[task_id] for task_id in requested]
    return CoverageMatrix(
        support_ids=coverage.support_ids,
        task_ids=requested,
        feasible=coverage.feasible[:, columns],
        roundtrip_distance_km=coverage.roundtrip_distance_km[:, columns],
    )
