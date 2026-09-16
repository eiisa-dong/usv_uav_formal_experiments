from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


def _finite_coordinate(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    x_km: float
    y_km: float
    level: int
    service_min: float
    source_id: str | None = None

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError("task id must be positive; id 0 is reserved for the port")
        _finite_coordinate("task.x_km", self.x_km)
        _finite_coordinate("task.y_km", self.y_km)
        if self.level not in {1, 2, 3}:
            raise ValueError("task level must be 1, 2, or 3")
        if self.service_min <= 0 or not isfinite(self.service_min):
            raise ValueError("task service_min must be finite and positive")

    @property
    def coordinate(self) -> tuple[float, float]:
        return self.x_km, self.y_km


@dataclass(frozen=True, slots=True)
class SupportPoint:
    id: int
    x_km: float
    y_km: float

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError("support id must be positive; id 0 is reserved for the port")
        _finite_coordinate("support.x_km", self.x_km)
        _finite_coordinate("support.y_km", self.y_km)

    @property
    def coordinate(self) -> tuple[float, float]:
        return self.x_km, self.y_km


@dataclass(frozen=True, slots=True)
class Port:
    x_km: float
    y_km: float

    def __post_init__(self) -> None:
        _finite_coordinate("port.x_km", self.x_km)
        _finite_coordinate("port.y_km", self.y_km)

    @property
    def id(self) -> int:
        return 0

    @property
    def coordinate(self) -> tuple[float, float]:
        return self.x_km, self.y_km


@dataclass(frozen=True, slots=True)
class SafePath:
    start: int
    end: int
    distance_km: float
    polyline: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.distance_km < 0 or not isfinite(self.distance_km):
            raise ValueError("safe path distance must be finite and non-negative")
        if len(self.polyline) < 2:
            raise ValueError("safe path polyline must contain at least two points")
        for x_km, y_km in self.polyline:
            _finite_coordinate("safe path x", x_km)
            _finite_coordinate("safe path y", y_km)


@dataclass(frozen=True, slots=True)
class Sortie:
    id: int
    origin_support: int
    task_sequence: tuple[int, ...]
    recovery_support: int
    flight_distance_km: float
    flight_time_min: float
    inspection_time_min: float
    nominal_energy_wh: float

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("sortie id must be non-negative")
        if self.origin_support <= 0 or self.recovery_support <= 0:
            raise ValueError("sortie origin and recovery must be support ids")
        if not self.task_sequence or len(set(self.task_sequence)) != len(self.task_sequence):
            raise ValueError("sortie tasks must be non-empty and unique")
        for name in (
            "flight_distance_km",
            "flight_time_min",
            "inspection_time_min",
            "nominal_energy_wh",
        ):
            value = float(getattr(self, name))
            if value < 0 or not isfinite(value):
                raise ValueError(f"sortie {name} must be finite and non-negative")

    @property
    def nominal_duration_min(self) -> float:
        return self.flight_time_min + self.inspection_time_min


@dataclass(frozen=True, slots=True, eq=False)
class Instance:
    instance_id: str
    tasks: tuple[Task, ...]
    support_points: tuple[SupportPoint, ...]
    selected_supports: tuple[int, ...]
    usv_route: tuple[int, ...]
    safe_distance_matrix: NDArray[np.float64]
    sortie_pool: tuple[Sortie, ...]
    uav_count: int
    charger_count: int
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id is required")
        if self.uav_count <= 0 or self.charger_count <= 0:
            raise ValueError("uav_count and charger_count must be positive")
        task_ids = [task.id for task in self.tasks]
        support_ids = [support.id for support in self.support_points]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task ids must be unique")
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("support ids must be unique")
        if len(set(self.selected_supports)) != len(self.selected_supports):
            raise ValueError("selected support ids must be unique")
        if not set(self.selected_supports) <= set(support_ids):
            raise ValueError("selected supports must exist in support_points")
        if len(self.usv_route) < 2 or self.usv_route[0] != 0 or self.usv_route[-1] != 0:
            raise ValueError("USV route must start and end at port id 0")
        if set(self.usv_route[1:-1]) != set(self.selected_supports):
            raise ValueError("USV route must visit every selected support")
        if len(self.usv_route[1:-1]) != len(self.selected_supports):
            raise ValueError("USV route must visit each selected support exactly once")

        matrix = np.array(self.safe_distance_matrix, dtype=np.float64, copy=True)
        expected = len(self.selected_supports) + 1
        if matrix.shape != (expected, expected):
            raise ValueError(f"safe distance matrix must have shape {(expected, expected)}")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("safe distance matrix must be finite and non-negative")
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0.0):
            raise ValueError("safe distance matrix must be symmetric with a zero diagonal")
        matrix.setflags(write=False)
        object.__setattr__(self, "safe_distance_matrix", matrix)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

        sortie_ids = [sortie.id for sortie in self.sortie_pool]
        if len(set(sortie_ids)) != len(sortie_ids):
            raise ValueError("sortie ids must be unique")
        selected = set(self.selected_supports)
        task_set = set(task_ids)
        for sortie in self.sortie_pool:
            if sortie.origin_support not in selected or sortie.recovery_support not in selected:
                raise ValueError(f"sortie {sortie.id} uses an unselected support")
            if not set(sortie.task_sequence) <= task_set:
                raise ValueError(f"sortie {sortie.id} uses an unknown task")
