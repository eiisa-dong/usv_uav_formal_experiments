from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Sequence

import numpy as np

from usv_uav.config import NavigationConfig
from usv_uav.core.models import SupportPoint, Task
from usv_uav.preprocessing.obstacle_builder import ObstacleMap


@dataclass(frozen=True, slots=True)
class SupportGenerationResult:
    supports: tuple[SupportPoint, ...]
    median_nearest_neighbour_km: float
    grid_spacing_km: float
    bounds_km: tuple[float, float, float, float]
    generated_count: int
    unsafe_count: int
    duplicate_count: int


def median_nearest_neighbour_distance(tasks: Sequence[Task]) -> float:
    if len(tasks) < 2:
        raise ValueError("at least two tasks are required to derive grid spacing")
    coordinates = np.asarray([(task.x_km, task.y_km) for task in tasks], dtype=np.float64)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    np.fill_diagonal(distances, np.inf)
    median = float(np.median(np.min(distances, axis=1)))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("task geometry has no positive nearest-neighbour distance")
    return median


def _grid_axis(lower: float, upper: float, spacing: float) -> np.ndarray:
    first = floor(lower / spacing) * spacing
    last = ceil(upper / spacing) * spacing
    count = int(round((last - first) / spacing)) + 1
    return first + np.arange(count, dtype=np.float64) * spacing


def generate_candidate_supports(
    tasks: Sequence[Task],
    obstacle_map: ObstacleMap,
    config: NavigationConfig,
    *,
    bounds_km: tuple[float, float, float, float] | None = None,
) -> SupportGenerationResult:
    if not tasks:
        raise ValueError("tasks must be non-empty")
    median_nn = median_nearest_neighbour_distance(tasks)
    spacing = config.candidate_grid_spacing_nn_factor * median_nn
    if bounds_km is None:
        margin = config.candidate_bounds_margin_nn_factor * median_nn
        xs = [task.x_km for task in tasks]
        ys = [task.y_km for task in tasks]
        bounds_km = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
    min_x, min_y, max_x, max_y = bounds_km
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("candidate bounds must have positive area")

    tolerance_km = config.coordinate_dedup_tolerance_m / 1000.0
    x_values = _grid_axis(min_x, max_x, spacing)
    y_values = _grid_axis(min_y, max_y, spacing)
    generated = len(x_values) * len(y_values)
    unsafe = 0
    duplicates = 0
    seen: set[tuple[int, int]] = set()
    coordinates: list[tuple[float, float]] = []
    for y_km in y_values:
        for x_km in x_values:
            if not obstacle_map.is_safe_point(float(x_km), float(y_km)):
                unsafe += 1
                continue
            key = (round(float(x_km) / tolerance_km), round(float(y_km) / tolerance_km))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            coordinates.append((float(x_km), float(y_km)))

    supports = tuple(
        SupportPoint(id=index, x_km=x_km, y_km=y_km)
        for index, (x_km, y_km) in enumerate(coordinates, start=1)
    )
    if not supports:
        raise ValueError("no safe support candidates were generated")
    return SupportGenerationResult(
        supports=supports,
        median_nearest_neighbour_km=median_nn,
        grid_spacing_km=spacing,
        bounds_km=bounds_km,
        generated_count=generated,
        unsafe_count=unsafe,
        duplicate_count=duplicates,
    )
