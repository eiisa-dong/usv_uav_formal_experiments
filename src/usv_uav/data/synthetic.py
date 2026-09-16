from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from collections import defaultdict
from math import floor

import numpy as np

from usv_uav.data.walney_loader import TurbineCoordinate, WalneyLayout


@dataclass(frozen=True, slots=True)
class GeometryCalibration:
    density_per_km2: float
    aspect_ratio: float
    principal_angle_deg: float
    median_nearest_neighbour_km: float
    centroid_x_km: float
    centroid_y_km: float


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    density_relative_error: float
    aspect_ratio_relative_error: float
    median_nn_relative_error: float


def _median_nn(coordinates: np.ndarray) -> float:
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.min(distances, axis=1)))


def _calibrate_coordinates(coordinates: np.ndarray) -> GeometryCalibration:
    centroid = np.mean(coordinates, axis=0)
    centered = coordinates - centroid
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    angle = atan2(principal[1], principal[0])
    rotation = np.array([[cos(-angle), -sin(-angle)], [sin(-angle), cos(-angle)]])
    aligned = centered @ rotation.T
    width, height = np.ptp(aligned[:, 0]), np.ptp(aligned[:, 1])
    major, minor = max(width, height), min(width, height)
    area = max(width * height, 1e-12)
    return GeometryCalibration(
        density_per_km2=len(coordinates) / area,
        aspect_ratio=major / max(minor, 1e-12),
        principal_angle_deg=degrees(angle),
        median_nearest_neighbour_km=_median_nn(coordinates),
        centroid_x_km=float(centroid[0]),
        centroid_y_km=float(centroid[1]),
    )


def calibrate_geometry(layout: WalneyLayout) -> GeometryCalibration:
    coordinates = np.asarray([(point.x_km, point.y_km) for point in layout.turbines], dtype=float)
    return _calibrate_coordinates(coordinates)


def _calibration_check(
    candidate: GeometryCalibration,
    target: GeometryCalibration,
) -> CalibrationCheck:
    return CalibrationCheck(
        density_relative_error=(candidate.density_per_km2 - target.density_per_km2) / target.density_per_km2,
        aspect_ratio_relative_error=(candidate.aspect_ratio - target.aspect_ratio) / target.aspect_ratio,
        median_nn_relative_error=(candidate.median_nearest_neighbour_km - target.median_nearest_neighbour_km) / target.median_nearest_neighbour_km,
    )


def _sample_field(calibration: GeometryCalibration, count: int, seed: int, min_spacing: float) -> np.ndarray | None:
    generator = np.random.default_rng(seed)
    area = count / calibration.density_per_km2
    width = sqrt(area * calibration.aspect_ratio)
    height = area / width
    points: list[tuple[float, float]] = []
    cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for _ in range(count * 20_000):
        candidate = (generator.uniform(-width / 2, width / 2), generator.uniform(-height / 2, height / 2))
        cell = (floor(candidate[0] / min_spacing), floor(candidate[1] / min_spacing))
        nearby = (
            point
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for point in cells.get((cell[0] + dx, cell[1] + dy), ())
        )
        if all(hypot(candidate[0] - x, candidate[1] - y) >= min_spacing for x, y in nearby):
            points.append(candidate)
            cells[cell].append(candidate)
            if len(points) == count:
                return np.asarray(points, dtype=float)
    return None


def generate_calibrated_master_field(
    real_layout: WalneyLayout,
    *,
    geometry_seed: int,
    task_count: int = 200,
) -> tuple[WalneyLayout, CalibrationCheck]:
    calibration = calibrate_geometry(real_layout)
    best: np.ndarray | None = None
    best_score = (float("inf"), float("inf"))
    for iteration, factor in enumerate(np.linspace(0.55, 1.05, 11)):
        sampled = _sample_field(
            calibration,
            task_count,
            geometry_seed * 100 + iteration,
            calibration.median_nearest_neighbour_km * float(factor),
        )
        if sampled is None:
            continue
        candidate_check = _calibration_check(_calibrate_coordinates(sampled), calibration)
        errors = (
            abs(candidate_check.density_relative_error),
            abs(candidate_check.aspect_ratio_relative_error),
            abs(candidate_check.median_nn_relative_error),
        )
        score = (max(errors), sum(errors))
        if score < best_score:
            best, best_score = sampled, score
    if best is None:
        raise RuntimeError("could not generate the calibrated master field")
    angle = radians(calibration.principal_angle_deg)
    rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    rotated = best @ rotation.T
    rotated[:, 0] += calibration.centroid_x_km
    rotated[:, 1] += calibration.centroid_y_km
    turbines = tuple(
        TurbineCoordinate(f"G{geometry_seed:02d}-{index:03d}", float(x), float(y))
        for index, (x, y) in enumerate(rotated, start=1)
    )
    layout = WalneyLayout(
        name=f"walney_calibrated_G{geometry_seed:02d}",
        crs=real_layout.crs,
        coordinate_units=real_layout.coordinate_units,
        port=real_layout.port,
        turbines=turbines,
        source_description=f"Walney-calibrated synthetic master field; geometry_seed={geometry_seed}",
        source_sha256=real_layout.source_sha256,
    )
    synthetic_calibration = calibrate_geometry(layout)
    check = _calibration_check(synthetic_calibration, calibration)
    return layout, check
