from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import GeometryCollection, LineString, Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from usv_uav.config import NavigationConfig
from usv_uav.core.models import Task
from usv_uav.data.walney_loader import TurbineCoordinate
from usv_uav.data.navigation_data import NavigationData, project_geometry_to_km


def _clean_union(geometries: Sequence[BaseGeometry]) -> BaseGeometry:
    nonempty = [geometry for geometry in geometries if not geometry.is_empty]
    if not nonempty:
        return GeometryCollection()
    merged = unary_union(nonempty)
    return merged if merged.is_valid else merged.buffer(0)


@dataclass(frozen=True, slots=True)
class ObstacleBuildStats:
    coastline_count: int
    land_polygon_count: int
    shallow_polygon_count: int
    turbine_buffer_count: int
    wreck_geometry_count: int


@dataclass(frozen=True, slots=True)
class ObstacleMap:
    geometry: BaseGeometry
    categories: Mapping[str, BaseGeometry]
    crs: str = "EPSG:32630"
    coordinate_units: str = "km"

    def __post_init__(self) -> None:
        if self.crs != "EPSG:32630" or self.coordinate_units != "km":
            raise ValueError("obstacle map must use EPSG:32630 kilometres")
        object.__setattr__(self, "categories", MappingProxyType(dict(self.categories)))

    def is_safe_point(self, x_km: float, y_km: float) -> bool:
        return not self.geometry.intersects(Point(float(x_km), float(y_km)))

    def is_safe_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        segment = LineString((start, end))
        return not self.geometry.intersects(segment)


def build_land_obstacles(
    coastline_lines: Sequence[LineString],
    *,
    buffer_km: float = 0.0,
) -> tuple[BaseGeometry, int]:
    """Polygonise closed OSM coastline rings.

    Open coastline fragments are deliberately not guessed into land polygons;
    the bathymetry-derived shallow-water layer supplies the conservative land
    and intertidal mask for those fragments.
    """
    polygons = tuple(polygonize(coastline_lines))
    land = _clean_union(polygons)
    if buffer_km > 0 and not land.is_empty:
        land = land.buffer(buffer_km)
    return land, len(polygons)


def build_shallow_water_obstacles(
    raster_path: str | Path,
    *,
    minimum_water_depth_m: float,
    positive_down: bool,
    target_crs: str = "EPSG:32630",
    clip_bounds_km: tuple[float, float, float, float] | None = None,
) -> tuple[BaseGeometry, int]:
    """Vectorise cells whose water depth is below the configured threshold.

    EMODnet mean bathymetry is elevation-like (negative below sea level), so
    the default rule is ``elevation > -minimum_depth``. The alternative flag
    supports positive-down rasters without changing this module.
    """
    source_path = Path(raster_path)
    projected: list[BaseGeometry] = []
    clip = box(*clip_bounds_km) if clip_bounds_km is not None else None
    with rasterio.open(source_path) as dataset:
        band = dataset.read(1, masked=True)
        values = np.asarray(band.filled(np.nan), dtype=np.float64)
        valid = ~np.ma.getmaskarray(band) & np.isfinite(values)
        shallow = values < minimum_water_depth_m if positive_down else values > -minimum_water_depth_m
        mask = valid & shallow
        if not np.any(mask):
            return GeometryCollection(), 0
        for raw_geometry, value in shapes(mask.astype(np.uint8), mask=mask, transform=dataset.transform):
            if int(value) != 1:
                continue
            geometry = project_geometry_to_km(
                shape(raw_geometry),
                source_crs=dataset.crs,
                target_crs=target_crs,
            )
            if clip is not None:
                geometry = geometry.intersection(clip)
            if not geometry.is_empty:
                projected.append(geometry)
    return _clean_union(projected), len(projected)


def build_obstacle_map(
    physical_turbines: Sequence[Task | TurbineCoordinate],
    navigation_data: NavigationData,
    config: NavigationConfig,
    *,
    clip_bounds_km: tuple[float, float, float, float] | None = None,
) -> tuple[ObstacleMap, ObstacleBuildStats]:
    categories: dict[str, BaseGeometry] = {}
    land_count = 0
    shallow_count = 0

    if config.use_land:
        land, land_count = build_land_obstacles(
            navigation_data.coastline_lines,
            buffer_km=config.land_buffer_m / 1000.0,
        )
        categories["land"] = land
    if config.use_shallow_water:
        shallow, shallow_count = build_shallow_water_obstacles(
            navigation_data.bathymetry_path,
            minimum_water_depth_m=config.minimum_water_depth_m,
            positive_down=config.bathymetry_positive_down,
            target_crs=config.target_crs,
            clip_bounds_km=clip_bounds_km,
        )
        categories["shallow_water"] = shallow
    if config.use_turbine_buffers:
        radius_km = config.turbine_buffer_m / 1000.0
        categories["turbines"] = _clean_union([
            Point(turbine.x_km, turbine.y_km).buffer(radius_km) for turbine in physical_turbines
        ])
    if config.use_wreck_obstruction_buffers:
        radius_km = config.wreck_obstruction_buffer_m / 1000.0
        wreck_geometries: list[BaseGeometry] = []
        if not navigation_data.wreck_points.is_empty:
            wreck_geometries.append(navigation_data.wreck_points.buffer(radius_km))
        if not navigation_data.wreck_areas.is_empty:
            wreck_geometries.append(navigation_data.wreck_areas.buffer(radius_km))
        categories["wreck_obstructions"] = _clean_union(wreck_geometries)

    geometry = _clean_union(list(categories.values()))
    stats = ObstacleBuildStats(
        coastline_count=len(navigation_data.coastline_lines),
        land_polygon_count=land_count,
        shallow_polygon_count=shallow_count,
        turbine_buffer_count=len(physical_turbines) if config.use_turbine_buffers else 0,
        wreck_geometry_count=(
            int(not navigation_data.wreck_points.is_empty)
            + int(not navigation_data.wreck_areas.is_empty)
            if config.use_wreck_obstruction_buffers
            else 0
        ),
    )
    return ObstacleMap(geometry=geometry, categories=categories), stats
