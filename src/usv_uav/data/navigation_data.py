from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from pyproj import CRS, Transformer
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from usv_uav.config import ProjectConfig


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_geometry_to_km(
    geometry: BaseGeometry,
    *,
    source_crs: str | CRS,
    target_crs: str | CRS = "EPSG:32630",
) -> BaseGeometry:
    """Project a geometry and convert projected metres to model kilometres."""
    if geometry.is_empty:
        return geometry
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_m = transform(transformer.transform, geometry)
    return affinity.scale(projected_m, xfact=0.001, yfact=0.001, origin=(0.0, 0.0))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_osm_coastlines(
    path: str | Path,
    *,
    target_crs: str = "EPSG:32630",
) -> tuple[LineString, ...]:
    source = Path(path)
    payload = _read_json(source)
    lines: list[LineString] = []
    for element in payload.get("elements", []):
        raw = element.get("geometry") or []
        if len(raw) < 2:
            continue
        line_wgs84 = LineString((float(point["lon"]), float(point["lat"])) for point in raw)
        projected = project_geometry_to_km(
            line_wgs84,
            source_crs="EPSG:4326",
            target_crs=target_crs,
        )
        if not projected.is_empty:
            lines.append(projected)
    return tuple(lines)


def load_geojson_union(
    path: str | Path,
    *,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32630",
) -> BaseGeometry:
    source = Path(path)
    payload = _read_json(source)
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"expected a GeoJSON FeatureCollection: {source}")
    geometries: list[BaseGeometry] = []
    for feature in payload.get("features", []):
        raw_geometry = feature.get("geometry")
        if raw_geometry is None:
            continue
        geometry = shape(raw_geometry)
        if not geometry.is_empty:
            geometries.append(project_geometry_to_km(
                geometry,
                source_crs=source_crs,
                target_crs=target_crs,
            ))
    return unary_union(geometries) if geometries else GeometryCollection()


@dataclass(frozen=True, slots=True)
class NavigationData:
    crs: str
    coordinate_units: str
    coastline_lines: tuple[LineString, ...]
    wreck_points: BaseGeometry
    wreck_areas: BaseGeometry
    bathymetry_path: Path
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.crs != "EPSG:32630" or self.coordinate_units != "km":
            raise ValueError("navigation data must be EPSG:32630 in kilometres")
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))


def load_navigation_data(config: ProjectConfig) -> NavigationData:
    paths = config.paths
    required = (paths.coastline, paths.wreck_points, paths.wreck_areas, paths.bathymetry)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"navigation source files missing: {missing}")
    target_crs = config.navigation.target_crs
    return NavigationData(
        crs=target_crs,
        coordinate_units="km",
        coastline_lines=load_osm_coastlines(paths.coastline, target_crs=target_crs),
        wreck_points=load_geojson_union(paths.wreck_points, target_crs=target_crs),
        wreck_areas=load_geojson_union(paths.wreck_areas, target_crs=target_crs),
        bathymetry_path=paths.bathymetry,
        source_hashes={
            "coastline": file_sha256(paths.coastline),
            "wreck_points": file_sha256(paths.wreck_points),
            "wreck_areas": file_sha256(paths.wreck_areas),
            "bathymetry": file_sha256(paths.bathymetry),
        },
    )
