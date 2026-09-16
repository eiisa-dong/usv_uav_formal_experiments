from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import heapq
from math import hypot, inf
import os
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from usv_uav.core.models import SafePath
from usv_uav.preprocessing.obstacle_builder import ObstacleMap


EPSILON = 1e-10


@dataclass(frozen=True, slots=True, eq=False)
class SafePathMatrix:
    node_ids: tuple[int, ...]
    distance_km: NDArray[np.float64]
    paths: tuple[SafePath, ...]

    def __post_init__(self) -> None:
        expected = (len(self.node_ids), len(self.node_ids))
        matrix = np.array(self.distance_km, dtype=np.float64, copy=True)
        if matrix.shape != expected:
            raise ValueError(f"safe path matrix must have shape {expected}")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("safe path distances must be finite and non-negative")
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0.0):
            raise ValueError("safe path matrix must be symmetric with a zero diagonal")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("safe path node ids must be unique")
        expected_paths = len(self.node_ids) * (len(self.node_ids) - 1) // 2
        if len(self.paths) != expected_paths:
            raise ValueError(f"expected {expected_paths} unordered safe paths")
        matrix.setflags(write=False)
        object.__setattr__(self, "distance_km", matrix)

    @property
    def path_by_pair(self) -> dict[tuple[int, int], SafePath]:
        result: dict[tuple[int, int], SafePath] = {}
        for path in self.paths:
            result[(path.start, path.end)] = path
            result[(path.end, path.start)] = SafePath(
                start=path.end,
                end=path.start,
                distance_km=path.distance_km,
                polyline=tuple(reversed(path.polyline)),
            )
        return result


@dataclass(frozen=True, slots=True)
class VisibilityBackbone:
    """Reusable obstacle-vertex graph for all scenarios of one master field."""
    coordinates: tuple[tuple[float, float], ...]
    adjacency: tuple[tuple[tuple[int, float], ...], ...]
    obstacles: BaseGeometry


def _polygon_rings(geometry: BaseGeometry) -> list[Sequence[tuple[float, float]]]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry.exterior.coords, *(ring.coords for ring in geometry.interiors)]
    if isinstance(geometry, MultiPolygon):
        return [ring for polygon in geometry.geoms for ring in _polygon_rings(polygon)]
    if isinstance(geometry, GeometryCollection):
        return [ring for part in geometry.geoms for ring in _polygon_rings(part)]
    return []


def _obstacle_vertices(
    geometry: BaseGeometry,
    *,
    dedup_tolerance_km: float = 1e-9,
) -> tuple[tuple[float, float], ...]:
    seen: set[tuple[int, int]] = set()
    vertices: list[tuple[float, float]] = []
    for ring in _polygon_rings(geometry):
        for raw_x, raw_y in list(ring)[:-1]:
            x, y = float(raw_x), float(raw_y)
            key = (round(x / dedup_tolerance_km), round(y / dedup_tolerance_km))
            if key not in seen:
                seen.add(key)
                vertices.append((x, y))
    return tuple(vertices)


def _visible(start: tuple[float, float], end: tuple[float, float], obstacles: BaseGeometry) -> bool:
    if start == end:
        return False
    if obstacles.is_empty:
        return True
    segment = LineString((start, end))
    # Visibility graph edges may touch or follow an obstacle boundary, but the
    # open segment may never enter an obstacle interior.
    return segment.relate_pattern(obstacles, "F********")


def _connect_visibility(coordinates: Sequence[tuple[float, float]], obstacles: BaseGeometry) -> list[list[tuple[int, float]]]:
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in coordinates]

    def fan(first: int) -> tuple[int, tuple[tuple[int, float], ...]]:
        remaining = coordinate_array[first + 1:]
        if not len(remaining):
            return first, ()
        line_coordinates = np.empty((len(remaining), 2, 2), dtype=np.float64)
        line_coordinates[:, 0, :] = coordinate_array[first]
        line_coordinates[:, 1, :] = remaining
        candidate_lines = shapely.linestrings(line_coordinates)
        visible = np.asarray(shapely.relate_pattern(candidate_lines, obstacles, "F********"), dtype=bool)
        offsets = np.flatnonzero(visible)
        deltas = remaining[offsets] - coordinate_array[first]
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        return first, tuple(
            (first + 1 + int(offset), float(distance))
            for offset, distance in zip(offsets, distances)
        )

    workers = min(8, max(1, os.cpu_count() or 1)) if len(coordinates) >= 64 else 1
    if workers == 1:
        fans = map(fan, range(len(coordinates)))
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="visibility")
        fans = executor.map(fan, range(len(coordinates)), chunksize=1)
    try:
        for first, edges in fans:
            for second, value in edges:
                adjacency[first].append((second, value))
                adjacency[second].append((first, value))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return adjacency


def build_visibility_backbone(
    obstacle_map: ObstacleMap,
    bounds_km: tuple[float, float, float, float],
    *,
    simplify_tolerance_km: float = 0.005,
) -> VisibilityBackbone:
    obstacles = obstacle_map.geometry.intersection(box(*bounds_km))
    if simplify_tolerance_km > 0 and not obstacles.is_empty:
        obstacles = obstacles.simplify(simplify_tolerance_km, preserve_topology=True)
    coordinates = _obstacle_vertices(obstacles)
    adjacency = _connect_visibility(coordinates, obstacles)
    return VisibilityBackbone(
        coordinates,
        tuple(tuple(edges) for edges in adjacency),
        obstacles,
    )


def _dijkstra(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    source: int,
) -> tuple[list[float], list[int | None]]:
    distance = [inf] * len(adjacency)
    previous: list[int | None] = [None] * len(adjacency)
    distance[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distance[node] + EPSILON:
            continue
        for neighbour, edge_distance in adjacency[node]:
            candidate = current_distance + edge_distance
            if candidate + EPSILON < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distance, previous


def _reconstruct(previous: Sequence[int | None], source: int, target: int) -> tuple[int, ...]:
    path = [target]
    cursor = target
    while cursor != source:
        predecessor = previous[cursor]
        if predecessor is None:
            raise RuntimeError(f"no visibility-graph path between nodes {source} and {target}")
        path.append(predecessor)
        cursor = predecessor
    return tuple(reversed(path))


def build_safe_path_matrix(
    nodes: Mapping[int, tuple[float, float]],
    obstacle_map: ObstacleMap,
    *,
    simplify_tolerance_km: float = 0.005,
    clip_margin_km: float = 2.0,
    backbone: VisibilityBackbone | None = None,
) -> SafePathMatrix:
    """Build one reusable visibility graph and all required Dijkstra paths."""
    if len(nodes) < 2:
        raise ValueError("at least two navigation nodes are required")
    node_ids = tuple(nodes)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("navigation node ids must be unique")
    required_coordinates = tuple((float(nodes[node][0]), float(nodes[node][1])) for node in node_ids)
    for node_id, coordinate in zip(node_ids, required_coordinates):
        if obstacle_map.geometry.contains(Point(coordinate)):
            raise ValueError(f"navigation node {node_id} is inside an obstacle")

    if backbone is None:
        xs = [coordinate[0] for coordinate in required_coordinates]
        ys = [coordinate[1] for coordinate in required_coordinates]
        bounds = (
            min(xs) - clip_margin_km, min(ys) - clip_margin_km,
            max(xs) + clip_margin_km, max(ys) + clip_margin_km,
        )
        active_backbone = build_visibility_backbone(
            obstacle_map,
            bounds,
            simplify_tolerance_km=simplify_tolerance_km,
        )
    else:
        active_backbone = backbone
    obstacles = active_backbone.obstacles
    coordinates = [*required_coordinates, *active_backbone.coordinates]
    required_count = len(required_coordinates)
    adjacency = [[] for _ in coordinates]
    for vertex, edges in enumerate(active_backbone.adjacency):
        adjacency[required_count + vertex].extend(
            (required_count + neighbour, distance) for neighbour, distance in edges
        )
    # Only required-node connections are scenario-specific; the expensive
    # obstacle-vertex graph above is reused across all J in the master field.
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    for first in range(required_count):
        remaining = coordinate_array[first + 1:]
        line_coordinates = np.empty((len(remaining), 2, 2), dtype=np.float64)
        line_coordinates[:, 0, :] = coordinate_array[first]
        line_coordinates[:, 1, :] = remaining
        visible = np.asarray(
            shapely.relate_pattern(shapely.linestrings(line_coordinates), obstacles, "F********"),
            dtype=bool,
        )
        for offset in np.flatnonzero(visible):
            second = first + 1 + int(offset)
            delta = coordinate_array[second] - coordinate_array[first]
            distance = float(np.hypot(delta[0], delta[1]))
            adjacency[first].append((second, distance))
            adjacency[second].append((first, distance))

    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=np.float64)
    paths: list[SafePath] = []
    for source in range(len(node_ids)):
        distances, previous = _dijkstra(adjacency, source)
        for target in range(source + 1, len(node_ids)):
            if not np.isfinite(distances[target]):
                raise RuntimeError(
                    f"no safe route between required nodes {node_ids[source]} and {node_ids[target]}"
                )
            graph_path = _reconstruct(previous, source, target)
            polyline = tuple(coordinates[index] for index in graph_path)
            distance = float(distances[target])
            matrix[source, target] = matrix[target, source] = distance
            paths.append(SafePath(
                start=node_ids[source],
                end=node_ids[target],
                distance_km=distance,
                polyline=polyline,
            ))
    return SafePathMatrix(node_ids=node_ids, distance_km=matrix, paths=tuple(paths))
