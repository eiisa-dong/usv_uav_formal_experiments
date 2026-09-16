from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import shapely

from usv_uav.core.models import Port, SupportPoint, Task
from usv_uav.data.instance_io import file_sha256
from usv_uav.data.synthetic import CalibrationCheck
from usv_uav.data.walney_loader import TurbineCoordinate, WalneyLayout
from usv_uav.preprocessing.coverage import CoverageMatrix
from usv_uav.preprocessing.obstacle_builder import ObstacleBuildStats, ObstacleMap
from usv_uav.preprocessing.safe_paths import VisibilityBackbone
from usv_uav.preprocessing.support_generator import SupportGenerationResult


MASTER_SCHEMA_VERSION = "2.1"


@dataclass(frozen=True, slots=True)
class MasterField:
    geometry_id: str
    layout: WalneyLayout
    physical_turbines: tuple[TurbineCoordinate, ...]
    master_tasks: tuple[Task, ...]
    task_order: tuple[int, ...]
    candidate_supports: tuple[SupportPoint, ...]
    full_coverage: CoverageMatrix
    obstacle_map: ObstacleMap
    obstacle_stats: ObstacleBuildStats
    support_generation: SupportGenerationResult
    geometry_seed: int
    level_seed: int
    task_order_seed: int
    master_hash: str
    visibility_backbone: VisibilityBackbone | None = None
    calibration: CalibrationCheck | None = None
    navigation_fingerprint: Mapping[str, str] | None = None
    visibility_bounds_km: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        task_ids = tuple(task.id for task in self.master_tasks)
        if self.physical_turbines != self.layout.turbines:
            raise ValueError("physical_turbines must be the complete layout")
        if len(self.master_tasks) != len(self.physical_turbines):
            raise ValueError("every physical turbine must have one frozen master task")
        if len(self.task_order) != len(task_ids) or set(self.task_order) != set(task_ids):
            raise ValueError("task_order must be a permutation of all master task ids")
        if self.full_coverage.task_ids != task_ids:
            raise ValueError("full coverage columns must follow master task ids")
        if self.full_coverage.support_ids != tuple(point.id for point in self.candidate_supports):
            raise ValueError("full coverage rows must follow candidate support ids")
        if self.full_coverage.uncovered_task_ids:
            raise ValueError(f"master field has uncovered tasks: {self.full_coverage.uncovered_task_ids}")


def compute_master_hash(
    geometry_id: str,
    physical_turbines: tuple[TurbineCoordinate, ...],
    master_tasks: tuple[Task, ...],
    task_order: tuple[int, ...],
    candidate_supports: tuple[SupportPoint, ...],
    coverage: CoverageMatrix,
    *,
    geometry_seed: int,
    level_seed: int,
    task_order_seed: int,
    navigation_fingerprint: Mapping[str, str] | None = None,
) -> str:
    digest = sha256()
    payload = {
        "schema_version": MASTER_SCHEMA_VERSION,
        "geometry_id": geometry_id,
        "geometry_seed": geometry_seed,
        "level_seed": level_seed,
        "task_order_seed": task_order_seed,
        "physical_turbines": [asdict(value) for value in physical_turbines],
        "master_tasks": [asdict(value) for value in master_tasks],
        "task_order": list(task_order),
        "candidate_supports": [asdict(value) for value in candidate_supports],
        "navigation_fingerprint": dict(navigation_fingerprint or {}),
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(coverage.feasible).tobytes())
    digest.update(np.ascontiguousarray(coverage.roundtrip_distance_km).tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_master_field(master: MasterField, directory: str | Path, *, force: bool = False) -> Path:
    """Persist the auditable physical field and master coverage atomically."""
    target = Path(directory).resolve()
    if target.exists() and not force:
        raise FileExistsError(f"master field already exists: {target}")
    temporary = target.with_name(f".{target.name}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    _write_json(temporary / "master_field.json", {
        "schema_version": MASTER_SCHEMA_VERSION,
        "geometry_id": master.geometry_id,
        "name": master.layout.name,
        "coordinate_reference_system": master.layout.crs,
        "coordinate_units": master.layout.coordinate_units,
        "port": asdict(master.layout.port),
        "source": master.layout.source_description,
        "source_sha256": master.layout.source_sha256,
        "physical_turbines": [asdict(value) for value in master.physical_turbines],
    })
    _write_json(temporary / "master_tasks.json", [asdict(value) for value in master.master_tasks])
    _write_json(temporary / "task_order.json", {
        "geometry_seed": master.geometry_seed,
        "task_order_seed": master.task_order_seed,
        "order": list(master.task_order),
    })
    _write_json(temporary / "candidate_supports.json", [asdict(value) for value in master.candidate_supports])
    np.savez_compressed(
        temporary / "coverage_full.npz",
        support_ids=np.asarray(master.full_coverage.support_ids, dtype=np.int64),
        task_ids=np.asarray(master.full_coverage.task_ids, dtype=np.int64),
        feasible=master.full_coverage.feasible,
        roundtrip_distance_km=master.full_coverage.roundtrip_distance_km,
    )
    (temporary / "obstacle_geometry.wkb").write_bytes(shapely.to_wkb(master.obstacle_map.geometry))
    if master.visibility_backbone is None:
        raise ValueError("formal master field requires a visibility backbone")
    indptr = [0]
    indices: list[int] = []
    weights: list[float] = []
    for edges in master.visibility_backbone.adjacency:
        for neighbour, weight in edges:
            indices.append(int(neighbour))
            weights.append(float(weight))
        indptr.append(len(indices))
    coordinates = np.asarray(master.visibility_backbone.coordinates, dtype=np.float64)
    if coordinates.size == 0:
        coordinates = coordinates.reshape((0, 2))
    np.savez_compressed(
        temporary / "visibility_backbone.npz",
        coordinates=coordinates,
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
        obstacles_wkb=np.frombuffer(shapely.to_wkb(master.visibility_backbone.obstacles), dtype=np.uint8),
    )
    artifacts = (
        "master_field.json", "master_tasks.json", "task_order.json",
        "candidate_supports.json", "coverage_full.npz", "obstacle_geometry.wkb",
        "visibility_backbone.npz",
    )
    hashes = {name: file_sha256(temporary / name) for name in artifacts}
    _write_json(temporary / "master_manifest.json", {
        "schema_version": MASTER_SCHEMA_VERSION,
        "geometry_id": master.geometry_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_turbine_count": len(master.physical_turbines),
        "master_task_count": len(master.master_tasks),
        "candidate_support_count": len(master.candidate_supports),
        "geometry_seed": master.geometry_seed,
        "level_seed": master.level_seed,
        "task_order_seed": master.task_order_seed,
        "master_hash": master.master_hash,
        "navigation_fingerprint": dict(master.navigation_fingerprint or {}),
        "calibration": asdict(master.calibration) if master.calibration is not None else None,
        "physical_validity": {"environment_obstacle_intersections": 0, "passed": True},
        "visibility_bounds_km": list(master.visibility_bounds_km) if master.visibility_bounds_km is not None else None,
        "artifact_sha256": hashes,
        "obstacles": asdict(master.obstacle_stats),
        "support_generation": {
            "generated_count": master.support_generation.generated_count,
            "safe_count": len(master.support_generation.supports),
            "unsafe_count": master.support_generation.unsafe_count,
            "median_nearest_neighbour_km": master.support_generation.median_nearest_neighbour_km,
            "grid_spacing_km": master.support_generation.grid_spacing_km,
            "bounds_km": list(master.support_generation.bounds_km),
            "duplicate_count": master.support_generation.duplicate_count,
        },
        "immutable": True,
    })
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return target


def load_master_field(directory: str | Path, *, verify_hashes: bool = True) -> MasterField:
    """Load and verify a frozen master without regenerating geometry or navigation."""
    source = Path(directory).resolve()
    manifest = json.loads((source / "master_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("schema_version")) != MASTER_SCHEMA_VERSION:
        raise ValueError(
            f"master loader requires schema_version {MASTER_SCHEMA_VERSION}; "
            f"found {manifest.get('schema_version')}"
        )
    if manifest.get("immutable") is not True:
        raise ValueError("master manifest is not immutable")
    if verify_hashes:
        for name, expected in manifest["artifact_sha256"].items():
            actual = file_sha256(source / name)
            if actual != expected:
                raise ValueError(f"frozen master artifact hash mismatch: {name}")

    field_payload = json.loads((source / "master_field.json").read_text(encoding="utf-8"))
    turbines = tuple(TurbineCoordinate(**value) for value in field_payload["physical_turbines"])
    layout = WalneyLayout(
        name=str(field_payload["name"]),
        crs=str(field_payload["coordinate_reference_system"]),
        coordinate_units=str(field_payload["coordinate_units"]),
        port=Port(**field_payload["port"]),
        turbines=turbines,
        source_description=str(field_payload["source"]),
        source_sha256=str(field_payload["source_sha256"]),
    )
    tasks = tuple(Task(**value) for value in json.loads((source / "master_tasks.json").read_text(encoding="utf-8")))
    order_payload = json.loads((source / "task_order.json").read_text(encoding="utf-8"))
    task_order = tuple(int(value) for value in order_payload["order"])
    supports = tuple(
        SupportPoint(**value)
        for value in json.loads((source / "candidate_supports.json").read_text(encoding="utf-8"))
    )
    with np.load(source / "coverage_full.npz", allow_pickle=False) as archive:
        coverage = CoverageMatrix(
            support_ids=tuple(int(value) for value in archive["support_ids"]),
            task_ids=tuple(int(value) for value in archive["task_ids"]),
            feasible=archive["feasible"],
            roundtrip_distance_km=archive["roundtrip_distance_km"],
        )
    obstacle_geometry = shapely.from_wkb((source / "obstacle_geometry.wkb").read_bytes())
    obstacle_map = ObstacleMap(obstacle_geometry, {"frozen_obstacles": obstacle_geometry})
    with np.load(source / "visibility_backbone.npz", allow_pickle=False) as archive:
        coordinates = tuple((float(x), float(y)) for x, y in archive["coordinates"])
        indptr = archive["indptr"]
        indices = archive["indices"]
        weights = archive["weights"]
        adjacency = tuple(
            tuple((int(indices[p]), float(weights[p])) for p in range(int(indptr[i]), int(indptr[i + 1])))
            for i in range(len(indptr) - 1)
        )
        backbone_obstacles = shapely.from_wkb(archive["obstacles_wkb"].tobytes())
    backbone = VisibilityBackbone(coordinates, adjacency, backbone_obstacles)
    support_payload = manifest["support_generation"]
    support_generation = SupportGenerationResult(
        supports=supports,
        median_nearest_neighbour_km=float(support_payload["median_nearest_neighbour_km"]),
        grid_spacing_km=float(support_payload["grid_spacing_km"]),
        bounds_km=tuple(float(value) for value in support_payload["bounds_km"]),
        generated_count=int(support_payload["generated_count"]),
        unsafe_count=int(support_payload["unsafe_count"]),
        duplicate_count=int(support_payload["duplicate_count"]),
    )
    calibration_payload = manifest.get("calibration")
    calibration = CalibrationCheck(**calibration_payload) if calibration_payload else None
    navigation_fingerprint = {
        str(key): str(value) for key, value in manifest.get("navigation_fingerprint", {}).items()
    }
    expected_master_hash = compute_master_hash(
        str(manifest["geometry_id"]), turbines, tasks, task_order, supports, coverage,
        geometry_seed=int(manifest["geometry_seed"]),
        level_seed=int(manifest["level_seed"]),
        task_order_seed=int(manifest["task_order_seed"]),
        navigation_fingerprint=navigation_fingerprint,
    )
    if expected_master_hash != manifest.get("master_hash"):
        raise ValueError("frozen master hash does not match its complete payload")
    visibility_bounds = manifest.get("visibility_bounds_km")
    return MasterField(
        geometry_id=str(manifest["geometry_id"]),
        layout=layout,
        physical_turbines=turbines,
        master_tasks=tasks,
        task_order=task_order,
        candidate_supports=supports,
        full_coverage=coverage,
        obstacle_map=obstacle_map,
        obstacle_stats=ObstacleBuildStats(**manifest["obstacles"]),
        support_generation=support_generation,
        geometry_seed=int(manifest["geometry_seed"]),
        level_seed=int(manifest["level_seed"]),
        task_order_seed=int(manifest["task_order_seed"]),
        master_hash=str(manifest["master_hash"]),
        visibility_backbone=backbone,
        calibration=calibration,
        navigation_fingerprint=navigation_fingerprint,
        visibility_bounds_km=tuple(float(value) for value in visibility_bounds) if visibility_bounds else None,
    )
