from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from usv_uav.core.models import Instance, Port, SafePath, Sortie, SupportPoint, Task
from usv_uav.preprocessing.safe_paths import SafePathMatrix


SCHEMA_VERSION = "2.1"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_paths_geojson(paths: SafePathMatrix) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": "safe_paths",
        "crs": {"type": "name", "properties": {"name": "EPSG:32630"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "start": path.start,
                    "end": path.end,
                    "distance_km": path.distance_km,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(point) for point in path.polyline],
                },
            }
            for path in paths.paths
        ],
    }


def freeze_instance(
    instance: Instance,
    directory: str | Path,
    *,
    port: Port,
    safe_paths: SafePathMatrix,
    preprocessing_metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Write an immutable, hashed instance bundle through an atomic rename."""
    target = Path(directory).resolve()
    if target.exists() and not force:
        raise FileExistsError(f"frozen instance already exists: {target}")
    temporary = target.with_name(f".{target.name}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    _write_json(temporary / "tasks.json", [asdict(task) for task in instance.tasks])
    _write_json(temporary / "supports.json", [asdict(support) for support in instance.support_points])
    _write_json(temporary / "selected_supports.json", list(instance.selected_supports))
    _write_json(temporary / "port.json", asdict(port))
    _write_json(temporary / "usv_route.json", list(instance.usv_route))
    np.savez_compressed(
        temporary / "safe_distance.npz",
        node_ids=np.asarray(safe_paths.node_ids, dtype=np.int64),
        distance_km=instance.safe_distance_matrix,
    )
    _write_json(temporary / "safe_paths.geojson", _safe_paths_geojson(safe_paths))
    pd.DataFrame([asdict(sortie) for sortie in instance.sortie_pool]).to_parquet(
        temporary / "sorties.parquet",
        index=False,
    )

    artifact_names = (
        "tasks.json",
        "supports.json",
        "selected_supports.json",
        "port.json",
        "usv_route.json",
        "safe_distance.npz",
        "safe_paths.geojson",
        "sorties.parquet",
    )
    hashes = {name: file_sha256(temporary / name) for name in artifact_names}
    instance_hash = sha256(
        "".join(f"{name}:{hashes[name]}\n" for name in sorted(hashes)).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "crs": "EPSG:32630",
        "coordinate_units": "km",
        "task_count": len(instance.tasks),
        "candidate_support_count": len(instance.support_points),
        "selected_support_count": len(instance.selected_supports),
        "sortie_count": len(instance.sortie_pool),
        "uav_count": instance.uav_count,
        "charger_count": instance.charger_count,
        "instance_hash": instance_hash,
        "master_geometry_id": (preprocessing_metadata or {}).get("master_geometry_id"),
        "master_hash": (preprocessing_metadata or {}).get("master_hash"),
        "master_schema_version": (preprocessing_metadata or {}).get("master_schema_version"),
        "artifact_sha256": hashes,
        "instance_metadata": dict(instance.metadata or {}),
        "preprocessing": preprocessing_metadata or {},
        "immutable": True,
    }
    _write_json(temporary / "instance_manifest.json", manifest)
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return target


def load_frozen_instance(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[Instance, Port, dict[str, Any]]:
    source = Path(directory).resolve()
    manifest = json.loads((source / "instance_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(
            f"V2.1 loader requires schema_version {SCHEMA_VERSION}; found {manifest.get('schema_version')}"
        )
    if verify_hashes:
        for name, expected in manifest["artifact_sha256"].items():
            actual = file_sha256(source / name)
            if actual != expected:
                raise ValueError(f"frozen artifact hash mismatch: {name}")
    tasks = tuple(Task(**payload) for payload in json.loads((source / "tasks.json").read_text(encoding="utf-8")))
    supports = tuple(SupportPoint(**payload) for payload in json.loads((source / "supports.json").read_text(encoding="utf-8")))
    selected = tuple(json.loads((source / "selected_supports.json").read_text(encoding="utf-8")))
    route = tuple(json.loads((source / "usv_route.json").read_text(encoding="utf-8")))
    port = Port(**json.loads((source / "port.json").read_text(encoding="utf-8")))
    with np.load(source / "safe_distance.npz") as archive:
        matrix = archive["distance_km"]
    frame = pd.read_parquet(source / "sorties.parquet")
    sorties = tuple(
        Sortie(
            id=int(row.id),
            origin_support=int(row.origin_support),
            task_sequence=tuple(int(value) for value in row.task_sequence),
            recovery_support=int(row.recovery_support),
            flight_distance_km=float(row.flight_distance_km),
            flight_time_min=float(row.flight_time_min),
            inspection_time_min=float(row.inspection_time_min),
            nominal_energy_wh=float(row.nominal_energy_wh),
        )
        for row in frame.itertuples(index=False)
    )
    instance = Instance(
        instance_id=manifest["instance_id"],
        tasks=tasks,
        support_points=supports,
        selected_supports=selected,
        usv_route=route,
        safe_distance_matrix=matrix,
        sortie_pool=sorties,
        uav_count=int(manifest["uav_count"]),
        charger_count=int(manifest["charger_count"]),
        metadata=manifest.get("instance_metadata", {}),
    )
    return instance, port, manifest


def load_frozen_safe_paths(directory: str | Path) -> SafePathMatrix:
    """Load the auditable safe-path matrix and polylines from a frozen bundle."""
    source = Path(directory).resolve()
    with np.load(source / "safe_distance.npz") as archive:
        node_ids = tuple(int(value) for value in archive["node_ids"])
        distance = archive["distance_km"]
    payload = json.loads((source / "safe_paths.geojson").read_text(encoding="utf-8"))
    paths = tuple(
        SafePath(
            start=int(feature["properties"]["start"]),
            end=int(feature["properties"]["end"]),
            distance_km=float(feature["properties"]["distance_km"]),
            polyline=tuple(
                (float(point[0]), float(point[1]))
                for point in feature["geometry"]["coordinates"]
            ),
        )
        for feature in payload["features"]
    )
    return SafePathMatrix(node_ids=node_ids, distance_km=distance, paths=paths)
