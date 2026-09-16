from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from usv_uav.config import InspectionConfig
from usv_uav.core.models import Port, Task


@dataclass(frozen=True, slots=True)
class TurbineCoordinate:
    source_id: str
    x_km: float
    y_km: float


@dataclass(frozen=True, slots=True)
class WalneyLayout:
    name: str
    crs: str
    coordinate_units: str
    port: Port
    turbines: tuple[TurbineCoordinate, ...]
    source_description: str
    source_sha256: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_walney_layout(path: str | Path, *, require_full_189: bool = False) -> WalneyLayout:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("coordinate_reference_system") != "EPSG:32630":
        raise ValueError("Walney layout must use EPSG:32630")
    if payload.get("coordinate_units") != "km":
        raise ValueError("Walney layout must use kilometre coordinates")

    raw_turbines = payload.get("turbines")
    if not isinstance(raw_turbines, list) or not raw_turbines:
        raise ValueError("Walney layout must contain a non-empty turbines list")
    turbines = tuple(
        TurbineCoordinate(str(item["id"]), float(item["x_km"]), float(item["y_km"]))
        for item in raw_turbines
    )
    ids = [item.source_id for item in turbines]
    coordinates = [(item.x_km, item.y_km) for item in turbines]
    if len(set(ids)) != len(ids):
        raise ValueError("Walney turbine ids must be unique")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Walney turbine coordinates must be unique")
    if require_full_189 and len(turbines) != 189:
        raise ValueError(f"expected Walney-189, found {len(turbines)} turbines")

    raw_port = payload["port"]
    return WalneyLayout(
        name=str(payload.get("name", source_path.stem)),
        crs="EPSG:32630",
        coordinate_units="km",
        port=Port(float(raw_port["x_km"]), float(raw_port["y_km"])),
        turbines=turbines,
        source_description=str(payload.get("source", "")),
        source_sha256=_file_sha256(source_path),
    )


def balanced_level_counts(total: int, shares: Mapping[int, float]) -> dict[int, int]:
    """Allocate integer counts with the deterministic largest-remainder rule."""
    if total <= 0:
        raise ValueError("total must be positive")
    levels = sorted(shares)
    exact = {level: total * float(shares[level]) for level in levels}
    counts = {level: int(np.floor(exact[level])) for level in levels}
    remaining = total - sum(counts.values())
    order = sorted(levels, key=lambda level: (-(exact[level] - counts[level]), level))
    for level in order[:remaining]:
        counts[level] += 1
    return counts


def balanced_level_sequence(
    total: int,
    shares: Mapping[int, float],
    *,
    seed: int,
) -> tuple[int, ...]:
    counts = balanced_level_counts(total, shares)
    levels = np.concatenate([
        np.full(counts[level], level, dtype=np.int64) for level in sorted(counts)
    ])
    generator = np.random.default_rng(seed)
    generator.shuffle(levels)
    return tuple(int(level) for level in levels)


def build_tasks(
    layout: WalneyLayout,
    inspection: InspectionConfig,
    *,
    level_seed: int,
    task_limit: int | None = None,
) -> tuple[Task, ...]:
    """Create immutable model tasks after assigning levels exactly once.

    The level permutation is generated for the complete layout before a prefix
    is selected, so increasing ``task_limit`` preserves nested task sets.
    """
    shares = {level: values.share for level, values in inspection.levels.items()}
    sequence = balanced_level_sequence(len(layout.turbines), shares, seed=level_seed)
    limit = len(layout.turbines) if task_limit is None else task_limit
    if not 1 <= limit <= len(layout.turbines):
        raise ValueError("task_limit must be within the layout size")
    tasks: list[Task] = []
    for index, (turbine, level) in enumerate(zip(layout.turbines[:limit], sequence[:limit]), start=1):
        tasks.append(Task(
            id=index,
            x_km=turbine.x_km,
            y_km=turbine.y_km,
            level=level,
            service_min=inspection.levels[level].service_min,
            source_id=turbine.source_id,
        ))
    return tuple(tasks)


def build_master_tasks(
    layout: WalneyLayout,
    inspection: InspectionConfig,
    *,
    level_seed: int,
) -> tuple[Task, ...]:
    """Freeze ids, levels and service times for every turbine in a field."""
    shares = {level: values.share for level, values in inspection.levels.items()}
    sequence = balanced_level_sequence(len(layout.turbines), shares, seed=level_seed)
    return tuple(
        Task(
            id=index,
            x_km=turbine.x_km,
            y_km=turbine.y_km,
            level=level,
            service_min=inspection.levels[level].service_min,
            source_id=turbine.source_id,
        )
        for index, (turbine, level) in enumerate(zip(layout.turbines, sequence), start=1)
    )


def generate_randomized_nested_order(
    master_tasks: Sequence[Task],
    *,
    seed: int,
    task_sizes: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Create a spatially independent, level-balanced nested activation order.

    Each level pool and each newly activated block is shuffled using only task
    identity, level and ``seed``. Coordinates are deliberately never read.
    The requested checkpoints receive largest-remainder level counts.
    """
    if not master_tasks:
        raise ValueError("master_tasks must be non-empty")
    ids = [task.id for task in master_tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("master task ids must be unique")
    total = len(master_tasks)
    checkpoints = sorted(set(int(value) for value in (task_sizes or range(1, total + 1))))
    if not checkpoints or checkpoints[0] <= 0 or checkpoints[-1] > total:
        raise ValueError("task_sizes must be within the master task count")
    if checkpoints[-1] != total:
        checkpoints.append(total)

    pools = {
        level: [task.id for task in master_tasks if task.level == level]
        for level in sorted({task.level for task in master_tasks})
    }
    generator = np.random.default_rng(seed)
    for pool in pools.values():
        generator.shuffle(pool)
    shares = {level: len(pool) / total for level, pool in pools.items()}
    used = {level: 0 for level in pools}
    order: list[int] = []
    for checkpoint in checkpoints:
        targets = balanced_level_counts(checkpoint, shares)
        block: list[int] = []
        for level in sorted(pools):
            required = targets[level] - used[level]
            if required < 0:
                raise ValueError("non-monotone balanced level targets")
            block.extend(pools[level][used[level] : used[level] + required])
            used[level] += required
        generator.shuffle(block)
        order.extend(block)
    if len(order) != total or set(order) != set(ids):
        raise RuntimeError("failed to construct a complete randomized nested order")
    return tuple(order)


def select_active_tasks(
    master_tasks: Sequence[Task],
    task_order: Sequence[int],
    task_size: int,
) -> tuple[Task, ...]:
    """Select a nested scenario without changing master-level task identity."""
    if not 1 <= task_size <= len(master_tasks):
        raise ValueError("task_size must be within the master task count")
    by_id = {task.id: task for task in master_tasks}
    selected_ids = tuple(int(value) for value in task_order[:task_size])
    if len(set(selected_ids)) != task_size or not set(selected_ids) <= set(by_id):
        raise ValueError("task_order is not a valid master-task permutation")
    return tuple(by_id[task_id] for task_id in selected_ids)
