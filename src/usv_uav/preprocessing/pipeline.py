from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from time import perf_counter
from typing import Any, Callable, Sequence

import shapely
from shapely.geometry import Point
from shapely.ops import unary_union

from usv_uav.config import ProjectConfig
from usv_uav.core.models import Instance
from usv_uav.data.master_field import MASTER_SCHEMA_VERSION, MasterField, compute_master_hash
from usv_uav.data.navigation_data import load_navigation_data
from usv_uav.data.synthetic import CalibrationCheck
from usv_uav.data.walney_loader import (
    WalneyLayout,
    build_master_tasks,
    generate_randomized_nested_order,
    load_walney_layout,
    select_active_tasks,
)
from usv_uav.preprocessing.coverage import CoverageMatrix, build_coverage_matrix, subset_coverage_matrix
from usv_uav.preprocessing.obstacle_builder import ObstacleBuildStats, ObstacleMap, build_obstacle_map
from usv_uav.preprocessing.safe_paths import SafePathMatrix, build_safe_path_matrix, build_visibility_backbone
from usv_uav.preprocessing.set_cover import SetCoverResult, solve_two_stage_set_cover
from usv_uav.preprocessing.sortie_generator import SortieGenerationStats, generate_sortie_pool
from usv_uav.preprocessing.sortie_index import SortieIndex, build_sortie_index
from usv_uav.preprocessing.support_generator import SupportGenerationResult, generate_candidate_supports
from usv_uav.preprocessing.tsp import GurobiTSPSolver, NN2OptTSPSolver, TSPSolution


FORMAL_TSP_UNAVAILABLE = "FORMAL_TSP_EXACT_SOLVER_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class MasterPreparationTimings:
    data_sec: float
    obstacle_sec: float
    candidate_sec: float
    coverage_sec: float

    @property
    def total_sec(self) -> float:
        return self.data_sec + self.obstacle_sec + self.candidate_sec + self.coverage_sec


@dataclass(frozen=True, slots=True)
class PipelineTimings:
    data_sec: float
    obstacle_sec: float
    candidate_sec: float
    set_cover_sec: float
    navigation_sec: float
    tsp_sec: float
    sortie_pool_sec: float

    @property
    def total_sec(self) -> float:
        return sum((self.data_sec, self.obstacle_sec, self.candidate_sec, self.set_cover_sec,
                    self.navigation_sec, self.tsp_sec, self.sortie_pool_sec))


@dataclass(frozen=True, slots=True)
class MasterPreparationResult:
    master_field: MasterField
    timings: MasterPreparationTimings


@dataclass(frozen=True, slots=True)
class PreprocessingPipelineResult:
    layout: WalneyLayout
    master_field: MasterField
    instance: Instance
    obstacle_map: ObstacleMap
    obstacle_stats: ObstacleBuildStats
    support_generation: SupportGenerationResult
    coverage: CoverageMatrix
    set_cover: SetCoverResult
    safe_paths: SafePathMatrix
    tsp: TSPSolution
    sortie_stats: SortieGenerationStats
    sortie_index: SortieIndex
    timings: PipelineTimings


def _clip_bounds(layout: WalneyLayout, margin: float) -> tuple[float, float, float, float]:
    coordinates = [layout.port.coordinate, *((value.x_km, value.y_km) for value in layout.turbines)]
    xs = [coordinate[0] for coordinate in coordinates]
    ys = [coordinate[1] for coordinate in coordinates]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def prepare_master_field(
    config: ProjectConfig,
    layout: WalneyLayout,
    *,
    geometry_seed: int,
    level_seed: int,
    task_order_seed: int | None = None,
    geometry_id: str | None = None,
    task_sizes: Sequence[int] | None = None,
    calibration: CalibrationCheck | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> MasterPreparationResult:
    """Build physical obstacles, candidates and coverage exactly once per field."""
    notify = progress or (lambda stage, values: None)
    order_seed = 20_000 + int(geometry_seed) if task_order_seed is None else int(task_order_seed)
    started = perf_counter()
    master_tasks = build_master_tasks(layout, config.inspection, level_seed=level_seed)
    task_order = generate_randomized_nested_order(master_tasks, seed=order_seed, task_sizes=task_sizes)
    navigation_data = load_navigation_data(config)
    data_sec = perf_counter() - started
    notify("master_data", {"physical_turbines": len(layout.turbines), "elapsed_sec": data_sec})

    started = perf_counter()
    obstacle_map, obstacle_stats = build_obstacle_map(
        layout.turbines,
        navigation_data,
        config.navigation,
        clip_bounds_km=_clip_bounds(layout, config.navigation.visibility_clip_margin_km),
    )
    environment_parts = [
        geometry for name, geometry in obstacle_map.categories.items()
        if name != "turbines" and not geometry.is_empty
    ]
    environment_geometry = unary_union(environment_parts) if environment_parts else shapely.GeometryCollection()
    invalid_turbines = tuple(
        turbine.source_id
        for turbine in layout.turbines
        if environment_geometry.intersects(Point(turbine.x_km, turbine.y_km))
    )
    if invalid_turbines:
        raise RuntimeError(f"MASTER_PHYSICAL_VALIDITY_GATE_FAILED:{invalid_turbines}")
    visibility_bounds = _clip_bounds(layout, config.navigation.visibility_clip_margin_km)
    backbone = build_visibility_backbone(
        obstacle_map,
        visibility_bounds,
        simplify_tolerance_km=config.navigation.visibility_simplify_m / 1000.0,
    )
    obstacle_sec = perf_counter() - started
    notify("master_obstacles", {"turbines": obstacle_stats.turbine_buffer_count, "elapsed_sec": obstacle_sec})

    started = perf_counter()
    support_generation = generate_candidate_supports(master_tasks, obstacle_map, config.navigation)
    candidate_sec = perf_counter() - started
    notify("master_supports", {"safe_candidates": len(support_generation.supports), "elapsed_sec": candidate_sec})

    started = perf_counter()
    full_coverage = build_coverage_matrix(support_generation.supports, master_tasks, config.physical.uav)
    if full_coverage.uncovered_task_ids:
        raise RuntimeError(f"F0_GEOMETRY_UNCOVERED:{full_coverage.uncovered_task_ids}")
    coverage_sec = perf_counter() - started
    notify("master_coverage", {"columns": len(master_tasks), "elapsed_sec": coverage_sec})

    active_geometry_id = geometry_id or (f"G{geometry_seed:02d}" if geometry_seed >= 0 else layout.name)
    navigation_payload = {
        "coastline_sha256": navigation_data.source_hashes["coastline"],
        "bathymetry_sha256": navigation_data.source_hashes["bathymetry"],
        "wreck_points_sha256": navigation_data.source_hashes["wreck_points"],
        "wreck_areas_sha256": navigation_data.source_hashes["wreck_areas"],
        "navigation_config_hash": sha256(
            json.dumps(asdict(config.navigation), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "obstacle_geometry_hash": sha256(shapely.to_wkb(obstacle_map.geometry)).hexdigest(),
    }
    navigation_payload["fingerprint_sha256"] = sha256(
        json.dumps(navigation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    master_hash = compute_master_hash(
        active_geometry_id,
        layout.turbines,
        master_tasks,
        task_order,
        support_generation.supports,
        full_coverage,
        geometry_seed=geometry_seed,
        level_seed=level_seed,
        task_order_seed=order_seed,
        navigation_fingerprint=navigation_payload,
    )
    master = MasterField(
        geometry_id=active_geometry_id,
        layout=layout,
        physical_turbines=layout.turbines,
        master_tasks=master_tasks,
        task_order=task_order,
        candidate_supports=support_generation.supports,
        full_coverage=full_coverage,
        obstacle_map=obstacle_map,
        obstacle_stats=obstacle_stats,
        support_generation=support_generation,
        geometry_seed=int(geometry_seed),
        level_seed=int(level_seed),
        task_order_seed=order_seed,
        master_hash=master_hash,
        visibility_backbone=backbone,
        calibration=calibration,
        navigation_fingerprint=navigation_payload,
        visibility_bounds_km=visibility_bounds,
    )
    return MasterPreparationResult(master, MasterPreparationTimings(data_sec, obstacle_sec, candidate_sec, coverage_sec))


def build_instance_from_master(
    config: ProjectConfig,
    master: MasterField,
    *,
    task_size: int,
    tsp_mode: str = "formal",
    tsp_time_limit_sec: float = 300.0,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> PreprocessingPipelineResult:
    """Build scenario-dependent set cover, safe route and sortie pool."""
    notify = progress or (lambda stage, values: None)
    tasks = select_active_tasks(master.master_tasks, master.task_order, task_size)
    coverage = subset_coverage_matrix(master.full_coverage, [task.id for task in tasks])

    started = perf_counter()
    set_cover = solve_two_stage_set_cover(coverage)
    set_cover_sec = perf_counter() - started
    support_by_id = {support.id: support for support in master.candidate_supports}
    selected = tuple(support_by_id[identifier] for identifier in set_cover.selected_support_ids)
    notify("set_cover", {"selected_supports": len(selected), "elapsed_sec": set_cover_sec})

    started = perf_counter()
    nodes = {0: master.layout.port.coordinate, **{support.id: support.coordinate for support in selected}}
    safe_paths = build_safe_path_matrix(
        nodes,
        master.obstacle_map,
        simplify_tolerance_km=config.navigation.visibility_simplify_m / 1000.0,
        clip_margin_km=config.navigation.visibility_clip_margin_km,
        backbone=master.visibility_backbone,
    )
    navigation_sec = perf_counter() - started
    notify("navigation", {"safe_paths": len(safe_paths.paths), "elapsed_sec": navigation_sec})

    started = perf_counter()
    normalised_mode = tsp_mode.lower()
    if normalised_mode == "formal":
        try:
            tsp = GurobiTSPSolver(time_limit_sec=tsp_time_limit_sec).solve(safe_paths.distance_km, depot=0)
        except Exception as error:
            raise RuntimeError(f"{FORMAL_TSP_UNAVAILABLE}: {error}") from error
        if not tsp.optimal:
            raise RuntimeError(f"FORMAL_TSP_NOT_OPTIMAL: gap={tsp.gap}")
    elif normalised_mode in {"smoke", "debug", "warm_start", "nn_2opt"}:
        tsp = NN2OptTSPSolver().solve(safe_paths.distance_km, depot=0)
    else:
        raise ValueError(f"unknown tsp_mode {tsp_mode!r}")
    usv_route = tuple(safe_paths.node_ids[index] for index in tsp.route)
    tsp_sec = perf_counter() - started
    notify("tsp", {"solver": tsp.solver, "optimal": tsp.optimal, "elapsed_sec": tsp_sec})

    started = perf_counter()
    sortie_pool, sortie_stats = generate_sortie_pool(
        tasks, selected, usv_route, config.physical.uav,
        max_tasks_per_sortie=config.inspection.max_tasks_per_sortie,
    )
    sortie_index = build_sortie_index(sortie_pool, usv_route[1:-1])
    uncovered = sortie_index.uncovered_tasks(tuple(task.id for task in tasks))
    if uncovered:
        raise RuntimeError(f"E0 sortie gate failed for tasks {uncovered}")
    sortie_sec = perf_counter() - started
    notify("sortie_pool", {"sorties": len(sortie_pool), "elapsed_sec": sortie_sec})

    instance = Instance(
        instance_id=f"{master.geometry_id}_J{task_size:03d}",
        tasks=tasks,
        support_points=master.candidate_supports,
        selected_supports=set_cover.selected_support_ids,
        usv_route=usv_route,
        safe_distance_matrix=safe_paths.distance_km,
        sortie_pool=sortie_pool,
        uav_count=config.physical.uav.baseline_count,
        charger_count=config.charging.chargers,
        metadata={
            "schema_version": "2.1", "master_hash": master.master_hash,
            "master_schema_version": MASTER_SCHEMA_VERSION,
            "geometry_id": master.geometry_id, "geometry_seed": str(master.geometry_seed),
            "level_seed": str(master.level_seed), "task_order_seed": str(master.task_order_seed),
            "layout_sha256": master.layout.source_sha256, "config_sha256": config.config_hash,
            "crs": master.layout.crs, "coordinate_units": master.layout.coordinate_units,
            "physical_turbine_count": str(len(master.physical_turbines)), "tsp_mode": normalised_mode,
        },
    )
    return PreprocessingPipelineResult(
        layout=master.layout,
        master_field=master,
        instance=instance,
        obstacle_map=master.obstacle_map,
        obstacle_stats=master.obstacle_stats,
        support_generation=master.support_generation,
        coverage=coverage,
        set_cover=set_cover,
        safe_paths=safe_paths,
        tsp=tsp,
        sortie_stats=sortie_stats,
        sortie_index=sortie_index,
        timings=PipelineTimings(0.0, 0.0, 0.0, set_cover_sec, navigation_sec, tsp_sec, sortie_sec),
    )


def build_layout_pipeline(
    config: ProjectConfig,
    layout: WalneyLayout,
    *,
    task_limit: int,
    level_seed: int,
    exact_tsp: bool = False,
    tsp_time_limit_sec: float = 60.0,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
    geometry_seed: int = 0,
    task_order_seed: int | None = None,
    tsp_mode: str | None = None,
) -> PreprocessingPipelineResult:
    """Compatibility facade over the V2 two-layer pipeline."""
    preparation = prepare_master_field(
        config, layout, geometry_seed=geometry_seed, level_seed=level_seed,
        task_order_seed=task_order_seed, geometry_id=layout.name, progress=progress,
    )
    active_mode = tsp_mode or ("formal" if exact_tsp else "smoke")
    result = build_instance_from_master(
        config, preparation.master_field, task_size=task_limit, tsp_mode=active_mode,
        tsp_time_limit_sec=tsp_time_limit_sec, progress=progress,
    )
    master_timing = preparation.timings
    scenario_timing = result.timings
    return replace(result, timings=PipelineTimings(
        master_timing.data_sec, master_timing.obstacle_sec,
        master_timing.candidate_sec + master_timing.coverage_sec,
        scenario_timing.set_cover_sec, scenario_timing.navigation_sec,
        scenario_timing.tsp_sec, scenario_timing.sortie_pool_sec,
    ))


def build_walney_instance(
    config: ProjectConfig,
    *,
    task_limit: int,
    level_seed: int,
    tsp_mode: str = "formal",
    tsp_time_limit_sec: float = 300.0,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> PreprocessingPipelineResult:
    layout = load_walney_layout(config.paths.walney_layout, require_full_189=True)
    return build_layout_pipeline(
        config, layout, task_limit=task_limit, level_seed=level_seed,
        tsp_time_limit_sec=tsp_time_limit_sec, progress=progress,
        geometry_seed=-1, tsp_mode=tsp_mode,
    )


build_walney_pipeline = build_walney_instance
