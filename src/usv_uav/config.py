from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration violates the data contract."""


SCHEMA_VERSION = "2.1"
LaunchPolicy = Literal["immediate", "dwell_window_synchronized"]
MODEL_LAUNCH_POLICIES: dict[str, LaunchPolicy] = {
    "2.1": "immediate",
    "3.0": "dwell_window_synchronized",
}


def _require_keys(section: str, data: Mapping[str, Any], keys: set[str]) -> None:
    missing = keys - set(data)
    if missing:
        raise ConfigError(f"{section}: missing keys: {sorted(missing)}")


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class USVConfig:
    count: int
    speed_km_min: float
    draft_m: float
    ukc_m: float

    def __post_init__(self) -> None:
        if self.count != 1:
            raise ConfigError("the current model requires exactly one USV")
        _positive("usv.speed_km_min", self.speed_km_min)
        _positive("usv.draft_m", self.draft_m)
        if self.ukc_m < 0:
            raise ConfigError("usv.ukc_m must be non-negative")


@dataclass(frozen=True, slots=True)
class UAVConfig:
    baseline_count: int
    speed_km_min: float
    battery_wh: float
    safety_soc_ratio: float
    flight_energy_wh_min: float
    inspection_energy_wh_min: float
    hovering_energy_wh_min: float
    # Retain the stored name for constructor/replace and historical config hashes.
    max_sortie_min: float

    def __post_init__(self) -> None:
        if self.baseline_count <= 0:
            raise ConfigError("uav.baseline_count must be positive")
        for name in (
            "speed_km_min",
            "battery_wh",
            "flight_energy_wh_min",
            "inspection_energy_wh_min",
            "hovering_energy_wh_min",
            "max_sortie_min",
        ):
            _positive(f"uav.{name}", float(getattr(self, name)))
        if not 0 < self.safety_soc_ratio < 1:
            raise ConfigError("uav.safety_soc_ratio must be in (0, 1)")

    @property
    def safety_soc_wh(self) -> float:
        return self.battery_wh * self.safety_soc_ratio

    @property
    def max_sortie_duration_min(self) -> float:
        """Single endurance limit used by nominal generation and actual decoding."""
        return self.max_sortie_min


@dataclass(frozen=True, slots=True)
class PhysicalConfig:
    schema_version: str
    model_semantics_version: str
    launch_policy: LaunchPolicy
    usv: USVConfig
    uav: UAVConfig

    def __post_init__(self) -> None:
        expected = MODEL_LAUNCH_POLICIES.get(self.model_semantics_version)
        if expected is None:
            raise ConfigError(
                "physical.model_semantics_version must be 2.1 or 3.0"
            )
        if self.launch_policy != expected:
            raise ConfigError(
                "physical launch policy does not match model semantics version: "
                f"{self.model_semantics_version} requires {expected}"
            )


@dataclass(frozen=True, slots=True)
class InspectionLevelConfig:
    service_min: float
    share: float

    def __post_init__(self) -> None:
        _positive("inspection service_min", self.service_min)
        if not 0 <= self.share <= 1:
            raise ConfigError("inspection share must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    schema_version: str
    levels: Mapping[int, InspectionLevelConfig]
    max_tasks_per_sortie: int
    mandatory_tasks: bool

    def __post_init__(self) -> None:
        if set(self.levels) != {1, 2, 3}:
            raise ConfigError("inspection levels must be exactly 1, 2, and 3")
        if abs(sum(level.share for level in self.levels.values()) - 1.0) > 1e-9:
            raise ConfigError("inspection level shares must sum to 1")
        if self.max_tasks_per_sortie <= 0:
            raise ConfigError("max_tasks_per_sortie must be positive")


@dataclass(frozen=True, slots=True)
class NavigationConfig:
    schema_version: str
    target_crs: str
    coordinate_units: str
    minimum_water_depth_m: float
    bathymetry_positive_down: bool
    turbine_buffer_m: float
    wreck_obstruction_buffer_m: float
    land_buffer_m: float
    use_land: bool
    use_shallow_water: bool
    use_turbine_buffers: bool
    use_wreck_obstruction_buffers: bool
    candidate_grid_spacing_nn_factor: float
    candidate_bounds_margin_nn_factor: float
    coordinate_dedup_tolerance_m: float
    visibility_simplify_m: float
    visibility_clip_margin_km: float

    def __post_init__(self) -> None:
        if self.target_crs.upper() != "EPSG:32630":
            raise ConfigError("the formal project CRS is fixed to EPSG:32630")
        if self.coordinate_units != "km":
            raise ConfigError("the model coordinate unit is fixed to km")
        for name in (
            "minimum_water_depth_m",
            "turbine_buffer_m",
            "wreck_obstruction_buffer_m",
            "candidate_grid_spacing_nn_factor",
            "coordinate_dedup_tolerance_m",
            "visibility_simplify_m",
        ):
            _positive(f"navigation.{name}", float(getattr(self, name)))
        if (
            self.land_buffer_m < 0
            or self.candidate_bounds_margin_nn_factor < 0
            or self.visibility_clip_margin_km < 0
        ):
            raise ConfigError("navigation buffers and margin factors must be non-negative")


@dataclass(frozen=True, slots=True)
class ChargingConfig:
    schema_version: str
    chargers: int
    effective_rate_wh_min: float
    policy: str
    non_idling: bool
    setup_time_min: float
    transfer_to_charger_min: float
    final_full_charge: bool
    charging_during_sailing: bool

    def __post_init__(self) -> None:
        if self.chargers <= 0:
            raise ConfigError("chargers must be positive")
        _positive("charging.effective_rate_wh_min", self.effective_rate_wh_min)
        if self.setup_time_min < 0 or self.transfer_to_charger_min < 0:
            raise ConfigError("charging setup and transfer times must be non-negative")
        if not self.policy:
            raise ConfigError("charging.policy is required")


@dataclass(frozen=True, slots=True)
class TSPConfig:
    schema_version: str
    formal_solver: str
    smoke_solver: str
    time_limit_sec: float

    def __post_init__(self) -> None:
        if self.formal_solver != "gurobi_exact":
            raise ConfigError("formal TSP solver is fixed to gurobi_exact")
        if self.smoke_solver != "nn_2opt":
            raise ConfigError("smoke TSP solver is fixed to nn_2opt")
        _positive("tsp.time_limit_sec", self.time_limit_sec)


@dataclass(frozen=True, slots=True)
class PathConfig:
    schema_version: str
    project_root: Path
    data_root: Path
    raw_data_root: Path
    processed_root: Path
    cache_root: Path
    results_root: Path
    walney_layout: Path
    coastline: Path
    bathymetry: Path
    wreck_points: Path
    wreck_areas: Path


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    physical: PhysicalConfig
    inspection: InspectionConfig
    navigation: NavigationConfig
    charging: ChargingConfig
    tsp: TSPConfig
    paths: PathConfig

    @property
    def config_hash(self) -> str:
        return _stable_hash(asdict(self))

    @property
    def decoder_semantics_hash(self) -> str:
        """Hash every configured input that changes authoritative decoding."""
        return _stable_hash(
            {
                "model_semantics_version": self.physical.model_semantics_version,
                "launch_policy": self.physical.launch_policy,
                "usv_speed_km_min": self.physical.usv.speed_km_min,
                "uav": asdict(self.physical.uav),
                "charging": asdict(self.charging),
            }
        )


def _normalise_hash_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _normalise_hash_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_hash_value(item) for item in value]
    return value


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        _normalise_hash_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_project_config(
    config_dir: str | Path | None = None,
    *,
    physical_filename: str = "physical.yaml",
) -> ProjectConfig:
    config_path = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
    config_path = config_path.resolve()
    project_root = config_path.parent

    physical_raw = _read_yaml(config_path / physical_filename)
    inspection_raw = _read_yaml(config_path / "inspection.yaml")
    navigation_raw = _read_yaml(config_path / "navigation.yaml")
    charging_raw = _read_yaml(config_path / "charging.yaml")
    tsp_raw = _read_yaml(config_path / "tsp.yaml")
    paths_raw = _read_yaml(config_path / "paths.yaml")
    schema_documents = {
        "physical": physical_raw,
        "inspection": inspection_raw,
        "navigation": navigation_raw,
        "charging": charging_raw,
        "tsp": tsp_raw,
        "paths": paths_raw,
    }
    mismatched = {
        name: str(document.get("schema_version"))
        for name, document in schema_documents.items()
        if str(document.get("schema_version")) != SCHEMA_VERSION
    }
    if mismatched:
        raise ConfigError(f"V2.1 requires schema_version {SCHEMA_VERSION}: {mismatched}")

    _require_keys(
        "physical",
        physical_raw,
        {
            "schema_version",
            "model_semantics_version",
            "launch_policy",
            "usv",
            "uav",
        },
    )
    _require_keys("inspection", inspection_raw, {"schema_version", "levels", "max_tasks_per_sortie", "mandatory_tasks"})

    uav_raw = dict(physical_raw["uav"])
    if "max_sortie_duration_min" in uav_raw:
        if "max_sortie_min" in uav_raw:
            raise ConfigError("specify only one UAV sortie duration parameter")
        uav_raw["max_sortie_min"] = uav_raw.pop("max_sortie_duration_min")
    physical = PhysicalConfig(
        schema_version=str(physical_raw["schema_version"]),
        model_semantics_version=str(physical_raw["model_semantics_version"]),
        launch_policy=str(physical_raw["launch_policy"]),
        usv=USVConfig(**physical_raw["usv"]),
        uav=UAVConfig(**uav_raw),
    )
    inspection = InspectionConfig(
        schema_version=str(inspection_raw["schema_version"]),
        levels={int(level): InspectionLevelConfig(**values) for level, values in inspection_raw["levels"].items()},
        max_tasks_per_sortie=int(inspection_raw["max_tasks_per_sortie"]),
        mandatory_tasks=bool(inspection_raw["mandatory_tasks"]),
    )
    navigation = NavigationConfig(**navigation_raw)
    charging = ChargingConfig(**charging_raw)
    tsp = TSPConfig(**tsp_raw)

    path_names = {
        "data_root",
        "raw_data_root",
        "processed_root",
        "cache_root",
        "results_root",
        "walney_layout",
        "coastline",
        "bathymetry",
        "wreck_points",
        "wreck_areas",
    }
    _require_keys("paths", paths_raw, {"schema_version", *path_names})
    paths = PathConfig(
        schema_version=str(paths_raw["schema_version"]),
        project_root=project_root,
        **{name: _resolve(project_root, str(paths_raw[name])) for name in path_names},
    )
    return ProjectConfig(physical, inspection, navigation, charging, tsp, paths)
