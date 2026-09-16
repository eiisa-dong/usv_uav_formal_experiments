from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True, slots=True)
class MultiScaleDestroyConfig:
    preserve_legacy_up_to_j: int = 30
    micro_min_tasks: int = 3
    micro_max_tasks: int = 6
    medium_after_stagnation: int = 20
    medium_min_tasks: int = 7
    medium_max_tasks: int = 12
    macro_every_stagnation: int = 50

    def __post_init__(self) -> None:
        if self.preserve_legacy_up_to_j < 1:
            raise ValueError("preserve_legacy_up_to_j must be >= 1")
        if self.micro_min_tasks < 1:
            raise ValueError("micro_min_tasks must be >= 1")
        if self.micro_max_tasks < self.micro_min_tasks:
            raise ValueError("micro_max_tasks must be >= micro_min_tasks")
        if self.medium_min_tasks <= self.micro_max_tasks:
            raise ValueError("medium_min_tasks must be > micro_max_tasks")
        if self.medium_max_tasks < self.medium_min_tasks:
            raise ValueError("medium_max_tasks must be >= medium_min_tasks")
        if self.medium_after_stagnation < 1:
            raise ValueError("medium_after_stagnation must be >= 1")
        if self.macro_every_stagnation <= self.medium_after_stagnation:
            raise ValueError(
                "macro_every_stagnation must be greater than medium_after_stagnation"
            )


@dataclass(frozen=True, slots=True)
class SearchCadenceConfig:
    """Numerical references from MS0 FULL; cadence modes belong to variants."""

    reference_horizon_sec: float = 300.0
    temperature_ratio_by_task_count: dict[int, float] = field(
        default_factory=lambda: {
            90: 1.2587703000043933e-12,
            150: 0.040848359296122246,
            200: 0.2892103616617183,
        }
    )
    local_search_interval_sec_by_task_count: dict[int, float] = field(
        default_factory=lambda: {
            90: 1.0937098974683446,
            150: 9.419743289729418,
            200: 24.391569884256512,
        }
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.reference_horizon_sec, bool)
            or not isinstance(self.reference_horizon_sec, (int, float))
            or not isfinite(self.reference_horizon_sec)
            or self.reference_horizon_sec <= 0
        ):
            raise ValueError("reference_horizon_sec must be finite and positive")
        for name, references in (
            ("temperature_ratio_by_task_count", self.temperature_ratio_by_task_count),
            ("local_search_interval_sec_by_task_count", self.local_search_interval_sec_by_task_count),
        ):
            if not isinstance(references, dict):
                raise TypeError(f"{name} must be a mapping")
            if not references:
                raise ValueError(f"{name} must not be empty")
            for task_count, value in references.items():
                if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0:
                    raise ValueError(f"{name} keys must be positive integers")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{name}[{task_count}] must be finite and positive")
        if any(ratio > 1 for ratio in self.temperature_ratio_by_task_count.values()):
            raise ValueError("temperature ratios must be in (0, 1]")
        if set(self.temperature_ratio_by_task_count) != set(self.local_search_interval_sec_by_task_count):
            raise ValueError("temperature and local-search reference maps must have the same task-count keys")


@dataclass(frozen=True, slots=True)
class RouteGuidedInitializationConfig:
    enabled: bool = False
    local_max_span: int = 1
    extended_max_span: int = 2
    global_fallback: bool = True
    candidate_scoring: Literal[
        "scope_only", "synchronization_lexicographic"
    ] = "scope_only"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("route-guided initialization enabled must be boolean")
        if (
            isinstance(self.local_max_span, bool)
            or not isinstance(self.local_max_span, int)
            or self.local_max_span < 0
        ):
            raise ValueError("local_max_span must be a non-negative integer")
        if (
            isinstance(self.extended_max_span, bool)
            or not isinstance(self.extended_max_span, int)
            or self.extended_max_span <= self.local_max_span
        ):
            raise ValueError("extended_max_span must be greater than local_max_span")
        if self.global_fallback is not True:
            raise ValueError("global_fallback must remain enabled for S5A")
        if self.candidate_scoring not in {
            "scope_only",
            "synchronization_lexicographic",
        }:
            raise ValueError(
                "candidate_scoring must be scope_only or synchronization_lexicographic"
            )
        if not self.enabled and self.candidate_scoring != "scope_only":
            raise ValueError(
                "synchronization_lexicographic candidate scoring requires route-guided initialization"
            )


@dataclass(frozen=True, slots=True)
class RouteGuidedV2Config:
    enabled: bool = False
    small_global_threshold: int = 30
    local_origin_radius: int = 1
    forward_origin_radius: int = 2
    local_max_recovery_span: int = 1
    forward_max_recovery_span: int = 2
    micro_corridor_forward: int = 2
    medium_corridor_backward: int = 1
    medium_corridor_forward: int = 3
    r4_span0_quota: int = 1
    r4_span1_quota: int = 1
    r4_span2_quota: int = 2
    global_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("route-guided V2 enabled must be boolean")
        integer_fields = {
            "small_global_threshold": self.small_global_threshold,
            "local_origin_radius": self.local_origin_radius,
            "forward_origin_radius": self.forward_origin_radius,
            "local_max_recovery_span": self.local_max_recovery_span,
            "forward_max_recovery_span": self.forward_max_recovery_span,
            "micro_corridor_forward": self.micro_corridor_forward,
            "medium_corridor_backward": self.medium_corridor_backward,
            "medium_corridor_forward": self.medium_corridor_forward,
            "r4_span0_quota": self.r4_span0_quota,
            "r4_span1_quota": self.r4_span1_quota,
            "r4_span2_quota": self.r4_span2_quota,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.small_global_threshold < 1:
            raise ValueError("small_global_threshold must be positive")
        if self.forward_origin_radius < self.local_origin_radius:
            raise ValueError("forward_origin_radius must cover the local radius")
        if self.forward_max_recovery_span < self.local_max_recovery_span:
            raise ValueError("forward recovery span must cover the local span")
        if self.r4_span0_quota + self.r4_span1_quota + self.r4_span2_quota != 4:
            raise ValueError("R4P span quotas must preserve the Top-M=4 budget")
        if self.global_fallback is not True:
            raise ValueError("RG-MS-ALNS V2 global fallback must remain enabled")


@dataclass(frozen=True, slots=True)
class RouteGuidedV21CandidateQuotaConfig:
    same: int = 1
    adjacent: int = 1
    forward_relocation: int = 1
    forward_absorption: int = 1

    def __post_init__(self) -> None:
        for name in (
            "same",
            "adjacent",
            "forward_relocation",
            "forward_absorption",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"candidate_quota.{name} must be non-negative")
        if sum(
            (
                self.same,
                self.adjacent,
                self.forward_relocation,
                self.forward_absorption,
            )
        ) != 4:
            raise ValueError("V2.1 candidate quotas must preserve the Top-M=4 budget")


@dataclass(frozen=True, slots=True)
class RouteGuidedV21Config:
    enabled: bool = False
    corridor_bundle_destroy: bool = True
    forward_partner_required: bool = False
    candidate_quota: RouteGuidedV21CandidateQuotaConfig = field(
        default_factory=RouteGuidedV21CandidateQuotaConfig
    )
    global_fallback: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "corridor_bundle_destroy",
            "forward_partner_required",
            "global_fallback",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"route-guided V2.1 {name} must be boolean")
        if self.enabled and not self.corridor_bundle_destroy:
            raise ValueError("V2.1 requires corridor_bundle_destroy")
        if self.global_fallback is not True:
            raise ValueError("RG-MS-ALNS V2.1 global fallback must remain enabled")


@dataclass(frozen=True, slots=True)
class RouteGuidedV22Config:
    enabled: bool = False
    guided_candidate_limit: int = 64
    candidate_quota: RouteGuidedV21CandidateQuotaConfig = field(
        default_factory=RouteGuidedV21CandidateQuotaConfig
    )
    global_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("route-guided V2.2 enabled must be boolean")
        if (
            isinstance(self.guided_candidate_limit, bool)
            or not isinstance(self.guided_candidate_limit, int)
            or self.guided_candidate_limit <= 0
        ):
            raise ValueError("guided_candidate_limit must be a positive integer")
        if self.global_fallback is not True:
            raise ValueError("RG-MS-ALNS V2.2 global fallback must remain enabled")


@dataclass(frozen=True, slots=True)
class RouteGuidedV3Config:
    enabled: bool = False
    guided_candidate_limit: int = 64
    global_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("route-guided V3 enabled must be boolean")
        if (
            isinstance(self.guided_candidate_limit, bool)
            or not isinstance(self.guided_candidate_limit, int)
            or self.guided_candidate_limit <= 0
        ):
            raise ValueError("guided_candidate_limit must be a positive integer")
        if self.global_fallback is not True:
            raise ValueError("RG-MS-ALNS V3 global fallback must remain enabled")


@dataclass(frozen=True, slots=True)
class RouteGuidedV31Config:
    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("route-guided V3.1 enabled must be boolean")


@dataclass(frozen=True, slots=True)
class RouteGuidedV32EscapeConfig:
    medium_budget_ratio: float = 0.05
    global_budget_ratio: float = 0.15
    descent_window: Literal["medium_threshold"] = "medium_threshold"
    # ``rearm_window`` reads legacy V3.2 files only.  V3.2.1 decisions use the
    # two explicit windows below and never consult the compatibility field.
    rearm_window: Literal["medium_threshold"] | None = "medium_threshold"
    guided_rearm_window: Literal["medium_threshold"] = "medium_threshold"
    global_rearm_window: Literal["global_threshold"] = "global_threshold"
    hard_global_escape: bool = True

    def __post_init__(self) -> None:
        if self.medium_budget_ratio != 0.05 or self.global_budget_ratio != 0.15:
            raise ValueError("V3.2 escape ratios are frozen at 0.05 and 0.15")
        if self.descent_window != "medium_threshold":
            raise ValueError("V3.2 descent window must equal the medium threshold")
        if self.rearm_window not in {None, "medium_threshold"}:
            raise ValueError("legacy V3.2 re-arm window must equal T_M")
        if self.guided_rearm_window != "medium_threshold":
            raise ValueError("V3.2.1 Guided re-arm window must equal T_M")
        if self.global_rearm_window != "global_threshold":
            raise ValueError("V3.2.1 Global re-arm window must equal T_G")
        if self.hard_global_escape is not True:
            raise ValueError("V3.2 requires hard global escape")


@dataclass(frozen=True, slots=True)
class RouteGuidedV32GuidedConfig:
    use_v3_1_target_selection: bool = True
    use_v3_1_b2_reconstruction: bool = True
    use_v3_1_one_shot_retry: bool = True

    def __post_init__(self) -> None:
        if not all(
            (
                self.use_v3_1_target_selection,
                self.use_v3_1_b2_reconstruction,
                self.use_v3_1_one_shot_retry,
            )
        ):
            raise ValueError("V3.2 must preserve every frozen V3.1 guided component")


@dataclass(frozen=True, slots=True)
class RouteGuidedV32GlobalEscapeConfig:
    destroy_pool: tuple[Literal["D1_random", "D5_related"], ...] = (
        "D1_random",
        "D5_related",
    )
    scope: Literal["global"] = "global"
    force_current_relocation: bool = True

    def __post_init__(self) -> None:
        if self.destroy_pool != ("D1_random", "D5_related"):
            raise ValueError("V3.2 global destroy pool is frozen to D1_random/D5_related")
        if self.scope != "global":
            raise ValueError("V3.2 global escape scope must be global")
        if self.force_current_relocation is not True:
            raise ValueError("V3.2 global escape must force current relocation")


@dataclass(frozen=True, slots=True)
class RouteGuidedV32Config:
    enabled: bool = False
    controller: Literal["descent_momentum_probabilistic_escape"] = (
        "descent_momentum_probabilistic_escape"
    )
    controller_mode: Literal[
        "escape_rearm", "normal_only_profile"
    ] = "escape_rearm"
    escape: RouteGuidedV32EscapeConfig = field(
        default_factory=RouteGuidedV32EscapeConfig
    )
    guided: RouteGuidedV32GuidedConfig = field(
        default_factory=RouteGuidedV32GuidedConfig
    )
    global_escape: RouteGuidedV32GlobalEscapeConfig = field(
        default_factory=RouteGuidedV32GlobalEscapeConfig
    )
    max_decoder_candidates: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("route-guided V3.2 enabled must be boolean")
        if self.controller != "descent_momentum_probabilistic_escape":
            raise ValueError("V3.2 requires the descent-momentum controller")
        if self.controller_mode not in {
            "escape_rearm",
            "normal_only_profile",
        }:
            raise ValueError("unknown V3.2.1 controller mode")
        if self.max_decoder_candidates != 4:
            raise ValueError("V3.2 Full Decoder candidate limit is frozen at four")

    @property
    def normal_only_profile(self) -> bool:
        return self.controller_mode == "normal_only_profile"

    @property
    def allow_guided_escape(self) -> bool:
        return not self.normal_only_profile

    @property
    def allow_global_escape(self) -> bool:
        return not self.normal_only_profile

    @property
    def allow_d6(self) -> bool:
        return not self.normal_only_profile

    @property
    def allow_b2(self) -> bool:
        return not self.normal_only_profile


@dataclass(frozen=True, slots=True)
class NormalProfileConfig:
    """Diagnostic-only switches for the frozen NORMAL search path."""

    disable_r4p: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.disable_r4p, bool):
            raise TypeError("normal_profile.disable_r4p must be boolean")


@dataclass(frozen=True, slots=True)
class ALNSConfig:
    schema_version: str
    name: str
    initial_temperature_fraction: float
    cooling_rate: float
    reaction_factor: float
    segment_length: int
    removal_fraction_min: float
    removal_fraction_max: float
    reward_global_best: float
    reward_improved: float
    reward_accepted: float
    local_search_interval: int
    search_engine: Literal[
        "legacy",
        "route_guided_v2",
        "route_guided_v2_1",
        "route_guided_v2_2",
        "route_guided_v3",
        "route_guided_v3_1",
        "route_guided_v3_2",
        "route_guided_v3_2_1",
    ] = "legacy"
    recovery_top_m: int = 4
    recovery_same_quota: int = 2
    recovery_diff_quota: int = 2
    multiscale_destroy: MultiScaleDestroyConfig = field(
        default_factory=MultiScaleDestroyConfig
    )
    search_cadence: SearchCadenceConfig = field(
        default_factory=SearchCadenceConfig
    )
    route_guided_initialization: RouteGuidedInitializationConfig = field(
        default_factory=RouteGuidedInitializationConfig
    )
    route_guided_v2: RouteGuidedV2Config = field(
        default_factory=RouteGuidedV2Config
    )
    route_guided_v2_1: RouteGuidedV21Config = field(
        default_factory=RouteGuidedV21Config
    )
    route_guided_v2_2: RouteGuidedV22Config = field(
        default_factory=RouteGuidedV22Config
    )
    route_guided_v3: RouteGuidedV3Config = field(
        default_factory=RouteGuidedV3Config
    )
    route_guided_v3_1: RouteGuidedV31Config = field(
        default_factory=RouteGuidedV31Config
    )
    route_guided_v3_2: RouteGuidedV32Config = field(
        default_factory=RouteGuidedV32Config
    )
    normal_profile: NormalProfileConfig = field(
        default_factory=NormalProfileConfig
    )

    @property
    def search_engine_version(self) -> str:
        return {
            "legacy": "legacy",
            "route_guided_v2": "2.0",
            "route_guided_v2_1": "2.1",
            "route_guided_v2_2": "2.2",
            "route_guided_v3": "3.0",
            "route_guided_v3_1": "3.1",
            "route_guided_v3_2": "3.2",
            "route_guided_v3_2_1": "3.2.1",
        }[self.search_engine]

    @property
    def controller_semantics_version(self) -> str:
        return {
            "route_guided_v3_2": "3.2",
            "route_guided_v3_2_1": "3.2.1",
        }.get(self.search_engine, "none")

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "2.1", "2.2", "2.3", "2.4", "3.0", "3.1", "3.2", "3.2.1"
        }:
            raise ValueError(
                "ALNS schema_version must be 2.1, 2.2, 2.3, 2.4, 3.0, "
                "3.1, 3.2, or 3.2.1"
            )
        if self.schema_version == "2.1" and (
            self.search_engine != "legacy"
            or self.route_guided_v2.enabled
            or self.route_guided_v2_1.enabled
            or self.route_guided_v2_2.enabled
            or self.route_guided_v3.enabled
            or self.route_guided_v3_1.enabled
            or self.route_guided_v3_2.enabled
        ):
            raise ValueError("schema_version 2.1 cannot enable route_guided_v2")
        if self.schema_version == "2.2" and (
            self.route_guided_v2_1.enabled
            or self.route_guided_v2_2.enabled
            or self.route_guided_v3.enabled
            or self.route_guided_v3_1.enabled
            or self.route_guided_v3_2.enabled
        ):
            raise ValueError("schema_version 2.2 cannot enable route_guided_v2_1")
        if self.schema_version == "2.3" and (
            self.route_guided_v2_2.enabled
            or self.route_guided_v3.enabled
            or self.route_guided_v3_1.enabled
            or self.route_guided_v3_2.enabled
        ):
            raise ValueError("schema_version 2.3 cannot enable route_guided_v2_2")
        if self.schema_version == "2.4" and (
            self.route_guided_v3.enabled
            or self.route_guided_v3_1.enabled
            or self.route_guided_v3_2.enabled
        ):
            raise ValueError("schema_version 2.4 cannot enable route_guided_v3")
        if self.schema_version == "3.0" and (
            self.search_engine != "route_guided_v3"
            or not self.route_guided_v3.enabled
            or self.route_guided_v3_1.enabled
            or self.route_guided_v3_2.enabled
        ):
            raise ValueError(
                "schema_version 3.0 requires the route_guided_v3 search engine"
            )
        if self.schema_version == "3.1" and (
            self.search_engine != "route_guided_v3_1"
            or not self.route_guided_v3.enabled
            or not self.route_guided_v3_1.enabled
            or self.route_guided_v3_2.enabled
        ):
            raise ValueError(
                "schema_version 3.1 requires the route_guided_v3_1 search engine"
            )
        if self.schema_version == "3.2" and (
            self.search_engine != "route_guided_v3_2"
            or not self.route_guided_v3.enabled
            or not self.route_guided_v3_1.enabled
            or not self.route_guided_v3_2.enabled
        ):
            raise ValueError(
                "schema_version 3.2 requires the route_guided_v3_2 search engine"
            )
        if self.schema_version == "3.2.1" and (
            self.search_engine != "route_guided_v3_2_1"
            or not self.route_guided_v3.enabled
            or not self.route_guided_v3_1.enabled
            or not self.route_guided_v3_2.enabled
        ):
            raise ValueError(
                "schema_version 3.2.1 requires the route_guided_v3_2_1 "
                "search engine"
            )
        route_engines = {
            "route_guided_v2",
            "route_guided_v2_1",
            "route_guided_v2_2",
            "route_guided_v3",
            "route_guided_v3_1",
            "route_guided_v3_2",
            "route_guided_v3_2_1",
        }
        if self.search_engine in route_engines and not self.route_guided_v2.enabled:
            raise ValueError("route-guided search requires the frozen V2 configuration")
        if self.route_guided_v2.enabled and self.search_engine not in route_engines:
            raise ValueError("route_guided_v2 configuration requires a route-guided engine")
        if (
            self.search_engine == "route_guided_v2_1"
            and not self.route_guided_v2_1.enabled
        ):
            raise ValueError("route_guided_v2_1 search engine requires its configuration")
        if self.route_guided_v2_1.enabled and self.search_engine != "route_guided_v2_1":
            raise ValueError("route_guided_v2_1 configuration requires its search engine")
        if (
            self.search_engine == "route_guided_v2_2"
            and not self.route_guided_v2_2.enabled
        ):
            raise ValueError("route_guided_v2_2 search engine requires its configuration")
        if self.route_guided_v2_2.enabled and self.search_engine != "route_guided_v2_2":
            raise ValueError("route_guided_v2_2 configuration requires its search engine")
        if (
            self.search_engine in {
                "route_guided_v3", "route_guided_v3_1", "route_guided_v3_2",
                "route_guided_v3_2_1"
            }
            and not self.route_guided_v3.enabled
        ):
            raise ValueError("route_guided_v3 search engine requires its configuration")
        if self.route_guided_v3.enabled and self.search_engine not in {
            "route_guided_v3", "route_guided_v3_1", "route_guided_v3_2",
            "route_guided_v3_2_1"
        }:
            raise ValueError("route_guided_v3 configuration requires its search engine")
        if (
            self.search_engine in {
                "route_guided_v3_1", "route_guided_v3_2",
                "route_guided_v3_2_1"
            }
            and not self.route_guided_v3_1.enabled
        ):
            raise ValueError(
                "route_guided_v3_1 search engine requires its configuration"
            )
        if self.route_guided_v3_1.enabled and self.search_engine not in {
            "route_guided_v3_1", "route_guided_v3_2", "route_guided_v3_2_1"
        }:
            raise ValueError(
                "route_guided_v3_1 configuration requires its search engine"
            )
        if (
            self.search_engine in {"route_guided_v3_2", "route_guided_v3_2_1"}
            and not self.route_guided_v3_2.enabled
        ):
            raise ValueError(
                "route_guided_v3_2 search engine requires its configuration"
            )
        if self.route_guided_v3_2.enabled and self.search_engine not in {
            "route_guided_v3_2", "route_guided_v3_2_1"
        }:
            raise ValueError(
                "route_guided_v3_2 configuration requires its search engine"
            )
        if (
            self.normal_profile.disable_r4p
            and not self.route_guided_v3_2.normal_only_profile
        ):
            raise ValueError(
                "normal_profile.disable_r4p requires NORMAL-only profile mode"
            )
        if self.initial_temperature_fraction <= 0:
            raise ValueError("initial_temperature_fraction must be positive")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if not 0 < self.reaction_factor <= 1:
            raise ValueError("reaction_factor must be in (0, 1]")
        if not 0 < self.removal_fraction_min <= self.removal_fraction_max <= 1:
            raise ValueError("invalid removal fractions")
        if self.local_search_interval < 0:
            raise ValueError("local_search_interval must be non-negative")
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if self.recovery_top_m <= 0:
            raise ValueError("recovery_top_m must be positive")
        if min(self.recovery_same_quota, self.recovery_diff_quota) < 0:
            raise ValueError("recovery candidate quotas must be non-negative")
        if self.recovery_same_quota + self.recovery_diff_quota > self.recovery_top_m:
            raise ValueError("recovery candidate quotas cannot exceed recovery_top_m")


def _normalise_cadence_task_counts(references: object, name: str) -> dict[int, float]:
    if not isinstance(references, dict):
        raise TypeError(f"{name} must be a mapping")
    normalised: dict[int, float] = {}
    for key, value in references.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError(f"{name} keys must be integers or integer strings")
        try:
            task_count = int(key)
        except ValueError as exc:
            raise ValueError(f"invalid task-count key in {name}: {key!r}") from exc
        if task_count in normalised:
            raise ValueError(f"duplicate normalised task-count key in {name}: {task_count}")
        normalised[task_count] = value
    return normalised


def alns_config_from_mapping(values: dict) -> ALNSConfig:
    payload = dict(values)
    payload = dict(payload)
    multiscale_payload = payload.pop("multiscale_destroy", None)
    if multiscale_payload is None:
        multiscale_config = MultiScaleDestroyConfig()
    elif isinstance(multiscale_payload, dict):
        multiscale_config = MultiScaleDestroyConfig(**multiscale_payload)
    else:
        raise TypeError("multiscale_destroy must be a mapping")
    cadence_payload = payload.pop("search_cadence", None)
    if cadence_payload is None:
        cadence_config = SearchCadenceConfig()
    elif isinstance(cadence_payload, dict):
        cadence_payload = dict(cadence_payload)
        for name in ("temperature_ratio_by_task_count", "local_search_interval_sec_by_task_count"):
            if name in cadence_payload:
                cadence_payload[name] = _normalise_cadence_task_counts(cadence_payload[name], name)
        cadence_config = SearchCadenceConfig(**cadence_payload)
    else:
        raise TypeError("search_cadence must be a mapping")
    route_initialization_payload = payload.pop("route_guided_initialization", None)
    if route_initialization_payload is None:
        route_initialization_config = RouteGuidedInitializationConfig()
    elif isinstance(route_initialization_payload, dict):
        route_initialization_config = RouteGuidedInitializationConfig(
            **route_initialization_payload
        )
    else:
        raise TypeError("route_guided_initialization must be a mapping")
    route_v2_payload = payload.pop("route_guided_v2", None)
    route_v21_payload = payload.pop("route_guided_v2_1", None)
    route_v22_payload = payload.pop("route_guided_v2_2", None)
    route_v3_payload = payload.pop("route_guided_v3", None)
    route_v31_payload = payload.pop("route_guided_v3_1", None)
    route_v32_payload = payload.pop("route_guided_v3_2", None)
    normal_profile_payload = payload.pop("normal_profile", None)
    search_engine = payload.get("search_engine", "legacy")
    if route_v2_payload is None:
        route_v2_config = RouteGuidedV2Config()
    elif isinstance(route_v2_payload, dict):
        route_v2_payload = dict(route_v2_payload)
        route_v2_payload.setdefault(
            "enabled", search_engine in {
                "route_guided_v2",
                "route_guided_v2_1",
                "route_guided_v2_2",
                "route_guided_v3",
                "route_guided_v3_1",
                "route_guided_v3_2",
                "route_guided_v3_2_1",
            }
        )
        route_v2_config = RouteGuidedV2Config(**route_v2_payload)
    else:
        raise TypeError("route_guided_v2 must be a mapping")
    if route_v21_payload is None:
        route_v21_config = RouteGuidedV21Config()
    elif isinstance(route_v21_payload, dict):
        route_v21_payload = dict(route_v21_payload)
        quota_payload = route_v21_payload.pop("candidate_quota", None)
        if quota_payload is None:
            quota_config = RouteGuidedV21CandidateQuotaConfig()
        elif isinstance(quota_payload, dict):
            quota_config = RouteGuidedV21CandidateQuotaConfig(**quota_payload)
        else:
            raise TypeError("route_guided_v2_1.candidate_quota must be a mapping")
        route_v21_payload.setdefault("enabled", search_engine == "route_guided_v2_1")
        route_v21_config = RouteGuidedV21Config(
            candidate_quota=quota_config,
            **route_v21_payload,
        )
    else:
        raise TypeError("route_guided_v2_1 must be a mapping")
    if route_v22_payload is None:
        route_v22_config = RouteGuidedV22Config()
    elif isinstance(route_v22_payload, dict):
        route_v22_payload = dict(route_v22_payload)
        quota_payload = route_v22_payload.pop("candidate_quota", None)
        if quota_payload is None:
            quota_config = RouteGuidedV21CandidateQuotaConfig()
        elif isinstance(quota_payload, dict):
            quota_config = RouteGuidedV21CandidateQuotaConfig(**quota_payload)
        else:
            raise TypeError("route_guided_v2_2.candidate_quota must be a mapping")
        route_v22_payload.setdefault("enabled", search_engine == "route_guided_v2_2")
        route_v22_config = RouteGuidedV22Config(
            candidate_quota=quota_config,
            **route_v22_payload,
        )
    else:
        raise TypeError("route_guided_v2_2 must be a mapping")
    if route_v3_payload is None:
        route_v3_config = RouteGuidedV3Config()
    elif isinstance(route_v3_payload, dict):
        route_v3_payload = dict(route_v3_payload)
        route_v3_payload.setdefault(
            "enabled", search_engine in {
                "route_guided_v3", "route_guided_v3_1", "route_guided_v3_2",
                "route_guided_v3_2_1"
            }
        )
        route_v3_config = RouteGuidedV3Config(**route_v3_payload)
    else:
        raise TypeError("route_guided_v3 must be a mapping")
    if route_v31_payload is None:
        route_v31_config = RouteGuidedV31Config()
    elif isinstance(route_v31_payload, dict):
        route_v31_payload = dict(route_v31_payload)
        route_v31_payload.setdefault(
            "enabled", search_engine in {
                "route_guided_v3_1", "route_guided_v3_2",
                "route_guided_v3_2_1"
            }
        )
        route_v31_config = RouteGuidedV31Config(**route_v31_payload)
    else:
        raise TypeError("route_guided_v3_1 must be a mapping")
    if route_v32_payload is None:
        route_v32_config = RouteGuidedV32Config()
    elif isinstance(route_v32_payload, dict):
        route_v32_payload = dict(route_v32_payload)
        escape_payload = route_v32_payload.pop("escape", {})
        guided_payload = route_v32_payload.pop("guided", {})
        global_escape_payload = route_v32_payload.pop("global_escape", {})
        if not isinstance(escape_payload, dict):
            raise TypeError("route_guided_v3_2.escape must be a mapping")
        if not isinstance(guided_payload, dict):
            raise TypeError("route_guided_v3_2.guided must be a mapping")
        if not isinstance(global_escape_payload, dict):
            raise TypeError("route_guided_v3_2.global_escape must be a mapping")
        global_escape_payload = dict(global_escape_payload)
        if "destroy_pool" in global_escape_payload:
            destroy_pool = global_escape_payload["destroy_pool"]
            if not isinstance(destroy_pool, (list, tuple)):
                raise TypeError("global_escape.destroy_pool must be a sequence")
            global_escape_payload["destroy_pool"] = tuple(destroy_pool)
        route_v32_payload.setdefault(
            "enabled",
            search_engine in {"route_guided_v3_2", "route_guided_v3_2_1"},
        )
        route_v32_config = RouteGuidedV32Config(
            escape=RouteGuidedV32EscapeConfig(**escape_payload),
            guided=RouteGuidedV32GuidedConfig(**guided_payload),
            global_escape=RouteGuidedV32GlobalEscapeConfig(
                **global_escape_payload
            ),
            **route_v32_payload,
        )
    else:
        raise TypeError("route_guided_v3_2 must be a mapping")
    if normal_profile_payload is None:
        normal_profile_config = NormalProfileConfig()
    elif isinstance(normal_profile_payload, dict):
        normal_profile_config = NormalProfileConfig(**normal_profile_payload)
    else:
        raise TypeError("normal_profile must be a mapping")
    return ALNSConfig(
        multiscale_destroy=multiscale_config,
        search_cadence=cadence_config,
        route_guided_initialization=route_initialization_config,
        route_guided_v2=route_v2_config,
        route_guided_v2_1=route_v21_config,
        route_guided_v2_2=route_v22_config,
        route_guided_v3=route_v3_config,
        route_guided_v3_1=route_v31_config,
        route_guided_v3_2=route_v32_config,
        normal_profile=normal_profile_config,
        **payload,
    )


def load_alns_config(path: str | Path) -> ALNSConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError("ALNS configuration must be a mapping")
    return alns_config_from_mapping(payload)
