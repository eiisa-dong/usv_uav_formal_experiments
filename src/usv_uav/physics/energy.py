from __future__ import annotations

from dataclasses import dataclass

from usv_uav.config import UAVConfig
from usv_uav.core.models import Sortie


@dataclass(frozen=True, slots=True)
class ActualSortieState:
    duration_min: float
    energy_wh: float
    soc_after_wh: float


def nominal_sortie_energy(sortie: Sortie) -> float:
    return float(sortie.nominal_energy_wh)


def hover_energy(hover_min: float, uav: UAVConfig) -> float:
    if hover_min < 0:
        raise ValueError("hover_min must be non-negative")
    return float(hover_min) * uav.hovering_energy_wh_min


def actual_sortie_state(
    sortie: Sortie,
    *,
    hover_min: float,
    soc_before_wh: float,
    uav: UAVConfig,
) -> ActualSortieState:
    energy = nominal_sortie_energy(sortie) + hover_energy(hover_min, uav)
    return ActualSortieState(
        duration_min=sortie.nominal_duration_min + hover_min,
        energy_wh=energy,
        soc_after_wh=soc_before_wh - energy,
    )


def has_soc_reserve(soc_after_wh: float, uav: UAVConfig, *, tolerance: float = 1e-9) -> bool:
    return soc_after_wh >= uav.safety_soc_wh - tolerance
