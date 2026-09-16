from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from usv_uav.config import ChargingConfig, UAVConfig
from usv_uav.core.models import Sortie


EPSILON = 1e-9


class ChargingPolicy(Protocol):
    name: str

    def target_soc_wh(
        self,
        current_soc_wh: float,
        next_sortie: Sortie | None,
        uav: UAVConfig,
        *,
        final_full_charge: bool,
    ) -> float: ...


class FCFSMinimumRequiredCharging:
    name = "fcfs_minimum_required"

    def target_soc_wh(
        self,
        current_soc_wh: float,
        next_sortie: Sortie | None,
        uav: UAVConfig,
        *,
        final_full_charge: bool,
    ) -> float:
        if next_sortie is None:
            return uav.battery_wh if final_full_charge else current_soc_wh
        required = next_sortie.nominal_energy_wh + uav.safety_soc_wh
        return min(uav.battery_wh, max(current_soc_wh, required))


class FullChargeFCFS:
    name = "full_charge_fcfs"

    def target_soc_wh(
        self,
        current_soc_wh: float,
        next_sortie: Sortie | None,
        uav: UAVConfig,
        *,
        final_full_charge: bool,
    ) -> float:
        if next_sortie is None and not final_full_charge:
            return current_soc_wh
        return uav.battery_wh


@dataclass(frozen=True, slots=True)
class ChargingOperation:
    uav_id: int
    after_sortie_id: int
    charger_id: int
    request_support: int
    ready_time_min: float
    start_time_min: float
    finish_time_min: float
    soc_before_wh: float
    soc_after_wh: float

    @property
    def energy_wh(self) -> float:
        return self.soc_after_wh - self.soc_before_wh

    @property
    def duration_min(self) -> float:
        return self.finish_time_min - self.start_time_min

    @property
    def waiting_min(self) -> float:
        return self.start_time_min - self.ready_time_min


class FCFSChargingScheduler:
    """Deterministic non-idling FCFS list scheduler on identical chargers."""

    def __init__(self, config: ChargingConfig) -> None:
        self.config = config
        self._available = [0.0] * config.chargers
        self.operations: list[ChargingOperation] = []

    def schedule(
        self,
        *,
        uav_id: int,
        after_sortie_id: int,
        request_support: int,
        ready_time_min: float,
        soc_before_wh: float,
        target_soc_wh: float,
    ) -> ChargingOperation | None:
        energy = max(0.0, target_soc_wh - soc_before_wh)
        if energy <= EPSILON:
            return None
        charger_id = min(
            range(len(self._available)),
            key=lambda item: (self._available[item], item),
        )
        start = max(ready_time_min, self._available[charger_id])
        duration = (
            self.config.setup_time_min
            + self.config.transfer_to_charger_min
            + energy / self.config.effective_rate_wh_min
        )
        finish = start + duration
        operation = ChargingOperation(
            uav_id=uav_id,
            after_sortie_id=after_sortie_id,
            charger_id=charger_id,
            request_support=request_support,
            ready_time_min=ready_time_min,
            start_time_min=start,
            finish_time_min=finish,
            soc_before_wh=soc_before_wh,
            soc_after_wh=target_soc_wh,
        )
        self._available[charger_id] = finish
        self.operations.append(operation)
        return operation
