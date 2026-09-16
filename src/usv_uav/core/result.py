from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PreprocessingTimings:
    obstacle_sec: float = 0.0
    candidates_sec: float = 0.0
    set_cover_sec: float = 0.0
    navigation_sec: float = 0.0
    tsp_sec: float = 0.0
    sortie_pool_sec: float = 0.0

    @property
    def total_sec(self) -> float:
        return sum((
            self.obstacle_sec,
            self.candidates_sec,
            self.set_cover_sec,
            self.navigation_sec,
            self.tsp_sec,
            self.sortie_pool_sec,
        ))


@dataclass(frozen=True, slots=True)
class PreprocessingSummary:
    instance_id: str
    candidate_supports: int
    safe_supports: int
    selected_supports: int
    covered_tasks: int
    total_tasks: int
    timings: PreprocessingTimings = field(default_factory=PreprocessingTimings)
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def coverage_complete(self) -> bool:
        return self.covered_tasks == self.total_tasks
