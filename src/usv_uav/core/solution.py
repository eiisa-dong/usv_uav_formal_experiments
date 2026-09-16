from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Solution:
    """Structural decisions only; no time, SOC, hover, or charging state."""

    uav_sequences: dict[int, list[int]] = field(default_factory=dict)

    def copy(self) -> "Solution":
        return Solution({uav_id: list(sequence) for uav_id, sequence in self.uav_sequences.items()})

    def canonical(self, uav_count: int) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(self.uav_sequences.get(uav_id, ())) for uav_id in range(uav_count))
