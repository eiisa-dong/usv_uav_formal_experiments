from __future__ import annotations

from dataclasses import dataclass

from usv_uav.core.solution import Solution


@dataclass(slots=True)
class PartialSolution:
    """A destroyed solution that preserves every unaffected UAV sequence."""

    uav_sequences: dict[int, list[int]]
    missing_tasks: set[int]

    def copy(self) -> "PartialSolution":
        return PartialSolution(
            {uav_id: list(sequence) for uav_id, sequence in self.uav_sequences.items()},
            set(self.missing_tasks),
        )

    def to_solution(self) -> Solution:
        if self.missing_tasks:
            raise ValueError(f"partial solution still has missing tasks: {sorted(self.missing_tasks)}")
        return Solution({uav_id: list(sequence) for uav_id, sequence in self.uav_sequences.items()})
