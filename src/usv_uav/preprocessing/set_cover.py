from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from usv_uav.preprocessing.coverage import CoverageMatrix


@dataclass(frozen=True, slots=True)
class SetCoverResult:
    selected_support_ids: tuple[int, ...]
    assignment_by_task: tuple[tuple[int, int], ...]
    minimum_support_count: int
    secondary_compactness_score: float
    posthoc_assignment_distance_km: float
    stage1_status: str
    stage2_status: str

    @property
    def assignment(self) -> dict[int, int]:
        return dict(self.assignment_by_task)

    @property
    def total_assignment_distance_km(self) -> float:
        """Deprecated compatibility alias for the post-hoc reporting metric."""
        return self.posthoc_assignment_distance_km


def _require_success(result, stage: str) -> None:
    if not result.success or result.x is None:
        raise RuntimeError(f"{stage} set-cover MILP failed: status={result.status}, message={result.message}")


def solve_two_stage_set_cover(coverage: CoverageMatrix) -> SetCoverResult:
    """Minimise support count, then a support-only compactness tie-break.

    Task assignments are deterministic post-processing and are deliberately not
    stage-2 decision variables. This keeps the formal J=200 model at ``|H|``
    binaries instead of ``|H| + |H||J|`` binaries.
    """
    uncovered = coverage.uncovered_task_ids
    if uncovered:
        raise ValueError(f"tasks have no nominally feasible support: {uncovered}")
    feasible = coverage.feasible
    support_count, task_count = feasible.shape

    # Stage 1: min sum(y_h), A^T y >= 1.
    stage1_constraint = LinearConstraint(
        feasible.T.astype(np.float64),
        lb=np.ones(task_count),
        ub=np.full(task_count, np.inf),
    )
    stage1 = milp(
        c=np.ones(support_count),
        integrality=np.ones(support_count),
        bounds=Bounds(np.zeros(support_count), np.ones(support_count)),
        constraints=stage1_constraint,
        options={"presolve": True},
    )
    _require_success(stage1, "stage 1")
    minimum_count = int(round(float(np.sum(stage1.x > 0.5))))

    # Stage 2 has only y_h. Each support receives the average round-trip
    # distance to tasks it can cover; the tiny support-id term is a stable
    # deterministic tie-break and has no modelling interpretation.
    feasible_counts = feasible.sum(axis=1)
    compactness = np.divide(
        np.where(feasible, coverage.roundtrip_distance_km, 0.0).sum(axis=1),
        feasible_counts,
        out=np.zeros(support_count, dtype=np.float64),
        where=feasible_counts > 0,
    )
    objective = compactness + 1e-10 * np.asarray(coverage.support_ids, dtype=np.float64)
    matrix = lil_matrix((1 + task_count, support_count), dtype=np.float64)
    matrix[0, :] = 1.0
    matrix[1:, :] = feasible.T.astype(np.float64)
    row_lower = np.concatenate(([minimum_count], np.ones(task_count)))
    row_upper = np.concatenate(([minimum_count], np.full(task_count, np.inf)))

    stage2 = milp(
        c=objective,
        integrality=np.ones(support_count),
        bounds=Bounds(np.zeros(support_count), (feasible_counts > 0).astype(np.float64)),
        constraints=LinearConstraint(matrix.tocsr(), row_lower, row_upper),
        options={"presolve": True},
    )
    _require_success(stage2, "stage 2")
    selected_indices = np.flatnonzero(stage2.x > 0.5)
    if len(selected_indices) != minimum_count:
        raise RuntimeError(
            f"stage 2 changed minimum support count: expected {minimum_count}, got {len(selected_indices)}"
        )
    selected_support_ids = tuple(coverage.support_ids[index] for index in selected_indices)
    assignments: list[tuple[int, int]] = []
    total_distance = 0.0
    for task_index, task_id in enumerate(coverage.task_ids):
        candidates = [index for index in selected_indices if feasible[index, task_index]]
        if not candidates:
            raise RuntimeError(f"selected cover has no feasible post-hoc assignment for task {task_id}")
        support_index = min(
            candidates,
            key=lambda index: (
                float(coverage.roundtrip_distance_km[index, task_index]),
                int(coverage.support_ids[index]),
            ),
        )
        assignments.append((task_id, coverage.support_ids[support_index]))
        total_distance += float(coverage.roundtrip_distance_km[support_index, task_index])

    return SetCoverResult(
        selected_support_ids=selected_support_ids,
        assignment_by_task=tuple(assignments),
        minimum_support_count=minimum_count,
        secondary_compactness_score=float(np.sum(compactness[selected_indices])),
        posthoc_assignment_distance_km=total_distance,
        stage1_status=str(stage1.message),
        stage2_status=str(stage2.message),
    )
