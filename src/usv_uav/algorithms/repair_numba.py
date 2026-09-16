from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

from usv_uav.core.partial_solution import PartialSolution
from usv_uav.preprocessing.numeric_sortie_index import NumericSortieIndex
from usv_uav.preprocessing.sortie_index import SortieIndex


RepairKernelBackend = Literal["auto", "python", "numba"]
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class RepairKernelResult:
    missing_slots: IntArray
    best_qids: IntArray
    best_uav_ids: IntArray
    best_positions: IntArray
    best_scores: FloatArray
    task_qids: IntArray
    task_uav_ids: IntArray
    task_positions: IntArray
    task_scores: FloatArray
    raw_candidates: int
    static_feasible: int
    options: int
    regret_memberships: int
    insertion_positions_raw: int
    insertion_windows_total: int
    insertion_windows_empty: int
    insertion_window_width_max: int
    input_candidate_sec: float
    input_route_sec: float
    kernel_sec: float


@dataclass(slots=True, eq=False)
class RepairNumbaWorkspace:
    """Branch-private scratch reused across one residual repair chain."""

    missing_slots: IntArray
    sequence_offsets: IntArray
    sequence_qids: IntArray
    missing_mask: NDArray[np.uint8]
    missing_row: IntArray
    candidate_marks: IntArray
    candidate_generation: int
    uav_loads: FloatArray
    sequence_monotone: NDArray[np.uint8]
    best: IntArray
    task_best: IntArray


def create_repair_numba_workspace(
    index: SortieIndex,
    uav_count: int,
) -> RepairNumbaWorkspace:
    data = index.numeric_repair_index
    task_total = len(data.task_ids)
    sortie_total = len(data.sortie_ids)
    return RepairNumbaWorkspace(
        missing_slots=np.empty(task_total, dtype=np.int64),
        sequence_offsets=np.empty(uav_count + 1, dtype=np.int64),
        sequence_qids=np.empty(task_total, dtype=np.int64),
        missing_mask=np.empty(task_total, dtype=np.uint8),
        missing_row=np.empty(task_total, dtype=np.int64),
        candidate_marks=np.zeros(sortie_total, dtype=np.int64),
        candidate_generation=0,
        uav_loads=np.empty(uav_count, dtype=np.float64),
        sequence_monotone=np.empty(uav_count, dtype=np.uint8),
        best=np.empty(3, dtype=np.int64),
        task_best=np.empty((task_total, 2, 3), dtype=np.int64),
    )


@njit(inline="always")
def _less(
    qid: int,
    uav_id: int,
    position: int,
    other_qid: int,
    other_uav_id: int,
    other_position: int,
    augmentation: bool,
    sortie_ids: IntArray,
    nominal_energy: FloatArray,
    proxy_base: FloatArray,
    task_count: IntArray,
    uav_loads: FloatArray,
) -> bool:
    if augmentation:
        left0 = -float(task_count[qid])
        right0 = -float(task_count[other_qid])
        if left0 != right0:
            return left0 < right0
        left1 = proxy_base[qid]
        right1 = proxy_base[other_qid]
        if left1 != right1:
            return left1 < right1
        left2 = 0.01 * uav_loads[uav_id]
        right2 = 0.01 * uav_loads[other_uav_id]
        if left2 != right2:
            return left2 < right2
        left3 = nominal_energy[qid]
        right3 = nominal_energy[other_qid]
        if left3 != right3:
            return left3 < right3
    else:
        left0 = proxy_base[qid]
        right0 = proxy_base[other_qid]
        if left0 != right0:
            return left0 < right0
        left1 = 0.01 * uav_loads[uav_id]
        right1 = 0.01 * uav_loads[other_uav_id]
        if left1 != right1:
            return left1 < right1
        left2 = nominal_energy[qid]
        right2 = nominal_energy[other_qid]
        if left2 != right2:
            return left2 < right2
        left3 = -float(task_count[qid])
        right3 = -float(task_count[other_qid])
        if left3 != right3:
            return left3 < right3
    if uav_id != other_uav_id:
        return uav_id < other_uav_id
    if position != other_position:
        return position < other_position
    return sortie_ids[qid] < sortie_ids[other_qid]


@njit(inline="always")
def _lower_bound_origin(
    sequence_qids: IntArray,
    start: int,
    length: int,
    origin_pos: IntArray,
    candidate_recovery: int,
) -> int:
    lo = 0
    hi = length
    while lo < hi:
        middle = (lo + hi) // 2
        if origin_pos[sequence_qids[start + middle]] < candidate_recovery:
            lo = middle + 1
        else:
            hi = middle
    return lo


@njit(inline="always")
def _upper_bound_recovery(
    sequence_qids: IntArray,
    start: int,
    length: int,
    recovery_pos: IntArray,
    candidate_origin: int,
) -> int:
    lo = 0
    hi = length
    while lo < hi:
        middle = (lo + hi) // 2
        if recovery_pos[sequence_qids[start + middle]] <= candidate_origin:
            lo = middle + 1
        else:
            hi = middle
    return lo


@njit(inline="always")
def _write_score(
    target: FloatArray,
    qid: int,
    uav_id: int,
    position: int,
    augmentation: bool,
    sortie_ids: IntArray,
    nominal_energy: FloatArray,
    proxy_base: FloatArray,
    task_count: IntArray,
    uav_loads: FloatArray,
) -> None:
    if augmentation:
        target[0] = -float(task_count[qid])
        target[1] = proxy_base[qid]
        target[2] = 0.01 * uav_loads[uav_id]
        target[3] = nominal_energy[qid]
    else:
        target[0] = proxy_base[qid]
        target[1] = 0.01 * uav_loads[uav_id]
        target[2] = nominal_energy[qid]
        target[3] = -float(task_count[qid])
    target[4] = float(uav_id)
    target[5] = float(position)
    target[6] = float(sortie_ids[qid])


@njit(inline="always")
def _consider_option(
    qid: int,
    uav_id: int,
    position: int,
    mode: int,
    sortie_ids: IntArray,
    nominal_energy: FloatArray,
    proxy_base: FloatArray,
    task_count: IntArray,
    sortie_task_offsets: IntArray,
    sortie_task_slots: IntArray,
    missing_row: IntArray,
    uav_loads: FloatArray,
    best: IntArray,
    task_best: IntArray,
) -> int:
    if mode != 1:
        if best[0] < 0 or _less(
            qid,
            uav_id,
            position,
            best[0],
            best[1],
            best[2],
            mode == 2,
            sortie_ids,
            nominal_energy,
            proxy_base,
            task_count,
            uav_loads,
        ):
            best[0] = qid
            best[1] = uav_id
            best[2] = position
        return 0

    memberships = 0
    for flat_index in range(
        sortie_task_offsets[qid], sortie_task_offsets[qid + 1]
    ):
        row = missing_row[sortie_task_slots[flat_index]]
        if row < 0:
            continue
        memberships += 1
        first_qid = task_best[row, 0, 0]
        if first_qid < 0 or _less(
            qid,
            uav_id,
            position,
            first_qid,
            task_best[row, 0, 1],
            task_best[row, 0, 2],
            False,
            sortie_ids,
            nominal_energy,
            proxy_base,
            task_count,
            uav_loads,
        ):
            task_best[row, 1, 0] = task_best[row, 0, 0]
            task_best[row, 1, 1] = task_best[row, 0, 1]
            task_best[row, 1, 2] = task_best[row, 0, 2]
            task_best[row, 0, 0] = qid
            task_best[row, 0, 1] = uav_id
            task_best[row, 0, 2] = position
            continue
        second_qid = task_best[row, 1, 0]
        if second_qid < 0 or _less(
            qid,
            uav_id,
            position,
            second_qid,
            task_best[row, 1, 1],
            task_best[row, 1, 2],
            False,
            sortie_ids,
            nominal_energy,
            proxy_base,
            task_count,
            uav_loads,
        ):
            task_best[row, 1, 0] = qid
            task_best[row, 1, 1] = uav_id
            task_best[row, 1, 2] = position
    return memberships


@njit(cache=True, fastmath=False)
def _rank_repair_options_kernel(
    sortie_ids: IntArray,
    origin_pos: IntArray,
    recovery_pos: IntArray,
    nominal_time: FloatArray,
    nominal_energy: FloatArray,
    proxy_base: FloatArray,
    task_count: IntArray,
    sortie_task_offsets: IntArray,
    sortie_task_slots: IntArray,
    task_qid_offsets: IntArray,
    task_qids: IntArray,
    missing_slots: IntArray,
    sequence_offsets: IntArray,
    sequence_qids: IntArray,
    mode: int,
    missing_mask: NDArray[np.uint8],
    missing_row: IntArray,
    candidate_marks: IntArray,
    candidate_generation: int,
    uav_loads: FloatArray,
    sequence_monotone: NDArray[np.uint8],
    best: IntArray,
    task_best: IntArray,
):
    sortie_total = len(sortie_ids)
    uav_count = len(sequence_offsets) - 1
    task_total = len(task_qid_offsets) - 1
    missing_mask[:] = 0
    missing_row[:] = -1
    for row in range(len(missing_slots)):
        missing_slot = missing_slots[row]
        missing_mask[missing_slot] = 1
        missing_row[missing_slot] = row
        begin = task_qid_offsets[missing_slot]
        end = task_qid_offsets[missing_slot + 1]
        for flat_index in range(begin, end):
            candidate_marks[task_qids[flat_index]] = candidate_generation

    uav_loads[:] = 0.0
    sequence_monotone[:] = 1
    for uav_id in range(uav_count):
        begin = sequence_offsets[uav_id]
        end = sequence_offsets[uav_id + 1]
        previous_recovery = -1
        for flat_index in range(begin, end):
            qid = sequence_qids[flat_index]
            origin = origin_pos[qid]
            recovery = recovery_pos[qid]
            uav_loads[uav_id] += nominal_time[qid]
            if origin > recovery or (
                flat_index > begin and previous_recovery > origin
            ):
                sequence_monotone[uav_id] = 0
            previous_recovery = recovery

    best[:] = -1
    task_best[: len(missing_slots), :, :] = -1
    raw_candidates = 0
    static_feasible = 0
    options = 0
    regret_memberships = 0
    insertion_positions_raw = 0
    insertion_windows_total = 0
    insertion_windows_empty = 0
    insertion_window_width_max = 0

    for qid in range(sortie_total):
        if candidate_marks[qid] != candidate_generation:
            continue
        raw_candidates += 1
        compatible = True
        for flat_index in range(
            sortie_task_offsets[qid], sortie_task_offsets[qid + 1]
        ):
            if missing_mask[sortie_task_slots[flat_index]] == 0:
                compatible = False
                break
        if not compatible:
            continue
        static_feasible += 1
        candidate_origin = origin_pos[qid]
        candidate_recovery = recovery_pos[qid]
        if candidate_origin > candidate_recovery:
            continue
        for uav_id in range(uav_count):
            sequence_start = sequence_offsets[uav_id]
            sequence_length = sequence_offsets[uav_id + 1] - sequence_start
            insertion_positions_raw += sequence_length + 1
            insertion_windows_total += 1
            if sequence_monotone[uav_id] != 0:
                lo = _lower_bound_origin(
                    sequence_qids,
                    sequence_start,
                    sequence_length,
                    origin_pos,
                    candidate_recovery,
                )
                hi = _upper_bound_recovery(
                    sequence_qids,
                    sequence_start,
                    sequence_length,
                    recovery_pos,
                    candidate_origin,
                )
                width = hi - lo + 1 if lo <= hi else 0
                if width == 0:
                    insertion_windows_empty += 1
                    continue
                if width > insertion_window_width_max:
                    insertion_window_width_max = width
                for position in range(lo, hi + 1):
                    options += 1
                    regret_memberships += _consider_option(
                        qid,
                        uav_id,
                        position,
                        mode,
                        sortie_ids,
                        nominal_energy,
                        proxy_base,
                        task_count,
                        sortie_task_offsets,
                        sortie_task_slots,
                        missing_row,
                        uav_loads,
                        best,
                        task_best,
                    )
            else:
                legal_positions = 0
                for position in range(sequence_length + 1):
                    previous_ok = position == 0 or (
                        recovery_pos[
                            sequence_qids[sequence_start + position - 1]
                        ]
                        <= candidate_origin
                    )
                    following_ok = position == sequence_length or (
                        candidate_recovery
                        <= origin_pos[
                            sequence_qids[sequence_start + position]
                        ]
                    )
                    if not previous_ok or not following_ok:
                        continue
                    legal_positions += 1
                    options += 1
                    regret_memberships += _consider_option(
                        qid,
                        uav_id,
                        position,
                        mode,
                        sortie_ids,
                        nominal_energy,
                        proxy_base,
                        task_count,
                        sortie_task_offsets,
                        sortie_task_slots,
                        missing_row,
                        uav_loads,
                        best,
                        task_best,
                    )
                if legal_positions == 0:
                    insertion_windows_empty += 1
                elif legal_positions > insertion_window_width_max:
                    insertion_window_width_max = legal_positions

    best_count = 1 if best[0] >= 0 else 0
    best_qids = np.empty(best_count, dtype=np.int64)
    best_uav_ids = np.empty(best_count, dtype=np.int64)
    best_positions = np.empty(best_count, dtype=np.int64)
    best_scores = np.empty((best_count, 7), dtype=np.float64)
    if best_count:
        best_qids[0] = best[0]
        best_uav_ids[0] = best[1]
        best_positions[0] = best[2]
        _write_score(
            best_scores[0],
            best[0],
            best[1],
            best[2],
            mode == 2,
            sortie_ids,
            nominal_energy,
            proxy_base,
            task_count,
            uav_loads,
        )

    task_qids_out = task_best[:, :, 0].copy()
    task_uav_ids = task_best[:, :, 1].copy()
    task_positions = task_best[:, :, 2].copy()
    task_scores = np.empty((len(missing_slots), 2, 7), dtype=np.float64)
    task_scores[:] = np.nan
    for row in range(len(missing_slots)):
        for rank in range(2):
            qid = task_qids_out[row, rank]
            if qid < 0:
                continue
            _write_score(
                task_scores[row, rank],
                qid,
                task_uav_ids[row, rank],
                task_positions[row, rank],
                False,
                sortie_ids,
                nominal_energy,
                proxy_base,
                task_count,
                uav_loads,
            )
    return (
        best_qids,
        best_uav_ids,
        best_positions,
        best_scores,
        task_qids_out,
        task_uav_ids,
        task_positions,
        task_scores,
        raw_candidates,
        static_feasible,
        options,
        regret_memberships,
        insertion_positions_raw,
        insertion_windows_total,
        insertion_windows_empty,
        insertion_window_width_max,
    )


def _partial_arrays(
    partial: PartialSolution,
    index: SortieIndex,
    data: NumericSortieIndex,
    uav_count: int,
    workspace: RepairNumbaWorkspace,
) -> tuple[IntArray, IntArray, IntArray, float, float]:
    candidate_started = perf_counter()
    ordered_missing_slots = sorted(
        data.task_slot_by_id[task_id]
        for task_id in partial.missing_tasks
        if task_id in data.task_slot_by_id
    )
    missing_count = len(ordered_missing_slots)
    workspace.missing_slots[:missing_count] = ordered_missing_slots
    missing_slots = workspace.missing_slots[:missing_count]
    candidate_sec = perf_counter() - candidate_started

    route_started = perf_counter()
    sequence_offsets = workspace.sequence_offsets
    sequence_offsets[0] = 0
    sequence_count = 0
    for uav_id in range(uav_count):
        for sortie_id in partial.uav_sequences.get(uav_id, ()):
            workspace.sequence_qids[sequence_count] = (
                index.sortie_rank_by_id[sortie_id]
            )
            sequence_count += 1
        sequence_offsets[uav_id + 1] = sequence_count
    sequence_qids = workspace.sequence_qids[:sequence_count]
    route_sec = perf_counter() - route_started
    return missing_slots, sequence_offsets, sequence_qids, candidate_sec, route_sec


def rank_repair_options(
    partial: PartialSolution,
    index: SortieIndex,
    uav_count: int,
    *,
    strategy: Literal["greedy", "regret2", "augmentation"],
    workspace: RepairNumbaWorkspace | None = None,
) -> RepairKernelResult:
    if type(index) is not SortieIndex:
        raise TypeError("shared Numba repair requires the canonical SortieIndex")
    data = index.numeric_repair_index
    if not data.route_positions_ready:
        raise ValueError("shared Numba repair requires an index with support order")
    mode = {"greedy": 0, "regret2": 1, "augmentation": 2}[strategy]
    active_workspace = workspace or create_repair_numba_workspace(
        index, uav_count
    )
    active_workspace.candidate_generation += 1
    if active_workspace.candidate_generation >= np.iinfo(np.int64).max:
        active_workspace.candidate_marks[:] = 0
        active_workspace.candidate_generation = 1
    (
        missing_slots,
        sequence_offsets,
        sequence_qids,
        candidate_sec,
        route_sec,
    ) = _partial_arrays(
        partial, index, data, uav_count, active_workspace
    )
    kernel_started = perf_counter()
    values = _rank_repair_options_kernel(
        data.sortie_ids,
        data.origin_pos,
        data.recovery_pos,
        data.nominal_time,
        data.nominal_energy,
        data.proxy_base,
        data.task_count,
        data.sortie_task_offsets,
        data.sortie_task_slots,
        data.task_qid_offsets,
        data.task_qids,
        missing_slots,
        sequence_offsets,
        sequence_qids,
        mode,
        active_workspace.missing_mask,
        active_workspace.missing_row,
        active_workspace.candidate_marks,
        active_workspace.candidate_generation,
        active_workspace.uav_loads,
        active_workspace.sequence_monotone,
        active_workspace.best,
        active_workspace.task_best,
    )
    kernel_sec = perf_counter() - kernel_started
    return RepairKernelResult(
        missing_slots=missing_slots,
        best_qids=values[0],
        best_uav_ids=values[1],
        best_positions=values[2],
        best_scores=values[3],
        task_qids=values[4],
        task_uav_ids=values[5],
        task_positions=values[6],
        task_scores=values[7],
        raw_candidates=int(values[8]),
        static_feasible=int(values[9]),
        options=int(values[10]),
        regret_memberships=int(values[11]),
        insertion_positions_raw=int(values[12]),
        insertion_windows_total=int(values[13]),
        insertion_windows_empty=int(values[14]),
        insertion_window_width_max=int(values[15]),
        input_candidate_sec=candidate_sec,
        input_route_sec=route_sec,
        kernel_sec=kernel_sec,
    )


def repair_numba_eligible(index: object) -> bool:
    return (
        type(index) is SortieIndex
        and index.numeric_repair_index.route_positions_ready
    )


def warmup_repair_numba_kernel() -> float:
    """Compile/load the production signature before the solver clock starts."""
    started = perf_counter()
    integer_arrays = [np.asarray([0], dtype=np.int64) for _ in range(9)]
    float_arrays = [np.asarray([1.0], dtype=np.float64) for _ in range(3)]
    for array in (*integer_arrays, *float_arrays):
        array.setflags(write=False)
    sortie_task_offsets = np.asarray([0, 1], dtype=np.int64)
    task_qid_offsets = np.asarray([0, 1], dtype=np.int64)
    for array in (sortie_task_offsets, task_qid_offsets):
        array.setflags(write=False)
    _rank_repair_options_kernel(
        integer_arrays[0],
        integer_arrays[1],
        integer_arrays[2],
        float_arrays[0],
        float_arrays[1],
        float_arrays[2],
        integer_arrays[3],
        sortie_task_offsets,
        integer_arrays[4],
        task_qid_offsets,
        integer_arrays[5],
        np.asarray([0], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        0,
        np.empty(1, dtype=np.uint8),
        np.empty(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        1,
        np.empty(1, dtype=np.float64),
        np.empty(1, dtype=np.uint8),
        np.empty(3, dtype=np.int64),
        np.empty((1, 2, 3), dtype=np.int64),
    )
    if not _rank_repair_options_kernel.nopython_signatures:
        raise RuntimeError("shared repair kernel did not compile in nopython mode")
    return perf_counter() - started


def compiled_signature_count() -> int:
    return len(_rank_repair_options_kernel.nopython_signatures)
