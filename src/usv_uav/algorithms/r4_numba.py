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


R4KernelBackend = Literal["auto", "python", "numba"]
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class R4KernelResult:
    qids: IntArray
    uav_ids: IntArray
    positions: IntArray
    scores: FloatArray
    raw_candidates: int
    static_feasible: int
    options: int
    same_options: int
    different_options: int
    insertion_positions_raw: int
    insertion_windows_total: int
    insertion_windows_empty: int
    insertion_window_width_max: int
    input_candidate_sec: float
    input_route_sec: float
    kernel_sec: float


@njit(inline="always")
def _candidate_less(
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
def _insert_top(
    qid: int,
    uav_id: int,
    position: int,
    count: int,
    capacity: int,
    top_qids: IntArray,
    top_uav_ids: IntArray,
    top_positions: IntArray,
    augmentation: bool,
    sortie_ids: IntArray,
    nominal_energy: FloatArray,
    proxy_base: FloatArray,
    task_count: IntArray,
    uav_loads: FloatArray,
) -> int:
    if capacity <= 0:
        return count
    insertion = count
    for index in range(count):
        if _candidate_less(
            qid,
            uav_id,
            position,
            top_qids[index],
            top_uav_ids[index],
            top_positions[index],
            augmentation,
            sortie_ids,
            nominal_energy,
            proxy_base,
            task_count,
            uav_loads,
        ):
            insertion = index
            break
    if count >= capacity and insertion == count:
        return count
    new_count = count + 1 if count < capacity else count
    last = new_count - 1
    for index in range(last, insertion, -1):
        top_qids[index] = top_qids[index - 1]
        top_uav_ids[index] = top_uav_ids[index - 1]
        top_positions[index] = top_positions[index - 1]
    top_qids[insertion] = qid
    top_uav_ids[insertion] = uav_id
    top_positions[insertion] = position
    return new_count


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
def _same_identity(
    qid: int,
    uav_id: int,
    position: int,
    selected_qids: IntArray,
    selected_uav_ids: IntArray,
    selected_positions: IntArray,
    selected_count: int,
) -> bool:
    for index in range(selected_count):
        if (
            selected_qids[index] == qid
            and selected_uav_ids[index] == uav_id
            and selected_positions[index] == position
        ):
            return True
    return False


@njit(cache=True, fastmath=False)
def _rank_options_kernel(
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
    augmentation: bool,
    top_m: int,
    same_quota: int,
    different_quota: int,
):
    sortie_total = len(sortie_ids)
    uav_count = len(sequence_offsets) - 1
    task_total = len(task_qid_offsets) - 1
    missing_mask = np.zeros(task_total, dtype=np.uint8)
    candidate_mask = np.zeros(sortie_total, dtype=np.uint8)
    for missing_slot in missing_slots:
        missing_mask[missing_slot] = 1
        begin = task_qid_offsets[missing_slot]
        end = task_qid_offsets[missing_slot + 1]
        for flat_index in range(begin, end):
            candidate_mask[task_qids[flat_index]] = 1

    uav_loads = np.zeros(uav_count, dtype=np.float64)
    sequence_monotone = np.ones(uav_count, dtype=np.uint8)
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

    global_qids = np.full(top_m, -1, dtype=np.int64)
    global_uav_ids = np.full(top_m, -1, dtype=np.int64)
    global_positions = np.full(top_m, -1, dtype=np.int64)
    same_qids = np.full(same_quota, -1, dtype=np.int64)
    same_uav_ids = np.full(same_quota, -1, dtype=np.int64)
    same_positions = np.full(same_quota, -1, dtype=np.int64)
    different_qids = np.full(different_quota, -1, dtype=np.int64)
    different_uav_ids = np.full(different_quota, -1, dtype=np.int64)
    different_positions = np.full(different_quota, -1, dtype=np.int64)
    global_count = 0
    same_top_count = 0
    different_top_count = 0
    raw_candidates = 0
    static_feasible = 0
    options = 0
    same_options = 0
    different_options = 0
    insertion_positions_raw = 0
    insertion_windows_total = 0
    insertion_windows_empty = 0
    insertion_window_width_max = 0

    for qid in range(sortie_total):
        if candidate_mask[qid] == 0:
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
        is_same = candidate_origin == candidate_recovery
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
                    if is_same:
                        same_options += 1
                    else:
                        different_options += 1
                    global_count = _insert_top(
                        qid, uav_id, position, global_count, top_m,
                        global_qids, global_uav_ids, global_positions,
                        augmentation, sortie_ids, nominal_energy, proxy_base,
                        task_count, uav_loads,
                    )
                    if not augmentation and is_same:
                        same_top_count = _insert_top(
                            qid, uav_id, position, same_top_count, same_quota,
                            same_qids, same_uav_ids, same_positions,
                            False, sortie_ids, nominal_energy, proxy_base,
                            task_count, uav_loads,
                        )
                    elif not augmentation:
                        different_top_count = _insert_top(
                            qid, uav_id, position, different_top_count,
                            different_quota, different_qids,
                            different_uav_ids, different_positions, False,
                            sortie_ids, nominal_energy, proxy_base, task_count,
                            uav_loads,
                        )
            else:
                legal_positions = 0
                for position in range(sequence_length + 1):
                    previous_ok = position == 0 or (
                        recovery_pos[sequence_qids[sequence_start + position - 1]]
                        <= candidate_origin
                    )
                    following_ok = position == sequence_length or (
                        candidate_recovery
                        <= origin_pos[sequence_qids[sequence_start + position]]
                    )
                    if not previous_ok or not following_ok:
                        continue
                    legal_positions += 1
                    options += 1
                    if is_same:
                        same_options += 1
                    else:
                        different_options += 1
                    global_count = _insert_top(
                        qid, uav_id, position, global_count, top_m,
                        global_qids, global_uav_ids, global_positions,
                        augmentation, sortie_ids, nominal_energy, proxy_base,
                        task_count, uav_loads,
                    )
                    if not augmentation and is_same:
                        same_top_count = _insert_top(
                            qid, uav_id, position, same_top_count, same_quota,
                            same_qids, same_uav_ids, same_positions,
                            False, sortie_ids, nominal_energy, proxy_base,
                            task_count, uav_loads,
                        )
                    elif not augmentation:
                        different_top_count = _insert_top(
                            qid, uav_id, position, different_top_count,
                            different_quota, different_qids,
                            different_uav_ids, different_positions, False,
                            sortie_ids, nominal_energy, proxy_base, task_count,
                            uav_loads,
                        )
                if legal_positions == 0:
                    insertion_windows_empty += 1
                elif legal_positions > insertion_window_width_max:
                    insertion_window_width_max = legal_positions

    selected_qids = np.full(top_m, -1, dtype=np.int64)
    selected_uav_ids = np.full(top_m, -1, dtype=np.int64)
    selected_positions = np.full(top_m, -1, dtype=np.int64)
    selected_count = 0
    if augmentation:
        for index in range(global_count):
            selected_qids[selected_count] = global_qids[index]
            selected_uav_ids[selected_count] = global_uav_ids[index]
            selected_positions[selected_count] = global_positions[index]
            selected_count += 1
    else:
        for index in range(same_top_count):
            if selected_count >= top_m:
                break
            selected_qids[selected_count] = same_qids[index]
            selected_uav_ids[selected_count] = same_uav_ids[index]
            selected_positions[selected_count] = same_positions[index]
            selected_count += 1
        for index in range(different_top_count):
            if selected_count >= top_m:
                break
            selected_qids[selected_count] = different_qids[index]
            selected_uav_ids[selected_count] = different_uav_ids[index]
            selected_positions[selected_count] = different_positions[index]
            selected_count += 1
        for index in range(global_count):
            if selected_count >= top_m:
                break
            qid = global_qids[index]
            uav_id = global_uav_ids[index]
            position = global_positions[index]
            if _same_identity(
                qid, uav_id, position, selected_qids, selected_uav_ids,
                selected_positions, selected_count,
            ):
                continue
            selected_qids[selected_count] = qid
            selected_uav_ids[selected_count] = uav_id
            selected_positions[selected_count] = position
            selected_count += 1

    scores = np.empty((selected_count, 7), dtype=np.float64)
    for index in range(selected_count):
        qid = selected_qids[index]
        uav_id = selected_uav_ids[index]
        position = selected_positions[index]
        if augmentation:
            scores[index, 0] = -float(task_count[qid])
            scores[index, 1] = proxy_base[qid]
            scores[index, 2] = 0.01 * uav_loads[uav_id]
            scores[index, 3] = nominal_energy[qid]
        else:
            scores[index, 0] = proxy_base[qid]
            scores[index, 1] = 0.01 * uav_loads[uav_id]
            scores[index, 2] = nominal_energy[qid]
            scores[index, 3] = -float(task_count[qid])
        scores[index, 4] = float(uav_id)
        scores[index, 5] = float(position)
        scores[index, 6] = float(sortie_ids[qid])
    return (
        selected_qids[:selected_count],
        selected_uav_ids[:selected_count],
        selected_positions[:selected_count],
        scores,
        raw_candidates,
        static_feasible,
        options,
        same_options,
        different_options,
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
) -> tuple[IntArray, IntArray, IntArray, float, float]:
    candidate_started = perf_counter()
    missing_slots = np.asarray(
        sorted(
            data.task_slot_by_id[task_id]
            for task_id in partial.missing_tasks
            if task_id in data.task_slot_by_id
        ),
        dtype=np.int64,
    )
    candidate_sec = perf_counter() - candidate_started

    route_started = perf_counter()
    sequence_offsets = np.empty(uav_count + 1, dtype=np.int64)
    sequence_offsets[0] = 0
    sequence_qids_list: list[int] = []
    for uav_id in range(uav_count):
        sequence_qids_list.extend(
            index.sortie_rank_by_id[sortie_id]
            for sortie_id in partial.uav_sequences.get(uav_id, ())
        )
        sequence_offsets[uav_id + 1] = len(sequence_qids_list)
    sequence_qids = np.asarray(sequence_qids_list, dtype=np.int64)
    route_sec = perf_counter() - route_started
    return missing_slots, sequence_offsets, sequence_qids, candidate_sec, route_sec


def rank_r4_options(
    partial: PartialSolution,
    index: SortieIndex,
    uav_count: int,
    *,
    augmentation: bool,
    top_m: int,
    same_quota: int = 0,
    different_quota: int = 0,
) -> R4KernelResult:
    if type(index) is not SortieIndex:
        raise TypeError("R4 Numba V1 requires the canonical SortieIndex")
    data = index.numeric_repair_index
    if not data.route_positions_ready:
        raise ValueError("R4 Numba V1 requires an index built with support order")
    if top_m < 1 or same_quota < 0 or different_quota < 0:
        raise ValueError("R4 kernel capacities must be positive/non-negative")
    (
        missing_slots,
        sequence_offsets,
        sequence_qids,
        candidate_sec,
        route_sec,
    ) = _partial_arrays(partial, index, data, uav_count)
    kernel_started = perf_counter()
    values = _rank_options_kernel(
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
        augmentation,
        top_m,
        same_quota,
        different_quota,
    )
    kernel_sec = perf_counter() - kernel_started
    return R4KernelResult(
        qids=values[0],
        uav_ids=values[1],
        positions=values[2],
        scores=values[3],
        raw_candidates=int(values[4]),
        static_feasible=int(values[5]),
        options=int(values[6]),
        same_options=int(values[7]),
        different_options=int(values[8]),
        insertion_positions_raw=int(values[9]),
        insertion_windows_total=int(values[10]),
        insertion_windows_empty=int(values[11]),
        insertion_window_width_max=int(values[12]),
        input_candidate_sec=candidate_sec,
        input_route_sec=route_sec,
        kernel_sec=kernel_sec,
    )


def r4_numba_eligible(index: object) -> bool:
    return (
        type(index) is SortieIndex
        and index.numeric_repair_index.route_positions_ready
    )


def warmup_r4_numba_kernel() -> float:
    """Compile/load the one production signature before the ALNS clock starts."""

    started = perf_counter()
    index = SortieIndex  # Keep the import live for static/package audits.
    del index
    integer_arrays = [np.asarray([0], dtype=np.int64) for _ in range(11)]
    float_arrays = [np.asarray([1.0], dtype=np.float64) for _ in range(3)]
    # Static index arrays are read-only in production and therefore form a
    # distinct Numba signature from per-call writable scratch arrays.
    for array in (*integer_arrays[:11], *float_arrays):
        array.setflags(write=False)
    sortie_ids = integer_arrays[0]
    origin_pos = integer_arrays[1]
    recovery_pos = integer_arrays[2]
    nominal_time, nominal_energy, proxy_base = float_arrays
    task_count = integer_arrays[3]
    sortie_task_offsets = np.asarray([0, 1], dtype=np.int64)
    sortie_task_slots = integer_arrays[4]
    task_qid_offsets = np.asarray([0, 1], dtype=np.int64)
    task_qids = integer_arrays[5]
    for array in (sortie_task_offsets, sortie_task_slots, task_qid_offsets, task_qids):
        array.setflags(write=False)
    missing_slots = np.asarray([0], dtype=np.int64)
    sequence_offsets = np.asarray([0, 0], dtype=np.int64)
    sequence_qids = np.asarray([], dtype=np.int64)
    _rank_options_kernel(
        sortie_ids,
        origin_pos,
        recovery_pos,
        nominal_time,
        nominal_energy,
        proxy_base,
        task_count,
        sortie_task_offsets,
        sortie_task_slots,
        task_qid_offsets,
        task_qids,
        missing_slots,
        sequence_offsets,
        sequence_qids,
        False,
        4,
        2,
        2,
    )
    if not _rank_options_kernel.nopython_signatures:
        raise RuntimeError("R4 kernel did not compile in nopython mode")
    return perf_counter() - started


def compiled_signature_count() -> int:
    return len(_rank_options_kernel.nopython_signatures)
