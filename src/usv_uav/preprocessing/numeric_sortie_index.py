from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from usv_uav.core.models import Sortie


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def _readonly(values: NDArray) -> NDArray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True, slots=True, eq=False)
class NumericSortieIndex:
    """Contiguous, object-free storage for compiled repair kernels.

    A qid is the sortie's position in the canonical feasible-pool order. Task
    membership and the inverse task-to-qid index use compressed task slots so
    task identifiers do not need to be contiguous.
    """

    sortie_ids: IntArray
    origin_pos: IntArray
    recovery_pos: IntArray
    nominal_time: FloatArray
    nominal_energy: FloatArray
    proxy_base: FloatArray
    task_count: IntArray
    sortie_task_offsets: IntArray
    sortie_task_slots: IntArray
    task_ids: IntArray
    task_qid_offsets: IntArray
    task_qids: IntArray
    task_slot_by_id: Mapping[int, int]
    route_positions_ready: bool


def build_numeric_sortie_index(
    sorties: Sequence[Sortie],
    support_position: Mapping[int, int],
    sortie_ranks_by_task: Mapping[int, Sequence[int]],
) -> NumericSortieIndex:
    task_ids_tuple = tuple(sorted(sortie_ranks_by_task))
    task_slot_by_id = {
        task_id: slot for slot, task_id in enumerate(task_ids_tuple)
    }

    sortie_task_offsets = [0]
    sortie_task_slots: list[int] = []
    for sortie in sorties:
        sortie_task_slots.extend(
            task_slot_by_id[task_id] for task_id in sortie.task_sequence
        )
        sortie_task_offsets.append(len(sortie_task_slots))

    task_qid_offsets = [0]
    task_qids: list[int] = []
    for task_id in task_ids_tuple:
        task_qids.extend(sortie_ranks_by_task[task_id])
        task_qid_offsets.append(len(task_qids))

    route_positions_ready = bool(support_position)
    if route_positions_ready:
        origin_pos = [support_position[sortie.origin_support] for sortie in sorties]
        recovery_pos = [
            support_position[sortie.recovery_support] for sortie in sorties
        ]
    else:
        origin_pos = [-1] * len(sorties)
        recovery_pos = [-1] * len(sorties)

    return NumericSortieIndex(
        sortie_ids=_readonly(
            np.asarray([sortie.id for sortie in sorties], dtype=np.int64)
        ),
        origin_pos=_readonly(np.asarray(origin_pos, dtype=np.int64)),
        recovery_pos=_readonly(np.asarray(recovery_pos, dtype=np.int64)),
        nominal_time=_readonly(
            np.asarray(
                [sortie.nominal_duration_min for sortie in sorties],
                dtype=np.float64,
            )
        ),
        nominal_energy=_readonly(
            np.asarray(
                [sortie.nominal_energy_wh for sortie in sorties],
                dtype=np.float64,
            )
        ),
        proxy_base=_readonly(
            np.asarray(
                [
                    sortie.nominal_duration_min / len(sortie.task_sequence)
                    for sortie in sorties
                ],
                dtype=np.float64,
            )
        ),
        task_count=_readonly(
            np.asarray(
                [len(sortie.task_sequence) for sortie in sorties],
                dtype=np.int64,
            )
        ),
        sortie_task_offsets=_readonly(
            np.asarray(sortie_task_offsets, dtype=np.int64)
        ),
        sortie_task_slots=_readonly(
            np.asarray(sortie_task_slots, dtype=np.int64)
        ),
        task_ids=_readonly(np.asarray(task_ids_tuple, dtype=np.int64)),
        task_qid_offsets=_readonly(
            np.asarray(task_qid_offsets, dtype=np.int64)
        ),
        task_qids=_readonly(np.asarray(task_qids, dtype=np.int64)),
        task_slot_by_id=MappingProxyType(task_slot_by_id),
        route_positions_ready=route_positions_ready,
    )
