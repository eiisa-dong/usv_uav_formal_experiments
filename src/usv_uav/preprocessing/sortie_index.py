from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from usv_uav.core.models import Sortie
from usv_uav.preprocessing.numeric_sortie_index import (
    NumericSortieIndex,
    build_numeric_sortie_index,
)


@dataclass(frozen=True, slots=True)
class SortieIndex:
    sortie_by_id: Mapping[int, Sortie]
    sortie_ids_in_original_order: tuple[int, ...]
    sortie_rank_by_id: Mapping[int, int]
    sortie_ranks_by_task: Mapping[int, tuple[int, ...]]
    sortie_ranks_by_span: Mapping[int, tuple[int, ...]]
    sorties_by_task: Mapping[int, tuple[int, ...]]
    sorties_by_task_origin: Mapping[tuple[int, int], tuple[int, ...]]
    sorties_by_task_origin_span: Mapping[tuple[int, int, int], tuple[int, ...]]
    support_position: Mapping[int, int]
    sorties_by_origin: Mapping[int, tuple[int, ...]]
    sorties_by_recovery: Mapping[int, tuple[int, ...]]
    sorties_by_task_count: Mapping[int, tuple[int, ...]]
    sorties_by_origin_recovery: Mapping[tuple[int, int], tuple[int, ...]]
    sorties_by_task_set: Mapping[frozenset[int], tuple[int, ...]]
    sorties_by_origin_sequence: Mapping[tuple[int, tuple[int, ...]], tuple[int, ...]]
    augmentation_index: Mapping[tuple[int, int], tuple[int, ...]]
    numeric_repair_index: NumericSortieIndex

    def __post_init__(self) -> None:
        for name in (
            "sortie_by_id",
            "sortie_rank_by_id",
            "sortie_ranks_by_task",
            "sortie_ranks_by_span",
            "sorties_by_task",
            "sorties_by_task_origin",
            "sorties_by_task_origin_span",
            "support_position",
            "sorties_by_origin",
            "sorties_by_recovery",
            "sorties_by_task_count",
            "sorties_by_origin_recovery",
            "sorties_by_task_set",
            "sorties_by_origin_sequence",
            "augmentation_index",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def uncovered_tasks(self, task_ids: Sequence[int]) -> tuple[int, ...]:
        return tuple(task_id for task_id in task_ids if not self.sorties_by_task.get(task_id))

    def by_task(self, task_id: int) -> tuple[int, ...]:
        return self.sorties_by_task.get(task_id, ())

    def by_task_origin(self, task_id: int, origin_position: int) -> tuple[int, ...]:
        return self.sorties_by_task_origin.get((task_id, origin_position), ())

    def by_task_origin_span(
        self,
        task_id: int,
        origin_position: int,
        span: int,
    ) -> tuple[int, ...]:
        return self.sorties_by_task_origin_span.get(
            (task_id, origin_position, span), ()
        )

    def candidate_sortie_ids(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        """Return the by-task union in the original feasible-pool order."""
        ranks: set[int] = set()
        for task_id in task_ids:
            ranks.update(self.sortie_ranks_by_task.get(task_id, ()))
        return tuple(self.sortie_ids_in_original_order[rank] for rank in sorted(ranks))

    def view(self, *, max_span: int | None = None) -> SortieIndex | SortieIndexView:
        """Return a route-span view without rebuilding or copying the sortie pool."""
        if max_span is None:
            return self
        if isinstance(max_span, bool) or not isinstance(max_span, int) or max_span < 0:
            raise ValueError("max_span must be a non-negative integer or None")
        if not self.support_position:
            raise ValueError("route-span views require a support order")
        ranks = frozenset(
            rank
            for span, span_ranks in self.sortie_ranks_by_span.items()
            if span <= max_span
            for rank in span_ranks
        )
        ordered_ids = tuple(
            sortie_id
            for rank, sortie_id in enumerate(self.sortie_ids_in_original_order)
            if rank in ranks
        )
        return SortieIndexView(
            base=self,
            max_span=max_span,
            allowed_ranks=ranks,
            sortie_ids_in_original_order=ordered_ids,
            sortie_by_id=MappingProxyType(
                {sortie_id: self.sortie_by_id[sortie_id] for sortie_id in ordered_ids}
            ),
            sortie_rank_by_id=MappingProxyType(
                {sortie_id: self.sortie_rank_by_id[sortie_id] for sortie_id in ordered_ids}
            ),
        )


@dataclass(frozen=True, slots=True)
class SortieIndexView:
    """Read-only subset of a :class:`SortieIndex` bounded by recovery span."""

    base: SortieIndex
    max_span: int
    allowed_ranks: frozenset[int]
    sortie_ids_in_original_order: tuple[int, ...]
    sortie_by_id: Mapping[int, Sortie]
    sortie_rank_by_id: Mapping[int, int]

    @property
    def support_position(self) -> Mapping[int, int]:
        return self.base.support_position

    @property
    def numeric_repair_index(self) -> NumericSortieIndex:
        return self.base.numeric_repair_index

    def uncovered_tasks(self, task_ids: Sequence[int]) -> tuple[int, ...]:
        return tuple(task_id for task_id in task_ids if not self.by_task(task_id))

    def _filter_ids(self, sortie_ids: Iterable[int]) -> tuple[int, ...]:
        return tuple(
            sortie_id
            for sortie_id in sortie_ids
            if self.base.sortie_rank_by_id[sortie_id] in self.allowed_ranks
        )

    def by_task(self, task_id: int) -> tuple[int, ...]:
        return self._filter_ids(self.base.sorties_by_task.get(task_id, ()))

    def by_task_origin(self, task_id: int, origin_position: int) -> tuple[int, ...]:
        return self._filter_ids(
            self.base.sorties_by_task_origin.get((task_id, origin_position), ())
        )

    def by_task_origin_span(
        self,
        task_id: int,
        origin_position: int,
        span: int,
    ) -> tuple[int, ...]:
        if span > self.max_span:
            return ()
        return self.base.by_task_origin_span(task_id, origin_position, span)

    def candidate_sortie_ids(self, task_ids: Iterable[int]) -> tuple[int, ...]:
        ranks: set[int] = set()
        for task_id in task_ids:
            ranks.update(self.base.sortie_ranks_by_task.get(task_id, ()))
        ranks.intersection_update(self.allowed_ranks)
        return tuple(
            self.base.sortie_ids_in_original_order[rank] for rank in sorted(ranks)
        )

    def view(self, *, max_span: int | None = None) -> SortieIndexView:
        if max_span is None or max_span >= self.max_span:
            return self
        return self.base.view(max_span=max_span)


def _freeze_lists(values: dict) -> dict:
    return {key: tuple(items) for key, items in values.items()}


def build_sortie_index(
    sorties: Sequence[Sortie],
    support_order: Sequence[int] | None = None,
) -> SortieIndex:
    sortie_by_id = {sortie.id: sortie for sortie in sorties}
    if len(sortie_by_id) != len(sorties):
        raise ValueError("sortie ids must be unique")
    sortie_ids_in_original_order = tuple(sortie.id for sortie in sorties)
    sortie_rank_by_id = {
        sortie_id: rank for rank, sortie_id in enumerate(sortie_ids_in_original_order)
    }
    ordered_supports = tuple(support_order or ())
    if len(set(ordered_supports)) != len(ordered_supports):
        raise ValueError("support order must contain unique support ids")
    support_position = {
        support_id: position for position, support_id in enumerate(ordered_supports)
    }
    if support_position:
        indexed_supports = {
            support_id
            for sortie in sorties
            for support_id in (sortie.origin_support, sortie.recovery_support)
        }
        unknown = indexed_supports - set(support_position)
        if unknown:
            raise ValueError(f"sortie supports are absent from support order: {sorted(unknown)}")
    by_task: dict[int, list[int]] = {}
    ranks_by_task: dict[int, list[int]] = {}
    ranks_by_span: dict[int, list[int]] = {}
    by_task_origin: dict[tuple[int, int], list[int]] = {}
    by_task_origin_span: dict[tuple[int, int, int], list[int]] = {}
    by_origin: dict[int, list[int]] = {}
    by_recovery: dict[int, list[int]] = {}
    by_count: dict[int, list[int]] = {}
    by_pair: dict[tuple[int, int], list[int]] = {}
    by_task_set: dict[frozenset[int], list[int]] = {}
    by_origin_sequence: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    by_signature: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
    for rank, sortie in enumerate(sorties):
        if support_position:
            origin_position = support_position[sortie.origin_support]
            span = support_position[sortie.recovery_support] - origin_position
            ranks_by_span.setdefault(span, []).append(rank)
        for task_id in sortie.task_sequence:
            by_task.setdefault(task_id, []).append(sortie.id)
            ranks_by_task.setdefault(task_id, []).append(rank)
            if support_position:
                origin_position = support_position[sortie.origin_support]
                span = support_position[sortie.recovery_support] - origin_position
                by_task_origin.setdefault((task_id, origin_position), []).append(sortie.id)
                by_task_origin_span.setdefault(
                    (task_id, origin_position, span), []
                ).append(sortie.id)
        by_origin.setdefault(sortie.origin_support, []).append(sortie.id)
        by_recovery.setdefault(sortie.recovery_support, []).append(sortie.id)
        by_count.setdefault(len(sortie.task_sequence), []).append(sortie.id)
        by_pair.setdefault((sortie.origin_support, sortie.recovery_support), []).append(sortie.id)
        by_task_set.setdefault(frozenset(sortie.task_sequence), []).append(sortie.id)
        by_origin_sequence.setdefault(
            (sortie.origin_support, sortie.task_sequence), []
        ).append(sortie.id)
        by_signature.setdefault(
            (sortie.origin_support, sortie.recovery_support, sortie.task_sequence), []
        ).append(sortie.id)

    augmentation: dict[tuple[int, int], list[int]] = {}
    for candidate in sorties:
        if len(candidate.task_sequence) <= 1:
            continue
        for position, added_task in enumerate(candidate.task_sequence):
            base_sequence = candidate.task_sequence[:position] + candidate.task_sequence[position + 1 :]
            signature = (candidate.origin_support, candidate.recovery_support, base_sequence)
            for base_id in by_signature.get(signature, ()):  # exact insertion variants only
                augmentation.setdefault((base_id, added_task), []).append(candidate.id)

    frozen_ranks_by_task = _freeze_lists(ranks_by_task)
    return SortieIndex(
        sortie_by_id=sortie_by_id,
        sortie_ids_in_original_order=sortie_ids_in_original_order,
        sortie_rank_by_id=sortie_rank_by_id,
        sortie_ranks_by_task=frozen_ranks_by_task,
        sortie_ranks_by_span=_freeze_lists(ranks_by_span),
        sorties_by_task=_freeze_lists(by_task),
        sorties_by_task_origin=_freeze_lists(by_task_origin),
        sorties_by_task_origin_span=_freeze_lists(by_task_origin_span),
        support_position=support_position,
        sorties_by_origin=_freeze_lists(by_origin),
        sorties_by_recovery=_freeze_lists(by_recovery),
        sorties_by_task_count=_freeze_lists(by_count),
        sorties_by_origin_recovery=_freeze_lists(by_pair),
        sorties_by_task_set=_freeze_lists(by_task_set),
        sorties_by_origin_sequence=_freeze_lists(by_origin_sequence),
        augmentation_index=_freeze_lists(augmentation),
        numeric_repair_index=build_numeric_sortie_index(
            sorties,
            support_position,
            frozen_ranks_by_task,
        ),
    )
