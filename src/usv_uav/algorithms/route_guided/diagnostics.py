from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite

from usv_uav.algorithms.route_guided.models import (
    BlockingAnalysis,
    CandidateClass,
    DwellGuidedAnalysis,
    EscapeDecision,
    ForwardFeasibility,
    NeighborhoodScope,
    SearchAction,
)
from usv_uav.algorithms.route_guided.scoring import RouteAffinityGain


def _span_bucket(span: int) -> str:
    return f"span{span}" if span in (0, 1, 2) else "span_gt2"


_V21_COUNTER_FIELDS = (
    "corridor_anchor_count",
    "corridor_partner_candidates",
    "corridor_partner_sorties_removed",
    "c0_generated",
    "c0_feasible",
    "c0_selected",
    "c0_accepted",
    "c0_improved",
    "c1_generated",
    "c1_feasible",
    "c1_selected",
    "c1_accepted",
    "c1_improved",
    "c2_generated",
    "c2_feasible",
    "c2_selected",
    "c2_accepted",
    "c2_improved",
    "c3_generated",
    "c3_feasible",
    "c3_selected",
    "c3_accepted",
    "c3_improved",
    "forward_absorbed_tasks",
    "forward_relocated_tasks",
    "span2_feasible",
    "no_span2_candidate",
    "span2_infeasible",
    "span2_feasible_but_worse",
    "span2_iteration_winner",
)


_V22_COUNTER_FIELDS = (
    "physical_forward_candidates",
    "static_impossible_forward_candidates",
    "dynamic_blocked_forward_candidates",
    "currently_feasible_forward_candidates",
    "guided_targets_selected",
    "guided_targets_reconstructed",
    "guided_targets_decoder_feasible",
    "guided_targets_iteration_best",
    "guided_targets_accepted",
    "guided_targets_improved",
    "blocking_sorties_released",
    "wasted_static_impossible_decoder_calls",
    "guided_hover_time_infeasible",
    "guided_hover_soc_infeasible",
    "guided_sequence_infeasible",
    "guided_other_infeasible",
    "guided_target_hover_time_infeasible",
    "guided_target_hover_soc_infeasible",
)


_V3_COUNTER_FIELDS = tuple(
    f"b{candidate}_{stage}"
    for candidate in range(4)
    for stage in ("generated", "feasible", "selected", "accepted", "improved")
)


_B2_TRANSACTION_COUNTER_FIELDS = (
    "b2_targets_selected",
    "b2_target_tasks_released",
    "b2_minimum_charge_unsafe_targets_skipped",
    "b2_destroy_target_inconsistent",
    "b2_structurally_uninsertable",
    "b2_reconstructed",
    "b2_decoder_feasible",
    "b2_decoder_infeasible",
    "b2_decoder_not_reached",
    "b2_fail_hover_time",
    "b2_fail_hover_soc",
    "b2_fail_ready",
    "b2_fail_route_sync",
    "b2_fail_charge",
    "b2_fail_other",
)


_V31_COUNTER_FIELDS = (
    "dwell_guided_d6_calls",
    "dwell_guided_positive_gain_targets",
    "b2_retry_used",
    "b2_retry_success",
    "b2_full_decoder_calls",
)


@dataclass(slots=True)
class RouteSearchDiagnostics:
    corridor_coupled: bool = False
    physical_feasibility_aware: bool = False
    final_candidate_classes: bool = False
    dwell_guided: bool = False
    scope_local_calls: int = 0
    scope_forward_calls: int = 0
    scope_global_calls: int = 0
    repair_local_to_forward_escalations: int = 0
    repair_forward_to_global_escalations: int = 0
    repair_calls: int = 0
    d6_calls: int = 0
    d6_affinity_gain_sum: float = 0.0
    d6_affinity_observations: int = 0
    d6_positive_affinity_tasks: int = 0
    d6_corridor_width_sum: int = 0
    d6_targetless_global_fallbacks: int = 0
    span0_candidates_generated: int = 0
    span1_candidates_generated: int = 0
    span2_candidates_generated: int = 0
    span_gt2_candidates_generated: int = 0
    span0_candidates_selected: int = 0
    span1_candidates_selected: int = 0
    span2_candidates_selected: int = 0
    span_gt2_candidates_selected: int = 0
    span0_candidates_accepted: int = 0
    span1_candidates_accepted: int = 0
    span2_candidates_accepted: int = 0
    span_gt2_candidates_accepted: int = 0
    span0_candidates_improved: int = 0
    span1_candidates_improved: int = 0
    span2_candidates_improved: int = 0
    span_gt2_candidates_improved: int = 0
    same_segment_relocations: int = 0
    forward_1_relocations: int = 0
    forward_2_relocations: int = 0
    backward_relocations: int = 0
    d6_relocation_observations: int = 0
    d6_forward_relocations: int = 0
    global_fallback_count: int = 0
    corridor_anchor_count: int = 0
    corridor_partner_candidates: int = 0
    corridor_partner_sorties_removed: int = 0
    c0_generated: int = 0
    c0_feasible: int = 0
    c0_selected: int = 0
    c0_accepted: int = 0
    c0_improved: int = 0
    c1_generated: int = 0
    c1_feasible: int = 0
    c1_selected: int = 0
    c1_accepted: int = 0
    c1_improved: int = 0
    c2_generated: int = 0
    c2_feasible: int = 0
    c2_selected: int = 0
    c2_accepted: int = 0
    c2_improved: int = 0
    c3_generated: int = 0
    c3_feasible: int = 0
    c3_selected: int = 0
    c3_accepted: int = 0
    c3_improved: int = 0
    b0_generated: int = 0
    b0_feasible: int = 0
    b0_selected: int = 0
    b0_accepted: int = 0
    b0_improved: int = 0
    b1_generated: int = 0
    b1_feasible: int = 0
    b1_selected: int = 0
    b1_accepted: int = 0
    b1_improved: int = 0
    b2_generated: int = 0
    b2_feasible: int = 0
    b2_selected: int = 0
    b2_accepted: int = 0
    b2_improved: int = 0
    b3_generated: int = 0
    b3_feasible: int = 0
    b3_selected: int = 0
    b3_accepted: int = 0
    b3_improved: int = 0
    b2_targets_selected: int = 0
    b2_target_tasks_released: int = 0
    b2_minimum_charge_unsafe_targets_skipped: int = 0
    b2_destroy_target_inconsistent: int = 0
    b2_structurally_uninsertable: int = 0
    b2_reconstructed: int = 0
    b2_decoder_feasible: int = 0
    b2_decoder_infeasible: int = 0
    b2_decoder_not_reached: int = 0
    b2_fail_hover_time: int = 0
    b2_fail_hover_soc: int = 0
    b2_fail_ready: int = 0
    b2_fail_route_sync: int = 0
    b2_fail_charge: int = 0
    b2_fail_other: int = 0
    forward_absorbed_tasks: int = 0
    forward_relocated_tasks: int = 0
    span2_feasible: int = 0
    no_span2_candidate: int = 0
    span2_infeasible: int = 0
    span2_feasible_but_worse: int = 0
    span2_iteration_winner: int = 0
    span2_best_objective_value: float = inf
    best_local_objective_value: float = inf
    span2_gap_to_best_local_sum: float = 0.0
    span2_gap_to_best_local_observations: int = 0
    physical_forward_candidates: int = 0
    static_impossible_forward_candidates: int = 0
    dynamic_blocked_forward_candidates: int = 0
    currently_feasible_forward_candidates: int = 0
    guided_targets_selected: int = 0
    guided_targets_reconstructed: int = 0
    guided_targets_decoder_feasible: int = 0
    guided_targets_iteration_best: int = 0
    guided_targets_accepted: int = 0
    guided_targets_improved: int = 0
    blocking_sorties_released: int = 0
    wasted_static_impossible_decoder_calls: int = 0
    guided_hover_time_infeasible: int = 0
    guided_hover_soc_infeasible: int = 0
    guided_sequence_infeasible: int = 0
    guided_other_infeasible: int = 0
    guided_target_hover_time_infeasible: int = 0
    guided_target_hover_soc_infeasible: int = 0
    physical_slack_sum_min: float = 0.0
    blocking_time_sum_min: float = 0.0
    blockers_per_target_sum: int = 0
    dwell_guided_d6_calls: int = 0
    dwell_guided_static_pool_sum: int = 0
    dwell_guided_dynamic_screened_sum: int = 0
    dwell_guided_positive_gain_targets: int = 0
    dwell_guided_estimated_gain_sum: float = 0.0
    b2_actual_dwell_reduction_sum: float = 0.0
    b2_actual_dwell_reduction_observations: int = 0
    b2_retry_used: int = 0
    b2_retry_success: int = 0
    b2_full_decoder_calls: int = 0
    b2_current_improvements: int = 0
    b2_basin_best_updates: int = 0
    b2_global_best_updates: int = 0
    b2_total_global_best_gain: float = 0.0

    def record_scope(self, scope: NeighborhoodScope) -> None:
        name = f"scope_{scope.value}_calls"
        setattr(self, name, getattr(self, name) + 1)

    def record_escalation(
        self,
        source: NeighborhoodScope,
        target: NeighborhoodScope,
    ) -> None:
        if source is NeighborhoodScope.LOCAL and target is NeighborhoodScope.FORWARD:
            self.repair_local_to_forward_escalations += 1
        elif source is NeighborhoodScope.FORWARD and target is NeighborhoodScope.GLOBAL:
            self.repair_forward_to_global_escalations += 1
            self.global_fallback_count += 1

    def record_d6(
        self,
        gains: tuple[RouteAffinityGain, ...],
        corridor_width: int,
    ) -> None:
        self.d6_calls += 1
        self.d6_corridor_width_sum += corridor_width
        for gain in gains:
            if isfinite(gain[0]):
                self.d6_affinity_gain_sum += gain[0]
                self.d6_affinity_observations += 1
            if gain > (0.0, 0.0, 0.0):
                self.d6_positive_affinity_tasks += 1

    def record_d6_targetless_global_fallback(self) -> None:
        if self.final_candidate_classes:
            self.d6_targetless_global_fallbacks += 1

    def record_corridor_bundle(
        self,
        *,
        partner_candidates: int,
        partner_sorties_removed: int,
    ) -> None:
        if not self.corridor_coupled:
            return
        self.corridor_anchor_count += 1
        self.corridor_partner_candidates += partner_candidates
        self.corridor_partner_sorties_removed += partner_sorties_removed

    def record_physical_pool(
        self,
        *,
        feasible_forward: int,
        static_impossible_forward: int,
    ) -> None:
        if not self.physical_feasibility_aware:
            return
        self.physical_forward_candidates = feasible_forward
        self.static_impossible_forward_candidates = static_impossible_forward

    def record_forward_analysis(self, analysis: BlockingAnalysis) -> None:
        if not self.physical_feasibility_aware:
            return
        if analysis.feasibility is ForwardFeasibility.DYNAMIC_BLOCKED:
            self.dynamic_blocked_forward_candidates += 1
        elif analysis.feasibility is ForwardFeasibility.CURRENTLY_FEASIBLE:
            self.currently_feasible_forward_candidates += 1

    def record_dwell_guided_screen(
        self,
        *,
        static_pool_size: int,
        dynamically_screened: int,
        positive_gain_targets: int,
    ) -> None:
        if not self.dwell_guided:
            return
        self.dwell_guided_d6_calls += 1
        self.dwell_guided_static_pool_sum += static_pool_size
        self.dwell_guided_dynamic_screened_sum += dynamically_screened
        self.dwell_guided_positive_gain_targets += positive_gain_targets

    def record_dwell_guided_target(self, analysis: DwellGuidedAnalysis) -> None:
        if not self.dwell_guided:
            return
        self.guided_targets_selected += 1
        self.blocking_sorties_released += len(analysis.blocking_sortie_ids)
        self.blockers_per_target_sum += len(analysis.blocking_sortie_ids)
        self.physical_slack_sum_min += analysis.dynamic_slack_min
        self.blocking_time_sum_min += analysis.current_intermediate_dwell_min
        self.dwell_guided_estimated_gain_sum += analysis.estimated_dwell_gain_min

    def record_b2_retry(self, *, success: bool) -> None:
        if not self.dwell_guided:
            return
        self.b2_retry_used += 1
        self.b2_retry_success += int(success)

    def record_b2_actual_dwell_reduction(self, reduction_min: float) -> None:
        if not self.dwell_guided:
            return
        self.b2_actual_dwell_reduction_sum += reduction_min
        self.b2_actual_dwell_reduction_observations += 1

    def record_b2_search_outcome(
        self,
        *,
        current_improvement: bool,
        basin_best_updated: bool,
        global_best_updated: bool,
        global_best_gain: float,
    ) -> None:
        if not self.dwell_guided:
            return
        self.b2_current_improvements += int(current_improvement)
        self.b2_basin_best_updates += int(basin_best_updated)
        self.b2_global_best_updates += int(global_best_updated)
        self.b2_total_global_best_gain += max(0.0, global_best_gain)

    def record_guided_target(self, analysis: BlockingAnalysis) -> None:
        if not self.physical_feasibility_aware:
            return
        self.guided_targets_selected += 1
        self.physical_slack_sum_min += analysis.physical_slack_min
        self.blocking_time_sum_min += analysis.estimated_blocking_time_min
        blocker_count = len(analysis.blocking_sortie_ids)
        self.blockers_per_target_sum += blocker_count
        self.blocking_sorties_released += blocker_count

    def record_b2_target_selected(self, *, target_task_count: int) -> None:
        if not self.final_candidate_classes:
            return
        self.b2_targets_selected += 1
        self.b2_target_tasks_released += target_task_count

    def record_b2_minimum_charge_unsafe_target(self) -> None:
        if self.final_candidate_classes:
            self.b2_minimum_charge_unsafe_targets_skipped += 1

    def record_b2_reconstruction(self, status: str) -> None:
        if not self.final_candidate_classes:
            return
        if status == "reconstructed":
            self.b2_reconstructed += 1
            self.guided_targets_reconstructed += 1
            return
        if status == "destroy_target_inconsistent":
            self.b2_destroy_target_inconsistent += 1
        elif status != "structurally_uninsertable":
            raise ValueError(f"unknown B2 reconstruction status {status!r}")
        self.b2_structurally_uninsertable += 1

    def record_b2_decoder(self, evaluations, *, target_sortie_id: int) -> None:
        if not self.final_candidate_classes:
            return
        decoded = tuple(evaluations)
        if self.dwell_guided:
            self.b2_full_decoder_calls += len(decoded)
        if not decoded:
            self.b2_decoder_not_reached += 1
            return
        if any(item.evaluation.feasible for item in decoded):
            self.b2_decoder_feasible += 1
            self.guided_targets_decoder_feasible += 1
            return
        self.b2_decoder_infeasible += 1
        codes = {
            detail.code
            for item in decoded
            for detail in item.evaluation.violation_details
        }
        classified = False
        if "HOVER_SORTIE_TIME" in codes:
            self.b2_fail_hover_time += 1
            classified = True
        if "HOVER_SOC_RESERVE" in codes:
            self.b2_fail_hover_soc += 1
            classified = True
        if codes & {"UAV_NOT_READY", "READY_TIME"}:
            self.b2_fail_ready += 1
            classified = True
        if codes & {
            "WRONG_SUPPORT_ORDER",
            "UAV_SEQUENCE_PRECEDENCE",
            "ROUTE_SYNC",
        }:
            self.b2_fail_route_sync += 1
            classified = True
        if codes & {"LAUNCH_NOMINAL_SOC", "CHARGE_DELAY"}:
            self.b2_fail_charge += 1
            classified = True
        if not classified:
            self.b2_fail_other += 1

    def record_guided_reconstructed(self) -> None:
        if self.physical_feasibility_aware:
            self.guided_targets_reconstructed += 1

    def record_guided_decoder(self, evaluation, *, target_sortie_id: int) -> None:
        if not self.physical_feasibility_aware:
            return
        if evaluation.feasible:
            if not self.final_candidate_classes:
                self.guided_targets_decoder_feasible += 1
            return
        codes = {detail.code for detail in evaluation.violation_details}
        classified = False
        if "HOVER_SORTIE_TIME" in codes:
            self.guided_hover_time_infeasible += 1
            classified = True
        if "HOVER_SOC_RESERVE" in codes:
            self.guided_hover_soc_infeasible += 1
            classified = True
        if "UAV_SEQUENCE_PRECEDENCE" in codes:
            self.guided_sequence_infeasible += 1
            classified = True
        if not classified:
            self.guided_other_infeasible += 1
        target_codes = {
            detail.code
            for detail in evaluation.violation_details
            if detail.sortie_id == target_sortie_id
        }
        if "HOVER_SORTIE_TIME" in target_codes:
            self.guided_target_hover_time_infeasible += 1
        if "HOVER_SOC_RESERVE" in target_codes:
            self.guided_target_hover_soc_infeasible += 1

    def record_guided_lifecycle(self, stage: str) -> None:
        if not self.physical_feasibility_aware:
            return
        name = f"guided_targets_{stage}"
        if name not in _V22_COUNTER_FIELDS:
            raise ValueError(f"unknown guided lifecycle stage {stage}")
        setattr(self, name, getattr(self, name) + 1)

    def record_candidate(self, stage: str, span: int) -> None:
        name = f"{_span_bucket(span)}_candidates_{stage}"
        setattr(self, name, getattr(self, name) + 1)

    def record_candidate_class(
        self,
        stage: str,
        candidate_class: CandidateClass | None,
    ) -> None:
        if not self.corridor_coupled or candidate_class is None:
            return
        name = f"{candidate_class.value}_{stage}"
        setattr(self, name, getattr(self, name) + 1)

    def record_feasible_candidate(
        self,
        *,
        candidate_class: CandidateClass | None,
        span: int,
    ) -> None:
        if not self.corridor_coupled:
            return
        self.record_candidate_class("feasible", candidate_class)
        if span == 2:
            self.span2_feasible += 1

    def record_r4p_outcome(
        self,
        *,
        generated_spans: tuple[int, ...],
        feasible_objectives: tuple[tuple[int, float], ...],
    ) -> None:
        if not self.corridor_coupled or self.final_candidate_classes:
            return
        span2_generated = any(span == 2 for span in generated_spans)
        feasible_span2 = tuple(
            objective for span, objective in feasible_objectives if span == 2
        )
        feasible_local = tuple(
            objective for span, objective in feasible_objectives if span in {0, 1}
        )
        if not span2_generated:
            self.no_span2_candidate += 1
            return
        if not feasible_span2:
            self.span2_infeasible += 1
            return

        best_span2 = min(feasible_span2)
        self.span2_best_objective_value = min(
            self.span2_best_objective_value, best_span2
        )
        if feasible_local:
            best_local = min(feasible_local)
            self.best_local_objective_value = min(
                self.best_local_objective_value, best_local
            )
            self.span2_gap_to_best_local_sum += (
                100.0 * (best_span2 - best_local) / best_local
            )
            self.span2_gap_to_best_local_observations += 1

        winner_span, _ = min(
            feasible_objectives,
            key=lambda value: value[1],
        )
        if winner_span == 2:
            self.span2_iteration_winner += 1
        else:
            self.span2_feasible_but_worse += 1

    def record_structural_tasks(
        self,
        *,
        absorbed_task_ids: tuple[int, ...],
        relocated_task_ids: tuple[int, ...],
    ) -> None:
        if not self.corridor_coupled:
            return
        self.forward_absorbed_tasks += len(absorbed_task_ids)
        self.forward_relocated_tasks += len(relocated_task_ids)

    def record_relocations(self, deltas: tuple[int, ...], *, from_d6: bool) -> None:
        for delta in deltas:
            if delta == 0:
                self.same_segment_relocations += 1
            elif delta == 1:
                self.forward_1_relocations += 1
            elif delta == 2:
                self.forward_2_relocations += 1
            elif delta < 0:
                self.backward_relocations += 1
            if from_d6:
                self.d6_relocation_observations += 1
                self.d6_forward_relocations += int(delta > 0)

    def to_dict(self) -> dict[str, int | float]:
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.endswith("_sum") and name not in {
                "corridor_coupled",
                "physical_feasibility_aware",
                "final_candidate_classes",
                "dwell_guided",
                "d6_affinity_observations",
                "d6_positive_affinity_tasks",
                "repair_calls",
                "d6_relocation_observations",
                "d6_forward_relocations",
                "corridor_anchor_count",
                "corridor_partner_candidates",
                "corridor_partner_sorties_removed",
                *_V21_COUNTER_FIELDS,
                *_V22_COUNTER_FIELDS,
                *_V3_COUNTER_FIELDS,
                *_B2_TRANSACTION_COUNTER_FIELDS,
                "span2_best_objective_value",
                "best_local_objective_value",
                "span2_gap_to_best_local_sum",
                "span2_gap_to_best_local_observations",
                "physical_slack_sum_min",
                "blocking_time_sum_min",
                "blockers_per_target_sum",
                "dwell_guided_static_pool_sum",
                "dwell_guided_dynamic_screened_sum",
                "dwell_guided_estimated_gain_sum",
                "b2_actual_dwell_reduction_sum",
                "b2_actual_dwell_reduction_observations",
            }
        }
        values.update(
            {
                "d6_mean_affinity_gain": (
                    self.d6_affinity_gain_sum / self.d6_affinity_observations
                    if self.d6_affinity_observations else 0.0
                ),
                "d6_positive_affinity_ratio": (
                    self.d6_positive_affinity_tasks / self.d6_affinity_observations
                    if self.d6_affinity_observations else 0.0
                ),
                "d6_corridor_width": (
                    self.d6_corridor_width_sum / self.d6_calls if self.d6_calls else 0.0
                ),
                "global_fallback_rate": (
                    self.global_fallback_count / self.repair_calls
                    if self.repair_calls else 0.0
                ),
                "forward_relocation_given_d6": (
                    self.d6_forward_relocations / self.d6_relocation_observations
                    if self.d6_relocation_observations else 0.0
                ),
                "span2_improvement_rate": (
                    self.span2_candidates_improved / self.span2_candidates_accepted
                    if self.span2_candidates_accepted else 0.0
                ),
            }
        )
        if self.corridor_coupled:
            values.update({name: getattr(self, name) for name in _V21_COUNTER_FIELDS})
            values.update(
                {
                    "span2_best_objective": (
                        self.span2_best_objective_value
                        if isfinite(self.span2_best_objective_value)
                        else float("nan")
                    ),
                    "best_local_objective": (
                        self.best_local_objective_value
                        if isfinite(self.best_local_objective_value)
                        else float("nan")
                    ),
                    "span2_gap_to_best_local": (
                        self.span2_gap_to_best_local_sum
                        / self.span2_gap_to_best_local_observations
                        if self.span2_gap_to_best_local_observations
                        else float("nan")
                    ),
                    "span2_gap_observations": (
                        self.span2_gap_to_best_local_observations
                    ),
                }
            )
        if self.physical_feasibility_aware:
            values.update({name: getattr(self, name) for name in _V22_COUNTER_FIELDS})
            target_count = self.guided_targets_selected
            values.update(
                {
                    "mean_physical_slack_min": (
                        self.physical_slack_sum_min / target_count
                        if target_count else 0.0
                    ),
                    "mean_blocking_time_min": (
                        self.blocking_time_sum_min / target_count
                        if target_count else 0.0
                    ),
                    "mean_blockers_per_target": (
                        self.blockers_per_target_sum / target_count
                        if target_count else 0.0
                    ),
                }
            )
        if self.final_candidate_classes:
            values.update({name: getattr(self, name) for name in _V3_COUNTER_FIELDS})
            values.update(
                {
                    name: getattr(self, name)
                    for name in _B2_TRANSACTION_COUNTER_FIELDS
                }
            )
        if self.dwell_guided:
            calls = self.dwell_guided_d6_calls
            selected = self.guided_targets_selected
            values.update({name: getattr(self, name) for name in _V31_COUNTER_FIELDS})
            values.update(
                {
                    "mean_static_pool_per_d6": (
                        self.dwell_guided_static_pool_sum / calls if calls else 0.0
                    ),
                    "mean_dynamically_screened_per_d6": (
                        self.dwell_guided_dynamic_screened_sum / calls
                        if calls else 0.0
                    ),
                    "mean_estimated_dwell_gain_min": (
                        self.dwell_guided_estimated_gain_sum / selected
                        if selected else 0.0
                    ),
                    "mean_actual_dwell_reduction_after_accepted_b2_min": (
                        self.b2_actual_dwell_reduction_sum
                        / self.b2_actual_dwell_reduction_observations
                        if self.b2_actual_dwell_reduction_observations else 0.0
                    ),
                    "b2_improvement_rate": (
                        self.b2_improved / self.b2_feasible
                        if self.b2_feasible else 0.0
                    ),
                    "mean_full_decoder_b2_per_d6": (
                        self.b2_full_decoder_calls / calls if calls else 0.0
                    ),
                }
            )
        return values


@dataclass(slots=True)
class NormalOnlySearchProfile:
    """Non-overlapping wall-time accounting for the NORMAL-only diagnostic."""

    normal_destroy_sec: float = 0.0
    normal_repair_sec: float = 0.0
    normal_candidate_build_sec: float = 0.0
    normal_decoder_sec: float = 0.0
    normal_local_search_sec: float = 0.0
    normal_operator_selection_sec: float = 0.0
    normal_diagnostics_sec: float = 0.0
    controller_sec: float = 0.0
    full_decoder_calls: int = 0

    def to_dict(
        self,
        *,
        wall_sec: float,
        iterations: int,
        generated_candidates: int,
        effective_evaluations: int,
    ) -> dict[str, int | float | str]:
        accounted = (
            self.normal_destroy_sec
            + self.normal_repair_sec
            + self.normal_decoder_sec
            + self.normal_local_search_sec
            + self.controller_sec
        )
        other_sec = max(0.0, wall_sec - accounted)
        denominator = wall_sec if wall_sec > 0 else 1.0
        return {
            "controller_mode": "normal_only_profile",
            "normal_destroy_sec": self.normal_destroy_sec,
            "normal_repair_sec": self.normal_repair_sec,
            "normal_candidate_build_sec": self.normal_candidate_build_sec,
            "normal_decoder_sec": self.normal_decoder_sec,
            "normal_local_search_sec": self.normal_local_search_sec,
            "normal_operator_selection_sec": (
                self.normal_operator_selection_sec
            ),
            "normal_diagnostics_sec": self.normal_diagnostics_sec,
            "controller_sec": self.controller_sec,
            "other_sec": other_sec,
            "profile_accounted_sec": accounted + other_sec,
            "profile_wall_sec": wall_sec,
            "iterations": iterations,
            "generated_candidates": generated_candidates,
            "full_decoder_calls": self.full_decoder_calls,
            "effective_evaluations": effective_evaluations,
            "iterations_per_sec": iterations / denominator,
            "candidates_per_sec": generated_candidates / denominator,
            "decoder_eval_per_sec": self.full_decoder_calls / denominator,
            "effective_eval_per_sec": effective_evaluations / denominator,
        }


@dataclass(slots=True)
class EscapeSearchDiagnostics:
    """V3.2.1 branch attribution and event logs; never feeds search weights."""

    action_calls: dict[SearchAction, int] = field(
        default_factory=lambda: {action: 0 for action in SearchAction}
    )
    action_wall_sec: dict[SearchAction, float] = field(
        default_factory=lambda: {action: 0.0 for action in SearchAction}
    )
    action_global_best_gain: dict[SearchAction, float] = field(
        default_factory=lambda: {action: 0.0 for action in SearchAction}
    )
    candidate_accepted: dict[SearchAction, int] = field(
        default_factory=lambda: {action: 0 for action in SearchAction}
    )
    current_improvements: dict[SearchAction, int] = field(
        default_factory=lambda: {action: 0 for action in SearchAction}
    )
    basin_best_updates: dict[SearchAction, int] = field(
        default_factory=lambda: {action: 0 for action in SearchAction}
    )
    global_best_updates: dict[SearchAction, int] = field(
        default_factory=lambda: {action: 0 for action in SearchAction}
    )
    descent_momentum_sum: float = 0.0
    stagnation_pressure_sum: float = 0.0
    decision_count: int = 0
    eligible_p_jump_sum: float = 0.0
    eligible_decision_count: int = 0
    global_escape_failed: int = 0
    normal_b2_calls: int = 0
    last_global_escape_elapsed_sec: float | None = None
    min_global_escape_interval_sec: float = inf
    global_escape_deterioration_sum: float = 0.0
    global_escape_deterioration_count: int = 0
    global_escape_worse_relocations: int = 0
    escape_events: list[dict[str, object]] = field(default_factory=list)
    global_best_events: list[dict[str, object]] = field(default_factory=list)

    def record_iteration(
        self,
        *,
        iteration: int,
        elapsed_sec: float,
        decision: EscapeDecision,
        current_objective: float,
        basin_best_objective: float,
        global_best_objective: float,
        wall_sec: float,
        destroy_operator: str,
        repair_operator: str,
        candidate_class: CandidateClass | None,
        accepted: bool,
        current_improvement: bool,
        basin_best_updated: bool,
        global_best_updated: bool,
        global_best_gain: float,
        objective_after: float,
        normal_b2_calls: int = 0,
    ) -> None:
        action = decision.action
        self.action_calls[action] += 1
        self.action_wall_sec[action] += max(0.0, wall_sec)
        self.action_global_best_gain[action] += max(0.0, global_best_gain)
        self.candidate_accepted[action] += int(accepted)
        self.current_improvements[action] += int(current_improvement)
        self.basin_best_updates[action] += int(basin_best_updated)
        self.global_best_updates[action] += int(global_best_updated)
        self.descent_momentum_sum += decision.descent_momentum
        self.stagnation_pressure_sum += decision.stagnation_pressure
        self.decision_count += 1
        if decision.probability_eligible:
            self.eligible_p_jump_sum += decision.p_jump
            self.eligible_decision_count += 1
        if action is SearchAction.GLOBAL_ESCAPE and not accepted:
            self.global_escape_failed += 1
        global_deterioration_pct: float | None = None
        if action is SearchAction.GLOBAL_ESCAPE:
            if self.last_global_escape_elapsed_sec is not None:
                self.min_global_escape_interval_sec = min(
                    self.min_global_escape_interval_sec,
                    elapsed_sec - self.last_global_escape_elapsed_sec,
                )
            self.last_global_escape_elapsed_sec = elapsed_sec
            if accepted:
                global_deterioration_pct = (
                    100.0
                    * (objective_after - current_objective)
                    / current_objective
                )
                self.global_escape_deterioration_sum += (
                    global_deterioration_pct
                )
                self.global_escape_deterioration_count += 1
                self.global_escape_worse_relocations += int(
                    global_deterioration_pct > 0.0
                )
        if action is SearchAction.NORMAL:
            self.normal_b2_calls += normal_b2_calls
        self.escape_events.append(
            {
                "elapsed_sec": elapsed_sec,
                "iteration": iteration,
                "current_objective": current_objective,
                "basin_best_objective": basin_best_objective,
                "global_best_objective": global_best_objective,
                "basin_stagnation_sec": decision.basin_stagnation_sec,
                "global_stagnation_sec": decision.global_stagnation_sec,
                "g_current": decision.descent_rate,
                "g_reference": decision.reference_descent_rate,
                "descent_momentum": decision.descent_momentum,
                "stagnation_pressure": decision.stagnation_pressure,
                "jump_hazard": decision.hazard_rate,
                "delta_t": decision.delta_t,
                "p_jump": decision.p_jump,
                "random_u": decision.random_draw,
                "guided_rearmed": decision.guided_rearmed,
                "global_rearmed": decision.global_rearmed,
                "guided_attempted_in_current_basin": (
                    decision.guided_attempted_in_current_basin
                ),
                "decision": action.value,
                "action": action.value,
                "descent_rate": decision.descent_rate,
                "reference_descent_rate": decision.reference_descent_rate,
                "destroy_operator": destroy_operator,
                "repair_operator": repair_operator,
                "candidate_class": (
                    candidate_class.value if candidate_class is not None else None
                ),
                "guided_accepted": (
                    accepted if action is SearchAction.GUIDED_ESCAPE else False
                ),
                "guided_global_best": (
                    global_best_updated
                    if action is SearchAction.GUIDED_ESCAPE
                    else False
                ),
                "global_forced": (
                    accepted if action is SearchAction.GLOBAL_ESCAPE else False
                ),
                "normal_b2_calls": (
                    normal_b2_calls if action is SearchAction.NORMAL else 0
                ),
                "objective_after": objective_after,
                "current_before": current_objective,
                "current_after": objective_after,
                "basin_best": basin_best_objective,
                "global_best": global_best_objective,
                "global_escape_objective_before": (
                    current_objective
                    if action is SearchAction.GLOBAL_ESCAPE else None
                ),
                "global_escape_objective_after": (
                    objective_after
                    if action is SearchAction.GLOBAL_ESCAPE and accepted else None
                ),
                "global_escape_deterioration_pct": global_deterioration_pct,
            }
        )

    def record_global_best(
        self,
        *,
        elapsed_sec: float,
        iteration: int,
        old_global_best: float,
        new_global_best: float,
        decision: EscapeDecision,
        destroy_operator: str,
        repair_operator: str,
        candidate_class: CandidateClass | None,
    ) -> None:
        self.global_best_events.append(
            {
                "elapsed_sec": elapsed_sec,
                "iteration": iteration,
                "old_global_best": old_global_best,
                "new_global_best": new_global_best,
                "gain_min": old_global_best - new_global_best,
                "source_action": decision.action.value,
                "destroy_operator": destroy_operator,
                "repair_operator": repair_operator,
                "candidate_class": (
                    candidate_class.value if candidate_class is not None else None
                ),
                "p_jump": decision.p_jump,
                "descent_momentum": decision.descent_momentum,
                "stagnation_pressure": decision.stagnation_pressure,
            }
        )

    def to_dict(self) -> dict[str, int | float]:
        total_wall = sum(self.action_wall_sec.values())
        values: dict[str, int | float] = {
            "normal_iterations": self.action_calls[SearchAction.NORMAL],
            "guided_escapes": self.action_calls[SearchAction.GUIDED_ESCAPE],
            "global_escapes": self.action_calls[SearchAction.GLOBAL_ESCAPE],
            "global_escape_failed": self.global_escape_failed,
            "normal_b2_calls": self.normal_b2_calls,
            "mean_descent_momentum": (
                self.descent_momentum_sum / self.decision_count
                if self.decision_count else 0.0
            ),
            "mean_stagnation_pressure": (
                self.stagnation_pressure_sum / self.decision_count
                if self.decision_count else 0.0
            ),
            "mean_p_jump_when_eligible": (
                self.eligible_p_jump_sum / self.eligible_decision_count
                if self.eligible_decision_count else 0.0
            ),
            "probability_eligible_decisions": self.eligible_decision_count,
            "min_global_escape_interval_sec": (
                self.min_global_escape_interval_sec
                if self.min_global_escape_interval_sec < inf
                else 0.0
            ),
            "global_escape_deterioration_mean_pct": (
                self.global_escape_deterioration_sum
                / self.global_escape_deterioration_count
                if self.global_escape_deterioration_count else 0.0
            ),
            "global_escape_deterioration_observations": (
                self.global_escape_deterioration_count
            ),
            "global_escape_worse_relocations": (
                self.global_escape_worse_relocations
            ),
        }
        for action in SearchAction:
            prefix = action.value
            wall_sec = self.action_wall_sec[action]
            gain = self.action_global_best_gain[action]
            values.update(
                {
                    f"{prefix}_wall_sec": wall_sec,
                    f"{prefix}_wall_ratio": (
                        wall_sec / total_wall if total_wall else 0.0
                    ),
                    f"{prefix}_global_best_gain": gain,
                    f"{prefix}_global_best_gain_per_sec": (
                        gain / wall_sec if wall_sec else 0.0
                    ),
                    f"{prefix}_candidate_accepted": self.candidate_accepted[action],
                    f"{prefix}_current_improvements": self.current_improvements[action],
                    f"{prefix}_basin_best_updates": self.basin_best_updates[action],
                    f"{prefix}_global_best_updates": self.global_best_updates[action],
                }
            )
        return values
