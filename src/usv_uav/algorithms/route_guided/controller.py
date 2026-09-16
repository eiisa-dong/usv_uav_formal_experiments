from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import expm1, isfinite
import random
from statistics import median
from typing import Literal, cast

from usv_uav.algorithms.alns_config import (
    RouteGuidedV2Config,
    RouteGuidedV32Config,
)
from usv_uav.algorithms.route_guided.models import (
    EscapeDecision,
    NeighborhoodScope,
    RepairPolicy,
    RouteCorridor,
    SearchAction,
)


_OBJECTIVE_TOLERANCE = 1e-9


class DescentMomentumEscapeController:
    """Wall-clock controller for V3.2.1 basin continuation and escape.

    The controller owns only scalar search-state semantics. Solutions and full
    evaluations remain in ``alns.py`` so the global incumbent is never exposed
    to a relocation operation.
    """

    def __init__(
        self,
        *,
        config: RouteGuidedV32Config,
        budget_sec: float,
        initial_objective: float,
        rng: random.Random,
        start_time: float = 0.0,
    ) -> None:
        if not config.enabled:
            raise ValueError("descent-momentum controller requires V3.2")
        if not isfinite(budget_sec) or budget_sec <= 0:
            raise ValueError("budget_sec must be finite and positive")
        if not isfinite(initial_objective) or initial_objective <= 0:
            raise ValueError("initial_objective must be finite and positive")
        if not isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and non-negative")
        self.config = config
        self.budget_sec = float(budget_sec)
        self.medium_threshold_sec = (
            self.budget_sec * config.escape.medium_budget_ratio
        )
        self.global_threshold_sec = (
            self.budget_sec * config.escape.global_budget_ratio
        )
        self.rng = rng
        self.basin_best_objective = float(initial_objective)
        self.global_best_objective = float(initial_objective)
        self.basin_start_time = float(start_time)
        self.last_basin_best_time = float(start_time)
        self.last_global_best_time = float(start_time)
        self.last_guided_escape_time: float | None = None
        self.last_global_escape_time: float | None = None
        self.guided_attempted_in_current_basin = False
        self.last_probability_decision_time = float(start_time)
        self._basin_history: deque[tuple[float, float]] = deque(
            [(float(start_time), float(initial_objective))]
        )
        self.positive_descent_rates: deque[float] = deque(maxlen=20)

    @staticmethod
    def _clip_unit(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _validate_elapsed(self, elapsed_sec: float) -> float:
        if not isfinite(elapsed_sec) or elapsed_sec < self.basin_start_time:
            raise ValueError("elapsed_sec must be finite and monotonic")
        return float(elapsed_sec)

    def _descent_rate(self, elapsed_sec: float) -> float:
        window_start = max(
            self.basin_start_time,
            elapsed_sec - self.medium_threshold_sec,
        )
        duration = elapsed_sec - window_start
        if duration <= 0:
            return 0.0
        reference_objective = self._basin_history[0][1]
        for observed_at, objective in self._basin_history:
            if observed_at > window_start + _OBJECTIVE_TOLERANCE:
                break
            reference_objective = objective
        gain = max(0.0, reference_objective - self.basin_best_objective)
        return gain / (reference_objective * duration)

    def _start_new_basin(self, *, elapsed_sec: float, objective: float) -> None:
        self.basin_start_time = elapsed_sec
        self.last_basin_best_time = elapsed_sec
        self.basin_best_objective = objective
        self.last_probability_decision_time = elapsed_sec
        self._basin_history.clear()
        self._basin_history.append((elapsed_sec, objective))
        self.positive_descent_rates.clear()
        self.guided_attempted_in_current_basin = False

    def _metrics(
        self,
        elapsed_sec: float,
    ) -> tuple[float, float, float, float, float, float]:
        descent_rate = self._descent_rate(elapsed_sec)
        reference_rate = (
            float(median(self.positive_descent_rates))
            if self.positive_descent_rates
            else 0.0
        )
        if reference_rate > 0:
            momentum = self._clip_unit(
                descent_rate / (reference_rate + _OBJECTIVE_TOLERANCE)
            )
        else:
            momentum = 1.0 if descent_rate > 0 else 0.0
        basin_stagnation = elapsed_sec - self.last_basin_best_time
        global_stagnation = elapsed_sec - self.last_global_best_time
        pressure = self._clip_unit(
            (basin_stagnation - self.medium_threshold_sec)
            / (self.global_threshold_sec - self.medium_threshold_sec)
        )
        return (
            descent_rate,
            reference_rate,
            momentum,
            basin_stagnation,
            global_stagnation,
            pressure,
        )

    def _decision(
        self,
        *,
        action: SearchAction,
        metrics: tuple[float, float, float, float, float, float],
        hazard_rate: float = 0.0,
        delta_t: float = 0.0,
        p_jump: float = 0.0,
        random_draw: float | None = None,
        guided_rearmed: bool = True,
        global_rearmed: bool = True,
    ) -> EscapeDecision:
        descent, reference, momentum, basin_stagnation, global_stagnation, pressure = metrics
        return EscapeDecision(
            action=action,
            descent_rate=descent,
            reference_descent_rate=reference,
            descent_momentum=momentum,
            basin_stagnation_sec=basin_stagnation,
            global_stagnation_sec=global_stagnation,
            stagnation_pressure=pressure,
            hazard_rate=hazard_rate,
            delta_t=delta_t,
            p_jump=p_jump,
            random_draw=random_draw,
            guided_rearmed=guided_rearmed,
            global_rearmed=global_rearmed,
            guided_attempted_in_current_basin=(
                self.guided_attempted_in_current_basin
            ),
        )

    @staticmethod
    def hazard_probability(
        *,
        stagnation_pressure: float,
        descent_momentum: float,
        medium_threshold_sec: float,
        delta_t: float,
    ) -> tuple[float, float]:
        if medium_threshold_sec <= 0 or delta_t < 0:
            raise ValueError("hazard time values must be non-negative and finite")
        pressure = DescentMomentumEscapeController._clip_unit(
            stagnation_pressure
        )
        momentum = DescentMomentumEscapeController._clip_unit(descent_momentum)
        hazard_rate = pressure * (1.0 - momentum) / medium_threshold_sec
        return hazard_rate, -expm1(-hazard_rate * delta_t)

    def decide(self, *, elapsed_sec: float) -> EscapeDecision:
        """Choose NORMAL, GUIDED_ESCAPE, or GLOBAL_ESCAPE in frozen order."""
        now = self._validate_elapsed(elapsed_sec)
        metrics = self._metrics(now)
        _, _, momentum, basin_stagnation, global_stagnation, pressure = metrics
        guided_rearmed = (
            self.last_guided_escape_time is None
            or now - self.last_guided_escape_time >= self.medium_threshold_sec
        )
        global_rearmed = (
            self.last_global_escape_time is None
            or now - self.last_global_escape_time >= self.global_threshold_sec
        )

        if self.config.normal_only_profile:
            self.last_probability_decision_time = now
            return self._decision(
                action=SearchAction.NORMAL,
                metrics=metrics,
                guided_rearmed=guided_rearmed,
                global_rearmed=global_rearmed,
            )

        # 1. Severe stagnation preserves the strict NORMAL -> GUIDED -> GLOBAL
        # hierarchy.  A Global action can never be the basin's first escape.
        if (
            global_stagnation >= self.global_threshold_sec
            and basin_stagnation >= self.medium_threshold_sec
        ):
            self.last_probability_decision_time = now
            if (
                not self.guided_attempted_in_current_basin
                and guided_rearmed
            ):
                action = SearchAction.GUIDED_ESCAPE
            elif self.guided_attempted_in_current_basin and global_rearmed:
                action = SearchAction.GLOBAL_ESCAPE
            else:
                action = SearchAction.NORMAL
            if action is SearchAction.GLOBAL_ESCAPE:
                assert self.last_global_escape_time is None or (
                    now - self.last_global_escape_time
                    >= self.global_threshold_sec
                ), "GLOBAL_ESCAPE cannot execute before its T_G re-arm"
            return self._decision(
                action=action,
                metrics=metrics,
                guided_rearmed=guided_rearmed,
                global_rearmed=global_rearmed,
            )

        # 2. The current basin receives at least T_M of ordinary exploitation,
        # and Guided has its own T_M re-arm window.
        if basin_stagnation < self.medium_threshold_sec or not guided_rearmed:
            self.last_probability_decision_time = now
            return self._decision(
                action=SearchAction.NORMAL,
                metrics=metrics,
                guided_rearmed=guided_rearmed,
                global_rearmed=global_rearmed,
            )

        # 4-6. Wall-clock hazard, then one independent controller draw.
        delta_t = max(0.0, now - self.last_probability_decision_time)
        hazard_rate, p_jump = self.hazard_probability(
            stagnation_pressure=pressure,
            descent_momentum=momentum,
            medium_threshold_sec=self.medium_threshold_sec,
            delta_t=delta_t,
        )
        random_draw = self.rng.random()
        self.last_probability_decision_time = now
        action = (
            SearchAction.GUIDED_ESCAPE
            if random_draw < p_jump
            else SearchAction.NORMAL
        )
        return self._decision(
            action=action,
            metrics=metrics,
            hazard_rate=hazard_rate,
            delta_t=delta_t,
            p_jump=p_jump,
            random_draw=random_draw,
            guided_rearmed=guided_rearmed,
            global_rearmed=global_rearmed,
        )

    def observe(
        self,
        *,
        action: SearchAction,
        elapsed_sec: float,
        accepted: bool,
        current_objective: float,
        basin_best_updated: bool,
        global_best_updated: bool,
        global_best_objective: float,
    ) -> None:
        """Update basin clocks after one action without ever relocating global best."""
        now = self._validate_elapsed(elapsed_sec)
        if not isfinite(current_objective) or current_objective <= 0:
            raise ValueError("current_objective must be finite and positive")
        if not isfinite(global_best_objective) or global_best_objective <= 0:
            raise ValueError("global_best_objective must be finite and positive")
        if global_best_objective > self.global_best_objective + _OBJECTIVE_TOLERANCE:
            raise ValueError("global best objective cannot worsen")

        if action in {SearchAction.GUIDED_ESCAPE, SearchAction.GLOBAL_ESCAPE}:
            if action is SearchAction.GUIDED_ESCAPE:
                self.last_guided_escape_time = now
                self.guided_attempted_in_current_basin = True
            else:
                self.last_global_escape_time = now
            if accepted:
                self._start_new_basin(
                    elapsed_sec=now,
                    objective=float(current_objective),
                )
        elif basin_best_updated:
            if current_objective >= self.basin_best_objective - _OBJECTIVE_TOLERANCE:
                raise ValueError("basin best update must strictly improve the basin")
            self.basin_best_objective = float(current_objective)
            self.last_basin_best_time = now
            self._basin_history.append((now, float(current_objective)))
            positive_rate = self._descent_rate(now)
            if positive_rate > 0:
                self.positive_descent_rates.append(positive_rate)

        if global_best_updated:
            if global_best_objective >= self.global_best_objective - _OBJECTIVE_TOLERANCE:
                raise ValueError("global best update must strictly improve the incumbent")
            self.last_global_best_time = now
        self.global_best_objective = float(global_best_objective)
        if self.global_best_objective > self.basin_best_objective + _OBJECTIVE_TOLERANCE:
            raise AssertionError("global best must not be worse than basin best")


class RouteSearchController:
    """Map one multi-scale decision to destroy geometry and repair scope."""

    def __init__(self, config: RouteGuidedV2Config) -> None:
        if not config.enabled:
            raise ValueError("route-guided controller requires an enabled V2 config")
        self.config = config

    def make_policy(
        self,
        *,
        n_tasks: int,
        destroy_scale: str,
        stagnation: int,
        repair_name: str,
    ) -> RepairPolicy:
        del stagnation, repair_name
        if destroy_scale not in {"legacy", "micro", "medium", "macro"}:
            raise ValueError(f"unknown destroy scale {destroy_scale}")
        if n_tasks <= self.config.small_global_threshold:
            scope = NeighborhoodScope.GLOBAL
            radius = max_span = None
            backward = forward = n_tasks
        elif destroy_scale == "micro":
            scope = NeighborhoodScope.LOCAL
            radius = self.config.local_origin_radius
            max_span = self.config.local_max_recovery_span
            backward = 0
            forward = self.config.micro_corridor_forward
        elif destroy_scale == "medium":
            scope = NeighborhoodScope.FORWARD
            radius = self.config.forward_origin_radius
            max_span = self.config.forward_max_recovery_span
            backward = self.config.medium_corridor_backward
            forward = self.config.medium_corridor_forward
        else:
            scope = NeighborhoodScope.GLOBAL
            radius = max_span = None
            backward = forward = n_tasks
        return RepairPolicy(
            scope=scope,
            origin_radius=radius,
            max_recovery_span=max_span,
            allow_global_fallback=self.config.global_fallback,
            destroy_scale=cast(
                Literal["legacy", "micro", "medium", "macro"], destroy_scale
            ),
            corridor_backward=backward,
            corridor_forward=forward,
            r4_span0_quota=self.config.r4_span0_quota,
            r4_span1_quota=self.config.r4_span1_quota,
            r4_span2_quota=self.config.r4_span2_quota,
        )

    def escalation_policies(self, policy: RepairPolicy) -> tuple[RepairPolicy, ...]:
        if policy.scope is NeighborhoodScope.GLOBAL:
            return (policy,)
        scopes = (
            (NeighborhoodScope.LOCAL, self.config.local_origin_radius,
             self.config.local_max_recovery_span),
            (NeighborhoodScope.FORWARD, self.config.forward_origin_radius,
             self.config.forward_max_recovery_span),
            (NeighborhoodScope.GLOBAL, None, None),
        )
        start = next(
            index for index, (scope, _, _) in enumerate(scopes)
            if scope is policy.scope
        )
        selected = scopes[start:] if policy.allow_global_fallback else scopes[start:start + 1]
        return tuple(
            replace(
                policy,
                scope=scope,
                origin_radius=radius,
                max_recovery_span=max_span,
            )
            for scope, radius, max_span in selected
        )

    @staticmethod
    def release_scale(required_release_tasks: int) -> Literal[
        "micro", "medium", "macro"
    ]:
        """Classify scale from the transformation's actual whole-sortie union."""
        if required_release_tasks <= 0:
            raise ValueError("required release size must be positive")
        if required_release_tasks <= 6:
            return "micro"
        if required_release_tasks <= 12:
            return "medium"
        return "macro"

    def policy_for_release_size(
        self,
        policy: RepairPolicy,
        required_release_tasks: int,
    ) -> RepairPolicy:
        scale = self.release_scale(required_release_tasks)
        if scale == "micro":
            return replace(
                policy,
                scope=NeighborhoodScope.LOCAL,
                origin_radius=self.config.local_origin_radius,
                max_recovery_span=self.config.local_max_recovery_span,
                destroy_scale=scale,
                corridor_backward=0,
                corridor_forward=self.config.micro_corridor_forward,
            )
        if scale == "medium":
            return replace(
                policy,
                scope=NeighborhoodScope.FORWARD,
                origin_radius=self.config.forward_origin_radius,
                max_recovery_span=self.config.forward_max_recovery_span,
                destroy_scale=scale,
                corridor_backward=self.config.medium_corridor_backward,
                corridor_forward=self.config.medium_corridor_forward,
            )
        return replace(
            policy,
            scope=NeighborhoodScope.GLOBAL,
            origin_radius=None,
            max_recovery_span=None,
            destroy_scale=scale,
        )

    @staticmethod
    def corridor_for(
        anchor_position: int,
        support_count: int,
        policy: RepairPolicy,
    ) -> RouteCorridor:
        if support_count <= 0:
            raise ValueError("support_count must be positive")
        if not 0 <= anchor_position < support_count:
            raise ValueError("anchor position is outside the support route")
        if policy.scope is NeighborhoodScope.GLOBAL or policy.destroy_scale in {
            "legacy", "macro"
        }:
            return RouteCorridor(0, support_count - 1)
        return RouteCorridor(
            max(0, anchor_position - policy.corridor_backward),
            min(support_count - 1, anchor_position + policy.corridor_forward),
        )
