from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum
import heapq
from math import inf
from typing import Iterable

from usv_uav.config import (
    MODEL_LAUNCH_POLICIES,
    ChargingConfig,
    LaunchPolicy,
    ProjectConfig,
    UAVConfig,
)
from usv_uav.core.models import Instance, Sortie
from usv_uav.core.solution import Solution
from usv_uav.physics.energy import actual_sortie_state, has_soc_reserve
from usv_uav.scheduling.charging import (
    ChargingOperation,
    ChargingPolicy,
    FCFSChargingScheduler,
    FCFSMinimumRequiredCharging,
    FullChargeFCFS,
)


EPSILON = 1e-9


class EventType(str, Enum):
    USV_ARRIVE = "USV_ARRIVE"
    USV_DEPART = "USV_DEPART"
    UAV_LAUNCH = "UAV_LAUNCH"
    UAV_AIR_ARRIVE = "UAV_AIR_ARRIVE"
    UAV_RECOVER = "UAV_RECOVER"
    CHARGE_START = "CHARGE_START"
    CHARGE_FINISH = "CHARGE_FINISH"


EVENT_ORDER = {
    EventType.USV_ARRIVE: 0,
    EventType.UAV_AIR_ARRIVE: 1,
    EventType.UAV_RECOVER: 2,
    EventType.CHARGE_START: 3,
    EventType.CHARGE_FINISH: 4,
    EventType.UAV_LAUNCH: 5,
    EventType.USV_DEPART: 6,
}


@dataclass(frozen=True, slots=True)
class Event:
    time_min: float
    event_type: EventType
    usv_node: int | None = None
    uav_id: int | None = None
    sortie_id: int | None = None
    charger_id: int | None = None
    soc_before: float | None = None
    soc_after: float | None = None


@dataclass(frozen=True, slots=True)
class SortieExecution:
    uav_id: int
    sequence_index: int
    sortie_id: int
    ready_for_launch_time_min: float
    launch_time_min: float
    launch_delay_on_deck_min: float
    air_arrival_time_min: float
    recovery_time_min: float
    hover_min: float
    actual_duration_min: float
    soc_before_wh: float
    soc_after_wh: float
    actual_energy_wh: float
    usv_arrival_at_recovery_min: float
    uav_lateness_to_usv_min: float


@dataclass(frozen=True, slots=True)
class SupportDwellSummary:
    support_id: int
    arrival_time_min: float
    departure_time_min: float
    dwell_time_min: float
    critical_event: str
    event_contributors: tuple["SupportEventContributor", ...] = ()

    @property
    def critical_events(self) -> tuple["SupportEventContributor", ...]:
        """All departure contributors tied at the latest completion time."""
        if not self.event_contributors:
            return ()
        latest = max(event.completion_time_min for event in self.event_contributors)
        return tuple(
            event
            for event in self.event_contributors
            if abs(event.completion_time_min - latest) <= EPSILON
        )

    @property
    def joint_critical_dwell_gain_min(self) -> float:
        """Dwell removed by jointly releasing all tied critical sorties."""
        if not self.event_contributors:
            return 0.0
        values = sorted(
            {
                round(event.completion_time_min, 9)
                for event in self.event_contributors
            },
            reverse=True,
        )
        if len(values) < 2:
            return max(0.0, values[0] - self.arrival_time_min)
        return max(0.0, values[0] - values[1])

    @property
    def marginal_dwell_gain_min(self) -> float:
        """Gain from removing one critical sortie; ties individually yield zero."""
        if len(self.critical_sortie_ids) != 1:
            return 0.0
        return self.joint_critical_dwell_gain_min

    @property
    def critical_sortie_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    event.sortie_id
                    for event in self.critical_events
                    if event.sortie_id is not None
                }
            )
        )


@dataclass(frozen=True, slots=True)
class SupportEventContributor:
    event_id: str
    sortie_id: int | None
    uav_id: int | None
    completion_time_min: float
    event_type: str


@dataclass(frozen=True, slots=True)
class ViolationDetail:
    code: str
    uav_id: int | None = None
    sortie_id: int | None = None
    actual: float | None = None
    limit: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    feasible: bool

    # Authoritative objective: time inside the field support network.
    makespan_min: float
    field_start_time_min: float
    field_end_time_min: float
    field_makespan_min: float

    # UAV inspection activity span (diagnostic only).
    first_launch_time_min: float
    last_recovery_time_min: float
    inspection_span_min: float

    # Whole operation, measured from port departure at time zero.
    deployment_time_min: float
    return_time_min: float
    port_arrival_time_min: float
    operation_total_min: float
    usv_deployment_time_min: float
    usv_mission_sail_min: float
    usv_return_time_min: float
    usv_total_travel_time_min: float
    usv_dwell_min: float
    usv_recovery_barrier_min: float
    usv_launch_readiness_barrier_min: float
    launch_delay_on_deck_min: float
    mean_launch_delay_on_deck_min: float
    mean_forward_launch_delay_on_deck_min: float
    uav_hover_min: float
    flight_time_min: float
    inspection_time_min: float
    flight_energy_wh: float
    inspection_energy_wh: float
    hover_energy_wh: float
    charging_wait_min: float
    charging_time_min: float
    charging_energy_wh: float
    charger_utilization: float
    operational_charger_utilization: float
    sortie_count: int
    avg_tasks_per_sortie: float
    same_point_recovery_ratio: float
    different_recovery_ratio: float
    violations: tuple[str, ...]
    violation_details: tuple[ViolationDetail, ...]
    event_log: tuple[Event, ...]
    executions: tuple[SortieExecution, ...]
    support_dwell_summaries: tuple[SupportDwellSummary, ...]
    charging_operations: tuple[ChargingOperation, ...]

    @property
    def objective(self) -> float:
        return self.makespan_min if self.feasible else inf

    @property
    def total_mission_time_min(self) -> float:
        """Port-to-port time; post-return battery restoration is not included."""
        return self.deployment_time_min + self.field_makespan_min + self.return_time_min

    @property
    def inspection_makespan_min(self) -> float:
        """Compatibility alias for the former inspection-span metric."""
        return self.inspection_span_min

    @property
    def usv_wait_min(self) -> float:
        """Compatibility alias; the quantity is support dwell, not pure waiting."""
        return self.usv_dwell_min

    @property
    def usv_travel_time_min(self) -> float:
        return self.usv_total_travel_time_min


@dataclass(slots=True)
class _ActiveSortie:
    sequence_index: int
    sortie: Sortie
    ready_for_launch_time: float
    launch_time: float
    air_arrival_time: float
    soc_before: float


class Evaluator:
    """The single authoritative structural solution decoder and evaluator."""

    def __init__(
        self,
        instance: Instance,
        *,
        uav: UAVConfig,
        usv_speed_km_min: float,
        charging: ChargingConfig,
        charging_policy: ChargingPolicy | None = None,
        model_semantics_version: str = "2.1",
        launch_policy: LaunchPolicy = "immediate",
    ) -> None:
        self.instance = instance
        self.uav = uav
        self.usv_speed_km_min = usv_speed_km_min
        self.charging = charging
        if launch_policy not in MODEL_LAUNCH_POLICIES.values():
            raise ValueError(f"unknown UAV launch policy {launch_policy!r}")
        expected_policy = MODEL_LAUNCH_POLICIES.get(model_semantics_version)
        if expected_policy is None:
            raise ValueError(
                f"unknown model semantics version {model_semantics_version!r}"
            )
        if launch_policy != expected_policy:
            raise ValueError(
                "model semantics version and UAV launch policy disagree: "
                f"{model_semantics_version} requires {expected_policy}"
            )
        self.model_semantics_version = model_semantics_version
        self.launch_policy = launch_policy
        if charging_policy is not None:
            self.charging_policy = charging_policy
        elif charging.policy == "full_charge_fcfs":
            self.charging_policy = FullChargeFCFS()
        else:
            self.charging_policy = FCFSMinimumRequiredCharging()
        self.sortie_by_id = {sortie.id: sortie for sortie in instance.sortie_pool}
        self.task_ids = tuple(task.id for task in instance.tasks)
        self.route_position = {
            support_id: position for position, support_id in enumerate(instance.usv_route[1:-1])
        }
        self.distance_index = {0: 0, **{
            support_id: index + 1 for index, support_id in enumerate(instance.selected_supports)
        }}
        self.evaluation_count = 0
        self.feasible_evaluation_count = 0
        self.violation_code_counts: dict[str, int] = {}

    def fork(self) -> "Evaluator":
        """Create an evaluator with independent counters for a worker thread."""

        return Evaluator(
            self.instance,
            uav=self.uav,
            usv_speed_km_min=self.usv_speed_km_min,
            charging=self.charging,
            charging_policy=copy(self.charging_policy),
            model_semantics_version=self.model_semantics_version,
            launch_policy=self.launch_policy,
        )

    def commit_external_results(
        self,
        results: Iterable[EvaluationResult],
    ) -> None:
        """Commit worker-produced results to authoritative counters in order."""

        for result in results:
            self.evaluation_count += 1
            self._record_result(result)

    def _record_result(self, result: EvaluationResult) -> EvaluationResult:
        if result.feasible:
            self.feasible_evaluation_count += 1
        else:
            for detail in result.violation_details:
                self.violation_code_counts[detail.code] = self.violation_code_counts.get(detail.code, 0) + 1
        return result

    @classmethod
    def from_config(
        cls,
        instance: Instance,
        config: ProjectConfig,
        *,
        charging_policy: ChargingPolicy | None = None,
    ) -> "Evaluator":
        return cls(
            instance,
            uav=config.physical.uav,
            usv_speed_km_min=config.physical.usv.speed_km_min,
            charging=config.charging,
            charging_policy=charging_policy,
            model_semantics_version=config.physical.model_semantics_version,
            launch_policy=config.physical.launch_policy,
        )

    def _travel_time(self, origin: int, destination: int) -> float:
        first = self.distance_index[origin]
        second = self.distance_index[destination]
        return float(self.instance.safe_distance_matrix[first, second]) / self.usv_speed_km_min

    def _forward_movement_time(self, origin: int, recovery: int) -> float:
        origin_position = self.route_position[origin]
        recovery_position = self.route_position[recovery]
        if recovery_position <= origin_position:
            raise ValueError("forward movement time requires a later recovery support")
        route_nodes = self.instance.usv_route[1:-1]
        return sum(
            self._travel_time(route_nodes[position], route_nodes[position + 1])
            for position in range(origin_position, recovery_position)
        )

    def _structural_violations(
        self,
        solution: Solution,
        *,
        require_complete: bool,
    ) -> list[str]:
        violations: list[str] = []
        invalid_uavs = sorted(set(solution.uav_sequences) - set(range(self.instance.uav_count)))
        if invalid_uavs:
            violations.append(f"UNKNOWN_UAV:{invalid_uavs}")
        task_occurrences = {task_id: 0 for task_id in self.task_ids}
        for uav_id in range(self.instance.uav_count):
            previous_recovery_position = -1
            for sortie_id in solution.uav_sequences.get(uav_id, ()):
                sortie = self.sortie_by_id.get(sortie_id)
                if sortie is None:
                    violations.append(f"UNKNOWN_SORTIE:uav={uav_id},sortie={sortie_id}")
                    continue
                origin_position = self.route_position.get(sortie.origin_support)
                recovery_position = self.route_position.get(sortie.recovery_support)
                if origin_position is None or recovery_position is None:
                    violations.append(f"UNKNOWN_SUPPORT:sortie={sortie.id}")
                    continue
                if origin_position > recovery_position:
                    violations.append(f"WRONG_SUPPORT_ORDER:sortie={sortie.id}")
                if origin_position < previous_recovery_position:
                    violations.append(f"UAV_SEQUENCE_PRECEDENCE:uav={uav_id},sortie={sortie.id}")
                previous_recovery_position = recovery_position
                for task_id in sortie.task_sequence:
                    if task_id not in task_occurrences:
                        violations.append(f"UNKNOWN_TASK:sortie={sortie.id},task={task_id}")
                    else:
                        task_occurrences[task_id] += 1
        duplicates = tuple(task_id for task_id, count in task_occurrences.items() if count > 1)
        if duplicates:
            violations.append(f"TASK_DUPLICATE:{duplicates}")
        if require_complete:
            uncovered = tuple(task_id for task_id, count in task_occurrences.items() if count == 0)
            if uncovered:
                violations.append(f"TASK_UNCOVERED:{uncovered}")
        return violations

    def _empty_result(self, violations: Iterable[str]) -> EvaluationResult:
        return EvaluationResult(
            feasible=False,
            makespan_min=inf,
            field_start_time_min=inf,
            field_end_time_min=-inf,
            field_makespan_min=inf,
            first_launch_time_min=inf,
            last_recovery_time_min=-inf,
            inspection_span_min=inf,
            deployment_time_min=0.0,
            return_time_min=0.0,
            port_arrival_time_min=inf,
            operation_total_min=inf,
            usv_deployment_time_min=0.0,
            usv_mission_sail_min=0.0,
            usv_return_time_min=0.0,
            usv_total_travel_time_min=0.0,
            usv_dwell_min=0.0,
            usv_recovery_barrier_min=0.0,
            usv_launch_readiness_barrier_min=0.0,
            launch_delay_on_deck_min=0.0,
            mean_launch_delay_on_deck_min=0.0,
            mean_forward_launch_delay_on_deck_min=0.0,
            uav_hover_min=0.0,
            flight_time_min=0.0,
            inspection_time_min=0.0,
            flight_energy_wh=0.0,
            inspection_energy_wh=0.0,
            hover_energy_wh=0.0,
            charging_wait_min=0.0,
            charging_time_min=0.0,
            charging_energy_wh=0.0,
            charger_utilization=0.0,
            operational_charger_utilization=0.0,
            sortie_count=0,
            avg_tasks_per_sortie=0.0,
            same_point_recovery_ratio=0.0,
            different_recovery_ratio=0.0,
            violations=tuple(violations),
            violation_details=tuple(
                ViolationDetail(str(value).split(":", 1)[0]) for value in violations
            ),
            event_log=(),
            executions=(),
            support_dwell_summaries=(),
            charging_operations=(),
        )

    def evaluate(self, solution: Solution, *, require_complete: bool = True) -> EvaluationResult:
        self.evaluation_count += 1
        violations = self._structural_violations(solution, require_complete=require_complete)
        if violations:
            return self._record_result(self._empty_result(violations))

        sequences = {
            uav_id: [self.sortie_by_id[sortie_id] for sortie_id in solution.uav_sequences.get(uav_id, ())]
            for uav_id in range(self.instance.uav_count)
        }
        next_index = [0] * self.instance.uav_count
        soc = [self.uav.battery_wh] * self.instance.uav_count
        onboard = [True] * self.instance.uav_count
        ready_time = [0.0] * self.instance.uav_count
        active: dict[int, _ActiveSortie] = {}
        scheduler = FCFSChargingScheduler(self.charging)
        violation_details: list[ViolationDetail] = []
        events: list[Event] = [Event(0.0, EventType.USV_DEPART, usv_node=0)]
        executions: list[SortieExecution] = []
        hover_total = 0.0
        hover_energy_total = 0.0
        usv_wait_total = 0.0
        recovery_barrier_total = 0.0
        launch_barrier_total = 0.0
        launch_delay_total = 0.0
        forward_launch_delay_total = 0.0
        forward_launch_count = 0
        usv_travel_total = 0.0
        usv_deployment_travel = 0.0
        usv_mission_sail = 0.0
        first_launch = inf
        last_recovery = -inf
        field_start = inf
        field_end = -inf
        support_dwell_summaries: list[SupportDwellSummary] = []
        previous_node = 0
        previous_departure = 0.0

        route_nodes = self.instance.usv_route[1:-1]
        for route_index, node in enumerate(route_nodes):
            travel = self._travel_time(previous_node, node)
            usv_travel_total += travel
            if route_index == 0:
                usv_deployment_travel += travel
            else:
                usv_mission_sail += travel
            arrival = previous_departure + travel
            if route_index == 0:
                field_start = arrival
            events.append(Event(arrival, EventType.USV_ARRIVE, usv_node=node))
            local_end = arrival
            local_recovery_barrier = 0.0
            local_launch_barrier = 0.0
            local_critical_event = EventType.USV_ARRIVE.value
            local_event_contributors: list[SupportEventContributor] = [
                SupportEventContributor(
                    event_id=f"support={node}:arrival",
                    sortie_id=None,
                    uav_id=None,
                    completion_time_min=arrival,
                    event_type=EventType.USV_ARRIVE.value,
                )
            ]
            # (time, priority, uav, serial, action); recovery wins ties.
            queue: list[tuple[float, int, int, int, str]] = []
            pending_forward_launches: set[int] = set()
            serial = 0

            def queue_next_launch(uav_id: int) -> None:
                nonlocal serial
                if not onboard[uav_id] or next_index[uav_id] >= len(sequences[uav_id]):
                    return
                next_sortie = sequences[uav_id][next_index[uav_id]]
                if next_sortie.origin_support != node:
                    return
                if (
                    self.launch_policy == "dwell_window_synchronized"
                    and next_sortie.recovery_support != node
                ):
                    pending_forward_launches.add(uav_id)
                    return
                heapq.heappush(
                    queue,
                    (max(arrival, ready_time[uav_id]), 1, uav_id, serial, "launch"),
                )
                serial += 1

            def launch_sortie(uav_id: int, requested_time: float) -> None:
                nonlocal first_launch, local_launch_barrier, serial
                nonlocal launch_delay_total, forward_launch_delay_total
                nonlocal forward_launch_count
                if not onboard[uav_id] or next_index[uav_id] >= len(sequences[uav_id]):
                    return
                sortie = sequences[uav_id][next_index[uav_id]]
                if sortie.origin_support != node:
                    return
                required_soc = sortie.nominal_energy_wh + self.uav.safety_soc_wh
                if soc[uav_id] + EPSILON < required_soc:
                    violations.append(
                        f"LAUNCH_NOMINAL_SOC:uav={uav_id},sortie={sortie.id},soc={soc[uav_id]:.6f},required={required_soc:.6f}"
                    )
                    violation_details.append(ViolationDetail(
                        "LAUNCH_NOMINAL_SOC", uav_id, sortie.id, soc[uav_id], required_soc,
                    ))
                earliest_launch = max(arrival, ready_time[uav_id])
                launch_time = max(requested_time, earliest_launch)
                if not (
                    self.launch_policy == "dwell_window_synchronized"
                    and sortie.recovery_support != node
                ):
                    local_event_contributors.append(SupportEventContributor(
                        event_id=(
                            f"support={node}:launch:sortie={sortie.id}:uav={uav_id}"
                        ),
                        sortie_id=sortie.id,
                        uav_id=uav_id,
                        completion_time_min=launch_time,
                        event_type=EventType.UAV_LAUNCH.value,
                    ))
                launch_delay = max(0.0, launch_time - earliest_launch)
                # The readiness barrier is physical waiting, not deliberate
                # synchronization inside an already-required dwell window.
                local_launch_barrier = max(
                    local_launch_barrier,
                    earliest_launch - arrival,
                )
                launch_delay_total += launch_delay
                if sortie.recovery_support != node:
                    forward_launch_delay_total += launch_delay
                    forward_launch_count += 1
                first_launch = min(first_launch, launch_time)
                air_arrival = launch_time + sortie.nominal_duration_min
                events.append(Event(
                    launch_time,
                    EventType.UAV_LAUNCH,
                    usv_node=node,
                    uav_id=uav_id,
                    sortie_id=sortie.id,
                    soc_before=soc[uav_id],
                    soc_after=soc[uav_id],
                ))
                events.append(Event(
                    air_arrival,
                    EventType.UAV_AIR_ARRIVE,
                    usv_node=sortie.recovery_support,
                    uav_id=uav_id,
                    sortie_id=sortie.id,
                ))
                active[uav_id] = _ActiveSortie(
                    sequence_index=next_index[uav_id],
                    sortie=sortie,
                    ready_for_launch_time=earliest_launch,
                    launch_time=launch_time,
                    air_arrival_time=air_arrival,
                    soc_before=soc[uav_id],
                )
                next_index[uav_id] += 1
                onboard[uav_id] = False
                if sortie.recovery_support == node:
                    heapq.heappush(
                        queue,
                        (air_arrival, 0, uav_id, serial, "recover"),
                    )
                    serial += 1

            for uav_id, execution in sorted(active.items()):
                if execution.sortie.recovery_support == node:
                    heapq.heappush(queue, (max(arrival, execution.air_arrival_time), 0, uav_id, serial, "recover"))
                    serial += 1
            for uav_id in range(self.instance.uav_count):
                queue_next_launch(uav_id)

            while queue:
                event_time, _, uav_id, _, action = heapq.heappop(queue)
                if event_time > local_end + EPSILON:
                    local_critical_event = (
                        EventType.UAV_RECOVER.value
                        if action == "recover"
                        else EventType.UAV_LAUNCH.value
                    )
                local_end = max(local_end, event_time)
                if action == "launch":
                    launch_sortie(uav_id, event_time)
                else:
                    execution = active.get(uav_id)
                    if execution is None or execution.sortie.recovery_support != node:
                        continue
                    sortie = execution.sortie
                    recovery_time = max(event_time, arrival, execution.air_arrival_time)
                    local_recovery_barrier = max(local_recovery_barrier, recovery_time - arrival)
                    hover = max(0.0, arrival - execution.air_arrival_time)
                    state = actual_sortie_state(
                        sortie, hover_min=hover, soc_before_wh=execution.soc_before, uav=self.uav,
                    )
                    hover_energy = state.energy_wh - sortie.nominal_energy_wh
                    actual_energy = state.energy_wh
                    soc_after = state.soc_after_wh
                    actual_duration = state.duration_min
                    hover_total += hover
                    hover_energy_total += hover_energy
                    soc[uav_id] = soc_after
                    onboard[uav_id] = True
                    del active[uav_id]
                    events.append(Event(
                        recovery_time,
                        EventType.UAV_RECOVER,
                        usv_node=node,
                        uav_id=uav_id,
                        sortie_id=sortie.id,
                        soc_before=execution.soc_before,
                        soc_after=soc_after,
                    ))
                    local_event_contributors.append(SupportEventContributor(
                        event_id=(
                            f"support={node}:recovery:sortie={sortie.id}:uav={uav_id}"
                        ),
                        sortie_id=sortie.id,
                        uav_id=uav_id,
                        completion_time_min=recovery_time,
                        event_type=EventType.UAV_RECOVER.value,
                    ))
                    last_recovery = max(last_recovery, recovery_time)
                    executions.append(SortieExecution(
                        uav_id=uav_id,
                        sequence_index=execution.sequence_index,
                        sortie_id=sortie.id,
                        ready_for_launch_time_min=execution.ready_for_launch_time,
                        launch_time_min=execution.launch_time,
                        launch_delay_on_deck_min=(
                            execution.launch_time - execution.ready_for_launch_time
                        ),
                        air_arrival_time_min=execution.air_arrival_time,
                        recovery_time_min=recovery_time,
                        hover_min=hover,
                        actual_duration_min=actual_duration,
                        soc_before_wh=execution.soc_before,
                        soc_after_wh=soc_after,
                        actual_energy_wh=actual_energy,
                        usv_arrival_at_recovery_min=arrival,
                        uav_lateness_to_usv_min=max(0.0, execution.air_arrival_time - arrival),
                    ))
                    if actual_duration > self.uav.max_sortie_duration_min + EPSILON:
                        violations.append(
                            f"HOVER_SORTIE_TIME:sortie={sortie.id},actual={actual_duration:.6f},limit={self.uav.max_sortie_duration_min:.6f}"
                        )
                        violation_details.append(ViolationDetail(
                            "HOVER_SORTIE_TIME", uav_id, sortie.id, actual_duration, self.uav.max_sortie_duration_min,
                        ))
                    if not has_soc_reserve(soc_after, self.uav, tolerance=EPSILON):
                        violations.append(
                            f"HOVER_SOC_RESERVE:sortie={sortie.id},soc={soc_after:.6f},reserve={self.uav.safety_soc_wh:.6f}"
                        )
                        violation_details.append(ViolationDetail(
                            "HOVER_SOC_RESERVE", uav_id, sortie.id, soc_after, self.uav.safety_soc_wh,
                        ))

                    next_sortie = (
                        sequences[uav_id][next_index[uav_id]]
                        if next_index[uav_id] < len(sequences[uav_id])
                        else None
                    )
                    target_soc = self.charging_policy.target_soc_wh(
                        soc_after,
                        next_sortie,
                        self.uav,
                        final_full_charge=self.charging.final_full_charge,
                    )
                    operation = scheduler.schedule(
                        uav_id=uav_id,
                        after_sortie_id=sortie.id,
                        request_support=node,
                        ready_time_min=recovery_time,
                        soc_before_wh=soc_after,
                        target_soc_wh=target_soc,
                    )
                    if operation is None:
                        ready_time[uav_id] = recovery_time
                    else:
                        soc[uav_id] = operation.soc_after_wh
                        ready_time[uav_id] = operation.finish_time_min
                        events.extend((
                            Event(
                                operation.start_time_min,
                                EventType.CHARGE_START,
                                usv_node=None,
                                uav_id=uav_id,
                                sortie_id=sortie.id,
                                charger_id=operation.charger_id,
                                soc_before=operation.soc_before_wh,
                                soc_after=operation.soc_before_wh,
                            ),
                            Event(
                                operation.finish_time_min,
                                EventType.CHARGE_FINISH,
                                usv_node=None,
                                uav_id=uav_id,
                                sortie_id=sortie.id,
                                charger_id=operation.charger_id,
                                soc_before=operation.soc_before_wh,
                                soc_after=operation.soc_after_wh,
                            ),
                        ))
                    if next_sortie is not None and next_sortie.origin_support == node:
                        queue_next_launch(uav_id)

            departure = local_end
            if (
                self.launch_policy == "dwell_window_synchronized"
                and pending_forward_launches
            ):
                valid_pending = tuple(
                    uav_id
                    for uav_id in sorted(pending_forward_launches)
                    if onboard[uav_id]
                    and next_index[uav_id] < len(sequences[uav_id])
                    and sequences[uav_id][next_index[uav_id]].origin_support == node
                    and sequences[uav_id][next_index[uav_id]].recovery_support != node
                )
                if valid_pending:
                    for uav_id in valid_pending:
                        sortie = sequences[uav_id][next_index[uav_id]]
                        readiness = max(arrival, ready_time[uav_id])
                        local_event_contributors.append(SupportEventContributor(
                            event_id=(
                                f"support={node}:forward-ready:"
                                f"sortie={sortie.id}:uav={uav_id}"
                            ),
                            sortie_id=sortie.id,
                            uav_id=uav_id,
                            completion_time_min=readiness,
                            event_type="FORWARD_LAUNCH_READY",
                        ))
                    latest_ready = max(
                        max(arrival, ready_time[uav_id])
                        for uav_id in valid_pending
                    )
                    if latest_ready > departure + EPSILON:
                        local_critical_event = "FORWARD_LAUNCH_READY"
                    departure = max(departure, latest_ready)
                    for uav_id in valid_pending:
                        sortie = sequences[uav_id][next_index[uav_id]]
                        earliest_launch = max(arrival, ready_time[uav_id])
                        ideal_launch = (
                            departure
                            + self._forward_movement_time(
                                node, sortie.recovery_support
                            )
                            - sortie.nominal_duration_min
                        )
                        synchronized_launch = min(
                            departure,
                            max(earliest_launch, ideal_launch),
                        )
                        launch_sortie(uav_id, synchronized_launch)
            usv_wait_total += max(0.0, departure - arrival)
            recovery_barrier_total += max(0.0, local_recovery_barrier)
            launch_barrier_total += max(0.0, local_launch_barrier)
            events.append(Event(departure, EventType.USV_DEPART, usv_node=node))
            support_dwell_summaries.append(SupportDwellSummary(
                support_id=node,
                arrival_time_min=arrival,
                departure_time_min=departure,
                dwell_time_min=max(0.0, departure - arrival),
                critical_event=local_critical_event,
                event_contributors=tuple(local_event_contributors),
            ))
            previous_node = node
            previous_departure = departure

        if route_nodes:
            field_end = previous_departure

        if active:
            violations.extend(f"UNRECOVERED_UAV:uav={uav_id}" for uav_id in sorted(active))
        unfinished = [
            uav_id for uav_id in range(self.instance.uav_count)
            if next_index[uav_id] != len(sequences[uav_id])
        ]
        if unfinished:
            violations.append(f"UNLAUNCHED_SORTIES:uavs={unfinished}")

        return_travel = self._travel_time(previous_node, 0)
        usv_travel_total += return_travel
        port_arrival_time = previous_departure + return_travel
        operational_completion = port_arrival_time
        if self.charging.final_full_charge and scheduler.operations:
            operational_completion = max(
                operational_completion,
                max(operation.finish_time_min for operation in scheduler.operations),
            )
        events.append(Event(port_arrival_time, EventType.USV_ARRIVE, usv_node=0))
        events.sort(key=lambda event: (
            event.time_min,
            EVENT_ORDER[event.event_type],
            -1 if event.uav_id is None else event.uav_id,
            -1 if event.sortie_id is None else event.sortie_id,
        ))

        sorties = [sortie for sequence in sequences.values() for sortie in sequence]
        flight_time = sum(sortie.flight_time_min for sortie in sorties)
        inspection_time = sum(sortie.inspection_time_min for sortie in sorties)
        flight_energy = flight_time * self.uav.flight_energy_wh_min
        inspection_energy = inspection_time * self.uav.inspection_energy_wh_min
        charging_time = sum(operation.duration_min for operation in scheduler.operations)
        charging_wait = sum(operation.waiting_min for operation in scheduler.operations)
        charging_energy = sum(operation.energy_wh for operation in scheduler.operations)
        task_count = sum(len(sortie.task_sequence) for sortie in sorties)
        sortie_count = len(sorties)
        different_recovery = sum(
            sortie.origin_support != sortie.recovery_support for sortie in sorties
        )
        same_point_recovery = sortie_count - different_recovery
        operational_utilization = (
            charging_time / (self.charging.chargers * operational_completion)
            if operational_completion > EPSILON
            else 0.0
        )
        field_charging_time = (
            sum(
                max(0.0, min(operation.finish_time_min, field_end) - max(operation.start_time_min, field_start))
                for operation in scheduler.operations
            )
            if field_start < inf and field_end > -inf
            else 0.0
        )
        field_makespan = field_end - field_start if field_start < inf and field_end > -inf else inf
        utilization = (
            field_charging_time / (self.charging.chargers * field_makespan)
            if field_makespan > EPSILON
            else 0.0
        )
        if first_launch == inf or last_recovery == -inf or field_makespan == inf:
            violations.append("NO_INSPECTION_EXECUTION")
            violation_details.append(ViolationDetail("NO_INSPECTION_EXECUTION"))
            inspection_span = inf
            deployment_time = 0.0
            return_time = 0.0
            operation_total = inf
        else:
            inspection_span = last_recovery - first_launch
            deployment_time = field_start
            return_time = return_travel
            operation_total = operational_completion
        return self._record_result(EvaluationResult(
            feasible=not violations,
            makespan_min=field_makespan,
            field_start_time_min=field_start,
            field_end_time_min=field_end,
            field_makespan_min=field_makespan,
            first_launch_time_min=first_launch,
            last_recovery_time_min=last_recovery,
            inspection_span_min=inspection_span,
            deployment_time_min=deployment_time,
            return_time_min=return_time,
            port_arrival_time_min=port_arrival_time,
            operation_total_min=operation_total,
            usv_deployment_time_min=usv_deployment_travel,
            usv_mission_sail_min=usv_mission_sail,
            usv_return_time_min=return_travel,
            usv_total_travel_time_min=usv_travel_total,
            usv_dwell_min=usv_wait_total,
            usv_recovery_barrier_min=recovery_barrier_total,
            usv_launch_readiness_barrier_min=launch_barrier_total,
            launch_delay_on_deck_min=launch_delay_total,
            mean_launch_delay_on_deck_min=(
                launch_delay_total / sortie_count if sortie_count else 0.0
            ),
            mean_forward_launch_delay_on_deck_min=(
                forward_launch_delay_total / forward_launch_count
                if forward_launch_count else 0.0
            ),
            uav_hover_min=hover_total,
            flight_time_min=flight_time,
            inspection_time_min=inspection_time,
            flight_energy_wh=flight_energy,
            inspection_energy_wh=inspection_energy,
            hover_energy_wh=hover_energy_total,
            charging_wait_min=charging_wait,
            charging_time_min=charging_time,
            charging_energy_wh=charging_energy,
            charger_utilization=utilization,
            operational_charger_utilization=operational_utilization,
            sortie_count=sortie_count,
            avg_tasks_per_sortie=task_count / sortie_count if sortie_count else 0.0,
            same_point_recovery_ratio=(
                same_point_recovery / sortie_count if sortie_count else 0.0
            ),
            different_recovery_ratio=different_recovery / sortie_count if sortie_count else 0.0,
            violations=tuple(violations),
            violation_details=tuple(violation_details),
            event_log=tuple(events),
            executions=tuple(sorted(executions, key=lambda item: (item.recovery_time_min, item.uav_id))),
            support_dwell_summaries=tuple(support_dwell_summaries),
            charging_operations=tuple(scheduler.operations),
        ))
