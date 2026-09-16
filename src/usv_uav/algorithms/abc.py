from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter

from usv_uav.algorithms.base import AlgorithmResult, ConvergencePoint, EvaluationBudget, Solver
from usv_uav.algorithms.construction import (
    RepairTimeBudgetExceeded,
    construct_initial_solution,
)
from usv_uav.algorithms.heuristic_config import ABCConfig
from usv_uav.algorithms.operators import generic_random_neighbor
from usv_uav.preprocessing.sortie_index import build_sortie_index


@dataclass(slots=True)
class _FoodSource:
    solution: object
    evaluation: object
    trials: int = 0


class ABCSolver(Solver):
    name = "abc"

    def __init__(self, config: ABCConfig | None = None) -> None:
        self.config = config or ABCConfig.default()

    def solve(self, instance, evaluator, seed, budget) -> AlgorithmResult:
        rng = random.Random(seed)
        limiter = EvaluationBudget(evaluator, budget)
        index = build_sortie_index(instance.sortie_pool, instance.usv_route[1:-1])
        initial = construct_initial_solution(instance, index)
        sources: list[_FoodSource] = []

        def create(base, iteration: int = 0) -> _FoodSource | None:
            if not limiter.available():
                return None
            neighbor_stage = "repair:generic_neighbor"
            neighbor_deadline = lambda: limiter.deadline_expired(neighbor_stage)
            started_neighbor = perf_counter()
            try:
                solution = generic_random_neighbor(
                    instance,
                    base,
                    index,
                    rng,
                    deadline_expired=neighbor_deadline,
                )
            except RepairTimeBudgetExceeded:
                solution = None
            except RuntimeError:
                solution = base.copy()
            neighbor_elapsed = perf_counter() - started_neighbor
            limiter.observe_stage(neighbor_stage, neighbor_elapsed)
            if limiter.deadline.expired():
                limiter.mark_deadline_abort(neighbor_stage)
                solution = None
            if solution is None:
                return None
            evaluation = limiter.try_evaluate(solution, iteration)
            if evaluation is None:
                return None
            return _FoodSource(solution, evaluation) if evaluation.feasible else None

        initial_evaluation = limiter.evaluate(initial)
        if not initial_evaluation.feasible:
            raise RuntimeError("ABC initial solution is infeasible")
        sources.append(_FoodSource(initial, initial_evaluation))
        while len(sources) < self.config.food_sources and limiter.available():
            source = create(initial)
            if source is not None:
                sources.append(source)
        best = min(sources, key=lambda source: source.evaluation.objective)
        best_solution = best.solution.copy()
        best_evaluation = best.evaluation
        time_to_best = limiter.elapsed_sec
        eval_to_best = limiter.evaluations
        convergence = [ConvergencePoint(0, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective)]
        cycle = 0
        scout_count = 0

        def improve(source: _FoodSource) -> None:
            nonlocal best_solution, best_evaluation, time_to_best, eval_to_best
            if not limiter.available(cycle):
                return
            neighbor_stage = "repair:generic_neighbor"
            neighbor_deadline = lambda: limiter.deadline_expired(neighbor_stage)
            started_neighbor = perf_counter()
            try:
                candidate = generic_random_neighbor(
                    instance,
                    source.solution,
                    index,
                    rng,
                    deadline_expired=neighbor_deadline,
                )
            except RepairTimeBudgetExceeded:
                candidate = None
            except RuntimeError:
                source.trials += 1
                candidate = None
            neighbor_elapsed = perf_counter() - started_neighbor
            limiter.observe_stage(neighbor_stage, neighbor_elapsed)
            if limiter.deadline.expired():
                limiter.mark_deadline_abort(neighbor_stage)
                candidate = None
            if candidate is None:
                return
            evaluation = limiter.try_evaluate(candidate, cycle)
            if evaluation is None:
                return
            if evaluation.feasible and evaluation.objective < source.evaluation.objective - 1e-9:
                source.solution, source.evaluation, source.trials = candidate, evaluation, 0
                if evaluation.objective < best_evaluation.objective - 1e-9:
                    best_solution = candidate.copy()
                    best_evaluation = evaluation
                    time_to_best = limiter.elapsed_sec
                    eval_to_best = limiter.evaluations
                    convergence.append(ConvergencePoint(cycle, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective))
            else:
                source.trials += 1

        while limiter.available(cycle):
            cycle += 1
            for source in sources:  # employed bees
                improve(source)
            if not limiter.available(cycle):
                break
            fitness = [1.0 / (1.0 + source.evaluation.objective) for source in sources]
            for _ in range(len(sources)):  # onlookers
                source = rng.choices(sources, weights=fitness, k=1)[0]
                improve(source)
                if not limiter.available(cycle):
                    break
            for position, source in enumerate(list(sources)):  # scouts
                if source.trials < self.config.scout_limit or not limiter.available(cycle):
                    continue
                replacement = create(initial, cycle)
                if replacement is not None:
                    sources[position] = replacement
                    scout_count += 1
        return AlgorithmResult(
            algorithm=self.name,
            best_solution=best_solution,
            best_evaluation=best_evaluation,
            wall_time_sec=limiter.elapsed_sec,
            evaluations=limiter.evaluations,
            time_to_best_sec=time_to_best,
            eval_to_best=eval_to_best,
            convergence=tuple(convergence),
            metadata={
                "seed": seed,
                "cycles": cycle,
                "food_sources": len(sources),
                "scouts": scout_count,
                **limiter.termination_metadata(cycle),
            },
        )
