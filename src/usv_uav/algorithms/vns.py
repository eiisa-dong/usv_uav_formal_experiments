from __future__ import annotations

import random
from time import perf_counter

from usv_uav.algorithms.base import AlgorithmResult, ConvergencePoint, EvaluationBudget, Solver, SolverBudget
from usv_uav.algorithms.construction import (
    RepairTimeBudgetExceeded,
    construct_initial_solution,
)
from usv_uav.algorithms.heuristic_config import VNSConfig
from usv_uav.algorithms.operators import (
    task_relocate,
    task_swap,
    relocate_sortie,
    split_sortie,
    merge_sorties,
    swap_sorties,
)
from usv_uav.preprocessing.sortie_index import build_sortie_index


class VNSSolver(Solver):
    name = "vns"

    def __init__(self, config: VNSConfig | None = None) -> None:
        self.config = config or VNSConfig.default()
        self.neighbourhoods = (
            ("N1_task_relocate", task_relocate),
            ("N2_task_swap", task_swap),
            ("N3_sortie_relocate", relocate_sortie),
            ("N4_sortie_swap", swap_sorties),
            ("N5_split", split_sortie),
            ("N6_merge", merge_sorties),
        )

    def solve(self, instance, evaluator, seed, budget) -> AlgorithmResult:
        rng = random.Random(seed)
        limiter = EvaluationBudget(evaluator, budget)
        index = build_sortie_index(instance.sortie_pool, instance.usv_route[1:-1])
        current_solution = construct_initial_solution(instance, index)
        current_evaluation = limiter.evaluate(current_solution)
        if not current_evaluation.feasible:
            raise RuntimeError(f"VNS initial solution infeasible: {current_evaluation.violations}")
        best_solution = current_solution.copy()
        best_evaluation = current_evaluation
        time_to_best = limiter.elapsed_sec
        eval_to_best = limiter.evaluations
        convergence = [ConvergencePoint(0, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective)]
        stats = {name: {"calls": 0.0, "improvements": 0.0} for name, _ in self.neighbourhoods}
        neighbourhood = 0
        iteration = 0
        while limiter.available(iteration):
            name, operator = self.neighbourhoods[neighbourhood]
            improved = False
            for _ in range(self.config.neighborhood_attempts):
                if not limiter.available(iteration):
                    break
                stats[name]["calls"] += 1
                neighbor_stage = f"local_search:{name}"
                neighbor_deadline = lambda: limiter.deadline_expired(
                    neighbor_stage
                )
                started_neighbor = perf_counter()
                try:
                    candidate = operator(
                        instance,
                        current_solution,
                        index,
                        rng,
                        deadline_expired=neighbor_deadline,
                    )
                except RepairTimeBudgetExceeded:
                    candidate = None
                neighbor_elapsed = perf_counter() - started_neighbor
                limiter.observe_stage(neighbor_stage, neighbor_elapsed)
                if limiter.deadline.expired():
                    limiter.mark_deadline_abort(neighbor_stage)
                    candidate = None
                if limiter.deadline_abort_stage is not None:
                    break
                if candidate is None:
                    continue
                candidate_evaluation = limiter.try_evaluate(candidate, iteration)
                if candidate_evaluation is None:
                    break
                if candidate_evaluation.feasible and candidate_evaluation.objective < current_evaluation.objective - 1e-9:
                    current_solution, current_evaluation = candidate, candidate_evaluation
                    stats[name]["improvements"] += 1
                    improved = True
                    if candidate_evaluation.objective < best_evaluation.objective - 1e-9:
                        best_solution = candidate.copy()
                        best_evaluation = candidate_evaluation
                        time_to_best = limiter.elapsed_sec
                        eval_to_best = limiter.evaluations
                        convergence.append(ConvergencePoint(iteration, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective))
                    break
            neighbourhood = 0 if improved else (neighbourhood + 1) % len(self.neighbourhoods)
            iteration += 1
        return AlgorithmResult(
            algorithm=self.name,
            best_solution=best_solution,
            best_evaluation=best_evaluation,
            wall_time_sec=limiter.elapsed_sec,
            evaluations=limiter.evaluations,
            time_to_best_sec=time_to_best,
            eval_to_best=eval_to_best,
            convergence=tuple(convergence),
            operator_stats=stats,
            metadata={
                "seed": seed,
                "iterations": iteration,
                "adaptive_weights": False,
                **limiter.termination_metadata(iteration),
            },
        )
