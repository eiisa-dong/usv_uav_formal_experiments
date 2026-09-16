from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter
from typing import Callable

from usv_uav.algorithms.base import AlgorithmResult, ConvergencePoint, EvaluationBudget, Solver
from usv_uav.algorithms.construction import (
    RepairTimeBudgetExceeded,
    construct_initial_solution,
)
from usv_uav.algorithms.heuristic_config import GAConfig
from usv_uav.algorithms.operators import generic_random_neighbor
from usv_uav.core.solution import Solution
from usv_uav.preprocessing.sortie_index import build_sortie_index


SEPARATOR = -1


@dataclass(slots=True)
class _Individual:
    chromosome: tuple[int, ...]
    solution: Solution
    evaluation: object


def encode_solution(solution: Solution, uav_count: int) -> tuple[int, ...]:
    genes: list[int] = []
    for uav_id in range(uav_count):
        if uav_id:
            genes.append(SEPARATOR)
        genes.extend(solution.uav_sequences.get(uav_id, ()))
    return tuple(genes)


def decode_chromosome_preserving_segments(
    chromosome,
    instance,
    index,
    deadline_expired: Callable[[], bool] | None = None,
) -> Solution:
    """Repair exact coverage while retaining crossover UAV segments and order."""
    segments = {uav_id: [] for uav_id in range(instance.uav_count)}
    owner = 0
    occupied: set[int] = set()
    preferred_owner: dict[int, int] = {}
    route_position = {
        support_id: position for position, support_id in enumerate(instance.usv_route[1:-1])
    }
    for gene in chromosome:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        if gene == SEPARATOR:
            owner = min(instance.uav_count - 1, owner + 1)
            continue
        if gene not in index.sortie_by_id:
            continue
        sortie = index.sortie_by_id[gene]
        tasks = set(sortie.task_sequence)
        for task_id in tasks:
            preferred_owner.setdefault(task_id, owner)
        if tasks & occupied:
            continue
        previous = segments[owner][-1] if segments[owner] else None
        if previous is not None:
            previous_recovery = route_position[index.sortie_by_id[previous].recovery_support]
            if previous_recovery > route_position[sortie.origin_support]:
                continue
        segments[owner].append(gene)
        occupied.update(tasks)

    missing = set(task.id for task in instance.tasks) - occupied

    def insertion_options(sortie_id: int) -> list[tuple[int, int]]:
        sortie = index.sortie_by_id[sortie_id]
        origin = route_position[sortie.origin_support]
        recovery = route_position[sortie.recovery_support]
        options: list[tuple[int, int]] = []
        for uav_id in range(instance.uav_count):
            sequence = segments[uav_id]
            for position in range(len(sequence) + 1):
                before = index.sortie_by_id[sequence[position - 1]] if position else None
                after = index.sortie_by_id[sequence[position]] if position < len(sequence) else None
                if before is not None and route_position[before.recovery_support] > origin:
                    continue
                if after is not None and recovery > route_position[after.origin_support]:
                    continue
                options.append((uav_id, position))
        return options

    while missing:
        if deadline_expired is not None and deadline_expired():
            raise RepairTimeBudgetExceeded
        anchor = min(missing)
        candidates = [
            index.sortie_by_id[identifier]
            for identifier in index.sorties_by_task[anchor]
            if set(index.sortie_by_id[identifier].task_sequence) <= missing
        ]
        ranked = sorted(
            candidates,
            key=lambda value: (
                -len(value.task_sequence), value.nominal_duration_min, value.id,
            ),
        )
        chosen = None
        chosen_option = None
        preferred = preferred_owner.get(anchor)
        for candidate in ranked:
            options = insertion_options(candidate.id)
            if preferred is not None:
                options.sort(key=lambda value: (value[0] != preferred, len(segments[value[0]]), value))
            else:
                options.sort(key=lambda value: (len(segments[value[0]]), value))
            if options:
                chosen, chosen_option = candidate, options[0]
                break
        if chosen is None or chosen_option is None:
            raise RuntimeError(f"GA chromosome repair cannot insert task {anchor}")
        uav_id, position = chosen_option
        segments[uav_id].insert(position, chosen.id)
        occupied.update(chosen.task_sequence)
        missing.difference_update(chosen.task_sequence)
    return Solution(segments)


repair_chromosome = decode_chromosome_preserving_segments


class GASolver(Solver):
    name = "ga"

    def __init__(self, config: GAConfig | None = None) -> None:
        self.config = config or GAConfig.default()

    def solve(self, instance, evaluator, seed, budget) -> AlgorithmResult:
        rng = random.Random(seed)
        limiter = EvaluationBudget(evaluator, budget)
        index = build_sortie_index(instance.sortie_pool, instance.usv_route[1:-1])
        initial = construct_initial_solution(instance, index)
        population: list[_Individual] = []

        def add(solution: Solution) -> None:
            evaluation = limiter.try_evaluate(solution)
            if evaluation is None:
                return
            if evaluation.feasible:
                population.append(_Individual(encode_solution(solution, instance.uav_count), solution, evaluation))

        add(initial)
        while len(population) < self.config.population_size and limiter.available():
            neighbor_stage = "repair:generic_neighbor"
            neighbor_deadline = lambda: limiter.deadline_expired(neighbor_stage)
            started_neighbor = perf_counter()
            try:
                candidate = generic_random_neighbor(
                    instance,
                    initial,
                    index,
                    rng,
                    deadline_expired=neighbor_deadline,
                )
            except RepairTimeBudgetExceeded:
                candidate = None
            except RuntimeError:
                candidate = initial.copy()
            neighbor_elapsed = perf_counter() - started_neighbor
            limiter.observe_stage(neighbor_stage, neighbor_elapsed)
            if limiter.deadline.expired():
                limiter.mark_deadline_abort(neighbor_stage)
                candidate = None
            if candidate is None:
                break
            else:
                add(candidate)
        if not population:
            raise RuntimeError("GA could not construct a feasible population")
        best = min(population, key=lambda item: item.evaluation.objective)
        best_solution = best.solution.copy()
        best_evaluation = best.evaluation
        time_to_best = limiter.elapsed_sec
        eval_to_best = limiter.evaluations
        convergence = [ConvergencePoint(0, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective)]
        generation = 0

        def tournament() -> _Individual:
            choices = [rng.choice(population) for _ in range(min(self.config.tournament_size, len(population)))]
            return min(choices, key=lambda item: item.evaluation.objective)

        while limiter.available(generation):
            generation += 1
            population.sort(key=lambda item: item.evaluation.objective)
            next_population = population[: self.config.elitism]
            while len(next_population) < self.config.population_size and limiter.available(generation):
                first, second = tournament(), tournament()
                if rng.random() < self.config.crossover_rate:
                    limit = min(len(first.chromosome), len(second.chromosome))
                    cut = rng.randrange(limit + 1) if limit else 0
                    chromosome = (*first.chromosome[:cut], *second.chromosome[cut:])
                else:
                    chromosome = first.chromosome
                genes = list(chromosome)
                nonseparators = [position for position, value in enumerate(genes) if value != SEPARATOR]
                if len(nonseparators) >= 2 and rng.random() < self.config.mutation_rate:
                    a, b = rng.sample(nonseparators, 2)
                    genes[a], genes[b] = genes[b], genes[a]
                repair_stage = "repair:ga_chromosome"
                repair_deadline = lambda: limiter.deadline_expired(repair_stage)
                started_repair = perf_counter()
                try:
                    candidate = repair_chromosome(
                        tuple(genes),
                        instance,
                        index,
                        deadline_expired=repair_deadline,
                    )
                except RepairTimeBudgetExceeded:
                    candidate = None
                repair_elapsed = perf_counter() - started_repair
                limiter.observe_stage(repair_stage, repair_elapsed)
                if limiter.deadline.expired():
                    limiter.mark_deadline_abort(repair_stage)
                    candidate = None
                if candidate is None:
                    break
                evaluation = limiter.try_evaluate(candidate, generation)
                if evaluation is None:
                    break
                if evaluation.feasible:
                    next_population.append(_Individual(encode_solution(candidate, instance.uav_count), candidate, evaluation))
                    if evaluation.objective < best_evaluation.objective - 1e-9:
                        best_solution = candidate.copy()
                        best_evaluation = evaluation
                        time_to_best = limiter.elapsed_sec
                        eval_to_best = limiter.evaluations
                        convergence.append(ConvergencePoint(generation, limiter.evaluations, limiter.elapsed_sec, best_evaluation.objective))
            if next_population:
                population = next_population
            else:
                break
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
                **limiter.termination_metadata(generation),
                "generations": generation,
                "chromosome": "flattened_uav_sortie_sequence_with_-1_separators",
                "population_size": self.config.population_size,
            },
        )
