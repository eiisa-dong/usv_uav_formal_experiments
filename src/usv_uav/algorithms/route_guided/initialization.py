from __future__ import annotations

from typing import Any, Callable

from usv_uav.algorithms.alns_config import RouteGuidedInitializationConfig
from usv_uav.algorithms.base import EvaluationBudget
from usv_uav.core.models import Instance
from usv_uav.preprocessing.sortie_index import SortieIndex


def initialize_route_guided_solution(
    *,
    instance: Instance,
    index: SortieIndex,
    limiter: EvaluationBudget,
    config: RouteGuidedInitializationConfig,
    frozen_s5a1_initializer: Callable[
        [Instance, SortieIndex, EvaluationBudget, RouteGuidedInitializationConfig], Any
    ],
):
    """Use the proven S5A1 initializer unchanged and assert the V2 contract."""
    if not config.enabled or config.candidate_scoring != "synchronization_lexicographic":
        raise ValueError("RG-MS-ALNS V2 requires the frozen S5A1 initialization")
    outcome = frozen_s5a1_initializer(instance, index, limiter, config)
    if outcome.metadata.get("initialization_candidate_scoring") != config.candidate_scoring:
        raise RuntimeError("S5A1 initialization scoring contract was not preserved")
    return outcome
