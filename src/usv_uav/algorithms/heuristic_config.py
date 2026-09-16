from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def _load(name: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    with (root / "configs" / "algorithms" / name).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@dataclass(frozen=True, slots=True)
class VNSConfig:
    schema_version: str
    name: str
    neighborhood_attempts: int

    def __post_init__(self) -> None:
        if self.neighborhood_attempts <= 0:
            raise ValueError("neighborhood_attempts must be positive")

    @classmethod
    def default(cls) -> "VNSConfig":
        return cls(**_load("vns.yaml"))


@dataclass(frozen=True, slots=True)
class GAConfig:
    schema_version: str
    name: str
    population_size: int
    tournament_size: int
    crossover_rate: float
    mutation_rate: float
    elitism: int

    def __post_init__(self) -> None:
        if self.population_size < 2 or not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid GA population or tournament size")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("GA probabilities must be in [0, 1]")
        if not 1 <= self.elitism < self.population_size:
            raise ValueError("GA elitism must be within the population")

    @classmethod
    def default(cls) -> "GAConfig":
        return cls(**_load("ga.yaml"))


@dataclass(frozen=True, slots=True)
class ABCConfig:
    schema_version: str
    name: str
    food_sources: int
    scout_limit: int

    def __post_init__(self) -> None:
        if self.food_sources < 2 or self.scout_limit <= 0:
            raise ValueError("invalid ABC source count or scout limit")

    @classmethod
    def default(cls) -> "ABCConfig":
        return cls(**_load("abc.yaml"))
