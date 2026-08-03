"""Experimental, preregistered, disabled-by-default TSH-CALO population schedule (Change F)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .environmental_selection import environmental_select


@dataclass(frozen=True, slots=True)
class PopulationScheduleConfig:
    enabled: bool = False
    experimental_mode: bool = False
    minimum_population: int = 8
    contraction_fraction: float = 0.15
    minimum_feasible_ratio: float = 0.50
    minimum_archive_coverage: float = 0.25
    minimum_diversity: float = 0.02
    maximum_diversity: float = 0.25
    maximum_remaining_budget: float = 0.75
    minimum_remaining_budget: float = 0.10
    minimum_evaluations_between_contractions: int = 200

    def validate(self) -> None:
        if self.enabled and not self.experimental_mode:
            raise ValueError("Population scheduling is experimental and requires experimental_mode")
        if self.minimum_population < 4:
            raise ValueError("Experimental population minimum must be at least four")
        for name, value in (
            ("contraction_fraction", self.contraction_fraction),
            ("minimum_feasible_ratio", self.minimum_feasible_ratio),
            ("minimum_archive_coverage", self.minimum_archive_coverage),
            ("minimum_diversity", self.minimum_diversity),
            ("maximum_diversity", self.maximum_diversity),
            ("maximum_remaining_budget", self.maximum_remaining_budget),
            ("minimum_remaining_budget", self.minimum_remaining_budget),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Population schedule {name} must be within [0, 1]")
        if self.contraction_fraction <= 0.0 or self.contraction_fraction >= 0.5:
            raise ValueError("Population contraction_fraction must be within (0, 0.5)")
        if self.minimum_diversity > self.maximum_diversity:
            raise ValueError("Population diversity thresholds are inverted")
        if self.minimum_remaining_budget > self.maximum_remaining_budget:
            raise ValueError("Population remaining-budget thresholds are inverted")
        if self.minimum_evaluations_between_contractions < 1:
            raise ValueError("Population contraction spacing must be positive")

    def design_hash(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PopulationScheduleMetrics:
    evaluations: int
    feasible_ratio: float
    archive_coverage: float
    diversity: float
    remaining_budget: float

    def validate(self) -> None:
        if self.evaluations < 0:
            raise ValueError("Population schedule evaluations cannot be negative")
        values = np.asarray(
            [
                self.feasible_ratio,
                self.archive_coverage,
                self.diversity,
                self.remaining_budget,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Population schedule metrics must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class PopulationScheduleDecision:
    contract: bool
    previous_size: int
    target_size: int
    reason: str
    design_hash: str
    evaluations: int


class ExperimentalPopulationSchedule:
    SCHEMA_VERSION = "tsh-calo-population-schedule-v1-experimental"

    def __init__(self, initial_population: int, config: PopulationScheduleConfig | None = None):
        self.config = config or PopulationScheduleConfig()
        self.config.validate()
        self.initial_population = int(initial_population)
        self.current_population = int(initial_population)
        if self.initial_population < self.config.minimum_population:
            raise ValueError("Initial population is smaller than the configured minimum")
        self.last_contraction_evaluation = -self.config.minimum_evaluations_between_contractions
        self.contractions = 0

    def decide(self, metrics: PopulationScheduleMetrics) -> PopulationScheduleDecision:
        metrics.validate()
        current = int(self.current_population)
        design_hash = self.config.design_hash()
        reasons: list[str] = []
        if not self.config.enabled:
            reasons.append("experimental population schedule disabled")
        if current <= self.config.minimum_population:
            reasons.append("minimum population reached")
        if metrics.feasible_ratio < self.config.minimum_feasible_ratio:
            reasons.append("feasibility condition not met")
        if metrics.archive_coverage < self.config.minimum_archive_coverage:
            reasons.append("archive coverage condition not met")
        if not self.config.minimum_diversity <= metrics.diversity <= self.config.maximum_diversity:
            reasons.append("diversity condition not met")
        if not (
            self.config.minimum_remaining_budget
            <= metrics.remaining_budget
            <= self.config.maximum_remaining_budget
        ):
            reasons.append("remaining-budget condition not met")
        if (
            metrics.evaluations - self.last_contraction_evaluation
            < self.config.minimum_evaluations_between_contractions
        ):
            reasons.append("contraction spacing condition not met")
        if reasons:
            return PopulationScheduleDecision(
                False, current, current, "; ".join(reasons), design_hash, metrics.evaluations
            )
        target = max(
            self.config.minimum_population,
            int(np.floor(current * (1.0 - self.config.contraction_fraction))),
        )
        target = min(target, current - 1)
        return PopulationScheduleDecision(
            True,
            current,
            target,
            "all preregistered contraction conditions met",
            design_hash,
            metrics.evaluations,
        )

    def apply(
        self,
        decision: PopulationScheduleDecision,
        population: np.ndarray,
        evaluations: list[object],
        *,
        epsilon: float,
        diversity_weight: float,
    ) -> tuple[np.ndarray, list[object], np.ndarray]:
        if decision.design_hash != self.config.design_hash():
            raise ValueError("Population schedule decision design hash is stale or incompatible")
        if decision.previous_size != self.current_population:
            raise ValueError("Population schedule decision does not match current state")
        population = np.asarray(population, dtype=float)
        if len(population) != self.current_population or len(evaluations) != len(population):
            raise ValueError("Population schedule inputs do not match current population state")
        if not decision.contract:
            return population.copy(), list(evaluations), np.arange(len(population), dtype=int)
        if not self.config.enabled or not self.config.experimental_mode:
            raise ValueError("Disabled experimental population schedule cannot contract")
        if not self.config.minimum_population <= decision.target_size < decision.previous_size:
            raise ValueError("Population schedule target size is outside preregistered bounds")
        selected_population, selected_evaluations, selected_indices = environmental_select(
            population,
            evaluations,
            decision.target_size,
            float(epsilon),
            diversity_weight=float(diversity_weight),
            return_indices=True,
        )
        self.current_population = int(decision.target_size)
        self.last_contraction_evaluation = int(decision.evaluations)
        self.contractions += 1
        return (
            np.asarray(selected_population),
            list(selected_evaluations),
            np.asarray(selected_indices, dtype=int),
        )

    def state_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": asdict(self.config),
            "design_hash": self.config.design_hash(),
            "initial_population": self.initial_population,
            "current_population": self.current_population,
            "last_contraction_evaluation": self.last_contraction_evaluation,
            "contractions": self.contractions,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "ExperimentalPopulationSchedule":
        if str(payload.get("schema_version", "")) != cls.SCHEMA_VERSION:
            raise ValueError("Population schedule checkpoint schema is incompatible")
        config = PopulationScheduleConfig(**dict(payload["config"]))
        if str(payload.get("design_hash", "")) != config.design_hash():
            raise ValueError("Population schedule checkpoint design hash is invalid")
        schedule = cls(int(payload["initial_population"]), config)
        schedule.current_population = int(payload["current_population"])
        schedule.last_contraction_evaluation = int(payload["last_contraction_evaluation"])
        schedule.contractions = int(payload["contractions"])
        if (
            not config.minimum_population
            <= schedule.current_population
            <= schedule.initial_population
        ):
            raise ValueError("Population schedule checkpoint size is invalid")
        if schedule.contractions < 0:
            raise ValueError("Population schedule checkpoint contraction count is invalid")
        return schedule
