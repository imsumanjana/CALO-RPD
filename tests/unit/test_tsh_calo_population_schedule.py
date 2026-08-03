"""Change-F experimental gating, preregistration, selection, and resume invariants."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_population_schedule import (
    ExperimentalPopulationSchedule,
    PopulationScheduleConfig,
    PopulationScheduleDecision,
    PopulationScheduleMetrics,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import DEFAULT_TSH_CALO_FEATURES


@dataclass
class _Evaluation:
    value: float
    feasible: bool
    violation: float


def _ready(evaluations: int = 500) -> PopulationScheduleMetrics:
    return PopulationScheduleMetrics(
        evaluations=evaluations,
        feasible_ratio=0.75,
        archive_coverage=0.60,
        diversity=0.10,
        remaining_budget=0.50,
    )


def _enabled(**changes) -> PopulationScheduleConfig:
    values = dict(
        enabled=True,
        experimental_mode=True,
        minimum_population=4,
        contraction_fraction=0.20,
        minimum_evaluations_between_contractions=100,
    )
    values.update(changes)
    return PopulationScheduleConfig(**values)


def test_population_schedule_is_doubly_gated_and_disabled_by_default():
    assert DEFAULT_TSH_CALO_FEATURES.population_schedule is False
    schedule = ExperimentalPopulationSchedule(12)
    decision = schedule.decide(_ready())

    assert decision.contract is False
    assert decision.target_size == 12
    assert "disabled" in decision.reason
    with pytest.raises(ValueError, match="experimental_mode"):
        PopulationScheduleConfig(enabled=True).validate()


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        (PopulationScheduleMetrics(500, 0.1, 0.6, 0.1, 0.5), "feasibility"),
        (PopulationScheduleMetrics(500, 0.8, 0.1, 0.1, 0.5), "archive"),
        (PopulationScheduleMetrics(500, 0.8, 0.6, 0.0, 0.5), "diversity"),
        (PopulationScheduleMetrics(500, 0.8, 0.6, 0.1, 0.05), "remaining-budget"),
    ],
)
def test_every_preregistered_condition_can_falsify_contraction(metrics, reason):
    decision = ExperimentalPopulationSchedule(12, _enabled()).decide(metrics)

    assert decision.contract is False
    assert reason in decision.reason


def test_eligible_contraction_is_bounded_deterministic_feasibility_first_and_has_no_fe():
    schedule = ExperimentalPopulationSchedule(12, _enabled())
    decision = schedule.decide(_ready())
    population = np.arange(36, dtype=float).reshape(12, 3) / 36.0
    evaluations = [
        _Evaluation(float(index), index < 10, 0.0 if index < 10 else float(index - 9))
        for index in range(12)
    ]

    selected, selected_evaluations, indices = schedule.apply(
        decision,
        population,
        evaluations,
        epsilon=0.0,
        diversity_weight=0.0,
    )

    assert decision.contract is True
    assert decision.target_size == 9
    assert selected.shape == (9, 3)
    assert len(selected_evaluations) == len(indices) == 9
    assert all(item.feasible for item in selected_evaluations)
    assert schedule.current_population == 9
    source = inspect.getsource(ExperimentalPopulationSchedule.apply)
    assert "evaluate_candidates" not in source
    assert "run_ac_power_flow" not in source


def test_schedule_checkpoint_restores_exact_state_and_spacing_decision():
    schedule = ExperimentalPopulationSchedule(12, _enabled())
    decision = schedule.decide(_ready(500))
    population = np.linspace(0.0, 1.0, 24).reshape(12, 2)
    evaluations = [_Evaluation(float(index), True, 0.0) for index in range(12)]
    schedule.apply(decision, population, evaluations, epsilon=0.0, diversity_weight=0.0)
    restored = ExperimentalPopulationSchedule.from_state_dict(schedule.state_dict())

    assert restored.state_dict() == schedule.state_dict()
    blocked = restored.decide(_ready(550))
    assert blocked.contract is False
    assert "spacing" in blocked.reason


def test_schedule_design_hash_changes_with_preregistered_threshold():
    first = _enabled()
    second = _enabled(minimum_feasible_ratio=0.60)

    assert first.design_hash() != second.design_hash()


def test_disabled_schedule_cannot_apply_fabricated_contraction_decision():
    schedule = ExperimentalPopulationSchedule(12)
    fabricated = PopulationScheduleDecision(
        True,
        12,
        10,
        "fabricated",
        schedule.config.design_hash(),
        500,
    )
    population = np.zeros((12, 2))
    evaluations = [_Evaluation(float(index), True, 0.0) for index in range(12)]

    with pytest.raises(ValueError, match="Disabled"):
        schedule.apply(
            fabricated,
            population,
            evaluations,
            epsilon=0.0,
            diversity_weight=0.0,
        )
