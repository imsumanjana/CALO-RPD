from __future__ import annotations

import pytest

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.study_strength import (
    StudyStrength,
    apply_study_strength,
    recommend_paired_runs,
    study_strength_plan,
    summarize_study_protocol_change,
)


@pytest.mark.parametrize(
    ("strength", "scenario_mode"),
    [
        (StudyStrength.LOW, "deterministic"),
        (StudyStrength.MODERATE, "deterministic"),
        (StudyStrength.GOOD, "monte_carlo"),
        (StudyStrength.STRONG, "monte_carlo"),
    ],
)
def test_study_strength_applies_one_coherent_valid_protocol(strength, scenario_mode):
    config = ExperimentConfig()
    plan = apply_study_strength(config, strength, case_name="case57")

    assert plan.strength is strength
    assert config.study_strength == strength.value
    assert config.case_name == "case57"
    recommendation = recommend_paired_runs(strength)
    assert config.runs == recommendation.runs
    assert config.runs == config.portfolio.required_runs()
    assert config.runs >= plan.runs
    assert config.scenarios.mode == scenario_mode
    assert config.budget.max_evaluations % config.population_size == 0
    assert config.max_iterations * config.population_size == config.budget.max_evaluations
    assert config.algorithms[0] == "CALO"
    assert config.portfolio.requested_outputs == list(plan.outputs)
    assert config.portfolio.require_independent_validation is True
    assert config.resume_enabled is True
    config.validate()


def test_study_strength_does_not_modify_governing_policy_binding():
    config = ExperimentConfig()
    config.algorithm_parameters["CALO"] = {
        "use_ai": True,
        "policy_id": "governing",
        "policy_sha256": "immutable-sha",
        "strict_policy_binding": True,
    }
    apply_study_strength(config, StudyStrength.STRONG, case_name="case118")
    assert config.algorithm_parameters["CALO"]["policy_id"] == "governing"
    assert config.algorithm_parameters["CALO"]["policy_sha256"] == "immutable-sha"
    assert config.algorithm_parameters["CALO"]["strict_policy_binding"] is True


def test_study_protocol_diff_is_scientist_readable_and_excludes_policy_internals():
    before = ExperimentConfig()
    after = ExperimentConfig.from_dict(before.to_dict())
    apply_study_strength(after, StudyStrength.GOOD, case_name="case57")
    changes = summarize_study_protocol_change(before, after)
    assert any(item.startswith("Reference case:") for item in changes)
    assert any(item.startswith("Paired runs per algorithm/case:") for item in changes)
    assert any(item.startswith("Required outputs:") for item in changes)
    assert not any("policy" in item.lower() for item in changes)


def test_strength_guidance_reports_outputs_runs_and_honest_limitations():
    guidance = study_strength_plan(StudyStrength.STRONG).guidance("case30")
    assert "paired independent runs" in guidance
    assert "power 95%" in guidance
    assert "paired differences" in guidance
    assert "Recommended outputs:" in guidance
    assert "No preset guarantees acceptance or universal superiority" in guidance
    assert "journal" not in guidance.lower()


def test_power_aware_run_planner_is_monotone_and_multiplicity_aware():
    moderate = recommend_paired_runs(StudyStrength.MODERATE)
    good = recommend_paired_runs(StudyStrength.GOOD)
    strong = recommend_paired_runs(StudyStrength.STRONG)
    assert moderate.runs < good.runs < strong.runs
    assert strong.comparisons == len(study_strength_plan(StudyStrength.STRONG).algorithms) - 1
    assert recommend_paired_runs(StudyStrength.STRONG, standardized_effect=0.40).runs > strong.runs
    assert recommend_paired_runs(StudyStrength.STRONG, planned_comparisons=19).runs > strong.runs
    with pytest.raises(ValueError, match="planned_comparisons"):
        recommend_paired_runs(StudyStrength.STRONG, planned_comparisons=0)


def test_low_strength_is_explicitly_screening_not_powered():
    recommendation = recommend_paired_runs(StudyStrength.LOW)
    assert recommendation.runs == 5
    assert recommendation.target_power is None
    assert "no inferential power claim" in recommendation.explanation


def test_study_strength_round_trip_is_serialized():
    config = ExperimentConfig()
    apply_study_strength(config, StudyStrength.MODERATE, case_name="case30")
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored.study_strength == "moderate"
    assert restored.study_case_plan == ["case30"]
    assert restored.study_standardized_effect == pytest.approx(config.study_standardized_effect)
    assert restored.study_target_power == pytest.approx(config.study_target_power)
    assert restored.study_run_planning_method == "paired_normal_holm_approximation"
    assert restored.portfolio.requested_outputs == config.portfolio.requested_outputs
