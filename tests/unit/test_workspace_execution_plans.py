from __future__ import annotations

from copy import deepcopy

import pytest

from calo_rpd_studio.experiments.execution_plans import (
    AlgorithmStage,
    IndividualExperimentPlan,
    WorkspaceStudyPlan,
    scientific_job_sha256,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.portfolio.models import ArticlePreset, PortfolioConfig
from calo_rpd_studio.portfolio.planner import PortfolioPlanner


def configured_stage() -> tuple[ExperimentConfig, AlgorithmStage]:
    config = ExperimentConfig()
    config.algorithms = ["CALO", "TLBO", "PSO"]
    config.algorithm_parameters = {
        "CALO": {
            "use_ai": False,
            "strict_policy_binding": False,
            "allow_unqualified_policy": False,
        },
        "TLBO": {"population_size": 40},
        "PSO": {"inertia": 0.7},
    }
    return config, AlgorithmStage.create(config)


def test_algorithm_stage_separates_content_identity_from_record_identity() -> None:
    config, first = configured_stage()
    second = AlgorithmStage.create(config)

    assert first.content_sha256 == second.content_sha256
    assert first.stage_id != second.stage_id
    assert first.record_sha256 != second.record_sha256


def test_workspace_plan_accepts_only_nonempty_stage_subset() -> None:
    config, stage = configured_stage()
    plan = WorkspaceStudyPlan.create(config, stage, ("CALO", "PSO"))

    assert plan.study_algorithm_names == ("CALO", "PSO")
    assert plan.config_payload["algorithms"] == ["CALO", "PSO"]
    assert set(plan.config_payload["algorithm_parameters"]) == {"CALO", "PSO"}

    with pytest.raises(ValueError, match="outside the submitted stage"):
        WorkspaceStudyPlan.create(config, stage, ("GWO",))
    with pytest.raises(ValueError, match="non-empty"):
        WorkspaceStudyPlan.create(config, stage, ())


def test_individual_plan_uses_complete_unchanged_stage() -> None:
    config, stage = configured_stage()
    plan = IndividualExperimentPlan.create(config, stage)

    assert plan.algorithm_names == stage.algorithm_names
    assert plan.config_payload["algorithms"] == list(stage.algorithm_names)
    assert plan.config_payload["execution_plan_kind"] == "individual_experiment"
    assert "portfolio" not in plan.config_payload
    assert plan.config_payload["portfolio_id"] == ""
    assert plan.config_payload["study_case_plan"] == [config.case_name]
    assert plan.config_payload["result_contract"]["owner"] == "individual_experiment"
    assert plan.config_payload["result_contract"]["required_fields"]

    changed = deepcopy(config)
    changed.algorithms = ["CALO", "TLBO"]
    with pytest.raises(ValueError, match="complete unchanged submitted algorithm stage"):
        IndividualExperimentPlan.create(changed, stage)


def test_individual_plan_identity_ignores_workspace_portfolio_and_study_metadata() -> None:
    config, stage = configured_stage()
    first = IndividualExperimentPlan.create(config, stage)

    changed = deepcopy(config)
    changed.portfolio.name = "Different Workspace evidence plan"
    changed.portfolio.custom_runs = 900
    changed.portfolio_id = "workspace-portfolio-not-owned-by-individual"
    changed.study_case_plan = ["case30", "case57"]
    changed.study_strength = "strong"
    second = IndividualExperimentPlan.create(changed, stage)

    assert first.config_payload == second.config_payload
    assert first.design_sha256 == second.design_sha256


def test_individual_validation_ignores_workspace_study_and_portfolio_requirements() -> None:
    config, _stage = configured_stage()
    config.runs = 1
    config.study_strength = "invalid-workspace-strength"
    config.study_case_plan = []
    config.execution_plan_kind = "workspace"

    config.validate(execution_plan_kind="individual_experiment")

    with pytest.raises(ValueError, match="study_strength"):
        config.validate(execution_plan_kind="workspace")

    config.study_strength = "custom"
    config.study_case_plan = [config.case_name]
    with pytest.raises(ValueError, match="portfolio-required minimum"):
        config.validate(execution_plan_kind="workspace")


def test_article_preset_never_mutates_authoritative_algorithm_pool() -> None:
    config, _stage = configured_stage()
    original_names = list(config.algorithms)
    original_parameters = deepcopy(config.algorithm_parameters)
    portfolio = PortfolioConfig(article_preset=ArticlePreset.CALO_DETERMINISTIC)

    requirements = PortfolioPlanner.apply_article_preset(config, portfolio)

    assert config.algorithms == original_names
    assert config.algorithm_parameters == original_parameters
    assert requirements.required_algorithms == ("CALO",)
    assert "TLBO" in requirements.recommended_algorithms


def test_scientific_job_identity_is_plan_and_cell_bound() -> None:
    config, _stage = configured_stage()
    payload = config.to_dict()
    seeds = {"algorithm_seed": 1, "scenario_seed": 2, "ai_inference_seed": 3}

    first = scientific_job_sha256(
        plan_id="workspace-plan-a",
        cell_id="cell-1",
        algorithm="TLBO",
        run_index=0,
        seed_payload=seeds,
        config=payload,
    )
    second = scientific_job_sha256(
        plan_id="workspace-plan-a",
        cell_id="cell-2",
        algorithm="TLBO",
        run_index=0,
        seed_payload=seeds,
        config=payload,
    )

    assert first != second
