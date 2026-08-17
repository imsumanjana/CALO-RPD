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
from calo_rpd_studio.portfolio.models import ArticlePreset, EvidenceProfile, PortfolioConfig
from calo_rpd_studio.portfolio.planner import PortfolioPlanner
from calo_rpd_studio.portfolio.study_planning import (
    PortfolioGoalPlanner,
    WorkspaceStudyPlanner,
)


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


def study_contracts(config, stage, subset, *, outputs=("objective_convergence",)):
    portfolio = deepcopy(config.portfolio)
    portfolio.requested_outputs = list(outputs)
    goal = PortfolioGoalPlanner.create(portfolio, stage, tuple(subset))
    config.portfolio = portfolio
    recommendation = WorkspaceStudyPlanner.recommend(goal, stage, config)
    config.runs = recommendation.recommended_runs
    selected = {
        "runs": config.runs,
        "study_case_plan": list(config.study_case_plan),
        "scenario_mode": str(config.scenarios.mode),
        "population_size": config.population_size,
        "max_evaluations": config.budget.max_evaluations,
        "master_seed": config.master_seed,
        "reuse_compatible_results": config.reuse_compatible_results,
        "resume_enabled": config.resume_enabled,
        "checkpoint_interval_evaluations": config.checkpoint_interval_evaluations,
    }
    setup = WorkspaceStudyPlanner.apply_selection(goal, stage, recommendation, selected)
    config.workspace_study_contract = {
        "schema_version": "calo-rpd-workspace-study-runtime-contract-v1",
        "portfolio_goal_id": goal.portfolio_goal_id,
        "portfolio_goal_sha256": goal.content_sha256,
        "recommendation_id": recommendation.recommendation_id,
        "recommendation_sha256": recommendation.recommendation_sha256,
        "study_setup_id": setup.study_setup_id,
        "study_setup_sha256": setup.content_sha256,
        "hard_minimum_runs": recommendation.hard_minimum_runs,
    }
    return goal, recommendation, setup


def test_algorithm_stage_separates_content_identity_from_record_identity() -> None:
    config, first = configured_stage()
    second = AlgorithmStage.create(config)

    assert first.content_sha256 == second.content_sha256
    assert first.stage_id != second.stage_id
    assert first.record_sha256 != second.record_sha256


def test_workspace_plan_accepts_only_nonempty_stage_subset() -> None:
    config, stage = configured_stage()
    goal, recommendation, setup = study_contracts(config, stage, ("CALO", "PSO"))
    plan = WorkspaceStudyPlan.create(
        config,
        stage,
        ("CALO", "PSO"),
        portfolio_goal=goal,
        recommendation=recommendation,
        applied_study_setup=setup,
    )

    assert plan.study_algorithm_names == ("CALO", "PSO")
    assert plan.config_payload["algorithms"] == ["CALO", "PSO"]
    assert set(plan.config_payload["algorithm_parameters"]) == {"CALO", "PSO"}

    with pytest.raises(ValueError, match="outside the submitted stage"):
        WorkspaceStudyPlan.create(
            config,
            stage,
            ("GWO",),
            portfolio_goal=goal,
            recommendation=recommendation,
            applied_study_setup=setup,
        )
    with pytest.raises(ValueError, match="non-empty"):
        WorkspaceStudyPlan.create(
            config,
            stage,
            (),
            portfolio_goal=goal,
            recommendation=recommendation,
            applied_study_setup=setup,
        )

    assert plan.portfolio_goal_sha256 == goal.content_sha256
    assert plan.study_recommendation_sha256 == recommendation.recommendation_sha256
    assert plan.study_setup_sha256 == setup.content_sha256
    assert plan.queue_task_count == setup.queue_task_count


def test_portfolio_goal_is_independent_of_concrete_study_values() -> None:
    config, stage = configured_stage()
    portfolio = deepcopy(config.portfolio)
    portfolio.requested_outputs = ["objective_convergence"]
    first = PortfolioGoalPlanner.create(portfolio, stage, ("CALO",))

    config.runs = 987
    config.case_name = "case57"
    config.study_case_plan = ["case57", "case118"]
    config.budget.max_evaluations = 123_456
    config.scenarios.mode = "monte_carlo"
    second = PortfolioGoalPlanner.create(portfolio, stage, ("CALO",))

    first_content = first.content_payload()
    second_content = second.content_payload()
    for key in ("portfolio_goal_id", "created_at"):
        first_content.pop(key)
        second_content.pop(key)
    assert first_content == second_content
    assert "runs" not in first.portfolio
    assert "case_name" not in first.portfolio


def test_recommended_five_may_be_changed_to_six_without_mutating_goal() -> None:
    config, stage = configured_stage()
    portfolio = deepcopy(config.portfolio)
    portfolio.evidence_profile = EvidenceProfile.CUSTOM
    portfolio.requested_outputs = ["median_convergence"]
    goal = PortfolioGoalPlanner.create(portfolio, stage, ("CALO",))
    recommendation = WorkspaceStudyPlanner.recommend(goal, stage, config)
    before = goal.content_payload()
    selected = {
        "runs": 6,
        "study_case_plan": list(config.study_case_plan),
        "scenario_mode": str(config.scenarios.mode),
        "population_size": config.population_size,
        "max_evaluations": config.budget.max_evaluations,
        "master_seed": config.master_seed,
        "reuse_compatible_results": True,
        "resume_enabled": True,
        "checkpoint_interval_evaluations": 500,
    }

    setup = WorkspaceStudyPlanner.apply_selection(goal, stage, recommendation, selected)

    assert recommendation.hard_minimum_runs == 5
    assert recommendation.recommended_runs == 5
    assert setup.selected_values["runs"] == 6
    assert setup.recommendation_delta["runs"] == 1
    assert setup.queue_task_count == 6
    assert goal.content_payload() == before

    selected["runs"] = 4
    with pytest.raises(ValueError, match="below the Portfolio hard minimum 5"):
        WorkspaceStudyPlanner.apply_selection(goal, stage, recommendation, selected)


def test_individual_plan_uses_complete_immutable_stage_when_editable_draft_drifts() -> None:
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
    changed.algorithm_parameters = {
        "CALO": {
            "use_ai": True,
            "strict_policy_binding": True,
            "allow_unqualified_policy": True,
        },
        "TLBO": {"population_size": 999},
    }
    changed.runs = 7

    stage_bound = IndividualExperimentPlan.create(changed, stage)

    assert stage_bound.algorithm_names == stage.algorithm_names
    assert stage_bound.config_payload["algorithms"] == list(stage.algorithm_names)
    assert stage_bound.config_payload["algorithm_parameters"] == stage.algorithm_parameters
    assert stage_bound.config_payload["runs"] == 7
    assert changed.algorithms == ["CALO", "TLBO"]
    assert changed.algorithm_parameters["TLBO"] == {"population_size": 999}


def test_individual_plan_identity_ignores_workspace_portfolio_and_study_metadata() -> None:
    config, stage = configured_stage()
    first = IndividualExperimentPlan.create(config, stage)

    changed = deepcopy(config)
    changed.portfolio.name = "Different Workspace evidence plan"
    changed.portfolio.custom_runs = 900
    changed.portfolio_id = "workspace-portfolio-not-owned-by-individual"
    changed.study_case_plan = ["case30", "case57"]
    changed.study_strength = "strong"
    changed.workspace_study_contract = {"portfolio_goal_id": "workspace-only"}
    second = IndividualExperimentPlan.create(changed, stage)

    assert first.config_payload == second.config_payload
    assert first.design_sha256 == second.design_sha256
    assert second.config_payload["workspace_study_contract"] == {}


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
