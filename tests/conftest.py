from __future__ import annotations
import numpy as np
import pytest
from calo_rpd_studio.power_system.case_model import PowerSystemCase
from calo_rpd_studio.portfolio.study_planning import (
    PortfolioGoalPlanner,
    WorkspaceStudyPlanner,
)


@pytest.fixture
def toy_case():
    # MATPOWER-compatible 3-bus case: slack, PV, PQ.
    bus = np.array(
        [
            [1, 3, 0, 0, 0, 0, 1, 1.04, 0, 230, 1, 1.10, 0.90],
            [2, 2, 20, 10, 0, 0, 1, 1.01, 0, 230, 1, 1.10, 0.90],
            [3, 1, 45, 15, 0, 0, 1, 1.00, 0, 230, 1, 1.10, 0.90],
        ],
        float,
    )
    gen = np.array(
        [
            [1, 40, 0, 100, -100, 1.04, 100, 1, 200, 0],
            [2, 30, 0, 100, -100, 1.01, 100, 1, 150, 0],
        ],
        float,
    )
    branch = np.array(
        [
            [1, 2, 0.02, 0.06, 0.03, 200, 200, 200, 0, 0, 1, -360, 360],
            [1, 3, 0.08, 0.24, 0.025, 200, 200, 200, 0, 0, 1, -360, 360],
            [2, 3, 0.06, 0.18, 0.02, 200, 200, 200, 1.0, 0, 1, -360, 360],
        ],
        float,
    )
    return PowerSystemCase("toy3", 100.0, bus, gen, branch)


@pytest.fixture
def apply_workspace_study():
    """Create the v3 draft only through the approved Goal -> recommendation -> Study path."""

    def apply(service, config, subset):
        from copy import deepcopy

        stage = service.active_stage()
        portfolio = deepcopy(config.portfolio)
        portfolio.requested_outputs = ["objective_convergence"]
        goal = PortfolioGoalPlanner.create(portfolio, stage, tuple(subset))
        service.database.replace_portfolio_goal(goal)
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
            "execution_backend": config.execution_backend,
            "execution_purpose": config.execution_purpose,
            "output_directory": config.output_directory,
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
        return service.create_workspace_draft(
            config,
            tuple(subset),
            portfolio_goal=goal,
            recommendation=recommendation,
            applied_study_setup=setup,
        )

    return apply
