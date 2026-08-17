from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.app.workflow_manager import WorkflowManager
from calo_rpd_studio.portfolio.study_planning import PortfolioGoalPlanner


def _policy_status(ready: bool):
    return SimpleNamespace(
        ready=ready,
        policy_name="Qualified policy" if ready else "",
        policy_sha256="abc123" if ready else "",
        grade="A" if ready else "",
        reason=(
            "Qualified governing policy is active."
            if ready
            else "No qualified active governing policy."
        ),
    )


def test_workflow_locks_power_system_until_governing_policy_then_prerequisites(tmp_path):
    state = AppState(str(tmp_path / "results.sqlite"))
    state.config.algorithms = ["CALO"]
    ready = {"value": False}
    state.governing_policy_status = lambda: _policy_status(ready["value"])
    workflow = WorkflowManager(state)

    assert workflow.next_descriptor().workspace_key == "algorithms"
    assert workflow.is_workspace_enabled("dashboard")
    assert workflow.is_workspace_enabled("calo_intelligence")
    assert workflow.is_workspace_enabled("algorithms")
    assert not workflow.is_workspace_enabled("power_system")
    stage = state.execution_control.submit_algorithm_stage(state.config)
    assert workflow.is_workspace_enabled("portfolio")
    assert not workflow.is_workspace_enabled("experiment")
    assert not workflow.is_workspace_enabled("scenarios")

    ready["value"] = True
    workflow.notify_governing_policy_changed()
    assert workflow.is_workspace_enabled("power_system")
    assert not workflow.is_workspace_enabled("orpd")

    workflow.mark_completed("power_system")
    assert workflow.is_workspace_enabled("orpd")
    assert workflow.is_workspace_enabled("algorithms")
    assert workflow.workspace_state_key("algorithms")[0] == "completed"
    assert state.execution_control.active_stage() == stage

    workflow.mark_completed("orpd")
    assert workflow.is_workspace_enabled("portfolio")
    assert not workflow.is_workspace_enabled("experiment")
    assert not workflow.is_workspace_enabled("scenarios")

    state.config.portfolio.requested_outputs = ["objective_convergence"]
    goal = PortfolioGoalPlanner.create(state.config.portfolio, stage, ("CALO",))
    state.database.replace_portfolio_goal(goal)
    assert workflow.is_workspace_enabled("experiment")
    assert workflow.is_workspace_enabled("scenarios")


def test_post_experiment_sequence_uses_keyed_workspace_gates(tmp_path):
    state = AppState(str(tmp_path / "results.sqlite"))
    state.config.algorithms = ["CALO"]
    state.governing_policy_status = lambda: _policy_status(True)
    workflow = WorkflowManager(state)
    workflow.notify_governing_policy_changed()
    stage = state.execution_control.submit_algorithm_stage(state.config)
    state.config.portfolio.requested_outputs = ["objective_convergence"]
    goal = PortfolioGoalPlanner.create(state.config.portfolio, stage, ("CALO",))
    state.database.replace_portfolio_goal(goal)
    for key in ("power_system", "orpd", "scenarios"):
        workflow.mark_completed(key)

    workflow.mark_experiment_started()
    assert workflow.is_workspace_enabled("experiment")
    assert workflow.is_workspace_enabled("live_optimization")
    assert not workflow.is_workspace_enabled("statistics")

    workflow.mark_experiment_completed()
    assert workflow.is_workspace_enabled("statistics")
    assert workflow.is_workspace_enabled("results")
    assert workflow.is_workspace_enabled("validation")
    assert workflow.is_workspace_enabled("publication")


def test_individual_setup_completion_is_separate_from_workspace_portfolio_sequence(tmp_path):
    state = AppState(str(tmp_path / "individual-workflow.sqlite"))
    state.governing_policy_status = lambda: _policy_status(True)
    workflow = WorkflowManager(state)
    workflow.notify_governing_policy_changed()

    workflow.mark_individual_completed("power_system")
    workflow.mark_individual_completed("orpd")

    assert workflow.individual_setup_state_key("scenarios")[0] == "recommended"
    assert workflow.workspace_state_key("scenarios") == (
        "locked",
        "Apply a Portfolio goal first. Portfolio defines what evidence is wanted; Study will "
        "then recommend how to produce it.",
    )

    workflow.mark_individual_completed("scenarios")
    assert workflow.individual_setup_state_key("scenarios")[0] == "completed"
    assert "scenarios" not in workflow.completed

    workflow.mark_completed("power_system")
    workflow.mark_completed("orpd")
    state.config.algorithms = ["TLBO"]
    state.config.algorithm_parameters = {"TLBO": {}}
    stage = state.execution_control.submit_algorithm_stage(state.config)
    portfolio = state.config.portfolio
    portfolio.requested_outputs = ["objective_convergence"]
    goal = PortfolioGoalPlanner.create(portfolio, stage, ("TLBO",))
    state.database.replace_portfolio_goal(goal)
    assert workflow.workspace_state_key("scenarios")[0] == "recommended"
    assert "scenarios" in workflow.individual_completed

    workflow.invalidate_individual_from("power_system")
    assert workflow.individual_completed == set()
    assert "power_system" in workflow.completed


def test_policy_free_submitted_stage_keeps_individual_case_setup_available(tmp_path):
    state = AppState(str(tmp_path / "policy-free-individual.sqlite"))
    state.config.algorithms = ["CALO", "TLBO"]
    state.config.algorithm_parameters = {
        "CALO": {
            "use_ai": False,
            "strict_policy_binding": False,
            "allow_unqualified_policy": False,
        },
        "TLBO": {},
    }
    state.execution_control.submit_algorithm_stage(state.config)
    state.governing_policy_status = lambda: _policy_status(False)
    workflow = WorkflowManager(state)

    assert workflow.individual_setup_state_key("power_system")[0] == "recommended"
