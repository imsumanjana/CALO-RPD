from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.app.workflow_manager import WorkflowManager


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

    assert workflow.is_workspace_enabled("dashboard")
    assert workflow.is_workspace_enabled("calo_intelligence")
    assert not workflow.is_workspace_enabled("power_system")

    ready["value"] = True
    workflow.notify_governing_policy_changed()
    assert workflow.is_workspace_enabled("power_system")
    assert not workflow.is_workspace_enabled("orpd")

    workflow.mark_completed("power_system")
    assert workflow.is_workspace_enabled("orpd")
    assert not workflow.is_workspace_enabled("algorithms")

    workflow.mark_completed("orpd")
    workflow.mark_completed("algorithms")
    assert workflow.is_workspace_enabled("portfolio")
    assert not workflow.is_workspace_enabled("scenarios")

    workflow.mark_completed("portfolio")
    assert workflow.is_workspace_enabled("scenarios")


def test_post_experiment_sequence_uses_keyed_workspace_gates(tmp_path):
    state = AppState(str(tmp_path / "results.sqlite"))
    state.config.algorithms = ["CALO"]
    state.governing_policy_status = lambda: _policy_status(True)
    workflow = WorkflowManager(state)
    workflow.notify_governing_policy_changed()
    for key in ("power_system", "orpd", "algorithms", "portfolio", "scenarios"):
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
