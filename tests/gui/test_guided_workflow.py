import pytest

pytest.importorskip("PyQt6")

from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.app.workflow_manager import WorkflowManager


def test_workflow_locks_downstream_until_prerequisites(tmp_path):
    state = AppState(str(tmp_path / "results.sqlite"))
    workflow = WorkflowManager(state)

    assert workflow.is_workspace_enabled(1)
    assert not workflow.is_workspace_enabled(2)

    workflow.mark_completed("power_system")
    assert workflow.is_workspace_enabled(2)
    assert not workflow.is_workspace_enabled(3)

    workflow.mark_completed("orpd")
    workflow.mark_completed("algorithms")
    assert workflow.is_workspace_enabled(4)
    assert not workflow.is_workspace_enabled(5)

    workflow.mark_completed("calo")
    assert workflow.is_workspace_enabled(5)


def test_post_experiment_sequence_requires_statistics_review_and_validation(tmp_path):
    state = AppState(str(tmp_path / "results.sqlite"))
    workflow = WorkflowManager(state)
    for key in ("power_system", "orpd", "algorithms", "calo", "scenarios"):
        workflow.mark_completed(key)

    workflow.mark_experiment_started()
    assert workflow.is_workspace_enabled(7)
    assert not workflow.is_workspace_enabled(8)

    workflow.mark_experiment_completed()
    assert workflow.is_workspace_enabled(8)
    assert not workflow.is_workspace_enabled(9)

    workflow.mark_statistics_completed()
    assert workflow.is_workspace_enabled(9)
    assert not workflow.is_workspace_enabled(10)

    workflow.mark_results_reviewed()
    assert workflow.is_workspace_enabled(10)
    assert not workflow.is_workspace_enabled(11)

    workflow.set_verified_results(1)
    assert workflow.is_workspace_enabled(11)
