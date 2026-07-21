from __future__ import annotations

from types import SimpleNamespace

from calo_rpd_studio.app.workflow_manager import WorkflowManager
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def test_workflow_snapshot_round_trip_restores_unlocked_experiment_state():
    config = ExperimentConfig()
    config.algorithms = ["CALO", "TLBO"]
    state = SimpleNamespace(config=config)
    first = WorkflowManager(state)
    for key in ("power_system", "orpd", "algorithms", "portfolio", "calo", "scenarios"):
        first.completed.add(key)
    first.experiment_started = True
    first.experiment_completed = True
    first.statistics_completed = True
    first.results_reviewed = True
    first.verified_results = 7
    payload = first.snapshot()

    restored = WorkflowManager(state)
    restored.restore(payload)
    assert restored.completed == first.completed
    assert restored.experiment_started is True
    assert restored.experiment_completed is True
    assert restored.statistics_completed is True
    assert restored.results_reviewed is True
    assert restored.verified_results == 7
    assert restored.is_workspace_enabled(12) is True


def test_old_experiment_without_workspace_snapshot_infers_setup_completion():
    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    state = SimpleNamespace(config=config)
    workflow = WorkflowManager(state)
    workflow.restore(None, infer_experiment=True, experiment_completed=False, verified_results=0)
    assert {"power_system", "orpd", "algorithms", "portfolio", "calo", "scenarios"} <= workflow.completed
    assert workflow.experiment_started is True
    assert workflow.is_workspace_enabled(7) is True
    assert workflow.is_workspace_enabled(8) is True
