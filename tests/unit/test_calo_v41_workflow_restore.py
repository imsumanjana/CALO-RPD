from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from calo_rpd_studio.app.workflow_manager import WorkflowManager
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def _state(config: ExperimentConfig, *, ready: bool = True, sha: str = "policy-sha"):
    status = SimpleNamespace(
        ready=ready,
        policy_name="Policy" if ready else "",
        policy_sha256=sha if ready else "",
        grade="A" if ready else "",
        reason="ready" if ready else "not ready",
    )
    return SimpleNamespace(config=config, governing_policy_status=lambda: status)


def test_workflow_snapshot_round_trip_restores_unlocked_experiment_state():
    config = ExperimentConfig()
    config.algorithms = ["CALO", "TLBO"]
    state = _state(config)
    first = WorkflowManager(state)
    first.notify_governing_policy_changed()
    for key in ("power_system", "orpd", "algorithms", "portfolio", "scenarios"):
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
    assert restored.is_workspace_enabled("publication") is True


def test_legacy_restore_never_infers_governing_policy_but_accepts_explicit_setup_inference():
    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    state = _state(config, ready=True)
    workflow = WorkflowManager(state)
    inferred = {"power_system", "orpd", "algorithms", "portfolio", "scenarios"}
    workflow.restore(
        None,
        infer_experiment=True,
        experiment_completed=False,
        verified_results=0,
        inferred_completed=inferred,
    )
    assert "calo_intelligence" in workflow.completed  # added only from live governing-policy readiness
    assert inferred <= workflow.completed
    assert workflow.experiment_started is True
    assert workflow.is_workspace_enabled("experiment") is True
    assert workflow.is_workspace_enabled("live_optimization") is True


def test_restored_downstream_setup_is_invalidated_when_governing_policy_sha_changed():
    config = ExperimentConfig()
    state = _state(config, ready=True, sha="new-sha")
    workflow = WorkflowManager(state)
    payload = {
        "schema_version": 2,
        "completed": ["calo_intelligence", "power_system", "orpd", "algorithms", "portfolio", "scenarios"],
        "governing_policy_sha": "old-sha",
    }
    workflow.restore(payload)
    assert workflow.completed == {"calo_intelligence"}
    assert workflow.is_workspace_enabled("power_system") is True
    assert workflow.is_workspace_enabled("orpd") is False
