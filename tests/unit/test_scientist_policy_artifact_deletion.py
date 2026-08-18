from __future__ import annotations

import pytest

from calo_rpd_studio.algorithms.calo.policy_artifact_deletion import (
    permanent_artifact_deletion_blocker,
    record_permanent_artifact_deletion,
)
from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.results.database import ResultDatabase
from tests.unit.test_calo_v41_policy_system import _write_native_policy


def _protected_active_policy(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    checkpoint = _write_native_policy(tmp_path / "historical-policy.pt")
    policy = registry.register(checkpoint, name="Historical policy")
    database.add_policy_qualification(
        qualification_id="assessment-1",
        policy_id=policy.id,
        config={"assessment": "retained"},
        metrics={"score": 50.0},
        passed=False,
        grade="U",
        score=50.0,
        qualification_status="assessed",
    )
    experiment_id = database.create_experiment(ExperimentConfig(), collect_provenance())
    database.bind_policy_to_experiment(
        experiment_id,
        {
            "policy_id": policy.id,
            "policy_name": policy.name,
            "policy_checkpoint": str(checkpoint),
            "policy_sha256": policy.sha256,
        },
    )
    database.set_active_policy(policy.id)
    return database, registry, checkpoint, registry.get(policy.id), experiment_id


def test_permanent_artifact_deletion_tombstones_protected_policy_and_retains_history(tmp_path):
    database, registry, checkpoint, active, experiment_id = _protected_active_policy(tmp_path)

    assert active.active is True
    assert registry.unqualified_candidate_removal_blocker(active.id)
    assert permanent_artifact_deletion_blocker(active) == ""

    checkpoint.unlink()
    tombstone = record_permanent_artifact_deletion(
        registry,
        active.id,
        expected_sha256=active.sha256,
        reason="scientist_test_deletion",
        deleted_scope="standalone_model_file",
    )

    assert tombstone.active is False
    assert tombstone.archived is True
    assert tombstone.usable is False
    assert registry.is_suppressed(active.sha256) is True
    assert len(database.list_policy_qualifications(active.id)) == 1
    binding = database.get_experiment_policy_binding(experiment_id)
    assert binding is not None
    assert binding["policy_id"] == active.id
    assert binding["sha256"] == active.sha256
    deletion = tombstone.metadata["artifact_deletion"]
    assert deletion["was_active"] is True
    assert deletion["qualification_record_count"] == 1
    assert deletion["experiment_binding_count"] == 1
    assert deletion["historical_records_retained"] is True


def test_permanent_artifact_deletion_rejects_identity_drift_without_changing_policy(tmp_path):
    _database, registry, _checkpoint, active, _experiment_id = _protected_active_policy(tmp_path)

    with pytest.raises(RuntimeError, match="identity changed"):
        record_permanent_artifact_deletion(
            registry,
            active.id,
            expected_sha256="0" * 64,
            reason="scientist_test_wrong_identity",
        )

    unchanged = registry.get(active.id)
    assert unchanged.active is True
    assert unchanged.archived is False
    assert registry.is_suppressed(active.sha256) is False


def test_permanent_artifact_deletion_requires_a_real_exact_model_target(tmp_path):
    _database, registry, checkpoint, active, _experiment_id = _protected_active_policy(tmp_path)

    checkpoint.unlink()
    blocker = permanent_artifact_deletion_blocker(registry.get(active.id))

    assert "no verified model file to delete" in blocker.lower()
