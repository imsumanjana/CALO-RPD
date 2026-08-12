from __future__ import annotations

import json
from pathlib import Path

import pytest

from calo_rpd_studio.algorithms.calo.policy_retirement import PolicyRetirementManager
from calo_rpd_studio.compute.source_identity import SourceIdentity
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.scripts import accept_development_freeze as acceptance


_FREEZE_COMMIT = "f" * 40


def _manager(tmp_path: Path, *, external_artifact: bool = False):
    source_root = tmp_path / "source"
    source_root.mkdir()
    store = source_root / "trained_models"
    store.mkdir()
    (store / "AGENTS.md").write_text("protected\n", encoding="utf-8")
    (store / "__init__.py").write_text('"""protected"""\n', encoding="utf-8")
    nested = store / "historical_scope"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested protected\n", encoding="utf-8")
    (nested / "__init__.py").write_text('"""nested protected"""\n', encoding="utf-8")
    checkpoint = store / "old-policy.pt"
    checkpoint.write_bytes(b"development-policy")
    (store / "old-policy.json").write_text('{"development_only": true}\n', encoding="utf-8")

    database = ResultDatabase(source_root / "results.sqlite")
    recorded_path = source_root / "outside.pt" if external_artifact else checkpoint
    if external_artifact:
        recorded_path.write_bytes(b"external-development-policy")
    database.upsert_policy(
        policy_id="development-policy",
        name="Development only",
        checkpoint_path=str(recorded_path),
        sha256="1" * 64,
        architecture_version="historical",
        state_schema_version="historical",
        action_schema_version="historical",
        training_environment_version="historical",
        metadata={"development_only": True},
    )
    identity = SourceIdentity(_FREEZE_COMMIT, True, "test-fixture")
    return (
        PolicyRetirementManager(
            store,
            database,
            source_root=source_root,
            source_identity=identity,
        ),
        database,
        store,
    )


def _accepted(path: Path) -> tuple[dict, Path]:
    stable = {
        "schema": acceptance.ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "decision_id": "phase4-retirement-test",
        "validation_run_id": "phase4-test-run",
        "validated_source_commit": _FREEZE_COMMIT,
        "validated_source_dirty": False,
        "claim_boundary": acceptance.ACCEPTANCE_CLAIM_BOUNDARY,
        "validation_summary_sha256": "1" * 64,
        "validation_log_manifest_sha256": "2" * 64,
        "development_freeze_candidate_sha256": "a" * 64,
        "development_source_contract_sha256": "4" * 64,
        "development_source_file_count": 1,
        "validator_sha256": "5" * 64,
    }
    receipt = {
        **stable,
        "created_at": "2026-08-12T00:00:00+00:00",
        "acceptance_receipt_sha256": acceptance._canonical_sha256(stable),
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt, path


def _authorized(plan: dict, acceptance_receipt: dict | None = None) -> dict:
    accepted = acceptance_receipt or {"acceptance_receipt_sha256": "b" * 64}
    authorization = PolicyRetirementManager.authorization_template(plan)
    authorization.update(
        {
            "authorization_granted": True,
            "authorization_id": "owner-approved-post-freeze-removal",
            "development_freeze_payload_sha256": "a" * 64,
            "phase4_acceptance_receipt_sha256": accepted["acceptance_receipt_sha256"],
            "acknowledge_phase4_development_freeze_accepted": True,
            "acknowledge_irreversible_artifact_deletion": True,
            "acknowledge_database_lifecycle_cleanup": True,
        }
    )
    return authorization


def test_phase4_inventory_and_plan_are_exact_and_non_destructive(tmp_path):
    manager, database, store = _manager(tmp_path)

    inventory = manager.inventory()
    plan = manager.dry_run(inventory)
    authorization = PolicyRetirementManager.authorization_template(plan)

    assert {item["path"] for item in inventory["protected_files"]} == {
        "AGENTS.md",
        "__init__.py",
        "historical_scope/AGENTS.md",
        "historical_scope/__init__.py",
    }
    assert {item["path"] for item in inventory["removable_files"]} == {
        "old-policy.json",
        "old-policy.pt",
    }
    assert plan["destructive_action_executed"] is False
    assert plan["requires_separate_post_freeze_authorization"] is True
    assert authorization["authorization_granted"] is False
    assert authorization["acknowledge_phase4_development_freeze_accepted"] is False
    assert authorization["development_freeze_payload_sha256"] == ""
    assert authorization["phase4_acceptance_receipt_sha256"] == ""
    assert (store / "old-policy.pt").is_file()
    assert database.policy_lifecycle_snapshot()["policies"]


def test_retirement_rejects_unapproved_or_tampered_documents_without_changes(tmp_path):
    manager, database, store = _manager(tmp_path)
    inventory = manager.inventory()
    plan = manager.dry_run(inventory)
    authorization = PolicyRetirementManager.authorization_template(plan)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    _receipt, acceptance_path = _accepted(tmp_path / "phase4-acceptance.json")

    with pytest.raises(PermissionError, match="explicit post-freeze authorization"):
        manager.execute(
            inventory=inventory,
            plan=plan,
            authorization=authorization,
            authorization_path=authorization_path,
            phase4_acceptance_path=acceptance_path,
            receipt_path=tmp_path / "receipt.json",
        )

    tampered = dict(plan)
    tampered["policy_store_root"] = str(tmp_path / "different-policy-store")
    with pytest.raises(ValueError, match="plan SHA-256 mismatch"):
        manager.execute(
            inventory=inventory,
            plan=tampered,
            authorization=_authorized(plan),
            authorization_path=authorization_path,
            phase4_acceptance_path=acceptance_path,
            receipt_path=tmp_path / "tampered-receipt.json",
        )
    assert (store / "old-policy.pt").is_file()
    assert database.policy_lifecycle_snapshot()["policies"]


def test_separately_authorized_synthetic_retirement_is_path_confined_and_receipted(tmp_path):
    manager, database, store = _manager(tmp_path)
    inventory = manager.inventory()
    plan = manager.dry_run(inventory)
    acceptance_receipt, acceptance_path = _accepted(tmp_path / "phase4-acceptance.json")
    authorization = _authorized(plan, acceptance_receipt)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    receipt_path = tmp_path / "retained" / "receipt.json"

    receipt = manager.execute(
        inventory=inventory,
        plan=plan,
        authorization=authorization,
        authorization_path=authorization_path,
        phase4_acceptance_path=acceptance_path,
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "completed"
    assert receipt_path.is_file()
    assert not (store / "old-policy.pt").exists()
    assert not (store / "old-policy.json").exists()
    assert (store / "AGENTS.md").is_file()
    assert (store / "__init__.py").is_file()
    assert (store / "historical_scope" / "AGENTS.md").is_file()
    assert (store / "historical_scope" / "__init__.py").is_file()
    assert not any(database.policy_lifecycle_snapshot().values())
    assert not Path(receipt["recovery_staging_directory"]).exists()
    assert all(not item["retained_for_recovery"] for item in receipt["staged_artifacts"])


def test_external_artifact_is_an_unresolved_path_confinement_blocker(tmp_path):
    manager, _database, _store = _manager(tmp_path, external_artifact=True)
    inventory = manager.inventory()
    plan = manager.dry_run(inventory)

    assert inventory["external_existing_artifacts"]
    assert plan["blockers"]


def test_read_only_inventory_does_not_create_or_initialize_a_missing_database(tmp_path):
    store = tmp_path / "trained_models"
    store.mkdir()
    (store / "__init__.py").write_text('"""empty"""\n', encoding="utf-8")
    database_path = tmp_path / "missing.sqlite"
    database = ResultDatabase(database_path, read_only=True)
    manager = PolicyRetirementManager(
        store,
        database,
        source_identity=SourceIdentity(_FREEZE_COMMIT, True, "test-fixture"),
    )

    inventory = manager.inventory()

    assert not database_path.exists()
    assert not any(inventory["database"].values())
