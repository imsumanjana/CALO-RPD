from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from calo_rpd_studio.algorithms.calo.policy_retirement import PolicyRetirementManager
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    TSHCALOCandidateArtifact,
)
from calo_rpd_studio.compute.source_identity import SourceIdentity
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.scripts import accept_development_freeze as acceptance
from calo_rpd_studio.scripts import create_development_freeze_candidate as freeze


_COMMIT = "9" * 40


def _complete_source_fixture(source: Path, *, clean: bool) -> dict:
    return {
        "schema": freeze.COMPLETE_SOURCE_MANIFEST_SCHEMA,
        "enumeration": "test_fixture",
        "source_status_sha256": "c" * 64,
        "source_status_clean": clean,
        "file_count": 4,
        "files": [
            freeze._file_record(source, relative)
            for relative in sorted(("Dockerfile", "exclude.txt", "interface.py", "lock.txt"))
        ],
    }


def _ensemble_artifact(*, freeze_commit: str, initialization_sha256: str = ""):
    members = []
    for index in range(2):
        members.append(
            {
                "source_candidate_sha256": str(index + 1) * 64,
                "training_provenance": {
                    "source_kind": "independent_policy_training",
                    "source_commit": _COMMIT,
                    "development_freeze_commit": freeze_commit,
                    "development_freeze_sha256": "d" * 64,
                    "phase4_acceptance_sha256": "a" * 64,
                    "initialization_policy_sha256": initialization_sha256,
                },
            }
        )
    return TSHCALOCandidateArtifact(
        path="candidate.pt",
        sha256="a" * 64,
        algorithm_id="TSH-CALO",
        algorithm_version="test",
        state_schema_version="test",
        action_schema_version="test",
        training_environment_version="test",
        artifact_kind="ensemble_policy",
        ensemble_size=2,
        feature_flags={},
        training_provenance={
            "source_kind": "independent_policy_training_ensemble",
            "members": members,
        },
    )


def test_production_artifact_eligibility_requires_exact_freeze_and_empty_initialization():
    assert _ensemble_artifact(freeze_commit=_COMMIT).post_development_eligible is True
    assert _ensemble_artifact(freeze_commit="").post_development_eligible is False
    assert (
        _ensemble_artifact(
            freeze_commit=_COMMIT,
            initialization_sha256="b" * 64,
        ).post_development_eligible
        is False
    )
    mixed = _ensemble_artifact(freeze_commit=_COMMIT)
    mixed.training_provenance["members"][1]["training_provenance"]["phase4_acceptance_sha256"] = (
        "b" * 64
    )
    assert mixed.post_development_eligible is False


def test_future_training_entrypoint_is_exact_freeze_bound_without_executing_training():
    command = Path("calo_rpd_studio/scripts/train_tsh_calo.py").read_text(encoding="utf-8")
    campaign = Path("calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py").read_text(
        encoding="utf-8"
    )

    assert '"--development-freeze"' in command
    assert "validate_development_freeze_for_plan" in command
    assert "require_training_eligible=True" in command
    assert "development_freeze_commit" in campaign
    assert "development_freeze_sha256" in campaign
    assert "phase4_acceptance_sha256" in campaign
    assert "v4-phase4-acceptance-bound" in campaign


def _fixture(tmp_path: Path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    for name in ("interface.py", "lock.txt", "Dockerfile", "exclude.txt", "validator.ps1"):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    store = root / "trained_models"
    store.mkdir()
    (store / "__init__.py").write_text('"""empty"""\n', encoding="utf-8")
    identity = SourceIdentity(_COMMIT, True, "test-fixture")
    database = ResultDatabase(root / "results.sqlite")
    inventory = PolicyRetirementManager(store, database, source_identity=identity).inventory()
    monkeypatch.setattr(freeze, "INTERFACE_FILES", {"interface": "interface.py"})
    monkeypatch.setattr(freeze, "DEPENDENCY_FILES", ("lock.txt",))
    monkeypatch.setattr(freeze, "CONTAINER_FILES", ("Dockerfile",))
    monkeypatch.setattr(freeze, "EXCLUSION_FILES", ("exclude.txt",))
    monkeypatch.setattr(freeze, "resolve_source_identity", lambda cwd: identity)
    monkeypatch.setattr(
        freeze,
        "_complete_source_manifest",
        lambda source: _complete_source_fixture(source, clean=True),
    )
    return root, inventory


def test_development_freeze_candidate_binds_source_interfaces_and_empty_policy_scope(
    tmp_path, monkeypatch
):
    root, inventory = _fixture(tmp_path, monkeypatch)

    report = freeze.build_development_freeze_candidate(
        root,
        policy_inventory=inventory,
        validator_path=root / "validator.ps1",
        require_clean=True,
    )

    assert report["status"] == "development_freeze_candidate"
    assert report["source_identity"]["source_commit"] == _COMMIT
    assert report["interfaces"]["interface"]["sha256"]
    assert report["complete_source_manifest"]["file_count"] == 4
    assert report["complete_source_manifest"]["source_status_sha256"] == "c" * 64
    assert report["complete_source_manifest"]["source_status_clean"] is True
    assert report["validator"]["sha256"]
    assert report["policy_inventory"]["release_scope_policy_count"] == 0
    assert report["policy_scope"]["qualified_policy_in_development_freeze"] is False
    assert report["policy_scope"]["future_policy_initialization_policy_sha256"] == ""
    assert report["post_transition_training_eligible"] is True
    assert "release publication" not in report["commands_executed_by_report"]

    verified = freeze.verify_development_freeze_source(report, root)
    assert verified["source_file_count"] == 4
    assert (
        verified["development_freeze_payload_sha256"] == report["development_freeze_payload_sha256"]
    )

    contract = freeze.development_source_contract(report)
    accepted_stable = {
        "schema": acceptance.ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "decision_id": "phase4-test-acceptance",
        "validation_run_id": "phase4-test-run",
        "validated_source_commit": _COMMIT,
        "validated_source_dirty": False,
        "claim_boundary": acceptance.ACCEPTANCE_CLAIM_BOUNDARY,
        "validation_summary_sha256": "1" * 64,
        "validation_log_manifest_sha256": "2" * 64,
        "development_freeze_candidate_sha256": report["development_freeze_payload_sha256"],
        "development_source_contract_sha256": contract["files_sha256"],
        "development_source_file_count": contract["file_count"],
        "validator_sha256": report["validator"]["sha256"],
    }
    accepted = {
        **accepted_stable,
        "created_at": "2026-08-12T00:00:00+00:00",
        "acceptance_receipt_sha256": acceptance._canonical_sha256(accepted_stable),
    }
    acceptance.acceptance_matches_freeze(accepted, report)


def test_accepted_source_contract_excludes_only_the_designated_policy_store(monkeypatch):
    files = [
        {"path": "ordinary/candidate.pt", "size_bytes": 1, "sha256": "1" * 64},
        {
            "path": "calo_rpd_studio/data/trained_models/old-policy.pt",
            "size_bytes": 2,
            "sha256": "2" * 64,
        },
        {
            "path": "calo_rpd_studio/data/trained_models/__init__.py",
            "size_bytes": 3,
            "sha256": "3" * 64,
        },
    ]
    report = {"complete_source_manifest": {"files": files}}
    monkeypatch.setattr(freeze, "validate_development_freeze_candidate", lambda _report: None)

    contract = freeze.development_source_contract(report)

    retained = [files[0], files[2]]
    assert contract["file_count"] == 2
    assert contract["files_sha256"] == freeze._canonical_sha256(retained)


def test_phase4_acceptance_requires_a_complete_hash_valid_passing_manual_run(tmp_path, monkeypatch):
    root, inventory = _fixture(tmp_path, monkeypatch)
    report = freeze.build_development_freeze_candidate(
        root,
        policy_inventory=inventory,
        validator_path=root / "validator.ps1",
        require_clean=True,
    )
    run = tmp_path / "phase4-test-run"
    (run / "evidence").mkdir(parents=True)
    (run / "commands").mkdir()
    summary = {
        "phase": "phase4",
        "passed": True,
        "failed_command_count": 0,
        "command_count": len(acceptance._REQUIRED_COMMAND_IDS),
        "source_commit": _COMMIT,
        "source_dirty": False,
        "validator_sha256": report["validator"]["sha256"],
        "results": [
            {"id": command_id, "status": "PASS"}
            for command_id in sorted(acceptance._REQUIRED_COMMAND_IDS)
        ],
    }
    files = {
        "validation-summary.json": json.dumps(summary),
        "VALIDATION_SUMMARY.md": "synthetic passing Phase 4 evidence\n",
        "phase4-source-manifest.json": "{}\n",
        "evidence/development-freeze-candidate.json": json.dumps(report),
        "evidence/old-policy-inventory.json": "{}\n",
        "evidence/old-policy-removal-plan.json": "{}\n",
        "evidence/old-policy-authorization-template-disabled.json": "{}\n",
        "evidence/git-status.txt": "\n",
        "evidence/git-status-final.txt": "\n",
        "commands/29-source-stability.txt": "Status: PASS\n",
        "commands/30-freeze-source-recheck.txt": "Status: PASS\n",
    }
    for relative, text in files.items():
        destination = run / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    manifest_rows = []
    for path in sorted(item for item in run.rglob("*") if item.is_file()):
        relative = path.relative_to(run).as_posix()
        manifest_rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    (run / "validation-log-sha256.txt").write_text(
        "\n".join(manifest_rows) + "\n",
        encoding="ascii",
    )

    accepted = acceptance.build_acceptance_receipt(
        run,
        decision_id="owner-accepted-phase4-test",
    )
    acceptance.validate_acceptance_receipt(accepted)
    acceptance.acceptance_matches_freeze(accepted, report)
    assert accepted["validation_run_id"] == run.name
    assert accepted["development_source_file_count"] == 4

    (run / "VALIDATION_SUMMARY.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acceptance.build_acceptance_receipt(run, decision_id="owner-accepted-phase4-test")


def test_dirty_development_freeze_is_evidence_but_not_new_policy_training_authority(
    tmp_path, monkeypatch
):
    root, inventory = _fixture(tmp_path, monkeypatch)
    dirty = SourceIdentity(_COMMIT, False, "test-fixture")
    inventory["source_identity"] = dirty.to_dict()
    stable = {
        key: value
        for key, value in inventory.items()
        if key not in {"created_at", "inventory_sha256"}
    }
    inventory["inventory_sha256"] = freeze._canonical_sha256(stable)
    monkeypatch.setattr(freeze, "resolve_source_identity", lambda cwd: dirty)
    monkeypatch.setattr(
        freeze,
        "_complete_source_manifest",
        lambda source: _complete_source_fixture(source, clean=False),
    )

    report = freeze.build_development_freeze_candidate(
        root,
        policy_inventory=inventory,
        validator_path=root / "validator.ps1",
    )

    assert report["post_transition_training_eligible"] is False
    freeze.validate_development_freeze_candidate(report)
    with pytest.raises(ValueError, match="clean, empty-policy"):
        freeze.validate_development_freeze_candidate(report, require_training_eligible=True)


def test_development_freeze_requires_matching_clean_source_when_requested(tmp_path, monkeypatch):
    root, inventory = _fixture(tmp_path, monkeypatch)
    dirty = SourceIdentity(_COMMIT, False, "test-fixture")
    inventory["source_identity"] = dirty.to_dict()
    stable = {
        key: value
        for key, value in inventory.items()
        if key not in {"created_at", "inventory_sha256"}
    }
    inventory["inventory_sha256"] = freeze._canonical_sha256(stable)
    monkeypatch.setattr(freeze, "resolve_source_identity", lambda cwd: dirty)

    with pytest.raises(RuntimeError, match="clean full Git source identity"):
        freeze.build_development_freeze_candidate(
            root,
            policy_inventory=inventory,
            validator_path=root / "validator.ps1",
            require_clean=True,
        )


def test_development_freeze_rejects_policy_inventory_from_another_source(tmp_path, monkeypatch):
    root, inventory = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        freeze,
        "resolve_source_identity",
        lambda cwd: SourceIdentity("8" * 40, True, "test-fixture"),
    )

    with pytest.raises(ValueError, match="different source identities"):
        freeze.build_development_freeze_candidate(
            root,
            policy_inventory=inventory,
            validator_path=root / "validator.ps1",
        )
