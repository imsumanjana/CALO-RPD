from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from calo_rpd_studio.compute.source_identity import SourceIdentity
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    build_tsh_calo_qualification_receipt,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import OODCalibration
from calo_rpd_studio.scripts import create_release_preparation as preparation
from calo_rpd_studio.scripts import generate_distribution_manifests as distributions
from calo_rpd_studio.scripts import finalize_release_records as final_records
from calo_rpd_studio.scripts import release_policy_scope as scope
from calo_rpd_studio.scripts import verify_release_ci_contract as release_ci


def _seal(payload: dict) -> dict:
    result = dict(payload)
    result["decision_payload_sha256"] = scope._canonical_sha256(payload)
    return result


def _policy_free_decision() -> dict:
    payload = scope.disabled_template()
    payload.update(
        {
            "approval_granted": True,
            "decision_id": "owner-policy-free-v12",
            "selected_scope": scope.POLICY_FREE,
            "phase4_acceptance_receipt_sha256": "1" * 64,
            "post_transition_freeze_sha256": "2" * 64,
            "development_source_contract_sha256": "3" * 64,
            "acknowledge_scope_is_exact_and_immutable": True,
            "acknowledge_no_automatic_policy_activation": True,
        }
    )
    payload.pop("decision_payload_sha256")
    return _seal(payload)


def test_disabled_scope_template_cannot_authorize_a_release_policy_scope():
    template = scope.disabled_template()

    with pytest.raises(PermissionError, match="not been explicitly approved"):
        scope.validate_scope_decision(template)


def test_disabled_final_record_template_cannot_generate_release_records():
    template = final_records.disabled_authorization_template()

    with pytest.raises(PermissionError, match="not been explicitly authorized"):
        final_records.validate_authorization(template)


def test_final_record_json_loader_requires_an_object(tmp_path):
    invalid = tmp_path / "array.json"
    invalid.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        final_records._load_json_object(invalid, label="Test evidence")


def test_policy_free_scope_forbids_policy_identity_and_benefit_claims():
    decision = _policy_free_decision()
    assert scope.validate_scope_decision(decision)["selected_scope"] == scope.POLICY_FREE

    contaminated = dict(decision)
    contaminated["policy_sha256"] = "4" * 64
    contaminated["decision_payload_sha256"] = scope._canonical_sha256(scope._stable(contaminated))
    with pytest.raises(ValueError, match="cannot identify or include"):
        scope.validate_scope_decision(contaminated)


def test_new_policy_scope_requires_a_confined_exact_manifest(tmp_path):
    artifact = tmp_path / "new-policy.pt"
    artifact.write_bytes(b"new policy fixture")
    policy_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    qualification = tmp_path / "qualification-receipt.json"
    qualification_payload = build_tsh_calo_qualification_receipt(
        qualification_run_id="synthetic-release-scope-fixture",
        source_policy_sha256=policy_sha256,
        source_commit="a" * 40,
        qualification_protocol_sha256="6" * 64,
        seed_manifest_sha256="7" * 64,
        evidence_artifact_sha256="8" * 64,
        development_cases=("case30",),
        ood_calibration=OODCalibration(
            mean=np.zeros(2, dtype=float),
            scale=np.ones(2, dtype=float),
        ),
    )
    qualification.write_text(json.dumps(qualification_payload) + "\n", encoding="utf-8")
    qualification_file_sha256 = hashlib.sha256(qualification.read_bytes()).hexdigest()
    qualification_sha256 = qualification_payload["receipt_sha256"]
    manifest = tmp_path / "policy-manifest.json"
    manifest_payload = {
        "schema": scope.POLICY_MANIFEST_SCHEMA,
        "policy_id": "new-ae-policy",
        "policy_sha256": policy_sha256,
        "policy_artifact_path": artifact.name,
        "algorithm_id": "TSH-CALO",
        "initialization_policy_sha256": "",
        "change_f_enabled": False,
        "qualification_receipt_sha256": qualification_sha256,
        "phase4_acceptance_receipt_sha256": "1" * 64,
        "post_transition_freeze_sha256": "2" * 64,
    }
    manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
    payload = scope.disabled_template()
    payload.update(
        {
            "approval_granted": True,
            "decision_id": "owner-new-policy-v12",
            "selected_scope": scope.NEWLY_QUALIFIED_POLICY,
            "phase4_acceptance_receipt_sha256": "1" * 64,
            "post_transition_freeze_sha256": "2" * 64,
            "development_source_contract_sha256": "3" * 64,
            "policy_id": "new-ae-policy",
            "policy_sha256": policy_sha256,
            "policy_manifest_path": manifest.name,
            "policy_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "qualification_receipt_path": qualification.name,
            "qualification_receipt_file_sha256": qualification_file_sha256,
            "qualification_receipt_sha256": qualification_sha256,
            "algorithm_id": "TSH-CALO",
            "acknowledge_scope_is_exact_and_immutable": True,
            "acknowledge_no_automatic_policy_activation": True,
        }
    )
    payload.pop("decision_payload_sha256")
    decision = _seal(payload)

    assert scope.validate_scope_decision(decision, evidence_root=tmp_path)["selected_scope"] == (
        scope.NEWLY_QUALIFIED_POLICY
    )


def _evidence_fixture(root: Path) -> dict[str, str]:
    mapping = {}
    for name in sorted(preparation.REQUIRED_EVIDENCE):
        relative = f"evidence/{name}.json"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        mapping[name] = relative
    return mapping


def test_release_preparation_binds_complete_evidence_without_claiming_release(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "retained"
    evidence.mkdir()
    mapping = _evidence_fixture(evidence)
    monkeypatch.setattr(
        preparation,
        "resolve_source_identity",
        lambda cwd: SourceIdentity("a" * 40, True, "test-fixture"),
    )

    report = preparation.build_release_preparation(
        source,
        evidence,
        mapping,
        require_clean=True,
    )

    preparation.validate_release_preparation(report)
    assert report["policy_scope"]["status"] == "pending_explicit_decision"
    assert report["release_candidate"] is False
    assert report["release_ready"] is False
    assert report["final_release"] is False
    assert set(report["evidence"]) == preparation.REQUIRED_EVIDENCE

    forged = dict(report)
    forged["release_ready"] = True
    stable = {
        key: value
        for key, value in forged.items()
        if key not in {"created_at", "release_preparation_payload_sha256"}
    }
    forged["release_preparation_payload_sha256"] = preparation._canonical_sha256(stable)
    with pytest.raises(ValueError, match="cannot claim"):
        preparation.validate_release_preparation(forged)


def test_release_preparation_refuses_approved_scope_without_transition_evidence(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "retained"
    evidence.mkdir()
    mapping = _evidence_fixture(evidence)
    monkeypatch.setattr(
        preparation,
        "resolve_source_identity",
        lambda cwd: SourceIdentity("a" * 40, True, "test-fixture"),
    )

    with pytest.raises(ValueError, match="requires Phase 4 acceptance"):
        preparation.build_release_preparation(
            source,
            evidence,
            mapping,
            scope_decision=_policy_free_decision(),
        )


def test_release_preparation_rejects_an_incomplete_evidence_map(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    with pytest.raises(ValueError, match="evidence map mismatch"):
        preparation.build_release_preparation(source, evidence, {})


def test_distribution_member_name_rejects_traversal():
    with pytest.raises(ValueError, match="Unsafe distribution member"):
        distributions._safe_name("package/../outside.txt")


def test_release_ci_contract_covers_required_jobs_and_boundaries():
    report = release_ci.verify(Path(".github/workflows/ci.yml"))

    assert report["missing_job_count"] == 0
    assert report["physical_cuda_dispatch_gated"] is True
    assert report["publication_or_release_executed"] is False
