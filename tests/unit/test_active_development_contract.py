from __future__ import annotations

import copy
from pathlib import Path

import pytest

from calo_rpd_studio.scripts import verify_active_version as verifier


ROOT = Path(__file__).resolve().parents[2]


def _report_with_status(monkeypatch, change):
    original = verifier._load_json
    status = copy.deepcopy(original(ROOT / "ACTIVE_DEVELOPMENT_STATUS.json"))
    change(status)

    def load(path):
        if path.name == "ACTIVE_DEVELOPMENT_STATUS.json":
            return status
        return original(path)

    monkeypatch.setattr(verifier, "_load_json", load)
    return verifier.verify_active_version(ROOT)


def test_reviewed_development_source_matches_its_status_contract():
    report = verifier.verify_active_version(ROOT)
    assert report["passed"], report
    assert report["details"]["active_status_runtime_contract_failures"] == []
    assert report["expected"]["stage"] == "development"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", 7),
        ("phase", 6.0),
        ("supported_execution_modes", ["cuda-preferred", "cpu-only", "xpu"]),
        ("supported_execution_purposes", ["exploratory"]),
        ("intel_xpu_executable", True),
        ("intel_xpu_executable", 0),
        ("safe_memory_admission_fraction", 0.81),
        ("phase_1_validation", None),
        ("phase_6_goal", "obsolete_goal"),
        ("phase_6_coding", "obsolete_implementation"),
        ("phase_6_validation", "passed_without_current_source_evidence"),
        ("phase_6_execution_controller", "ownership_checks_removed"),
    ],
)
def test_status_drift_is_rejected_with_field_diagnostics(monkeypatch, field, value):
    report = _report_with_status(monkeypatch, lambda status: status.update({field: value}))
    assert report["passed"] is False
    assert report["checks"]["active_status_runtime_contract"] is False
    failures = report["details"]["active_status_runtime_contract_failures"]
    assert field in {item["field"] for item in failures}


def test_missing_and_multiple_status_requirements_are_all_reported(monkeypatch):
    def change(status):
        status.pop("phase_6_validation")
        status["safe_memory_admission_fraction"] = 0.95

    report = _report_with_status(monkeypatch, change)
    failures = {
        item["field"]: item for item in report["details"]["active_status_runtime_contract_failures"]
    }
    assert report["passed"] is False
    assert failures["phase_6_validation"]["present"] is False
    assert failures["safe_memory_admission_fraction"]["actual"] == 0.95


@pytest.mark.parametrize(
    "field",
    [
        "release_candidate",
        "final_release",
        "final_freeze_available",
        "release_qualification_complete",
        "protected_case_evidence_open",
        "policy_training_authorized_by_status",
        "policy_evaluation_authorized_by_status",
    ],
)
def test_status_reconciliation_never_grants_execution_or_release_authority(monkeypatch, field):
    report = _report_with_status(monkeypatch, lambda status: status.update({field: True}))
    assert report["passed"] is False
