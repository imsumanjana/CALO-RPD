"""Authenticated, read-only evidence binding for TSH-CALO parameter studies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .tsh_calo_feasibility_assessment import validate_tsh_calo_feasibility_assessment
from .tsh_calo_training_campaign import (
    TSHCALOTrainingCampaignPlan,
    parse_tsh_calo_extension_plan,
    tsh_calo_training_compatibility_contract,
    validate_tsh_calo_training_compatibility_contract,
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256_text(value: Any, label: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return text


@dataclass(frozen=True, slots=True)
class VerifiedTrainingInfluenceCampaign:
    """One candidate whose independent variable and outcome evidence are mutually authenticated."""

    candidate_sha256: str
    plan: TSHCALOTrainingCampaignPlan
    ratings: dict
    training_execution_plan_sha256: str
    training_compatibility_sha256: str
    assessment_comparison_protocol_sha256: str
    assessment_evidence_sha256: str

    def cohort_identity(self) -> dict:
        return {
            "training_compatibility_sha256": self.training_compatibility_sha256,
            "assessment_comparison_protocol_sha256": self.assessment_comparison_protocol_sha256,
        }


def verify_training_influence_campaign(
    campaign: dict,
    assessment: dict,
    *,
    expected_candidate_sha256: str,
) -> VerifiedTrainingInfluenceCampaign:
    """Bind a saved training plan to the exact candidate and exact assessment design.

    Source commits are deliberately not used as a scientific compatibility key.  They remain
    provenance.  Compatibility is determined by the authenticated architecture/training contract,
    while outcome comparability is determined by the exact assessment comparison protocol.
    """

    expected_sha = _sha256_text(expected_candidate_sha256, "candidate SHA-256")
    directory_text = str(campaign.get("directory", "")).strip()
    plan_text = str(campaign.get("plan", "")).strip()
    candidate_text = str(campaign.get("policy_candidate", "")).strip()
    if not directory_text or not plan_text or not candidate_text:
        raise ValueError("completed training campaign is missing retained paths")
    directory = Path(directory_text).expanduser().resolve(strict=True)
    plan_path = Path(plan_text).expanduser().resolve(strict=True)
    candidate_path = Path(candidate_text).expanduser().resolve(strict=True)
    if not directory.is_dir() or directory not in plan_path.parents and plan_path.parent != directory:
        raise ValueError("training plan is outside the completed campaign")
    if not candidate_path.is_file():
        raise ValueError("training candidate is unavailable")
    actual_sha = _file_sha256(candidate_path)
    if actual_sha != expected_sha:
        raise ValueError("training candidate checksum differs from the registered policy")

    plan_payload = _read_object(plan_path, "training plan")
    plan = parse_tsh_calo_extension_plan(plan_payload)
    plan_sha = plan.execution_plan_sha256()
    status = _read_object(directory / "training_status.json", "training status")
    if status.get("state") != "completed":
        raise ValueError("training campaign is not complete")
    if str(status.get("plan_sha256", "")).lower() != plan_sha:
        raise ValueError("training status does not bind the retained plan")

    manifest_path = candidate_path.parent / "training_manifest.json"
    if not manifest_path.is_file():
        manifest_path = directory / "training_manifest.json"
    manifest = _read_object(manifest_path, "training manifest")
    if str(manifest.get("execution_plan_sha256", "")).lower() != plan_sha:
        raise ValueError("training manifest does not bind the retained plan")
    if str(manifest.get("scientific_design_sha256", "")).lower() != plan.scientific_design_hash():
        raise ValueError("training manifest scientific design differs from the retained plan")
    if str(manifest.get("seed_manifest_sha256", "")).lower() != plan.seed_manifest_sha256():
        raise ValueError("training manifest seed design differs from the retained plan")
    ensemble = dict(manifest.get("ensemble_candidate", {}) or {})
    if ensemble:
        if str(ensemble.get("sha256", "")).lower() != expected_sha:
            raise ValueError("training manifest candidate checksum differs from the selected model")
    recorded_contract = manifest.get("training_compatibility_contract")
    if recorded_contract is None:
        raise ValueError("training compatibility evidence is unavailable")
    validate_tsh_calo_training_compatibility_contract(recorded_contract, plan)
    expected_contract = tsh_calo_training_compatibility_contract(plan)
    training_compatibility_sha = _canonical_sha256(expected_contract)

    assessment_candidate = str(assessment.get("candidate_sha256", "")).strip().lower()
    if assessment_candidate and assessment_candidate != expected_sha:
        raise ValueError("assessment belongs to another candidate")
    protocol = dict(assessment.get("comparison_protocol", {}) or {})
    protocol_sha = _sha256_text(
        assessment.get("comparison_protocol_sha256", ""),
        "assessment comparison protocol SHA-256",
    )
    if not protocol or _canonical_sha256(protocol) != protocol_sha:
        raise ValueError("assessment comparison protocol checksum is inconsistent")
    ratings = dict(assessment.get("feasibility_assessment", {}) or {})
    validate_tsh_calo_feasibility_assessment(ratings)
    evidence_sha = _sha256_text(
        assessment.get("evidence_artifact_sha256", ""),
        "assessment evidence SHA-256",
    )
    return VerifiedTrainingInfluenceCampaign(
        candidate_sha256=expected_sha,
        plan=plan,
        ratings=ratings,
        training_execution_plan_sha256=plan_sha,
        training_compatibility_sha256=training_compatibility_sha,
        assessment_comparison_protocol_sha256=protocol_sha,
        assessment_evidence_sha256=evidence_sha,
    )
