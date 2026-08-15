"""Fail-closed admission and comparison of independent TSH-CALO qualification evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tsh_calo_qualification import (
    qualification_config,
    load_tsh_calo_qualification_receipt,
)
from .tsh_calo_qualification_campaign import (
    LEGACY_TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
    QUALIFICATION_CELL_INDEX_FILE,
    QUALIFICATION_COMPLETION_FILE,
    QUALIFICATION_EVENT_LOG_FILE,
    QUALIFICATION_INFRASTRUCTURE_DIRECTORY,
    QUALIFICATION_STATUS_FILE,
    TSH_CALO_QUALIFICATION_CELL_INDEX_SCHEMA,
    TSH_CALO_QUALIFICATION_CELL_SUCCESS_SCHEMA,
    TSH_CALO_QUALIFICATION_COMPLETION_SCHEMA,
    TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
    TSH_CALO_QUALIFICATION_STATUS_SCHEMA,
    TSHCALOQualificationPlan,
    grade_tsh_calo_qualification_evidence,
    tsh_calo_qualification_cell_identity,
)
from .tsh_calo_feasibility_assessment import (
    TSH_CALO_FEASIBILITY_ADMISSION_SCHEMA,
    TSH_CALO_FEASIBILITY_ASSESSMENT_SCHEMA,
    TSH_CALO_FEASIBILITY_COMPLETION_SCHEMA,
    build_tsh_calo_feasibility_assessment,
    validate_tsh_calo_feasibility_assessment,
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json_file(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Qualification artifact is not a regular file: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Qualification artifact is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Qualification artifact must contain one JSON object: {path.name}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_transactional_completion(
    source: Path,
    *,
    plan: TSHCALOQualificationPlan,
    evidence: dict,
    expected_policy_sha256: str,
    assessment: bool = False,
) -> dict:
    """Require one complete unique-cell set and the final atomic completion authority."""

    completion_path = source / QUALIFICATION_COMPLETION_FILE
    index_path = source / QUALIFICATION_CELL_INDEX_FILE
    status_path = source / QUALIFICATION_STATUS_FILE
    event_path = source / QUALIFICATION_EVENT_LOG_FILE
    for path in (completion_path, index_path, status_path, event_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Transactional qualification artifact is missing: {path.name}")
    completion = _read_json_file(completion_path)
    index = _read_json_file(index_path)
    status = _read_json_file(status_path)
    expected_completion_schema = (
        TSH_CALO_FEASIBILITY_COMPLETION_SCHEMA
        if assessment
        else TSH_CALO_QUALIFICATION_COMPLETION_SCHEMA
    )
    if completion.get("schema_version") != expected_completion_schema:
        raise ValueError("Qualification/assessment completion schema is incompatible")
    if index.get("schema_version") != TSH_CALO_QUALIFICATION_CELL_INDEX_SCHEMA:
        raise ValueError("Qualification terminal-cell index schema is incompatible")
    if status.get("schema_version") != TSH_CALO_QUALIFICATION_STATUS_SCHEMA:
        raise ValueError("Qualification status schema is incompatible")
    plan_sha256 = plan.execution_plan_sha256()
    seed_sha256 = plan.seed_manifest_sha256()
    bindings = {
        "qualification_run_id": plan.qualification_run_id,
        "qualification_plan_sha256": plan_sha256,
        "source_policy_sha256": expected_policy_sha256,
    }
    for artifact_name, payload in (
        ("completion", completion),
        ("terminal-cell index", index),
    ):
        for field_name, expected in bindings.items():
            if payload.get(field_name) != expected:
                raise ValueError(
                    f"Qualification {artifact_name} has an invalid {field_name} binding"
                )
    expected_cells = len(plan.development_cases) * plan.runs * 2
    if (
        int(completion.get("committed_unique_cells", -1)) != expected_cells
        or int(completion.get("failed_scientific_cells", -1)) != 0
        or int(completion.get("infrastructure_incident_count", -1)) != 0
    ):
        raise ValueError("Qualification/assessment completion cardinality is not admissible")
    if assessment:
        if (
            completion.get("assessment_complete") is not True
            or completion.get("automated_suitability_decision") is not None
            or completion.get("authority_boundary")
            != "assessment_completion_only_scientist_decides_no_selection_activation_or_binding"
        ):
            raise ValueError("Feasibility completion assigned or implied an automated decision")
    elif (
        completion.get("passed") is not True
        or completion.get("authority_boundary")
        != "completion_only_no_registration_activation_or_experiment_binding"
    ):
        raise ValueError("Qualification completion does not authorize passed evidence admission")
    if (
        int(index.get("expected_cells", -1)) != expected_cells
        or int(index.get("committed_unique_cells", -1)) != expected_cells
        or completion.get("seed_manifest_sha256") != seed_sha256
        or completion.get("evidence_artifact_sha256")
        != _file_sha256(source / "qualification_evidence.json")
        or completion.get("terminal_cell_index_sha256") != _file_sha256(index_path)
        or completion.get("qualification_event_log_sha256") != _file_sha256(event_path)
        or completion.get("qualification_status_sha256") != _file_sha256(status_path)
    ):
        raise ValueError("Qualification completion checksums or cardinality are inconsistent")
    expected_status = "completed_assessed" if assessment else "completed_qualified"
    if (
        status.get("state") != expected_status
        or status.get("qualification_plan_sha256") != plan_sha256
        or status.get("evidence_sha256") != completion.get("evidence_artifact_sha256")
        or status.get("receipt_sha256") != completion.get("receipt_sha256")
    ):
        raise ValueError("Qualification/assessment completed status is not admission-authoritative")
    if assessment:
        if (
            status.get("qualification_receipt_permitted") is not False
            or status.get("feasibility_receipt_permitted") is not True
            or status.get("scientist_decision") != "not_recorded"
        ):
            raise ValueError("Feasibility status contains an automated or missing authority state")
    elif status.get("qualification_receipt_permitted") is not True:
        raise ValueError("Qualification completed status does not permit its receipt")
    incident_directory = source / QUALIFICATION_INFRASTRUCTURE_DIRECTORY
    if incident_directory.is_dir() and any(incident_directory.glob("*.json")):
        raise ValueError("Qualification retains an infrastructure incident")

    paired_runs = list(plan.seed_manifest().get("paired_runs", []))
    expected_by_identity: dict[str, dict] = {}
    cell_index = 0
    for case_name in plan.development_cases:
        for run_index, seeds in enumerate(paired_runs):
            for label in ("baseline", "candidate"):
                cell_index += 1
                identity = tsh_calo_qualification_cell_identity(
                    plan,
                    case_name=case_name,
                    run_index=run_index,
                    label=label,
                    seeds=seeds,
                )
                expected_by_identity[identity] = {
                    "case": case_name,
                    "run_index": run_index,
                    "label": label,
                    "seeds": seeds,
                    "cell_index": cell_index,
                    "artifact_path": f"records/{case_name}-{run_index:03d}-{label}.json",
                }
    entries = list(index.get("entries", []))
    if len(entries) != expected_cells or not all(isinstance(item, dict) for item in entries):
        raise ValueError("Qualification terminal-cell index is incomplete")
    observed_identities = [str(item.get("cell_identity", "")) for item in entries]
    if len(set(observed_identities)) != expected_cells or set(observed_identities) != set(
        expected_by_identity
    ):
        raise ValueError("Qualification terminal-cell identities are duplicate or unexpected")
    for entry in entries:
        identity = str(entry["cell_identity"])
        expected = expected_by_identity[identity]
        if (
            entry.get("terminal_state") != "committed_success"
            or int(entry.get("cell_index", -1)) != expected["cell_index"]
            or entry.get("case") != expected["case"]
            or int(entry.get("run_index", -1)) != expected["run_index"]
            or entry.get("label") != expected["label"]
            or entry.get("artifact_path") != expected["artifact_path"]
        ):
            raise ValueError("Qualification terminal-cell index contains an invalid disposition")
        artifact_path = (source / str(entry["artifact_path"])).resolve()
        try:
            artifact_path.relative_to(source)
        except ValueError as exc:
            raise ValueError(
                "Qualification terminal-cell artifact escaped its evidence root"
            ) from exc
        payload = _read_json_file(artifact_path)
        if _file_sha256(artifact_path) != entry.get("artifact_sha256"):
            raise ValueError("Qualification terminal-cell artifact checksum changed")
        required = {
            "schema_version": TSH_CALO_QUALIFICATION_CELL_SUCCESS_SCHEMA,
            "terminal_state": "committed_success",
            "cell_identity": identity,
            "cell_index": expected["cell_index"],
            "total_cells": expected_cells,
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan_sha256,
            "source_policy_sha256": expected_policy_sha256,
            "case": expected["case"],
            "run_index": expected["run_index"],
            "label": expected["label"],
            "seeds": expected["seeds"],
        }
        if any(payload.get(key) != value for key, value in required.items()) or int(
            payload.get("evaluations", -1)
        ) != int(plan.max_evaluations):
            raise ValueError("Qualification terminal success artifact violates its frozen binding")
    terminal_index = dict(evidence.get("terminal_cell_index", {}) or {})
    if (
        terminal_index.get("schema_version") != TSH_CALO_QUALIFICATION_CELL_INDEX_SCHEMA
        or terminal_index.get("sha256") != _file_sha256(index_path)
        or int(terminal_index.get("committed_unique_cells", -1)) != expected_cells
        or list(evidence.get("infrastructure_incidents", []))
    ):
        raise ValueError("Qualification evidence does not bind its clean terminal-cell index")
    return completion


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Qualification evidence has a non-finite {label}")
    return number


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _minimum(values: list[Any], label: str) -> float:
    if not values:
        raise ValueError(f"Qualification evidence lacks {label}")
    return min(_finite(value, label) for value in values)


def _maximum(values: list[Any], label: str) -> float:
    if not values:
        raise ValueError(f"Qualification evidence lacks {label}")
    return max(_finite(value, label) for value in values)


def _confidence_lower(case: dict) -> Any:
    interval = list(case.get("feasible_probability_difference_ci95", []))
    if len(interval) != 2:
        raise ValueError("Qualification evidence has an invalid feasibility confidence interval")
    return interval[0]


def qualification_comparison_protocol(plan: TSHCALOQualificationPlan) -> dict:
    """Return the frozen design fields required for a direct policy comparison.

    Candidate identity, run labels, source checkout identity, and local execution device are
    intentionally excluded. The paired seeds, cases, budgets, analysis definitions, calibration
    design, decision thresholds, and architecture contract remain part of the comparison identity.
    """

    payload = plan.to_dict()
    for key in (
        "qualification_run_id",
        "source_commit",
        "source_tracked_clean",
        "candidate_path",
        "candidate_sha256",
        "inference_device",
        "allow_cpu_fallback",
    ):
        payload.pop(key, None)
    candidate_contract = dict(payload.pop("candidate_contract", {}) or {})
    component_evidence = dict(payload.pop("component_evidence", {}) or {})
    if candidate_contract:
        for key in (
            "candidate_sha256",
            "member_candidate_sha256",
            "member_training_design_sha256",
            "training_provenance_sha256",
        ):
            candidate_contract.pop(key, None)
        payload["candidate_architecture_contract"] = candidate_contract
    elif component_evidence:
        # Retained legacy evidence remains comparable without governing new qualification runs.
        payload["legacy_qualified_components"] = sorted(component_evidence)
    return payload


def _evidence_summary(evidence: dict) -> dict:
    cases = list(evidence.get("case_evidence", []) or [])
    if not cases or not all(isinstance(item, dict) for item in cases):
        raise ValueError("Qualification evidence lacks per-case results")
    anytime_rows = [
        dict(row)
        for case in cases
        for row in dict(case.get("anytime", {}) or {}).values()
        if isinstance(row, dict)
    ]
    anytime_objectives = [
        row.get("median_relative_objective_improvement")
        for row in anytime_rows
        if row.get("median_relative_objective_improvement") is not None
    ]
    return {
        "case_count": len(cases),
        "minimum_candidate_feasible_probability": _minimum(
            [item.get("candidate_feasible_probability") for item in cases],
            "candidate feasible probability",
        ),
        "minimum_feasibility_ci_lower": _minimum(
            [_confidence_lower(item) for item in cases],
            "feasibility confidence lower bound",
        ),
        "minimum_relative_objective_improvement": _minimum(
            [item.get("median_relative_objective_improvement") for item in cases],
            "relative objective improvement",
        ),
        "minimum_objective_win_rate": _minimum(
            [item.get("objective_win_rate") for item in cases], "objective win rate"
        ),
        "minimum_rank_biserial": _minimum(
            [item.get("paired_rank_biserial") for item in cases], "paired effect size"
        ),
        "maximum_holm_p": _maximum(
            [item.get("holm_p") for item in cases], "Holm-controlled p-value"
        ),
        "minimum_anytime_feasibility_difference": _minimum(
            [row.get("feasible_probability_difference") for row in anytime_rows],
            "anytime feasibility difference",
        ),
        "minimum_anytime_objective_improvement": (
            _minimum(anytime_objectives, "anytime objective improvement")
            if anytime_objectives
            else None
        ),
    }


@dataclass(frozen=True, slots=True)
class VerifiedQualificationEvidence:
    directory: str
    qualification_id: str
    policy_sha256: str
    grade: str
    score: float
    config: dict
    metrics: dict


@dataclass(frozen=True, slots=True)
class VerifiedFeasibilityAssessment:
    directory: str
    assessment_id: str
    policy_sha256: str
    score: float
    config: dict
    metrics: dict


def inspect_qualification_evidence(
    directory: str | Path,
    *,
    expected_policy_sha256: str,
) -> VerifiedQualificationEvidence:
    """Verify one completed formal evidence directory without changing registry state."""

    source = Path(directory).expanduser()
    if source.is_symlink():
        raise ValueError("Symbolic-link qualification directories are not accepted")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Qualification evidence location is not a directory")

    plan_path = source / "qualification_plan.json"
    seed_path = source / "seed_manifest.json"
    evidence_path = source / "qualification_evidence.json"
    receipt_path = source / "qualification_receipt.json"
    plan = TSHCALOQualificationPlan.from_dict(_read_json_file(plan_path))
    seed_manifest = _read_json_file(seed_path)
    evidence = _read_json_file(evidence_path)
    receipt_payload = _read_json_file(receipt_path)

    expected_sha = str(expected_policy_sha256).strip().lower()
    if plan.mode != "formal" or plan.source_tracked_clean is not True:
        raise ValueError("Only a formal qualification from a clean tracked source can be admitted")
    if plan.candidate_sha256.lower() != expected_sha:
        raise ValueError("Qualification plan belongs to a different policy checkpoint")
    if _canonical_sha256(seed_manifest) != plan.seed_manifest_sha256():
        raise ValueError("Qualification seed manifest checksum does not match the frozen plan")
    evidence_schema = str(evidence.get("schema_version", ""))
    if evidence_schema not in {
        LEGACY_TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
        TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
    }:
        raise ValueError("Qualification evidence schema is incompatible")
    completion = (
        _verify_transactional_completion(
            source,
            plan=plan,
            evidence=evidence,
            expected_policy_sha256=expected_sha,
        )
        if evidence_schema == TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA
        else {}
    )
    if evidence.get("analysis_schema_version") != plan.analysis_schema_version:
        raise ValueError("Qualification evidence paired-analysis schema is incompatible")
    if evidence.get("relative_improvement_version") != plan.relative_improvement_version:
        raise ValueError("Qualification evidence improvement definition is incompatible")
    if float(evidence.get("objective_scale_floor", float("nan"))) != float(
        plan.objective_scale_floor
    ):
        raise ValueError("Qualification evidence objective scale is inconsistent")
    if str(evidence.get("qualification_run_id", "")) != plan.qualification_run_id:
        raise ValueError("Qualification evidence run identity does not match its plan")
    if str(evidence.get("source_commit", "")).lower() != plan.source_commit.lower():
        raise ValueError("Qualification evidence source identity does not match its plan")
    if evidence.get("source_tracked_clean") is not True:
        raise ValueError("Qualification evidence did not retain a clean tracked source state")
    if str(evidence.get("source_policy_sha256", "")).lower() != expected_sha:
        raise ValueError("Qualification evidence belongs to a different policy checkpoint")
    if str(evidence.get("qualification_plan_sha256", "")).lower() != (plan.execution_plan_sha256()):
        raise ValueError("Qualification evidence plan checksum is inconsistent")
    if str(evidence.get("scientific_design_sha256", "")).lower() != (
        plan.scientific_design_sha256()
    ):
        raise ValueError("Qualification evidence scientific design checksum is inconsistent")
    if str(evidence.get("seed_manifest_sha256", "")).lower() != (plan.seed_manifest_sha256()):
        raise ValueError("Qualification evidence seed identity is inconsistent")
    if tuple(evidence.get("development_cases", ())) != tuple(plan.development_cases):
        raise ValueError("Qualification evidence used a different development-case design")
    if evidence.get("protected_cases_opened") is not False:
        raise ValueError("Qualification evidence did not prove protected-case closure")
    if evidence.get("authority_boundary") != (
        "independent_qualification_only_no_registration_or_activation"
    ):
        raise ValueError("Qualification evidence authority boundary is incompatible")
    candidate_contract = dict(evidence.get("candidate_contract", {}) or {})
    if plan.candidate_contract:
        if candidate_contract != dict(plan.candidate_contract):
            raise ValueError(
                "Qualification evidence candidate architecture contract differs from its plan"
            )
        if str(candidate_contract.get("candidate_sha256", "")).lower() != expected_sha:
            raise ValueError("Qualification candidate contract belongs to another checkpoint")
    else:
        component_evidence = dict(evidence.get("component_evidence", {}) or {})
        if set(component_evidence) != set("ABCDE") or any(
            not isinstance(item, dict)
            or item.get("accepted") is not True
            or not _is_sha256(item.get("sha256", ""))
            for item in component_evidence.values()
        ):
            raise ValueError("Legacy qualification evidence lacks its accepted component records")
        for component, item in component_evidence.items():
            planned = dict(plan.component_evidence.get(component, {}) or {})
            if str(item.get("sha256", "")).lower() != str(planned.get("sha256", "")).lower():
                raise ValueError(
                    f"Legacy qualification component {component} differs from the frozen plan"
                )

    records = dict(evidence.get("records", {}) or {})
    expected_records = len(plan.development_cases) * plan.runs * 2
    if (
        int(records.get("expected", -1)) != expected_records
        or int(records.get("completed", -1)) != expected_records
        or int(records.get("failed", -1)) != 0
    ):
        raise ValueError("Qualification evidence is incomplete or retains failed paired cells")
    decision = dict(evidence.get("decision", {}) or {})
    cases = list(evidence.get("case_evidence", []) or [])
    if [str(item.get("case", "")) for item in cases] != list(plan.development_cases):
        raise ValueError("Qualification per-case evidence does not match the frozen case order")
    recalculated_decision = grade_tsh_calo_qualification_evidence(plan, cases, [])
    if decision != recalculated_decision:
        raise ValueError("Qualification decision does not match the canonical frozen gates")
    if decision.get("passed") is not True or decision.get("grade") != "A":
        raise ValueError("Qualification evidence did not pass every frozen formal gate")
    if list(decision.get("reasons", []) or []):
        raise ValueError("Passed qualification evidence contains unresolved rejection reasons")
    score = _finite(decision.get("score"), "qualification score")

    evidence_sha = _file_sha256(evidence_path)
    receipt = load_tsh_calo_qualification_receipt(
        {"tsh_calo_qualification_receipt": receipt_payload},
        expected_policy_sha256=expected_sha,
    )
    if completion and completion.get("receipt_sha256") != _file_sha256(receipt_path):
        raise ValueError("Qualification completion does not bind the retained receipt")
    if receipt.qualification_run_id != plan.qualification_run_id:
        raise ValueError("Qualification receipt run identity does not match its plan")
    if receipt.source_commit.lower() != plan.source_commit.lower():
        raise ValueError("Qualification receipt source identity does not match its plan")
    if receipt.qualification_protocol_sha256 != plan.scientific_design_sha256():
        raise ValueError("Qualification receipt protocol checksum does not match its plan")
    if receipt.seed_manifest_sha256 != plan.seed_manifest_sha256():
        raise ValueError("Qualification receipt seed checksum does not match its plan")
    if receipt.evidence_artifact_sha256 != evidence_sha:
        raise ValueError("Qualification receipt does not bind the retained evidence file")
    if tuple(receipt.development_cases) != tuple(plan.development_cases):
        raise ValueError("Qualification receipt development cases do not match its plan")
    if str(evidence.get("ood_calibration_sha256", "")).lower() != (receipt.ood_calibration_sha256):
        raise ValueError("Qualification evidence and receipt bind different OOD calibration")

    comparison_protocol = qualification_comparison_protocol(plan)
    metrics = {
        "admission_schema_version": "tsh-calo-policy-qualification-admission-v1",
        "evidence_directory": str(source),
        "qualification_run_id": plan.qualification_run_id,
        "qualification_mode": plan.mode,
        "source_commit": plan.source_commit,
        "qualification_plan_sha256": plan.execution_plan_sha256(),
        "scientific_design_sha256": plan.scientific_design_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
        "evidence_artifact_sha256": evidence_sha,
        "receipt_sha256": receipt.receipt_sha256,
        "comparison_protocol": comparison_protocol,
        "comparison_protocol_sha256": _canonical_sha256(comparison_protocol),
        "development_cases": list(plan.development_cases),
        "runs_per_case": int(plan.runs),
        "population_size": int(plan.population_size),
        "max_evaluations": int(plan.max_evaluations),
        "summary": _evidence_summary(evidence),
        "decision": decision,
        "claim_scope": str(decision.get("claim_scope", "")),
    }
    return VerifiedQualificationEvidence(
        directory=str(source),
        qualification_id=plan.qualification_run_id,
        policy_sha256=expected_sha,
        grade="A",
        score=score,
        config=qualification_config(receipt.as_dict()),
        metrics=metrics,
    )


def inspect_feasibility_assessment(
    directory: str | Path,
    *,
    expected_policy_sha256: str,
) -> VerifiedFeasibilityAssessment:
    """Verify a complete measurement dossier without making a suitability decision."""

    source = Path(directory).expanduser()
    if source.is_symlink():
        raise ValueError("Symbolic-link feasibility directories are not accepted")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Feasibility evidence location is not a directory")
    plan = TSHCALOQualificationPlan.from_dict(_read_json_file(source / "qualification_plan.json"))
    seed_manifest = _read_json_file(source / "seed_manifest.json")
    evidence_path = source / "qualification_evidence.json"
    receipt_path = source / "qualification_receipt.json"
    evidence = _read_json_file(evidence_path)
    receipt_payload = _read_json_file(receipt_path)
    expected_sha = str(expected_policy_sha256).strip().lower()
    if plan.mode != "formal" or plan.source_tracked_clean is not True:
        raise ValueError("A selectable feasibility assessment requires a clean formal source")
    if plan.candidate_sha256.lower() != expected_sha:
        raise ValueError("Feasibility plan belongs to a different policy checkpoint")
    if _canonical_sha256(seed_manifest) != plan.seed_manifest_sha256():
        raise ValueError("Feasibility seed manifest checksum does not match the frozen plan")
    if evidence.get("schema_version") != TSH_CALO_FEASIBILITY_ASSESSMENT_SCHEMA:
        raise ValueError("Feasibility assessment schema is incompatible")
    completion = _verify_transactional_completion(
        source,
        plan=plan,
        evidence=evidence,
        expected_policy_sha256=expected_sha,
        assessment=True,
    )
    expected_bindings = {
        "analysis_schema_version": plan.analysis_schema_version,
        "relative_improvement_version": plan.relative_improvement_version,
        "qualification_run_id": plan.qualification_run_id,
        "source_commit": plan.source_commit,
        "source_policy_sha256": expected_sha,
        "qualification_plan_sha256": plan.execution_plan_sha256(),
        "scientific_design_sha256": plan.scientific_design_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
    }
    for key, expected in expected_bindings.items():
        observed = evidence.get(key)
        if key in {"source_commit", "source_policy_sha256"}:
            observed, expected = str(observed).lower(), str(expected).lower()
        if observed != expected:
            raise ValueError(f"Feasibility evidence has an invalid {key} binding")
    if float(evidence.get("objective_scale_floor", float("nan"))) != float(
        plan.objective_scale_floor
    ):
        raise ValueError("Feasibility evidence objective scale is inconsistent")
    if evidence.get("source_tracked_clean") is not True:
        raise ValueError("Feasibility evidence did not retain a clean tracked source")
    if tuple(evidence.get("development_cases", ())) != tuple(plan.development_cases):
        raise ValueError("Feasibility evidence used a different case design")
    if evidence.get("protected_cases_opened") is not False:
        raise ValueError("Feasibility evidence did not prove protected-case closure")
    if evidence.get("authority_boundary") != (
        "measurement_only_scientist_decides_no_selection_activation_or_experiment_binding"
    ):
        raise ValueError("Feasibility evidence assigned incompatible decision authority")
    candidate_contract = dict(evidence.get("candidate_contract", {}) or {})
    if candidate_contract != dict(plan.candidate_contract):
        raise ValueError("Feasibility candidate architecture contract differs from its plan")
    if str(candidate_contract.get("candidate_sha256", "")).lower() != expected_sha:
        raise ValueError("Feasibility candidate contract belongs to another checkpoint")
    records = dict(evidence.get("records", {}) or {})
    expected_records = len(plan.development_cases) * plan.runs * 2
    if (
        int(records.get("expected", -1)) != expected_records
        or int(records.get("completed", -1)) != expected_records
        or int(records.get("failed", -1)) != 0
        or int(records.get("committed_unique", -1)) != expected_records
    ):
        raise ValueError("Feasibility assessment lacks a complete unique paired-cell set")
    cases = list(evidence.get("case_evidence", []) or [])
    if [str(item.get("case", "")) for item in cases] != list(plan.development_cases):
        raise ValueError("Feasibility per-case evidence does not match the frozen case order")
    assessment = dict(evidence.get("feasibility_assessment", {}) or {})
    validate_tsh_calo_feasibility_assessment(assessment)
    recalculated = build_tsh_calo_feasibility_assessment(
        cases=cases,
        expected_case_order=plan.development_cases,
    )
    if assessment != recalculated:
        raise ValueError("Feasibility ratings do not match the retained cell measurements")
    evidence_sha = _file_sha256(evidence_path)
    receipt = load_tsh_calo_qualification_receipt(
        {"tsh_calo_qualification_receipt": receipt_payload},
        expected_policy_sha256=expected_sha,
    )
    if completion.get("receipt_sha256") != _file_sha256(receipt_path):
        raise ValueError("Feasibility completion does not bind the retained receipt")
    if (
        receipt.qualification_run_id != plan.qualification_run_id
        or receipt.source_commit.lower() != plan.source_commit.lower()
        or receipt.qualification_protocol_sha256 != plan.scientific_design_sha256()
        or receipt.seed_manifest_sha256 != plan.seed_manifest_sha256()
        or receipt.evidence_artifact_sha256 != evidence_sha
        or tuple(receipt.development_cases) != tuple(plan.development_cases)
        or str(evidence.get("ood_calibration_sha256", "")).lower() != receipt.ood_calibration_sha256
    ):
        raise ValueError("Feasibility receipt does not bind the frozen assessment")
    protocol = qualification_comparison_protocol(plan)
    metrics = {
        "admission_schema_version": TSH_CALO_FEASIBILITY_ADMISSION_SCHEMA,
        "evidence_directory": str(source),
        "assessment_id": plan.qualification_run_id,
        "candidate_sha256": expected_sha,
        "source_commit": plan.source_commit,
        "assessment_plan_sha256": plan.execution_plan_sha256(),
        "scientific_design_sha256": plan.scientific_design_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
        "evidence_artifact_sha256": evidence_sha,
        "receipt_sha256": receipt.receipt_sha256,
        "comparison_protocol": protocol,
        "comparison_protocol_sha256": _canonical_sha256(protocol),
        "development_cases": list(plan.development_cases),
        "runs_per_case": int(plan.runs),
        "population_size": int(plan.population_size),
        "max_evaluations": int(plan.max_evaluations),
        "feasibility_assessment": assessment,
        "case_evidence": cases,
        "decision_authority": "scientist_only",
        "automated_suitability_decision": None,
    }
    return VerifiedFeasibilityAssessment(
        directory=str(source),
        assessment_id=plan.qualification_run_id,
        policy_sha256=expected_sha,
        score=float(assessment["overall_feasibility_score"]),
        config=qualification_config(receipt.as_dict()),
        metrics=metrics,
    )


def performance_vector(summary: dict) -> tuple[float, ...]:
    """Higher is better for every coordinate; significance is a confidence tie-breaker."""

    anytime_objective = summary.get("minimum_anytime_objective_improvement")
    return (
        float(summary["minimum_candidate_feasible_probability"]),
        float(summary["minimum_feasibility_ci_lower"]),
        float(summary["minimum_relative_objective_improvement"]),
        float(summary["minimum_objective_win_rate"]),
        float(summary["minimum_rank_biserial"]),
        float(summary["minimum_anytime_feasibility_difference"]),
        float(anytime_objective) if anytime_objective is not None else float("-inf"),
        -float(summary["maximum_holm_p"]),
    )


def pareto_dominates(left: dict, right: dict) -> bool:
    left_vector = performance_vector(left)
    right_vector = performance_vector(right)
    return all(a >= b for a, b in zip(left_vector, right_vector, strict=True)) and any(
        a > b for a, b in zip(left_vector, right_vector, strict=True)
    )
