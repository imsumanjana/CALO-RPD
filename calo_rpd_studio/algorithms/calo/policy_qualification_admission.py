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
    TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
    TSHCALOQualificationPlan,
    grade_tsh_calo_qualification_evidence,
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


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Qualification evidence has a non-finite {label}")
    return number


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


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
    design, decision thresholds, and component set remain part of the comparison identity.
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
    component_evidence = dict(payload.pop("component_evidence", {}) or {})
    payload["qualified_components"] = sorted(component_evidence)
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
        raise ValueError(
            "Only a formal qualification from a clean tracked source can be admitted"
        )
    if plan.candidate_sha256.lower() != expected_sha:
        raise ValueError("Qualification plan belongs to a different policy checkpoint")
    if _canonical_sha256(seed_manifest) != plan.seed_manifest_sha256():
        raise ValueError("Qualification seed manifest checksum does not match the frozen plan")
    if evidence.get("schema_version") != TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA:
        raise ValueError("Qualification evidence schema is incompatible")
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
    if str(evidence.get("qualification_plan_sha256", "")).lower() != (
        plan.execution_plan_sha256()
    ):
        raise ValueError("Qualification evidence plan checksum is inconsistent")
    if str(evidence.get("scientific_design_sha256", "")).lower() != (
        plan.scientific_design_sha256()
    ):
        raise ValueError("Qualification evidence scientific design checksum is inconsistent")
    if str(evidence.get("seed_manifest_sha256", "")).lower() != (
        plan.seed_manifest_sha256()
    ):
        raise ValueError("Qualification evidence seed identity is inconsistent")
    if tuple(evidence.get("development_cases", ())) != tuple(plan.development_cases):
        raise ValueError("Qualification evidence used a different development-case design")
    if evidence.get("protected_cases_opened") is not False:
        raise ValueError("Qualification evidence did not prove protected-case closure")
    if evidence.get("authority_boundary") != (
        "independent_qualification_only_no_registration_or_activation"
    ):
        raise ValueError("Qualification evidence authority boundary is incompatible")
    component_evidence = dict(evidence.get("component_evidence", {}) or {})
    if set(component_evidence) != set("ABCDE") or any(
        not isinstance(item, dict)
        or item.get("accepted") is not True
        or not _is_sha256(item.get("sha256", ""))
        for item in component_evidence.values()
    ):
        raise ValueError("Qualification evidence lacks accepted frozen A-E component evidence")
    for component, item in component_evidence.items():
        planned = dict(plan.component_evidence.get(component, {}) or {})
        if str(item.get("sha256", "")).lower() != str(
            planned.get("sha256", "")
        ).lower():
            raise ValueError(
                f"Qualification Change {component} evidence does not match the frozen plan"
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
    if str(evidence.get("ood_calibration_sha256", "")).lower() != (
        receipt.ood_calibration_sha256
    ):
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
