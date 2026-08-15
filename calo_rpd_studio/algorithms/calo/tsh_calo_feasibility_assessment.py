"""Non-decisional feasibility ratings for transactional TSH-CALO evidence.

The functions in this module summarize retained measurements.  They do not qualify, reject,
select, activate, train, or modify a policy.  A scientist remains responsible for deciding whether
an integrity-valid, compatible candidate is suitable for an intended use.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


TSH_CALO_FEASIBILITY_RATING_SCHEMA = "tsh-calo-feasibility-ratings-v1"
TSH_CALO_FEASIBILITY_ASSESSMENT_SCHEMA = (
    "tsh-calo-feasibility-assessment-v1-transactional-cells"
)
TSH_CALO_FEASIBILITY_COMPLETION_SCHEMA = "tsh-calo-feasibility-completion-v1"
TSH_CALO_FEASIBILITY_ADMISSION_SCHEMA = "tsh-calo-policy-feasibility-admission-v1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bounded_percentage(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"Feasibility evidence has an invalid {label}")
    return 100.0 * number


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_tsh_calo_feasibility_assessment(
    *,
    cases: list[dict],
    expected_case_order: tuple[str, ...],
) -> dict:
    """Build transparent ratings without applying an automated suitability threshold."""

    if [str(item.get("case", "")) for item in cases] != list(expected_case_order):
        raise ValueError("Feasibility evidence does not match the frozen case order")
    case_ratings: list[dict] = []
    total_pairs = 0
    weighted_feasible = 0.0
    weighted_reach = 0.0
    weighted_efficiency = 0.0
    weighted_validation = 0.0
    weighted_objective_coverage = 0.0
    for item in cases:
        pairs = int(item.get("n_pairs", 0))
        if pairs <= 0:
            raise ValueError("Feasibility evidence requires at least one retained pair per case")
        feasible = _bounded_percentage(
            item.get("candidate_feasible_probability"), "candidate full-feasibility rate"
        )
        baseline_feasible = _bounded_percentage(
            item.get("baseline_feasible_probability"), "baseline full-feasibility rate"
        )
        reached = _bounded_percentage(
            item.get("candidate_first_feasible_reached_probability"),
            "candidate first-feasible reach rate",
        )
        efficiency = _bounded_percentage(
            item.get("candidate_first_feasible_efficiency"),
            "candidate first-feasible efficiency",
        )
        validation = _bounded_percentage(
            item.get("candidate_independent_validation_probability"),
            "candidate independent-validation rate",
        )
        objective_coverage = _bounded_percentage(
            item.get("paired_feasible_objective_fraction"),
            "paired feasible-objective coverage",
        )
        case_name = str(item["case"])
        case_ratings.append(
            {
                "case": case_name,
                "n_candidate_cells": pairs,
                "ratings": {
                    "full_feasibility": feasible,
                    "baseline_full_feasibility": baseline_feasible,
                    "first_feasible_reached": reached,
                    "first_feasible_efficiency": efficiency,
                    "independent_validation": validation,
                    "paired_feasible_objective_coverage": objective_coverage,
                },
                "candidate_first_feasible_evaluation_median": _optional_number(
                    item.get("candidate_first_feasible_evaluation_median")
                ),
                "definition": (
                    "Full feasibility is the percentage of candidate cells satisfying every frozen "
                    "physical and numerical feasibility requirement within the exact FE budget."
                ),
            }
        )
        total_pairs += pairs
        weighted_feasible += feasible * pairs
        weighted_reach += reached * pairs
        weighted_efficiency += efficiency * pairs
        weighted_validation += validation * pairs
        weighted_objective_coverage += objective_coverage * pairs
    if total_pairs <= 0:
        raise ValueError("Feasibility assessment has no candidate cells")
    overall = {
        "full_feasibility": weighted_feasible / total_pairs,
        "first_feasible_reached": weighted_reach / total_pairs,
        "first_feasible_efficiency": weighted_efficiency / total_pairs,
        "independent_validation": weighted_validation / total_pairs,
        "paired_feasible_objective_coverage": weighted_objective_coverage / total_pairs,
    }
    assessment = {
        "schema_version": TSH_CALO_FEASIBILITY_RATING_SCHEMA,
        "decision_authority": "scientist_only",
        "automated_suitability_decision": None,
        "candidate_cell_count": total_pairs,
        "overall_feasibility_score": overall["full_feasibility"],
        "overall_ratings": overall,
        "case_ratings": case_ratings,
        "score_definition": (
            "overall_feasibility_score is exactly 100 times the proportion of candidate cells "
            "that reached complete physical feasibility; it is not a recommendation or use gate"
        ),
        "first_feasible_efficiency_definition": (
            "For each candidate cell, zero means feasibility was not reached within budget and "
            "one means it was reached at the first FE; retained values are averaged before conversion "
            "to a percentage."
        ),
    }
    assessment["rating_payload_sha256"] = _canonical_sha256(assessment)
    return assessment


def validate_tsh_calo_feasibility_assessment(payload: dict) -> None:
    """Fail closed when a retained rating payload is malformed or self-inconsistent."""

    assessment = dict(payload or {})
    digest = str(assessment.pop("rating_payload_sha256", "")).lower()
    if assessment.get("schema_version") != TSH_CALO_FEASIBILITY_RATING_SCHEMA:
        raise ValueError("Feasibility rating schema is incompatible")
    if assessment.get("decision_authority") != "scientist_only":
        raise ValueError("Feasibility assessment assigned decision authority to software")
    if assessment.get("automated_suitability_decision") is not None:
        raise ValueError("Feasibility assessment contains an automated suitability decision")
    if digest != _canonical_sha256(assessment):
        raise ValueError("Feasibility rating payload checksum is inconsistent")
    _bounded_percentage(
        float(assessment.get("overall_feasibility_score", -1.0)) / 100.0,
        "overall feasibility score",
    )
    case_ratings = list(assessment.get("case_ratings", []))
    if not case_ratings or not all(isinstance(item, dict) for item in case_ratings):
        raise ValueError("Feasibility assessment lacks per-case ratings")
