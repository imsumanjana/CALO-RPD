"""Read-only training-parameter influence analysis for assessed TSH-CALO candidates."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Any

import numpy as np

from .tsh_calo_feasibility_assessment import (
    TSH_CALO_FEASIBILITY_RATING_SCHEMA,
    validate_tsh_calo_feasibility_assessment,
)
from .tsh_calo_training_campaign import TSHCALOTrainingCampaignPlan
from .tsh_calo_parameter_registry import TSH_CALO_PARAMETER_REGISTRY


TSH_CALO_TRAINING_INFLUENCE_SCHEMA = "tsh-calo-training-parameter-influence-v1"
MINIMUM_INFLUENCE_CAMPAIGNS = 3
MINIMUM_RANKABLE_CAMPAIGNS = 6
MINIMUM_DISTINCT_PARAMETER_VALUES = 2


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def training_parameter_values(plan: TSHCALOTrainingCampaignPlan) -> dict[str, float | int | bool]:
    """Return scientist-controlled values without changing or interpreting the training plan."""

    values: dict[str, float | int | bool] = {
        "population_size": int(plan.population_size),
        "max_evaluations": int(plan.max_evaluations),
        "ensemble_members": len(plan.members),
        "deterministic_policy": bool(plan.deterministic_policy),
        "environment_deterministic": bool(plan.environment_deterministic),
    }
    values.update({f"training.{key}": value for key, value in asdict(plan.training).items()})
    values.update({f"environment.{key}": value for key, value in asdict(plan.environment).items()})
    return values


def training_comparability_protocol(plan: TSHCALOTrainingCampaignPlan) -> dict:
    """Retain fixed architecture/protocol fields while excluding tunable values and identities."""

    return {
        "training_schema_version": plan.schema_version,
        "development_cases": list(plan.development_cases),
        "feature_flags": asdict(plan.feature_flags),
        "episode_case_sequence": [episode.case_identity for episode in plan.members[0].episodes],
    }


def _continuous_association(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < MINIMUM_INFLUENCE_CAMPAIGNS or len(np.unique(x)) < 2:
        raise ValueError("Insufficient parameter variation")
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _binary_contrast(xs: list[bool], ys: list[float]) -> float:
    if len(xs) < MINIMUM_INFLUENCE_CAMPAIGNS or len(set(xs)) < 2:
        raise ValueError("Insufficient parameter variation")
    enabled = [value for flag, value in zip(xs, ys) if flag]
    disabled = [value for flag, value in zip(xs, ys) if not flag]
    if not enabled or not disabled:
        raise ValueError("Both binary parameter states are required")
    pooled = float(np.std(np.asarray(ys, dtype=float)))
    if pooled <= 0.0:
        return 0.0
    return float((np.mean(enabled) - np.mean(disabled)) / pooled)


def _direction(value: float) -> str:
    return "positive" if value > 0.0 else ("negative" if value < 0.0 else "flat")

def _rating_values(ratings: dict) -> dict[str, float]:
    overall = dict(ratings.get("overall_ratings", {}) or {})
    values = {
        "overall_full_feasibility": ratings.get("overall_feasibility_score"),
        "first_feasible_reached": overall.get("first_feasible_reached"),
        "first_feasible_efficiency": overall.get("first_feasible_efficiency"),
        "independent_validation": overall.get("independent_validation"),
        "paired_feasible_objective_coverage": overall.get("paired_feasible_objective_coverage"),
    }
    output: dict[str, float] = {}
    for name, value in values.items():
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 100.0:
            raise ValueError(f"feasibility rating {name} is invalid")
        output[name] = number
    return output


def build_training_parameter_influence(
    *,
    selected_candidate_sha256: str,
    selected_plan: TSHCALOTrainingCampaignPlan,
    cohort: list[dict],
    selected_assessment_comparison_protocol_sha256: str = "",
    selected_training_compatibility_sha256: str = "",
) -> dict:
    """Estimate transparent observational associations over compatible assessed campaigns.

    Each cohort row requires ``candidate_sha256``, a parsed ``plan``, and a feasibility ``ratings``
    payload.  The result is deliberately marked observational: existing campaign differences are
    useful for scientist-directed follow-up but cannot prove causal hyperparameter effects.
    """

    if not _is_sha256(selected_candidate_sha256):
        raise ValueError("Selected influence candidate SHA-256 is invalid")
    if not isinstance(selected_plan, TSHCALOTrainingCampaignPlan):
        raise TypeError("Selected influence training plan was not parsed")
    selected_protocol = training_comparability_protocol(selected_plan)
    selected_protocol_sha256 = _canonical_sha256(selected_protocol)
    selected_assessment_protocol = str(selected_assessment_comparison_protocol_sha256).lower()
    selected_training_compatibility = str(selected_training_compatibility_sha256).lower()
    if selected_assessment_protocol and not _is_sha256(selected_assessment_protocol):
        raise ValueError("Selected assessment comparison protocol SHA-256 is invalid")
    if selected_training_compatibility and not _is_sha256(selected_training_compatibility):
        raise ValueError("Selected training compatibility SHA-256 is invalid")
    compatible: list[dict] = []
    exclusions: list[dict] = []
    observed_candidates: set[str] = set()
    for row in cohort:
        try:
            plan = row["plan"]
            ratings = dict(row["ratings"])
            candidate_sha256 = str(row["candidate_sha256"]).lower()
            if not _is_sha256(candidate_sha256):
                raise ValueError("candidate SHA-256 is invalid")
            if candidate_sha256 in observed_candidates:
                raise ValueError("candidate is duplicated in the influence cohort")
            if not isinstance(plan, TSHCALOTrainingCampaignPlan):
                raise ValueError("training plan was not parsed")
            if ratings.get("schema_version") != TSH_CALO_FEASIBILITY_RATING_SCHEMA:
                raise ValueError("rating schema differs")
            validate_tsh_calo_feasibility_assessment(ratings)
            if training_comparability_protocol(plan) != selected_protocol:
                raise ValueError("training comparison protocol differs")
            row_assessment_protocol = str(
                row.get("assessment_comparison_protocol_sha256", "")
            ).lower()
            row_training_compatibility = str(row.get("training_compatibility_sha256", "")).lower()
            if selected_assessment_protocol:
                if row_assessment_protocol != selected_assessment_protocol:
                    raise ValueError("assessment comparison protocol differs")
            if selected_training_compatibility:
                if row_training_compatibility != selected_training_compatibility:
                    raise ValueError("training compatibility contract differs")
            rating_values = _rating_values(ratings)
            observed_candidates.add(candidate_sha256)
            compatible.append(
                {
                    "candidate_sha256": candidate_sha256,
                    "plan": plan,
                    "rating_values": rating_values,
                    "ratings": ratings,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            exclusions.append(
                {
                    "candidate_sha256": str(row.get("candidate_sha256", "")),
                    "reason": str(exc),
                }
            )
    parameter_rows: list[dict] = []
    selected_values = training_parameter_values(selected_plan)
    for name, selected_value in selected_values.items():
        definition = TSH_CALO_PARAMETER_REGISTRY.get(name)
        observations: list[tuple[Any, dict[str, float]]] = []
        for row in compatible:
            value = training_parameter_values(row["plan"]).get(name)
            if not isinstance(value, (bool, int, float)):
                continue
            observations.append((value, dict(row["rating_values"])))
        distinct = sorted({item[0] for item in observations}, key=lambda value: str(value))
        result = {
            "parameter": name,
            "label": definition.label if definition else name,
            "domain": definition.domain if definition else "unclassified",
            "selected_value": selected_value,
            "observations": len(observations),
            "distinct_values": distinct,
            "affected_rating": "not_estimated",
            "evidence_classification": "insufficient_comparative_evidence",
            "association": None,
            "association_measure": "not_estimated",
            "direction": "not_estimated",
            "rating_effects": [],
            "rankable": False,
        }
        if len(observations) >= MINIMUM_INFLUENCE_CAMPAIGNS and len(distinct) >= 2:
            rating_effects = []
            is_binary = all(isinstance(item[0], bool) for item in observations)
            for rating_name in next(iter(observations))[1]:
                ys = [item[1][rating_name] for item in observations]
                if is_binary:
                    association = _binary_contrast(
                        [bool(item[0]) for item in observations], ys
                    )
                    measure = "standardized_binary_contrast"
                else:
                    association = _continuous_association(
                        [float(item[0]) for item in observations], ys
                    )
                    measure = "pearson_correlation"
                rating_effects.append(
                    {
                        "rating": rating_name,
                        "association": association,
                        "association_measure": measure,
                        "direction": _direction(association),
                    }
                )
            rating_effects.sort(key=lambda item: -abs(float(item["association"])))
            strongest = rating_effects[0]
            rankable = len(observations) >= MINIMUM_RANKABLE_CAMPAIGNS
            result.update(
                {
                    "evidence_classification": (
                        "preliminary_observational_association"
                        if rankable
                        else "exploratory_observational_signal"
                    ),
                    "affected_rating": strongest["rating"],
                    "association": strongest["association"],
                    "association_measure": strongest["association_measure"],
                    "direction": strongest["direction"],
                    "rating_effects": rating_effects,
                    "rankable": rankable,
                }
            )
        parameter_rows.append(result)
    parameter_rows.sort(
        key=lambda item: (
            not bool(item.get("rankable", False)),
            item["association"] is None,
            -abs(float(item["association"] or 0.0)),
            str(item["parameter"]),
        )
    )
    payload = {
        "schema_version": TSH_CALO_TRAINING_INFLUENCE_SCHEMA,
        "selected_candidate_sha256": str(selected_candidate_sha256).lower(),
        "rating_schema_version": TSH_CALO_FEASIBILITY_RATING_SCHEMA,
        "training_comparison_protocol_sha256": selected_protocol_sha256,
        "assessment_comparison_protocol_sha256": selected_assessment_protocol,
        "training_compatibility_sha256": selected_training_compatibility,
        "analysis_authority": "scientist_decision_support_only",
        "automatic_training_or_parameter_change": False,
        "evidence_classification": (
            "preliminary_observational_association"
            if any(item.get("rankable", False) for item in parameter_rows)
            else (
                "exploratory_observational_signal"
                if any(item.get("association") is not None for item in parameter_rows)
                else "insufficient_comparative_evidence"
            )
        ),
        "compatible_campaign_count": len(compatible),
        "excluded_campaigns": exclusions,
        "parameters": parameter_rows,
        "limitations": [
            "Associations across retained campaigns are not causal effects.",
            "The displayed rating is the largest absolute univariate association, not proof of mechanism.",
            "Fewer than six compatible campaigns are treated as exploratory and are not ranked as parameter evidence.",
            "Reliable tuning requires controlled parameter variation and repeated independent campaigns.",
            "Only campaigns with the same authenticated training contract and assessment design are directly comparable when those identities are supplied.",
            "Protected final-assessment cases must not be used to tune a later model.",
        ],
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    return payload
