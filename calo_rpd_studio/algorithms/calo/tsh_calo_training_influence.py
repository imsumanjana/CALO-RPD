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


TSH_CALO_TRAINING_INFLUENCE_SCHEMA = "tsh-calo-training-parameter-influence-v1"
MINIMUM_INFLUENCE_CAMPAIGNS = 3
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
        "episode_case_sequence": [
            episode.case_identity for episode in plan.members[0].episodes
        ],
    }


def _standardized_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < MINIMUM_INFLUENCE_CAMPAIGNS or len(np.unique(x)) < 2:
        raise ValueError("Insufficient parameter variation")
    x_scale = float(np.std(x))
    y_scale = float(np.std(y))
    if x_scale <= 0.0 or y_scale <= 0.0:
        return 0.0, 0.0
    standardized_x = (x - float(np.mean(x))) / x_scale
    standardized_y = (y - float(np.mean(y))) / y_scale
    effect = float(np.dot(standardized_x, standardized_y) / np.dot(standardized_x, standardized_x))
    correlation = float(np.corrcoef(x, y)[0, 1])
    return effect, correlation


def _rating_values(ratings: dict) -> dict[str, float]:
    overall = dict(ratings.get("overall_ratings", {}) or {})
    values = {
        "overall_full_feasibility": ratings.get("overall_feasibility_score"),
        "first_feasible_reached": overall.get("first_feasible_reached"),
        "first_feasible_efficiency": overall.get("first_feasible_efficiency"),
        "independent_validation": overall.get("independent_validation"),
        "paired_feasible_objective_coverage": overall.get(
            "paired_feasible_objective_coverage"
        ),
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
        observations: list[tuple[float, dict[str, float]]] = []
        for row in compatible:
            value = training_parameter_values(row["plan"]).get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            observations.append((float(value), dict(row["rating_values"])))
        distinct = sorted({item[0] for item in observations})
        result = {
            "parameter": name,
            "selected_value": selected_value,
            "observations": len(observations),
            "distinct_values": distinct,
            "affected_rating": "not_estimated",
            "evidence_classification": "insufficient_comparative_evidence",
            "standardized_effect": None,
            "correlation": None,
            "direction": "not_estimated",
            "rating_effects": [],
        }
        if (
            len(observations) >= MINIMUM_INFLUENCE_CAMPAIGNS
            and len(distinct) >= MINIMUM_DISTINCT_PARAMETER_VALUES
        ):
            rating_effects = []
            for rating_name in next(iter(observations))[1]:
                effect, correlation = _standardized_slope(
                    [item[0] for item in observations],
                    [item[1][rating_name] for item in observations],
                )
                rating_effects.append(
                    {
                        "rating": rating_name,
                        "standardized_effect": effect,
                        "correlation": correlation,
                        "direction": (
                            "positive"
                            if effect > 0.0
                            else ("negative" if effect < 0.0 else "flat")
                        ),
                    }
                )
            rating_effects.sort(
                key=lambda item: -abs(float(item["standardized_effect"]))
            )
            strongest = rating_effects[0]
            result.update(
                {
                    "evidence_classification": "observational_association",
                    "affected_rating": strongest["rating"],
                    "standardized_effect": strongest["standardized_effect"],
                    "correlation": strongest["correlation"],
                    "direction": strongest["direction"],
                    "rating_effects": rating_effects,
                }
            )
        parameter_rows.append(result)
    parameter_rows.sort(
        key=lambda item: (
            item["standardized_effect"] is None,
            -abs(float(item["standardized_effect"] or 0.0)),
            str(item["parameter"]),
        )
    )
    payload = {
        "schema_version": TSH_CALO_TRAINING_INFLUENCE_SCHEMA,
        "selected_candidate_sha256": str(selected_candidate_sha256).lower(),
        "rating_schema_version": TSH_CALO_FEASIBILITY_RATING_SCHEMA,
        "training_comparison_protocol_sha256": selected_protocol_sha256,
        "analysis_authority": "scientist_decision_support_only",
        "automatic_training_or_parameter_change": False,
        "evidence_classification": (
            "observational_association"
            if any(item["standardized_effect"] is not None for item in parameter_rows)
            else "insufficient_comparative_evidence"
        ),
        "compatible_campaign_count": len(compatible),
        "excluded_campaigns": exclusions,
        "parameters": parameter_rows,
        "limitations": [
            "Associations across retained campaigns are not causal effects.",
            "The displayed affected rating is the largest absolute univariate association, not proof of mechanism.",
            "Reliable tuning requires controlled parameter variation and repeated independent campaigns.",
            "Protected final-assessment cases must not be used to tune a later model.",
        ],
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    return payload
