"""Immutable controlled-study design for TSH-CALO parameter research.

This module creates deterministic factor assignments only.  It never starts training, opens a
protected holdout, changes a qualified policy, or mutates a scientist's retained campaign plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

from .tsh_calo_parameter_registry import parameter_definition
from .tsh_calo_training_campaign import (
    TSHCALOEnvironmentHyperparameters,
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingHyperparameters,
)


TSH_CALO_PARAMETER_STUDY_SCHEMA = "tsh-calo-parameter-study-v1"
TSH_CALO_PARAMETER_DESIGN_SCHEMA = "tsh-calo-parameter-design-v1"

DesignMethod = Literal["latin_hypercube", "sobol"]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParameterStudyFactor:
    key: str
    lower: float
    upper: float
    values: tuple[float | int | bool, ...] = ()

    def validate(self) -> None:
        definition = parameter_definition(self.key)
        if not definition.scientist_tunable:
            raise ValueError(f"{definition.label} is not available for ordinary parameter studies")
        if self.values:
            if len(set(self.values)) < 2:
                raise ValueError(f"{definition.label} requires at least two distinct study values")
            if definition.kind == "binary" and (
                set(self.values) != {False, True}
                or any(type(value) is not bool for value in self.values)
            ):
                raise ValueError(f"{definition.label} binary study values must be False and True")
            if definition.kind == "integer" and any(
                isinstance(value, bool) or int(value) != float(value) for value in self.values
            ):
                raise ValueError(f"{definition.label} study values must be integers")
            return
        if definition.kind == "binary":
            raise ValueError(f"{definition.label} requires explicit False/True values")
        if not math.isfinite(float(self.lower)) or not math.isfinite(float(self.upper)):
            raise ValueError(f"{definition.label} study bounds must be finite")
        if float(self.lower) >= float(self.upper):
            raise ValueError(f"{definition.label} study lower bound must be below the upper bound")
        if definition.scale == "log" and float(self.lower) <= 0.0:
            raise ValueError(f"{definition.label} log-scale study values must be positive")


@dataclass(frozen=True, slots=True)
class ParameterStudyPlan:
    study_id: str
    base_execution_plan_sha256: str
    development_cases: tuple[str, ...]
    factors: tuple[ParameterStudyFactor, ...]
    design_method: DesignMethod
    design_points: int
    independent_replicates: int
    design_seed: int
    protected_cases_opened: bool = False
    schema_version: str = TSH_CALO_PARAMETER_STUDY_SCHEMA

    def validate(self) -> None:
        if self.schema_version != TSH_CALO_PARAMETER_STUDY_SCHEMA:
            raise ValueError("Parameter study schema is incompatible")
        if not str(self.study_id).strip():
            raise ValueError("Parameter study requires an identity")
        sha = str(self.base_execution_plan_sha256).strip().lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ValueError("Parameter study requires the exact base training-plan SHA-256")
        if not self.development_cases:
            raise ValueError("Parameter study requires development cases")
        leaked = protected_holdout_matches(self.development_cases)
        if leaked or self.protected_cases_opened:
            raise ValueError(
                "Protected holdout cases cannot enter parameter studies"
                + (f": {', '.join(leaked)}" if leaked else "")
            )
        if not self.factors:
            raise ValueError("Parameter study requires at least one factor")
        keys = [factor.key for factor in self.factors]
        if len(keys) != len(set(keys)):
            raise ValueError("Parameter study factors must be unique")
        for factor in self.factors:
            factor.validate()
        if self.design_method not in {"latin_hypercube", "sobol"}:
            raise ValueError("Parameter study design must be Latin hypercube or Sobol")
        if self.design_points < 3:
            raise ValueError("Parameter study requires at least three design points")
        if self.independent_replicates < 1:
            raise ValueError("Parameter study requires at least one independent replicate")
        if self.design_method == "sobol" and self.design_points & (self.design_points - 1):
            raise ValueError("Sobol study size must be a power of two")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["development_cases"] = list(self.development_cases)
        payload["factors"] = [asdict(factor) for factor in self.factors]
        return payload

    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ParameterStudyAssignment:
    design_index: int
    replicate_index: int
    block_id: str
    values: dict[str, float | int | bool]


def _map_dimension(unit: np.ndarray, factor: ParameterStudyFactor) -> list[float | int | bool]:
    definition = parameter_definition(factor.key)
    if factor.values:
        values = tuple(factor.values)
        indices = np.minimum((unit * len(values)).astype(int), len(values) - 1)
        return [values[int(index)] for index in indices]
    low = float(factor.lower)
    high = float(factor.upper)
    if definition.scale == "log":
        mapped = np.exp(np.log(low) + unit * (np.log(high) - np.log(low)))
    else:
        mapped = low + unit * (high - low)
    if definition.kind == "integer":
        return [int(round(float(value))) for value in mapped]
    return [float(value) for value in mapped]


def generate_parameter_design(plan: ParameterStudyPlan) -> dict:
    """Return deterministic assignments grouped into identical replicate blocks."""

    plan.validate()
    dimensions = len(plan.factors)
    if plan.design_method == "latin_hypercube":
        sample = qmc.LatinHypercube(d=dimensions, seed=plan.design_seed).random(plan.design_points)
    else:
        exponent = int(math.log2(plan.design_points))
        sample = qmc.Sobol(d=dimensions, scramble=True, seed=plan.design_seed).random_base2(exponent)
    mapped = {
        factor.key: _map_dimension(sample[:, index], factor)
        for index, factor in enumerate(plan.factors)
    }
    for factor in plan.factors:
        if len(set(mapped[factor.key])) < 2:
            raise ValueError(
                f"Study factor {factor.key} collapses to one effective value; widen its range"
            )
    assignments: list[ParameterStudyAssignment] = []
    for replicate in range(plan.independent_replicates):
        for design_index in range(plan.design_points):
            values = {key: column[design_index] for key, column in mapped.items()}
            assignments.append(
                ParameterStudyAssignment(
                    design_index=design_index,
                    replicate_index=replicate,
                    block_id=f"replicate-{replicate + 1:02d}",
                    values=values,
                )
            )
    payload = {
        "schema_version": TSH_CALO_PARAMETER_DESIGN_SCHEMA,
        "study_sha256": plan.sha256(),
        "design_method": plan.design_method,
        "design_points": plan.design_points,
        "independent_replicates": plan.independent_replicates,
        "factor_order": [factor.key for factor in plan.factors],
        "assignments": [asdict(item) for item in assignments],
        "protected_cases_opened": False,
        "execution_authority": "none",
    }
    payload["design_sha256"] = _canonical_sha256(payload)
    return payload


def apply_parameter_assignment(
    base: TSHCALOTrainingCampaignPlan,
    assignment: ParameterStudyAssignment,
    *,
    campaign_id: str,
) -> TSHCALOTrainingCampaignPlan:
    """Create an unexecuted campaign plan with one controlled factor assignment applied."""

    base.validate()
    if protected_holdout_matches(base.development_cases):
        raise ValueError("Protected holdout cases cannot enter parameter studies")
    top_level: dict[str, Any] = {}
    training = asdict(base.training)
    environment = asdict(base.environment)
    for key, value in assignment.values.items():
        definition = parameter_definition(key)
        if not definition.scientist_tunable:
            raise ValueError(f"{definition.label} is not available for ordinary parameter studies")
        if key.startswith("training."):
            training[key.split(".", 1)[1]] = value
        elif key.startswith("environment."):
            environment[key.split(".", 1)[1]] = value
        elif key in {"population_size", "max_evaluations", "deterministic_policy", "environment_deterministic"}:
            top_level[key] = value
        elif key == "ensemble_members":
            raise ValueError("Ensemble member count requires an explicit seed/curriculum redesign")
        else:
            raise ValueError(f"Parameter study does not know how to apply {key}")
    candidate = replace(
        base,
        campaign_id=str(campaign_id),
        training=TSHCALOTrainingHyperparameters(**training),
        environment=TSHCALOEnvironmentHyperparameters(**environment),
        **top_level,
    )
    candidate.validate()
    return candidate
