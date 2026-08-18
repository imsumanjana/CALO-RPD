"""Robust response analysis and preregistered local-response plans for TSH-CALO parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .ai_controller import PARAMETER_HIGH, PARAMETER_LOW, PARAMETER_NAMES
from .tsh_calo_parameter_registry import parameter_definition


TSH_CALO_ROBUST_PARAMETER_RESPONSE_SCHEMA = "tsh-calo-robust-parameter-response-v1"
TSH_CALO_LOCAL_RESPONSE_PLAN_SCHEMA = "tsh-calo-local-parameter-response-plan-v1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256(value: str, label: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be a SHA-256 value")
    return text


@dataclass(frozen=True, slots=True)
class ParameterStudyObservation:
    design_index: int
    replicate_index: int
    case_identity: str
    values: dict[str, float | int | bool]
    full_feasible: bool
    final_objective: float | None
    first_feasible_evaluations: int | None
    convergence_auc: float | None
    runtime_seconds: float

    def validate(self) -> None:
        if self.design_index < 0 or self.replicate_index < 0:
            raise ValueError("Parameter study observation indices cannot be negative")
        if not str(self.case_identity).strip():
            raise ValueError("Parameter study observation requires a case identity")
        if not self.values:
            raise ValueError("Parameter study observation requires parameter values")
        for key in self.values:
            parameter_definition(key)
        if self.full_feasible:
            if self.final_objective is None or not math.isfinite(float(self.final_objective)):
                raise ValueError("Feasible observations require a finite final objective")
        if self.first_feasible_evaluations is not None and self.first_feasible_evaluations < 0:
            raise ValueError("First-feasible evaluation count cannot be negative")
        if self.convergence_auc is not None and not math.isfinite(float(self.convergence_auc)):
            raise ValueError("Convergence AUC must be finite when retained")
        if not math.isfinite(float(self.runtime_seconds)) or self.runtime_seconds < 0.0:
            raise ValueError("Runtime must be finite and non-negative")


def _iqr(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def summarize_robust_parameter_response(
    *,
    study_sha256: str,
    design_sha256: str,
    observations: list[ParameterStudyObservation],
    required_cases: tuple[str, ...],
    independent_replicates: int,
    minimum_full_feasibility: float = 0.95,
) -> dict[str, Any]:
    """Rank complete design points lexicographically after a declared feasibility gate.

    No weighted objective is invented.  Feasibility is a hard gate; remaining summaries are exposed
    separately so the scientist can inspect objective quality, dispersion, convergence, and cost.
    """

    _sha256(study_sha256, "Study identity")
    _sha256(design_sha256, "Design identity")
    if not required_cases or len(set(required_cases)) != len(required_cases):
        raise ValueError("Robust response analysis requires unique cases")
    if independent_replicates < 1:
        raise ValueError("Robust response analysis requires independent replicates")
    if not 0.0 <= minimum_full_feasibility <= 1.0:
        raise ValueError("Minimum feasibility must be within [0, 1]")
    groups: dict[int, list[ParameterStudyObservation]] = {}
    for observation in observations:
        observation.validate()
        groups.setdefault(observation.design_index, []).append(observation)
    expected_pairs = {
        (replicate, case)
        for replicate in range(independent_replicates)
        for case in required_cases
    }
    summaries: list[dict[str, Any]] = []
    for design_index, rows in sorted(groups.items()):
        pairs = {(row.replicate_index, row.case_identity) for row in rows}
        if pairs != expected_pairs or len(rows) != len(expected_pairs):
            continue
        reference_values = rows[0].values
        if any(row.values != reference_values for row in rows[1:]):
            raise ValueError("One design point contains inconsistent parameter assignments")
        feasibility = float(np.mean([1.0 if row.full_feasible else 0.0 for row in rows]))
        objectives = [float(row.final_objective) for row in rows if row.full_feasible]
        first_feasible = [
            float(row.first_feasible_evaluations)
            for row in rows
            if row.first_feasible_evaluations is not None
        ]
        auc = [float(row.convergence_auc) for row in rows if row.convergence_auc is not None]
        runtimes = [float(row.runtime_seconds) for row in rows]
        summaries.append(
            {
                "design_index": design_index,
                "values": dict(reference_values),
                "full_feasibility_rate": feasibility,
                "passes_feasibility_gate": feasibility >= minimum_full_feasibility,
                "median_final_objective": float(np.median(objectives)) if objectives else None,
                "final_objective_iqr": _iqr(objectives) if len(objectives) >= 2 else 0.0 if objectives else None,
                "median_first_feasible_evaluations": (
                    float(np.median(first_feasible)) if first_feasible else None
                ),
                "median_convergence_auc": float(np.median(auc)) if auc else None,
                "median_runtime_seconds": float(np.median(runtimes)),
                "case_count": len(required_cases),
                "replicate_count": independent_replicates,
            }
        )
    eligible = [row for row in summaries if row["passes_feasibility_gate"]]
    eligible.sort(
        key=lambda row: (
            float(row["median_final_objective"])
            if row["median_final_objective"] is not None
            else float("inf"),
            float(row["final_objective_iqr"])
            if row["final_objective_iqr"] is not None
            else float("inf"),
            float(row["median_first_feasible_evaluations"])
            if row["median_first_feasible_evaluations"] is not None
            else float("inf"),
            float(row["median_runtime_seconds"]),
        )
    )
    payload = {
        "schema_version": TSH_CALO_ROBUST_PARAMETER_RESPONSE_SCHEMA,
        "study_sha256": study_sha256.lower(),
        "design_sha256": design_sha256.lower(),
        "required_cases": list(required_cases),
        "independent_replicates": independent_replicates,
        "minimum_full_feasibility": minimum_full_feasibility,
        "complete_design_points": len(summaries),
        "eligible_design_points": len(eligible),
        "design_points": summaries,
        "robust_order": [row["design_index"] for row in eligible],
        "selection_rule": (
            "feasibility_gate_then_objective_median_then_objective_iqr_then_"
            "first_feasible_then_runtime"
        ),
        "automatic_parameter_change": False,
        "protected_holdout_selection_authority": False,
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class LocalParameterResponsePlan:
    analysis_id: str
    source_run_id: str
    policy_sha256: str
    trajectory_row_sha256: str
    rng_state_sha256: str
    parameter: str
    candidate_values: tuple[float, ...]
    analysis_fe_budget_per_value: int
    analysis_fe_ledger_id: str
    protected_case: bool = False
    schema_version: str = TSH_CALO_LOCAL_RESPONSE_PLAN_SCHEMA

    def validate(self) -> None:
        if self.schema_version != TSH_CALO_LOCAL_RESPONSE_PLAN_SCHEMA:
            raise ValueError("Local parameter-response plan schema is incompatible")
        if not self.analysis_id.strip() or not self.source_run_id.strip():
            raise ValueError("Local parameter-response plan requires analysis and source-run identities")
        _sha256(self.policy_sha256, "Policy identity")
        _sha256(self.trajectory_row_sha256, "Trajectory-state identity")
        _sha256(self.rng_state_sha256, "Random-state identity")
        definition = parameter_definition(self.parameter)
        if not definition.local_response_allowed or not self.parameter.startswith("policy."):
            raise ValueError(
                "Local response grids are limited to adaptive policy parameters observed in a run"
            )
        if definition.kind == "binary":
            raise ValueError("Local response grids require a numeric parameter")
        policy_name = self.parameter.split(".", 1)[1]
        try:
            parameter_index = PARAMETER_NAMES.index(policy_name)
        except ValueError as exc:
            raise ValueError("Local response parameter is not part of the policy action") from exc
        lower = float(PARAMETER_LOW[parameter_index])
        upper = float(PARAMETER_HIGH[parameter_index])
        if len(self.candidate_values) < 3 or len(set(self.candidate_values)) != len(self.candidate_values):
            raise ValueError("Local parameter response requires at least three distinct values")
        if any(not math.isfinite(float(value)) for value in self.candidate_values):
            raise ValueError("Local parameter response values must be finite")
        if any(not lower <= float(value) <= upper for value in self.candidate_values):
            raise ValueError(
                f"Local parameter response values for {policy_name} must be within "
                f"[{lower:g}, {upper:g}]"
            )
        if self.analysis_fe_budget_per_value < 1:
            raise ValueError("Local parameter response requires a separate positive analysis FE budget")
        if not self.analysis_fe_ledger_id.strip():
            raise ValueError("Local parameter response requires a separate analysis FE ledger")
        if self.protected_case:
            raise ValueError("Protected holdout states cannot be used for parameter selection")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["candidate_values"] = list(self.candidate_values)
        payload["official_experiment_fe_accounting"] = False
        payload["automatic_parameter_change"] = False
        payload["plan_sha256"] = _canonical_sha256(payload)
        return payload
