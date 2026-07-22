"""Leakage-aware historical experience repository.

The repository separates three kinds of reusable knowledge:

* complete CALO policy trajectories, which may be used for offline policy pretraining;
* validated cross-algorithm solution exemplars, which may seed a practical warm start;
* CALO parameter priors estimated from successful historical training experiments.

Only experiments explicitly marked ``train`` and learning-eligible are included.  Validation,
benchmark/test, and excluded experiments are never admitted into the training repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ROLES = ("train", "validation", "test", "excluded")
REPOSITORY_SCHEMA_VERSION = 1


def _json_load(value: str | dict | None, default):
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _clean_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _problem_identity(config: dict, result: dict) -> dict:
    metadata = dict(result.get("metadata") or {})
    solution_state = dict(metadata.get("solution_state") or {})
    vector = list(result.get("best_vector") or [])
    return {
        "case_name": str(config.get("case_name", "unknown")),
        "case_checksum": str(solution_state.get("case_checksum", "")),
        "dimension": len(vector),
        "scenario_mode": str((config.get("scenarios") or {}).get("mode", "deterministic")),
        "objective": dict(config.get("objective") or {}),
        "variables": dict(config.get("variables") or {}),
    }


def _problem_key(identity: dict) -> str:
    checksum = identity.get("case_checksum") or identity.get("case_name") or "unknown"
    return f"{checksum}|d={int(identity.get('dimension', 0))}|scenario={identity.get('scenario_mode', 'deterministic')}"


def _repository_checksum(payload: dict) -> str:
    stable = dict(payload)
    stable.pop("repository_sha256", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _relative_improvement(previous: float | None, current: float | None) -> float:
    if previous is None or current is None:
        return 0.0
    if not (math.isfinite(previous) and math.isfinite(current)):
        return 0.0
    scale = max(abs(previous), abs(current), 1e-12)
    return float(np.clip((previous - current) / scale, -1.0, 1.0))


def _reconstruct_legacy_calo_trajectory(result: dict) -> list[dict]:
    """Reconstruct a conservative partial trajectory from v1.2 summary histories.

    The reconstruction is explicitly marked as partial. It can supervise regime/operator choices
    and approximate state/value learning, but never supervises the continuous parameter head because
    the original per-step parameter actions were not stored.
    """
    metadata = dict(result.get("metadata") or {})
    diagnostics = dict(metadata.get("diagnostics_history") or {})
    regimes = list(metadata.get("regime_history") or [])
    usage = list(metadata.get("operator_usage_history") or [])
    rewards = list(metadata.get("reward_history") or [])
    evaluations = list(metadata.get("convergence_evaluations") or [])
    objective_history = list(metadata.get("best_feasible_objective_history") or [])
    operator_success = list(metadata.get("operator_success_history") or [])
    operator_names = list(metadata.get("operator_names") or [])
    component_names = (
        "bus_voltage",
        "generator_q",
        "generator_p",
        "branch_thermal",
        "power_flow",
    )
    lengths = [len(regimes), len(usage), len(rewards)]
    for key in (
        "population_diversity",
        "elite_diversity",
        "feasible_ratio",
        "epsilon_feasible_ratio",
        "mean_total_violation",
        "best_total_violation",
    ):
        lengths.append(len(diagnostics.get(key) or []))
    count = min(lengths) if lengths and all(lengths) else 0
    if count <= 0:
        return []

    regime_map = {
        "feasibility": 0,
        "transition": 1,
        "objective_refinement": 2,
        "recovery": 3,
    }
    previous_violation: float | None = None
    previous_objective: float | None = None
    constraint_stagnation = 0
    objective_stagnation = 0
    max_evaluations = max(int(result.get("evaluations", 0)), 1)
    n_operators = max(len(operator_names), 1)
    expected_state_dim = 18 + n_operators
    output: list[dict] = []
    for index in range(count):
        best_violation = _clean_float((diagnostics.get("best_total_violation") or [1e12] * count)[index], 1e12)
        mean_violation = _clean_float((diagnostics.get("mean_total_violation") or [1e12] * count)[index], 1e12)
        feasible_ratio = float((diagnostics.get("feasible_ratio") or [0.0] * count)[index])
        epsilon_ratio = float((diagnostics.get("epsilon_feasible_ratio") or [0.0] * count)[index])
        diversity = float((diagnostics.get("population_diversity") or [0.0] * count)[index])
        elite = float((diagnostics.get("elite_diversity") or [0.0] * count)[index])
        objective = None
        if index < len(objective_history):
            objective = _clean_float(objective_history[index])

        if (
            previous_violation is not None
            and best_violation is not None
            and best_violation < previous_violation - 1e-12
        ):
            constraint_stagnation = 0
        else:
            constraint_stagnation += 1
        if (
            objective is not None
            and previous_objective is not None
            and objective < previous_objective - 1e-12
        ):
            objective_stagnation = 0
        elif objective is not None:
            objective_stagnation += 1

        component_values = []
        for name in component_names:
            values = diagnostics.get(f"best_{name}") or []
            raw = _clean_float(values[index], 0.0) if index < len(values) else 0.0
            component_values.append(float(math.tanh(max(raw or 0.0, 0.0))))

        success = operator_success[index] if index < len(operator_success) else {}
        credit = np.asarray(
            [float(success.get(name, 0.0)) for name in operator_names[:n_operators]], dtype=float
        )
        if credit.size != n_operators or credit.sum() <= 0:
            credit = np.full(n_operators, 1 / n_operators)
        else:
            credit = credit + 1e-6
            credit /= credit.sum()

        evaluation = (
            int(evaluations[index])
            if index < len(evaluations)
            else int((index + 1) * max_evaluations / count)
        )
        state = np.r_[
            np.clip(diversity, -1.0, 1.0),
            np.clip(elite, -1.0, 1.0),
            np.clip(feasible_ratio, 0.0, 1.0),
            np.clip(epsilon_ratio, 0.0, 1.0),
            math.tanh(max(mean_violation or 0.0, 0.0)),
            math.tanh(max(best_violation or 0.0, 0.0)),
            component_values,
            _relative_improvement(previous_violation, best_violation),
            _relative_improvement(previous_objective, objective),
            min(constraint_stagnation / 12.0, 1.0),
            min(objective_stagnation / 12.0, 1.0),
            max(0.0, 1.0 - evaluation / max_evaluations),
            0.0,
            0.0,
            credit,
        ].astype(float)
        if state.shape != (expected_state_dim,):
            return []
        usage_row = usage[index] if index < len(usage) else {}
        if usage_row:
            dominant_name = max(usage_row, key=lambda name: usage_row.get(name, 0))
            operator = operator_names.index(dominant_name) if dominant_name in operator_names else 0
        else:
            operator = 0
        regime = regime_map.get(str(regimes[index]), 1)
        output.append(
            {
                "state": np.clip(state, -1.0, 1.0).tolist(),
                "regime": int(regime),
                "operator": int(operator),
                "parameter": [0.5] * 6,
                "parameter_supervision": False,
                "reward": float(rewards[index]),
                "evaluations": evaluation,
                "source_policy": "legacy_reconstructed",
                "reconstruction_quality": "partial_v1.2_summary",
                "quality_weight": 0.35,
            }
        )
        previous_violation = best_violation
        if objective is not None:
            previous_objective = objective
    return output


@dataclass(slots=True)
class HistoricalExperienceRepository:
    payload: dict
    path: str = ""

    @property
    def summary(self) -> dict:
        return dict(self.payload.get("summary") or {})

    @property
    def policy_trajectories(self) -> list[dict]:
        return list(self.payload.get("policy_trajectories") or [])

    @property
    def cross_algorithm_solutions(self) -> list[dict]:
        return list(self.payload.get("cross_algorithm_solutions") or [])

    @property
    def parameter_priors(self) -> dict:
        return dict(self.payload.get("parameter_priors") or {})

    def compatible_solutions(
        self, *, case_checksum: str, case_name: str, dimension: int
    ) -> list[dict]:
        output = []
        for item in self.cross_algorithm_solutions:
            identity = item.get("problem") or {}
            same_case = bool(case_checksum and identity.get("case_checksum") == case_checksum)
            same_name = str(identity.get("case_name", "")).lower() == str(case_name).lower()
            if int(identity.get("dimension", -1)) != int(dimension):
                continue
            if same_case or (not case_checksum and same_name) or same_name:
                output.append(item)
        return output

    def calo_parameter_prior(self, *, case_checksum: str, case_name: str, dimension: int) -> dict:
        candidates = []
        for key, item in self.parameter_priors.items():
            identity = item.get("problem") or {}
            same_case = bool(case_checksum and identity.get("case_checksum") == case_checksum)
            same_name = str(identity.get("case_name", "")).lower() == str(case_name).lower()
            if int(identity.get("dimension", -1)) != int(dimension):
                continue
            if same_case or same_name:
                candidates.append((key, item))
        if not candidates:
            return {}
        # Exact checksum match wins; otherwise use the most supported same-name prior.
        candidates.sort(
            key=lambda pair: (
                0
                if pair[1].get("problem", {}).get("case_checksum") == case_checksum
                and case_checksum
                else 1,
                -int(pair[1].get("support", 0)),
            )
        )
        return dict(candidates[0][1].get("parameters") or {})


def load_experience_repository(path: str | Path) -> HistoricalExperienceRepository:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != REPOSITORY_SCHEMA_VERSION:
        raise ValueError("Unsupported historical experience repository schema")
    expected = str(payload.get("repository_sha256", ""))
    actual = _repository_checksum(payload)
    if expected and expected != actual:
        raise ValueError("Historical experience repository checksum verification failed")
    return HistoricalExperienceRepository(payload=payload, path=str(source.resolve()))


def _parameter_priors(entries: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        if entry.get("algorithm") != "CALO" or not entry.get("feasible"):
            continue
        grouped.setdefault(_problem_key(entry["problem"]), []).append(entry)

    priors: dict[str, dict] = {}
    for key, group in grouped.items():
        group = sorted(
            group,
            key=lambda item: (
                item.get("best_objective") is None,
                float(item.get("best_objective") or 0.0),
            ),
        )
        top = group[: max(1, math.ceil(len(group) / 2))]
        numerical: dict[str, list[float]] = {}
        for item in top:
            for name, value in (item.get("parameters") or {}).items():
                if _finite_number(value):
                    numerical.setdefault(str(name), []).append(float(value))
        parameters = {
            name: float(np.median(values)) for name, values in numerical.items() if values
        }
        priors[key] = {
            "problem": dict(group[0]["problem"]),
            "support": len(group),
            "top_support": len(top),
            "estimator": "median of top 50% feasible CALO training runs by final objective",
            "parameters": parameters,
        }
    return priors


def build_experience_repository(
    database,
    output_path: str | Path,
    *,
    verified_only: bool = True,
    max_solutions_per_problem: int = 64,
) -> HistoricalExperienceRepository:
    """Build a leakage-aware repository from explicitly eligible historical experiments."""

    experiments = {
        row["id"]: row
        for row in database.list_learning_experiments(role="train", eligible_only=True)
    }
    policy_trajectories: list[dict] = []
    solution_entries: list[dict] = []
    summary_only_calo_runs = 0
    skipped_unverified = 0

    for experiment_id, experiment in experiments.items():
        config = _json_load(experiment.get("config_json"), {})
        for row in database.list_runs(experiment_id=experiment_id, verified_only=False):
            if verified_only and row.get("validation_status") != "verified":
                skipped_unverified += 1
                continue
            result = _json_load(row.get("result_json"), {})
            problem = _problem_identity(config, result)
            entry = {
                "experiment_id": experiment_id,
                "run_id": row["id"],
                "algorithm": str(row.get("algorithm", result.get("algorithm", ""))),
                "run_index": int(row.get("run_index", 0)),
                "validation_status": str(row.get("validation_status", "unverified")),
                "problem": problem,
                "best_vector": list(result.get("best_vector") or []),
                "best_objective": _clean_float(result.get("best_objective")),
                "total_constraint_violation": _clean_float(
                    result.get("total_constraint_violation")
                ),
                "feasible": bool(result.get("feasible", False)),
                "evaluations": int(result.get("evaluations", 0)),
                "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
                "parameters": dict(result.get("parameters") or {}),
            }
            solution_entries.append(entry)

            if entry["algorithm"] == "CALO":
                trajectory = list((result.get("metadata") or {}).get("policy_trajectory") or [])
                trajectory_source = "exact_v1.3_runtime"
                if not trajectory:
                    trajectory = _reconstruct_legacy_calo_trajectory(result)
                    trajectory_source = "reconstructed_v1.2_summary"
                if trajectory:
                    policy_trajectories.append(
                        {
                            "experiment_id": experiment_id,
                            "run_id": row["id"],
                            "problem": problem,
                            "validation_status": row.get("validation_status", "unverified"),
                            "trajectory_source": trajectory_source,
                            "transitions": trajectory,
                        }
                    )
                    if trajectory_source != "exact_v1.3_runtime":
                        summary_only_calo_runs += 1
                else:
                    summary_only_calo_runs += 1

    # Keep the best feasible exemplars and then diverse near-feasible fallbacks per problem.
    grouped_solutions: dict[str, list[dict]] = {}
    for entry in solution_entries:
        grouped_solutions.setdefault(_problem_key(entry["problem"]), []).append(entry)
    cross_algorithm_solutions: list[dict] = []
    for _key, group in grouped_solutions.items():
        group.sort(
            key=lambda item: (
                0 if item["feasible"] else 1,
                (item["best_objective"] if item["best_objective"] is not None else float("inf"))
                if item["feasible"]
                else (
                    item["total_constraint_violation"]
                    if item["total_constraint_violation"] is not None
                    else float("inf")
                ),
            )
        )
        cross_algorithm_solutions.extend(group[: max(1, int(max_solutions_per_problem))])

    parameter_priors = _parameter_priors(solution_entries)
    transition_count = sum(len(item.get("transitions") or []) for item in policy_trajectories)
    exact_trajectories = sum(
        1 for item in policy_trajectories if item.get("trajectory_source") == "exact_v1.3_runtime"
    )
    reconstructed_trajectories = len(policy_trajectories) - exact_trajectories
    payload = {
        "schema_version": REPOSITORY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_database": str(Path(database.path).expanduser().resolve()),
        "selection_policy": {
            "experiment_role": "train",
            "learning_eligible": True,
            "verified_only": bool(verified_only),
            "test_and_validation_experiments_excluded": True,
        },
        "summary": {
            "eligible_training_experiments": len(experiments),
            "included_runs": len(solution_entries),
            "skipped_unverified_runs": skipped_unverified,
            "calo_policy_trajectories": len(policy_trajectories),
            "exact_calo_policy_trajectories": exact_trajectories,
            "reconstructed_legacy_calo_trajectories": reconstructed_trajectories,
            "policy_transitions": transition_count,
            "summary_only_calo_runs": summary_only_calo_runs,
            "cross_algorithm_solutions": len(cross_algorithm_solutions),
            "parameter_prior_groups": len(parameter_priors),
        },
        "policy_trajectories": policy_trajectories,
        "cross_algorithm_solutions": cross_algorithm_solutions,
        "parameter_priors": parameter_priors,
    }
    payload["repository_sha256"] = _repository_checksum(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return HistoricalExperienceRepository(payload=payload, path=str(destination.resolve()))
