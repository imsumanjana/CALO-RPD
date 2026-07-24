"""CALO v5.9 competitive multi-branch policy evolution.

v5.9 treats a competitive training session as a transaction. Branches train independently and
never average neural-network parameters. Exact optimizer/RNG resume states are staged privately;
a complete branch generation becomes authoritative only after every branch state and the new root
manifest are validated and durably committed.

Training-time champion comparison is feasibility-first, hardware-neutral and validation-bundle
fingerprinted. Final Base selection re-evaluates every eligible candidate under one common bundle
and uses a deterministic order-independent ranking protocol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import tempfile
import time
import uuid
from typing import Any, Iterable

import numpy as np
import torch

from calo_rpd_studio.ai.model_io import (
    durable_torch_save,
    durable_write_bytes,
    load_checkpoint,
    load_trusted_resume,
    write_trusted_resume_hash,
)

from .policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_DIM,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
)

_LOG = logging.getLogger(__name__)

_COMPARATOR_SCHEMA = "calo-champion-comparator-v5.9"
_MANIFEST_SCHEMA = 3


class TrainingSessionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    SAFE_STOPPED = "SAFE_STOPPED"
    SAFE_STOPPED_DEGRADED = "SAFE_STOPPED_DEGRADED"
    SAFE_STOPPED_PROTECTION = "SAFE_STOPPED_PROTECTION"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class CompetitiveTrainingResult:
    output_path: str
    history: list[dict]
    status: TrainingSessionStatus
    common_resume_epoch: int
    manifest_path: str
    selected_artifact_path: str = ""
    degraded_branches: tuple[str, ...] = ()

    # Backward compatible with ``path, history = train_policy_parallel(...)``. The unpacked path
    # remains the logical Base namespace so its ``.branches.json`` authoritative manifest is stable;
    # callers that need the saved candidate use ``selected_artifact_path`` explicitly.
    def __iter__(self):
        yield self.output_path
        yield self.history


@dataclass(frozen=True, slots=True)
class BranchSeed:
    branch_id: str
    seed: int
    strategy: str


def competitive_progress_snapshot(
    branch_payloads: list[dict],
    current_epochs,
    *,
    active_indices: Iterable[int] = (),
    finished_indices: Iterable[int] = (),
    concurrency: int = 1,
    common_safe_epoch: int = -1,
    training_mode: str = "cumulative",
) -> tuple[int, str, dict]:
    """Return truthful user-facing competitive training progress.

    The percentage is based on *scientific session branch-epochs*, not process leases or safe
    checkpoint cadence. For indefinite training the percentage is ``-1`` (indeterminate). The
    detail deliberately distinguishes session progress, cumulative epoch, device assignment and
    durable exact-safe checkpoints so a status such as ``epochs [2]`` can never be mistaken for a
    configured two-epoch target.
    """

    payloads = list(branch_payloads or [])
    epochs = [int(current_epochs[i]) for i in range(len(payloads))]
    active = {int(i) for i in active_indices}
    finished = {int(i) for i in finished_indices}
    mode = str(training_mode or "cumulative").lower()
    fixed = mode != "indefinite"

    rows: list[dict] = []
    completed_units = 0
    total_units = 0
    for index, payload in enumerate(payloads):
        start = int(payload.get("start_epoch", 0) or 0)
        target = int(payload.get("scientific_session_target_epoch", 0) or 0)
        current = int(epochs[index])
        session_target = max(0, target - start) if target > 0 else 0
        session_done = max(0, current - start)
        if session_target > 0:
            session_done = min(session_done, session_target)
            completed_units += session_done
            total_units += session_target
        state = "active" if index in active else "completed" if index in finished else "queued"
        rows.append(
            {
                "branch_id": str(payload.get("branch_id", f"B{index + 1:02d}")),
                "state": state,
                "device": str(payload.get("assigned_device", "") or "unassigned"),
                "start_epoch": start,
                "current_epoch": current,
                "target_epoch": target,
                "session_done": session_done,
                "session_target": session_target,
            }
        )

    if fixed and total_units > 0:
        percent = max(0, min(100, int(round(100.0 * completed_units / total_units))))
    else:
        percent = -1

    safe = int(common_safe_epoch)
    if len(rows) == 1:
        row = rows[0]
        current = int(row["current_epoch"])
        start = int(row["start_epoch"])
        target = int(row["target_epoch"])
        session_done = int(row["session_done"])
        session_target = int(row["session_target"])
        last_safe = safe if safe >= 0 else start
        next_rolling = ((max(current, last_safe) // 10) + 1) * 10
        if target > 0 and next_rolling > target:
            next_safe_text = f"terminal {target}"
        else:
            next_safe_text = str(next_rolling)
        if fixed and session_target > 0:
            session_text = f"session {session_done}/{session_target} epoch(s)"
            cumulative_text = (
                f" · cumulative {current}/{target}" if start > 0 else f" · epoch {current}/{target}"
            )
        else:
            session_text = f"cumulative epoch {current} · indefinite"
            cumulative_text = ""
        detail = (
            f"{row['branch_id']} {row['state']} · {session_text}{cumulative_text} · {row['device']} · "
            f"last exact safe {last_safe} · next exact safe {next_safe_text}"
        )
    else:
        active_count = sum(1 for row in rows if row["state"] == "active")
        completed_count = sum(1 for row in rows if row["state"] == "completed")
        queued_count = max(0, len(rows) - active_count - completed_count)
        if fixed and total_units > 0:
            overall = f"overall {completed_units}/{total_units} branch-epochs"
        else:
            overall = "indefinite branch rotation"
        branch_bits = []
        for row in rows[:8]:
            if int(row["session_target"]) > 0:
                branch_bits.append(
                    f"{row['branch_id']} {row['session_done']}/{row['session_target']}"
                )
            else:
                branch_bits.append(f"{row['branch_id']} e{row['current_epoch']}")
        if len(rows) > 8:
            branch_bits.append(f"+{len(rows) - 8} more")
        detail = (
            f"Competitive queue · {active_count}/{max(1, int(concurrency))} active · {queued_count} queued · "
            f"{completed_count} completed · {overall} · "
            + ", ".join(branch_bits)
            + f" · last common exact safe {safe if safe >= 0 else 0}"
        )

    payload = {
        "overall_percent": int(percent),
        "completed_branch_epochs": int(completed_units),
        "total_branch_epochs": int(total_units),
        "branches": rows,
        "common_safe_epoch": int(safe),
    }
    return int(percent), detail, payload


@dataclass(frozen=True, slots=True)
class ChampionDecision:
    superior: bool
    wins: int
    losses: int
    ties: int
    critical_wins: int
    critical_losses: int
    reason: str
    verdict: str = "INFERIOR"


# Runtime is deliberately diagnostic-only; hardware timing cannot vote a policy into the Base.
_QUALITY_METRIC_DIRECTIONS: dict[str, str] = {
    "feasible_episode_rate": "max",
    "mean_final_feasible_ratio": "max",
    "median_final_feasible_objective": "min",
    "mean_final_feasible_objective": "min",
    "best_final_feasible_objective": "min",
    "worst_final_feasible_objective": "min",
    "convergence_auc": "min",
    "median_constraint_violation": "min",
    "median_steps_to_first_feasibility": "min",
    "mean_validation_return": "max",
    "median_validation_return": "max",
    "worst_validation_return": "max",
    "objective_iqr": "min",
}
_CRITICAL_METRICS = (
    "feasible_episode_rate",
    "median_final_feasible_objective",
    "convergence_auc",
    "median_constraint_violation",
)


def build_branch_seed_plan(config, parallel_runs: int | None = None) -> list[BranchSeed]:
    """Build the explicit same/increment/decrement/custom branch seed plan."""

    base_seed = int(getattr(config, "seed", 0))
    same = max(0, int(getattr(config, "parallel_same_seed_branches", 0) or 0))
    inc = max(0, int(getattr(config, "parallel_incremental_branches", 0) or 0))
    dec = max(0, int(getattr(config, "parallel_decremental_branches", 0) or 0))
    custom_raw = getattr(config, "parallel_custom_seeds", ()) or ()
    if isinstance(custom_raw, str):
        custom = [int(item.strip()) for item in custom_raw.split(",") if item.strip()]
    else:
        custom = [int(item) for item in custom_raw]

    requested = int(parallel_runs or getattr(config, "parallel_runs", 1) or 1)
    if same + inc + dec + len(custom) == 0:
        same = 1
        inc = max(0, requested - 1)
    total = same + inc + dec + len(custom)
    if total <= 0:
        raise ValueError("At least one policy-training branch is required")

    seeds: list[tuple[int, str]] = []
    seeds.extend((base_seed, "same") for _ in range(same))
    seeds.extend((base_seed + offset, "incremental") for offset in range(1, inc + 1))
    seeds.extend((base_seed - offset, "decremental") for offset in range(1, dec + 1))
    seeds.extend((seed, "custom") for seed in custom)
    return [
        BranchSeed(branch_id=f"B{index + 1:02d}", seed=int(seed), strategy=strategy)
        for index, (seed, strategy) in enumerate(seeds)
    ]


def _metric_value(metrics: dict, key: str) -> float:
    direction = _QUALITY_METRIC_DIRECTIONS.get(key, "min")
    value = metrics.get(key)
    if value is None:
        return math.inf if direction == "min" else -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf if direction == "min" else -math.inf
    if math.isnan(number):
        return math.inf if direction == "min" else -math.inf
    return number


def _compare_one(candidate: float, incumbent: float, direction: str) -> int:
    if not math.isfinite(candidate) and not math.isfinite(incumbent):
        return 0
    if math.isfinite(candidate) and not math.isfinite(incumbent):
        return 1
    if not math.isfinite(candidate) and math.isfinite(incumbent):
        return -1
    scale = max(abs(candidate), abs(incumbent), 1.0)
    tol = 1e-7 * scale
    if abs(candidate - incumbent) <= tol:
        return 0
    return 1 if ((candidate < incumbent) if direction == "min" else (candidate > incumbent)) else -1


def _eligible(metrics: dict) -> bool:
    if not bool(metrics.get("valid", False)):
        return False
    if "eligible" in metrics:
        return bool(metrics.get("eligible"))
    feas = _metric_value(metrics, "feasible_episode_rate")
    if not math.isfinite(feas) or feas <= 0.0:
        return False
    # Old tests/legacy metrics may not contain objective evidence. Do not manufacture invalidity
    # solely because an older sparse metric dictionary omitted it.
    if "median_final_feasible_objective" in metrics:
        return math.isfinite(_metric_value(metrics, "median_final_feasible_objective"))
    return True


def _deployable_eligible(metrics: dict) -> bool:
    return bool(_eligible(metrics) and metrics.get("deployable_eligible", False))


def _bundle_compatible(candidate: dict, incumbent: dict) -> bool:
    a = str(candidate.get("validation_bundle_fingerprint", "") or "")
    b = str(incumbent.get("validation_bundle_fingerprint", "") or "")
    return not a or not b or a == b


def compare_champion_metrics(candidate: dict, incumbent: dict | None) -> ChampionDecision:
    """Feasibility-first, hardware-neutral, deterministic branch champion comparator.

    Final global Base selection does *not* use sequential pairwise promotion; all candidates are
    re-evaluated and ranked together. This pairwise comparator is used only for one branch's temporal
    champion tracking under the same fixed validation bundle.
    """

    if not _eligible(candidate):
        return ChampionDecision(False, 0, 1, 0, 0, 1, "candidate failed validity/feasibility eligibility gates", "INVALID")
    if incumbent is None or not _eligible(incumbent):
        return ChampionDecision(True, 1, 0, 0, 1, 0, "first eligible champion", "SUPERIOR")
    if not _bundle_compatible(candidate, incumbent):
        return ChampionDecision(False, 0, 0, 1, 0, 0, "validation bundle changed; incumbent/candidate must be re-evaluated together", "INVALID")

    cand_feas = _metric_value(candidate, "feasible_episode_rate")
    base_feas = _metric_value(incumbent, "feasible_episode_rate")
    if cand_feas + 0.02 < base_feas:
        return ChampionDecision(False, 0, 1, 0, 0, 1, "candidate materially reduces feasible-episode probability", "INFERIOR")

    wins = losses = ties = critical_wins = critical_losses = 0
    for key, direction in _QUALITY_METRIC_DIRECTIONS.items():
        result = _compare_one(_metric_value(candidate, key), _metric_value(incumbent, key), direction)
        if result > 0:
            wins += 1
            if key in _CRITICAL_METRICS:
                critical_wins += 1
        elif result < 0:
            losses += 1
            if key in _CRITICAL_METRICS:
                critical_losses += 1
        else:
            ties += 1

    cand_obj = _metric_value(candidate, "median_final_feasible_objective")
    base_obj = _metric_value(incumbent, "median_final_feasible_objective")
    if math.isfinite(cand_obj) and math.isfinite(base_obj):
        if cand_obj > base_obj + 0.01 * max(abs(base_obj), 1.0):
            return ChampionDecision(False, wins, losses, ties, critical_wins, critical_losses, "candidate worsens median final feasible objective by more than 1%", "INFERIOR")

    # Predeclared scientific lexicographic hierarchy. Correlated metrics remain reported as evidence
    # but do not get independent majority votes that can overwhelm feasibility/objective quality.
    hierarchy = (
        ("feasible_episode_rate", "max"),
        ("median_final_feasible_objective", "min"),
        ("median_constraint_violation", "min"),
        ("convergence_auc", "min"),
        ("objective_iqr", "min"),
        ("median_validation_return", "max"),
    )
    for key, direction in hierarchy:
        result = _compare_one(_metric_value(candidate, key), _metric_value(incumbent, key), direction)
        if result > 0:
            return ChampionDecision(True, wins, losses, ties, critical_wins, critical_losses, f"candidate is superior on predeclared hierarchy at {key}", "SUPERIOR")
        if result < 0:
            return ChampionDecision(False, wins, losses, ties, critical_wins, critical_losses, f"candidate is inferior on predeclared hierarchy at {key}", "INFERIOR")
    return ChampionDecision(False, wins, losses, ties, critical_wins, critical_losses, "candidate is scientifically equivalent within comparator tolerances", "EQUIVALENT")


def _deterministic_action(network, state: np.ndarray, device: torch.device):
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    with torch.inference_mode():
        regime_logits, operator_logits, alpha, beta, _value = network(tensor)
        regime = int(torch.argmax(regime_logits).item())
        operator = int(torch.argmax(operator_logits).item())
        parameter = (alpha / torch.clamp(alpha + beta, min=1e-8)).detach().cpu().numpy()
    return regime, operator, parameter


def _development_case_identity(items: Iterable[str]) -> list[dict[str, str]]:
    identities = []
    for raw in items:
        path = Path(str(raw)).expanduser()
        digest = ""
        if path.is_file():
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                digest = "unreadable"
        identities.append({"source": str(raw), "sha256": digest})
    return identities


def validation_bundle_fingerprint(config) -> str:
    payload = {
        "schema": _COMPARATOR_SCHEMA,
        "seed": int(getattr(config, "champion_validation_seed", 918273)),
        "episodes_per_stage": int(getattr(config, "champion_validation_episodes", 5) or 5),
        "horizon": int(getattr(config, "champion_validation_horizon", 12) or 12),
        "population_size": int(getattr(config, "population_size", 20) or 20),
        "minimum_feasible_rate": float(0.80 if getattr(config, "champion_min_feasible_rate", None) is None else getattr(config, "champion_min_feasible_rate")),
        "development_cases": _development_case_identity(getattr(config, "development_cases", ()) or ()),
        "development_experiment_config": _development_case_identity(
            [getattr(config, "development_experiment_config_path", "")]
            if str(getattr(config, "development_experiment_config_path", "") or "").strip() else []
        ),
        "state_schema": POLICY_STATE_SCHEMA,
        "action_schema": POLICY_ACTION_SCHEMA,
        "training_environment": TRAINING_ENVIRONMENT_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_development_problem(config, source: str, scenario_seed: int):
    """Build a real ORPD development problem from the exact declared experiment formulation."""
    from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
    from calo_rpd_studio.experiments.experiment_runner import build_scenarios
    from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
    from calo_rpd_studio.power_system.case_loader import CaseLoader

    config_path = str(getattr(config, "development_experiment_config_path", "") or "").strip()
    if not config_path:
        raise ValueError("A development_experiment_config_path is required for deployable Base evidence")
    experiment = ExperimentConfig.load(config_path)
    experiment.case_name = str(source)
    experiment.validate_policy_development()
    case = CaseLoader.load(experiment.case_name)
    scenarios = build_scenarios(experiment, int(scenario_seed), case)
    problem_config = ORPDProblemConfig(
        objective=experiment.objective,
        variables=experiment.variables,
        robust=experiment.robust_objective,
        power_flow=experiment.power_flow,
        constraint_tolerances=experiment.constraint_tolerances,
    )
    return ORPDProblem(case, problem_config, scenarios)


def _run_policy_episode(network, env, horizon: int, device, inference_accumulator: list[float]) -> dict:
    episode_return = 0.0
    first_feasible = horizon + 1
    any_feasible = False
    quality_curve: list[float] = []
    for step in range(horizon):
        state = env.policy_state(horizon)
        started = time.perf_counter()
        regime, operator, parameter = _deterministic_action(network, state, device)
        inference_accumulator[0] += time.perf_counter() - started
        inference_accumulator[1] += 1
        episode_return += float(env.step(regime, operator, parameter, horizon))
        violation, objective, feasible_ratio = env._diagnostics(env.evaluations)
        if feasible_ratio > 0.0:
            any_feasible = True
            if first_feasible == horizon + 1:
                first_feasible = step + 1
        quality = float(objective) if math.isfinite(float(objective)) else 1.0e9 + 1.0e6 * max(float(violation), 0.0)
        quality_curve.append(quality)
    violation, objective, feasible_ratio = env._diagnostics(env.evaluations)
    final_best_feasible = bool(feasible_ratio > 0.0 and math.isfinite(float(objective)))
    return {
        "return": float(episode_return),
        "final_feasible_ratio": float(feasible_ratio),
        "final_best_feasible": final_best_feasible,
        "any_feasible": bool(any_feasible),
        "objective": float(objective) if final_best_feasible else float("inf"),
        "violation": float(violation) if math.isfinite(float(violation)) else 1.0e12,
        "first_feasible": float(first_feasible),
        "auc": float(np.mean(quality_curve)) if quality_curve else 1.0e12,
    }


def evaluate_policy_multimetric(network, config, *, validation_seed: int | None = None) -> dict:
    """Evaluate training quality separately from deployable real-ORPD scientific evidence.

    Synthetic curriculum tasks are screening evidence only. A candidate is deployable only when a
    fixed real ORPD development suite is present *and* an exact ExperimentConfig reproduces the
    objective, controls, robust scenarios, PF options and constraint tolerances. Raw objectives from
    unrelated tasks are never pooled; real-case objectives are normalized by a fixed center-control
    reference value for that case/formulation.
    """
    from .training import SyntheticCALOEnvironment

    base_seed = int(validation_seed if validation_seed is not None else getattr(config, "champion_validation_seed", 918273))
    episodes_per_stage = max(1, int(getattr(config, "champion_validation_episodes", 5) or 5))
    horizon = max(2, min(int(getattr(config, "horizon", 28)), int(getattr(config, "champion_validation_horizon", 12) or 12)))
    device = next(network.parameters()).device
    was_training = bool(network.training)
    network.eval()
    inference = [0.0, 0]
    synthetic_rows: list[dict] = []
    real_rows: list[dict] = []
    real_case_rows: dict[str, list[dict]] = {}
    dev_cases = tuple(getattr(config, "development_cases", ()) or ())
    exact_config_path = str(getattr(config, "development_experiment_config_path", "") or "").strip()

    try:
        # Training Champion screening bundle: fixed synthetic stages only.
        for stage in (0, 1, 2, 3):
            for rep in range(episodes_per_stage):
                seed = base_seed + stage * 100_003 + rep * 10_007
                env = SyntheticCALOEnvironment(np.random.default_rng(seed), stage, int(config.population_size))
                synthetic_rows.append(_run_policy_episode(network, env, horizon, device, inference))

        # Deployable Scientific Base bundle: fixed real ORPD cases under the exact declared formulation.
        if dev_cases and exact_config_path and Path(exact_config_path).expanduser().is_file():
            for case_index, source in enumerate(dev_cases):
                case_name = str(source)
                for rep in range(episodes_per_stage):
                    seed = base_seed + 900_001 + case_index * 100_003 + rep * 10_007
                    problem = _exact_development_problem(config, case_name, seed)
                    # Fixed formulation-specific reference scale prevents raw objective units/case size
                    # from dominating a pooled median across heterogeneous systems.
                    center = np.full(problem.dimension, 0.5, dtype=float)
                    reference = problem.evaluate(center)
                    reference_scale = max(abs(float(reference.value)), 1e-12) if math.isfinite(float(reference.value)) else 1.0
                    env = SyntheticCALOEnvironment(np.random.default_rng(seed), 4, int(config.population_size), problem=problem)
                    row = _run_policy_episode(network, env, horizon, device, inference)
                    row["case"] = case_name
                    row["reference_scale"] = float(reference_scale)
                    row["normalized_objective"] = (
                        float(row["objective"]) / reference_scale if row["final_best_feasible"] else float("inf")
                    )
                    real_rows.append(row)
                    real_case_rows.setdefault(case_name, []).append(row)
    finally:
        network.train(was_training)

    def summarize(rows: list[dict], *, normalized: bool = False) -> dict:
        returns = [float(r["return"]) for r in rows]
        final_ratios = [float(r["final_feasible_ratio"]) for r in rows]
        violations = [float(r["violation"]) for r in rows]
        first = [float(r["first_feasible"]) for r in rows]
        aucs = [float(r["auc"]) for r in rows]
        key = "normalized_objective" if normalized else "objective"
        objectives = np.asarray([float(r[key]) for r in rows if math.isfinite(float(r.get(key, float("inf"))))], dtype=float)
        return {
            "episodes": len(rows),
            "valid": bool(rows) and all(math.isfinite(v) for v in returns),
            "any_feasible_found_rate": float(np.mean([bool(r["any_feasible"]) for r in rows])) if rows else 0.0,
            "final_best_feasible_rate": float(np.mean([bool(r["final_best_feasible"]) for r in rows])) if rows else 0.0,
            "mean_final_feasible_ratio": float(np.mean(final_ratios)) if rows else 0.0,
            "median_objective": float(np.median(objectives)) if len(objectives) else 1.0e12,
            "mean_objective": float(np.mean(objectives)) if len(objectives) else 1.0e12,
            "best_objective": float(np.min(objectives)) if len(objectives) else 1.0e12,
            "worst_objective": float(np.max(objectives)) if len(objectives) else 1.0e12,
            "objective_iqr": float(np.percentile(objectives, 75) - np.percentile(objectives, 25)) if len(objectives) >= 2 else 0.0,
            "median_constraint_violation": float(np.median(violations)) if rows else 1.0e12,
            "median_steps_to_first_feasibility": float(np.median(first)) if rows else float(horizon + 1),
            "convergence_auc": float(np.mean(aucs)) if rows else 1.0e12,
            "mean_validation_return": float(np.mean(returns)) if rows else -1.0e12,
            "median_validation_return": float(np.median(returns)) if rows else -1.0e12,
            "worst_validation_return": float(np.min(returns)) if rows else -1.0e12,
        }

    synthetic = summarize(synthetic_rows, normalized=False)
    real = summarize(real_rows, normalized=True) if real_rows else {}
    minimum_rate = float(0.80 if getattr(config, "champion_min_feasible_rate", None) is None else getattr(config, "champion_min_feasible_rate"))
    screening_eligible = bool(synthetic.get("valid")) and float(synthetic.get("final_best_feasible_rate", 0.0)) >= minimum_rate
    per_case = {
        name: summarize(rows, normalized=True)
        for name, rows in sorted(real_case_rows.items())
    }
    all_cases_meet_feasibility = bool(per_case) and all(
        float(row.get("final_best_feasible_rate", 0.0)) >= minimum_rate for row in per_case.values()
    )
    deployable_eligible = bool(
        screening_eligible
        and exact_config_path
        and len(per_case) == len(dev_cases)
        and all_cases_meet_feasibility
        and bool(real.get("valid", False))
        and math.isfinite(float(real.get("median_objective", float("inf"))))
    )
    primary = real if real_rows else synthetic
    return {
        "valid": bool(synthetic.get("valid", False)),
        "eligible": bool(screening_eligible),  # Training Champion eligibility only.
        "screening_eligible": bool(screening_eligible),
        "deployable_eligible": bool(deployable_eligible),
        "deployment_evidence_kind": "exact_real_orpd" if real_rows else "synthetic_screening_only",
        "comparator_schema_version": _COMPARATOR_SCHEMA,
        "validation_bundle_fingerprint": validation_bundle_fingerprint(config),
        "validation_seed": base_seed,
        "validation_episodes": int(synthetic.get("episodes", 0)) + int(real.get("episodes", 0) if real else 0),
        "feasible_episode_rate": float(primary.get("final_best_feasible_rate", 0.0)),
        "any_feasible_found_rate": float(primary.get("any_feasible_found_rate", 0.0)),
        "final_best_feasible_rate": float(primary.get("final_best_feasible_rate", 0.0)),
        "mean_final_feasible_ratio": float(primary.get("mean_final_feasible_ratio", 0.0)),
        "median_final_feasible_objective": float(primary.get("median_objective", 1.0e12)),
        "mean_final_feasible_objective": float(primary.get("mean_objective", 1.0e12)),
        "best_final_feasible_objective": float(primary.get("best_objective", 1.0e12)),
        "worst_final_feasible_objective": float(primary.get("worst_objective", 1.0e12)),
        "convergence_auc": float(primary.get("convergence_auc", 1.0e12)),
        "median_constraint_violation": float(primary.get("median_constraint_violation", 1.0e12)),
        "median_steps_to_first_feasibility": float(primary.get("median_steps_to_first_feasibility", horizon + 1)),
        "mean_validation_return": float(primary.get("mean_validation_return", -1.0e12)),
        "median_validation_return": float(primary.get("median_validation_return", -1.0e12)),
        "worst_validation_return": float(primary.get("worst_validation_return", -1.0e12)),
        "objective_iqr": float(primary.get("objective_iqr", 0.0)),
        "synthetic_screening": synthetic,
        "real_orpd_normalized": real,
        "real_orpd_by_case": per_case,
        "development_experiment_config_path": exact_config_path,
        "policy_inference_ms": float(1000.0 * inference[0] / max(int(inference[1]), 1)),
    }


def _rank_key(metrics: dict, *, source_priority: int = 1, stable_id: str = "") -> tuple:
    """Order-independent global Base ranking key; lower tuple is better."""
    return (
        0 if _eligible(metrics) else 1,
        -_metric_value(metrics, "feasible_episode_rate"),
        _metric_value(metrics, "median_final_feasible_objective"),
        _metric_value(metrics, "median_constraint_violation"),
        _metric_value(metrics, "convergence_auc"),
        _metric_value(metrics, "objective_iqr"),
        -_metric_value(metrics, "median_validation_return"),
        int(source_priority),
        str(stable_id),
    )


class BranchChampionTracker:
    def __init__(self, *, base_payload: dict | None = None, base_metrics: dict | None = None, decision_limit: int = 200):
        self.state_dict: dict[str, torch.Tensor] | None = None
        self.metrics: dict | None = None
        self.epoch = 0
        self.source = "none"
        self.decisions: deque[dict] = deque(maxlen=max(10, int(decision_limit)))
        if base_payload is not None and base_metrics is not None and _eligible(base_metrics):
            self.state_dict = {k: v.detach().cpu().clone() for k, v in dict(base_payload.get("model_state_dict", base_payload)).items()}
            self.metrics = dict(base_metrics)
            self.epoch = int(dict(base_payload.get("metadata", {}) or {}).get("cumulative_epoch", 0) or 0)
            self.source = "base_threshold"

    def restore_from_extra(self, extra: dict) -> None:
        champion = dict(extra.get("branch_champion", {}) or {})
        state = champion.get("model_state_dict")
        metrics = champion.get("metrics")
        if isinstance(state, dict) and isinstance(metrics, dict):
            # Never compare stale validation evidence to a different bundle.
            if self.metrics is not None and not _bundle_compatible(metrics, self.metrics):
                return
            candidate = {k: v.detach().cpu().clone() for k, v in state.items() if torch.is_tensor(v)}
            decision = compare_champion_metrics(metrics, self.metrics)
            if decision.superior or self.metrics is None:
                self.state_dict = candidate
                self.metrics = dict(metrics)
                self.epoch = int(champion.get("epoch", 0) or 0)
                self.source = str(champion.get("source", "restored_branch"))

    def consider(self, network, metrics: dict, epoch: int, *, source: str) -> ChampionDecision:
        decision = compare_champion_metrics(metrics, self.metrics)
        self.decisions.append({"epoch": int(epoch), "source": str(source), "decision": asdict(decision), "metrics": dict(metrics)})
        if decision.superior:
            self.state_dict = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            self.metrics = dict(metrics)
            self.epoch = int(epoch)
            self.source = str(source)
        return decision

    def extra_payload(self) -> dict:
        return {
            "branch_champion": {
                "model_state_dict": self.state_dict or {},
                "metrics": dict(self.metrics or {}),
                "epoch": int(self.epoch),
                "source": self.source,
                "decision_history_tail": list(self.decisions),
            }
        }


class RollingSafeStore:
    """Disk-backed exact-state snapshots used only during an active session."""

    def __init__(self, root: Path, branch_id: str):
        self.directory = Path(root) / branch_id
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, epoch: int) -> Path:
        return self.directory / f"safe_{int(epoch):012d}.resume.pt"

    def epochs(self) -> list[int]:
        output = []
        for path in self.directory.glob("safe_*.resume.pt"):
            try:
                output.append(int(path.name.split("_")[1].split(".")[0]))
            except (IndexError, ValueError):
                _LOG.warning("Ignoring malformed rolling-safe snapshot filename: %s", path)
        return sorted(output)

    def cleanup_before(self, epoch: int) -> None:
        for old_epoch in self.epochs():
            if old_epoch < int(epoch):
                path = self.path(old_epoch)
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)


def _copy_trusted_resume(source: Path, target: Path) -> None:
    payload = load_trusted_resume(source, map_location="cpu")
    durable_torch_save(payload, target)
    write_trusted_resume_hash(target)


def _config_payload(config) -> dict:
    return asdict(config) if is_dataclass(config) else dict(config)


def _rebuild_config(config_dict: dict):
    from .training import TrainingConfig
    from .heterogeneous_training import HeterogeneousTrainingConfig

    hetero_fields = {f.name for f in fields(HeterogeneousTrainingConfig)}
    base_fields = {f.name for f in fields(TrainingConfig)}
    use_hetero = bool(config_dict.get("heterogeneous_rollouts", False))
    cls = HeterogeneousTrainingConfig if use_hetero else TrainingConfig
    allowed = hetero_fields if use_hetero else base_fields
    payload = {key: value for key, value in config_dict.items() if key in allowed}
    for name in ("development_cases", "parallel_custom_seeds", "curriculum_stage_milestones"):
        if name in payload and not isinstance(payload[name], tuple):
            payload[name] = tuple(payload[name] or ())
    return cls(**payload)


def _load_base_payload(path: str | Path | None) -> tuple[dict | None, dict | None]:
    if not path:
        return None, None
    source = Path(path)
    if not source.is_file():
        return None, None
    payload = load_checkpoint(source, map_location="cpu")
    metrics = dict(dict(payload.get("metadata", {}) or {}).get("champion_metrics", {}) or {})
    return payload, metrics or None


def _network_from_payload(payload: dict, config):
    from .policy_network import CALOPolicyNetwork

    arch = dict(payload.get("architecture", {}) or {})
    network = CALOPolicyNetwork(POLICY_STATE_DIM, int(arch.get("hidden_dim", getattr(config, "hidden_dim", 96))))
    network.load_state_dict(payload.get("model_state_dict", payload))
    return network


def _evaluate_payload(payload: dict, config) -> dict:
    return evaluate_policy_multimetric(_network_from_payload(payload, config), config)


def _plan_branch_resources(config, total_branches: int):
    """Return the authoritative v6.1 protected branch resource plan.

    Total scientific branch count is intentionally independent of simultaneous concurrency. The
    planner consumes the same ComputeTopology/Safe-80 profile shown on Dashboard and never
    performs implicit accelerator-to-CPU spillover.
    """
    from calo_rpd_studio.compute.training_resources import build_training_resource_plan

    return build_training_resource_plan(config, int(total_branches))


def _plan_branch_devices(config, count: int) -> list[str]:
    """Compatibility helper returning simultaneous primary devices only.

    ``count`` is interpreted as total scientific branches; callers that require the richer queue/
    capability plan should use :func:`_plan_branch_resources`.
    """
    plan = _plan_branch_resources(config, int(count))
    return [slot.primary_device for slot in plan.slots]


def _branch_worker_main(
    config_dict: dict,
    branch_payload: dict,
    scratch_root: str,
    cancel_event,
    protection_level,
    current_epochs,
    last_safe_epochs,
    global_safe_epoch,
    last_progress,
    status_queue,
) -> None:
    from .training import TrainingCancelled, save_training_resume, train_policy
    from .heterogeneous_training import train_policy_heterogeneous

    index = int(branch_payload["index"])
    branch_id = str(branch_payload["branch_id"])
    lease_id = str(branch_payload.get("lease_id", "") or "")
    lease_number = max(1, int(branch_payload.get("lease_number", 1) or 1))
    config = _rebuild_config(config_dict)
    config.seed = int(branch_payload["seed"])
    config.parallel_runs = 1
    config.checkpoint_each_epoch = False
    config.resume_checkpoint = str(branch_payload.get("resume_path", "") or "")
    config.initial_policy_checkpoint = str(branch_payload.get("initial_policy_checkpoint", "") or "")
    config.ppo_device = str(branch_payload.get("assigned_device", config.ppo_device))
    config.lease_target_epoch = max(0, int(branch_payload.get("lease_target_epoch", 0) or 0))
    branch_cpu_budget = max(1, int(branch_payload.get("cpu_worker_budget", 1) or 1))
    requested_rollout_workers = int(getattr(config, "rollout_workers", 0) or 0)
    config.rollout_workers = (
        branch_cpu_budget
        if requested_rollout_workers <= 0
        else max(1, min(requested_rollout_workers, branch_cpu_budget))
    )

    # v6.1 beta3: one global CPU budget is divided across active branch slots. Prevent native BLAS/
    # OpenMP libraries inside each branch/actor process from multiplying that budget again.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _native_thread_limiter = None
    try:
        # Runtime threadpool limiting also constrains BLAS libraries that were imported during the
        # spawn bootstrap before the target function could set environment variables.
        from threadpoolctl import threadpool_limits

        _native_thread_limiter = threadpool_limits(limits=1)
    except (ImportError, RuntimeError):
        _LOG.debug("threadpoolctl unavailable; relying on environment/PyTorch thread caps")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (RuntimeError, AttributeError):
        _LOG.debug("Could not apply per-branch PyTorch host thread cap", exc_info=True)

    # v6.1 beta4 uses explicit device capabilities. A direct XPU may own a full branch. A sidecar
    # XPU is an auxiliary actor/evaluator only and may be leased to at most one simultaneous slot.
    # No branch may silently instantiate a sibling branch's accelerator.
    if bool(getattr(config, "heterogeneous_rollouts", False)):
        from calo_rpd_studio.compute.training_resources import protected_rollout_shares

        config.strict_resource_binding = True
        effective = protected_rollout_shares(
            cuda_share=int(getattr(config, "cuda_rollout_share", 0) or 0),
            xpu_share=int(getattr(config, "xpu_rollout_share", 0) or 0),
            cpu_share=int(getattr(config, "cpu_rollout_share", 0) or 0),
            primary_device=str(config.ppo_device),
            auxiliary_xpu_runtime=str(branch_payload.get("auxiliary_xpu_runtime", "") or ""),
        )
        config.cuda_rollout_share = int(effective["cuda"])
        config.xpu_rollout_share = int(effective["xpu"])
        config.cpu_rollout_share = int(effective["cpu"])
    output_path = Path(branch_payload["working_output"])
    scratch = RollingSafeStore(Path(scratch_root), branch_id)
    base_payload, _stored_metrics = _load_base_payload(branch_payload.get("base_model_checkpoint"))
    base_metrics = dict(branch_payload.get("base_metrics", {}) or {}) or None
    tracker = BranchChampionTracker(
        base_payload=base_payload,
        base_metrics=base_metrics,
        decision_limit=int(getattr(config, "champion_decision_history_limit", 200) or 200),
    )

    resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    branch_initial_epoch = 0
    if resume_path is not None and resume_path.is_file():
        resume_payload = load_trusted_resume(resume_path, map_location="cpu")
        branch_initial_epoch = int(resume_payload.get("next_epoch", 0) or 0)
        tracker.restore_from_extra(dict(resume_payload.get("extra", {}) or {}))

    validation_interval = max(1, int(getattr(config, "champion_validation_interval_epochs", 10) or 10))
    safe_interval = 10  # v5.9 fixed rolling cadence; the starting exact state is also a valid safe point.
    max_lead = max(safe_interval, int(getattr(config, "max_branch_lead_epochs", 30) or 30))
    screening_best_by_stage: dict[int, float] = {}
    session_target_raw = int(branch_payload.get("scientific_session_target_epoch", 0) or 0)
    session_target = session_target_raw if session_target_raw > 0 else None

    def touch() -> None:
        last_progress[index] = time.monotonic()

    def extra_provider() -> dict:
        return {
            **tracker.extra_payload(),
            "branch_id": branch_id,
            "branch_seed": int(config.seed),
            "branch_seed_strategy": str(branch_payload.get("strategy", "")),
            "branch_start_mode": str(branch_payload.get("start_mode", "new")),
            "assigned_device": str(config.ppo_device),
            "validation_bundle_fingerprint": validation_bundle_fingerprint(config),
        }

    def observer(state: dict) -> None:
        completed_epoch = int(state["epoch"])
        current_epochs[index] = completed_epoch
        touch()
        stage = int(state.get("stage", 0))
        returns = [float(value) for value in state.get("episode_returns", []) if math.isfinite(float(value))]
        screen_value = float(np.mean(returns)) if returns else -1.0e12
        previous_screen = screening_best_by_stage.get(stage, -math.inf)
        screen_promising = screen_value > previous_screen + 1e-12
        if screen_promising:
            screening_best_by_stage[stage] = screen_value
        if lease_number > 1 and completed_epoch == branch_initial_epoch:
            # Exact process-lease continuation begins from a state already durably persisted at the
            # prior lease boundary. Avoid rewriting the same checkpoint and rerunning an identical
            # champion bundle merely because the process was rotated by the scheduler.
            status_queue.put({"type": "resumed", "branch_id": branch_id, "lease_id": lease_id, "epoch": completed_epoch})
            return
        # Persist the exact safe state before any potentially expensive champion evaluation.  This
        # ordering is essential for immediate Safe Stop: cancellation must never wait for a full
        # validation bundle merely to make epoch-0/start-epoch recoverable.
        if completed_epoch == branch_initial_epoch or completed_epoch % safe_interval == 0:
            path = scratch.path(completed_epoch)
            save_training_resume(
                path,
                network=state["network"],
                optimizer=state["optimizer"],
                next_epoch=completed_epoch,
                history=state["history"],
                rng=state["rng"],
                historical_pretraining=state["historical_pretraining"],
                config=config,
                extra={**extra_provider(), "temporary_safe_snapshot": True, "curriculum_encoding": "zero_based_0_4"},
            )
            last_safe_epochs[index] = completed_epoch
            committed = int(global_safe_epoch.value)
            if committed >= 0:
                scratch.cleanup_before(committed)
            status_queue.put({"type": "safe", "branch_id": branch_id, "lease_id": lease_id, "epoch": completed_epoch})
            while (
                all(int(last_safe_epochs[pos]) >= 0 for pos in range(len(last_safe_epochs)))
                and completed_epoch - int(global_safe_epoch.value) > max_lead
                and not bool(cancel_event.value)
            ):
                touch()
                time.sleep(0.05)

        if bool(cancel_event.value):
            status_queue.put({"type": "screen", "branch_id": branch_id, "lease_id": lease_id, "epoch": completed_epoch, "screening_mean_episode_return": screen_value, "deep_validation": False, "cancelled_before_validation": True})
            return

        deep_due = (
            completed_epoch == branch_initial_epoch
            or completed_epoch % validation_interval == 0
            or screen_promising
            or (session_target is not None and completed_epoch >= session_target)
        )
        if deep_due:
            metrics = evaluate_policy_multimetric(state["network"], config)
            metrics["screening_mean_episode_return"] = screen_value
            metrics["deep_validation_trigger"] = (
                "initial" if completed_epoch == branch_initial_epoch
                else "session_terminal" if session_target is not None and completed_epoch >= session_target
                else "screen_improvement" if screen_promising
                else "periodic"
            )
            decision = tracker.consider(state["network"], metrics, completed_epoch, source=f"{branch_id}@{completed_epoch}")
            status_queue.put({"type": "champion", "branch_id": branch_id, "lease_id": lease_id, "epoch": completed_epoch, "promoted": bool(decision.superior), "verdict": decision.verdict, "reason": decision.reason, "metrics": metrics})
        else:
            status_queue.put({"type": "screen", "branch_id": branch_id, "lease_id": lease_id, "epoch": completed_epoch, "screening_mean_episode_return": screen_value, "deep_validation": False})

    def cancelled() -> bool:
        return bool(bool(cancel_event.value))

    try:
        touch()
        status_queue.put({"type": "started", "branch_id": branch_id, "lease_id": lease_id, "seed": int(config.seed), "assigned_device": str(config.ppo_device), "cpu_worker_budget": branch_cpu_budget, "auxiliary_xpu_runtime": str(branch_payload.get("auxiliary_xpu_runtime", "") or "")})
        trainer = train_policy_heterogeneous if bool(getattr(config, "heterogeneous_rollouts", False)) else train_policy
        trainer(
            config,
            output_path,
            progress_callback=None,
            cancel_callback=cancelled,
            epoch_observer=observer,
            resume_extra_provider=extra_provider,
            cancel_during_rollout=False,
            suppress_cancel_persistence=True,
            protection_callback=lambda: int(protection_level.value),
        )
        touch()
        status_queue.put({"type": "completed", "branch_id": branch_id, "lease_id": lease_id, "epoch": int(current_epochs[index]), "terminal_resume": str(Path(config.resume_checkpoint))})
    except TrainingCancelled:
        touch()
        status_queue.put({"type": "cancelled", "branch_id": branch_id, "lease_id": lease_id, "epoch": int(current_epochs[index])})
    except BaseException as exc:
        touch()
        status_queue.put({"type": "fatal", "branch_id": branch_id, "lease_id": lease_id, "error": f"{type(exc).__name__}: {exc}"})
        raise


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".branches.json")


def load_branch_manifest(output_path: str | Path) -> dict:
    path = _manifest_path(Path(output_path))
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    durable_write_bytes(path, (json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf-8"))


def _recovery_directory(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_branches" / "recovery"


def list_recoverable_sessions(output_path: str | Path) -> list[dict]:
    directory = _recovery_directory(Path(output_path))
    sessions = []
    if not directory.is_dir():
        return sessions
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("status", "")) not in {"COMMITTED", "DISCARDED"}:
            payload["recovery_index_path"] = str(path)
            sessions.append(payload)
    return sessions


def discard_recovery_session(output_path: str | Path, session_id: str) -> None:
    index = _recovery_directory(Path(output_path)) / f"{session_id}.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    payload = json.loads(index.read_text(encoding="utf-8"))
    scratch_raw = str(payload.get("scratch_root", "") or "").strip()
    if scratch_raw:
        scratch = Path(scratch_raw).expanduser().resolve()
        # Every competitive scratch root is created as <scratch-base>/<session_id>. Refuse any
        # malformed recovery record that could otherwise broaden deletion beyond one session.
        if scratch.name != str(session_id):
            raise ValueError("Recovery scratch path does not match the requested session; discard refused")
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=False)
    payload["status"] = "DISCARDED"
    _atomic_json(index, payload)
    index.unlink(missing_ok=True)


def _policy_payload_from_champion(champion: dict, config, *, branch_id: str, seed: int, session: dict) -> dict:
    state = champion.get("model_state_dict") or {}
    metrics = dict(champion.get("metrics", {}) or {})
    epoch = int(champion.get("epoch", 0) or 0)
    return {
        "model_state_dict": state,
        "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": int(config.hidden_dim)},
        "metadata": {
            "algorithm": "CALO",
            "calo_core": "v5.0",
            "policy_training_architecture": "v5.9",
            "training_method": "transactional competitive multi-branch PPO; no neural weight averaging",
            "training_config": _config_payload(config),
            "training_seed": int(seed),
            "cumulative_epoch": epoch,
            "champion_epoch": epoch,
            "champion_metrics": metrics,
            "champion_validation_bundle_fingerprint": metrics.get("validation_bundle_fingerprint", ""),
            "champion_comparator_schema": _COMPARATOR_SCHEMA,
            "base_source_branch": branch_id,
            "parallel_branches": int(session.get("branch_count", 1)),
            "parallel_seed_plan": session.get("seed_plan", []),
            "common_resume_epoch": int(session.get("common_resume_epoch", 0)),
            "policy_lineage_id": str(getattr(config, "policy_lineage_id", "")),
            "policy_lineage_name": str(getattr(config, "policy_lineage_name", "")),
            "policy_phase_index": int(getattr(config, "policy_phase_index", 1) or 1),
            "checkpoint_role": "competitive_base_model",
            "state_dimension": POLICY_STATE_DIM,
            "state_schema_version": POLICY_STATE_SCHEMA,
            "action_schema_version": POLICY_ACTION_SCHEMA,
            "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
            "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
        },
    }


def _save_immutable_base(output_path: Path, payload: dict) -> tuple[Path, str]:
    """Create an immutable candidate/Base artifact without mutating the logical Base alias.

    The logical alias is refreshed only *after* the authoritative branch manifest commits.  This
    prevents a failed multi-file finalization from leaving a new Base alias paired with an old
    exact-resume generation.  Provisional/non-eligible candidates therefore never overwrite the
    logical Base alias.
    """
    artifact_dir = output_path.parent / f"{output_path.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"base_{int(payload['metadata'].get('champion_epoch', 0)):012d}_{uuid.uuid4().hex[:10]}.pt"
    payload = {**payload, "metadata": dict(payload.get("metadata", {}))}
    payload["metadata"]["immutable_artifact_path"] = str(artifact_path.resolve())
    payload["metadata"]["immutable_terminal_checkpoint"] = str(artifact_path.resolve())
    durable_torch_save(payload, artifact_path)
    sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _atomic_json(artifact_path.with_suffix(".json"), {**payload["metadata"], "sha256": sha})
    return artifact_path, sha


def _candidate_from_resume(path: Path) -> dict | None:
    payload = load_trusted_resume(path, map_location="cpu")
    extra = dict(payload.get("extra", {}) or {})
    champion = dict(extra.get("branch_champion", {}) or {})
    if not champion.get("model_state_dict") or not champion.get("metrics"):
        return None
    if str(champion.get("source", "")) == "base_threshold":
        return None
    return champion


def _commit_branch_generation(branch_dir: Path, session_id: str, sources: list[tuple[dict, Path]], common_epoch: int) -> list[dict]:
    generations = branch_dir / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    staging = generations / f".{session_id}.staging"
    final_dir = generations / session_id
    shutil.rmtree(staging, ignore_errors=True)
    if final_dir.exists():
        raise FileExistsError(f"Branch generation already exists: {final_dir}")
    staging.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    try:
        for payload, source in sources:
            branch_id = str(payload["branch_id"])
            target = staging / f"{branch_id}.resume.pt"
            _copy_trusted_resume(source, target)
            verified = load_trusted_resume(target, map_location="cpu")
            actual_epoch = int(verified.get("next_epoch", 0) or 0)
            if actual_epoch != int(common_epoch):
                raise RuntimeError(f"Branch {branch_id} generation epoch mismatch: expected {common_epoch}, got {actual_epoch}")
            # Stream-copy bounded telemetry segments; never read a potentially large telemetry file
            # into RAM during generation finalization. Legacy single-file telemetry is accepted only
            # for migration/compatibility.
            resume_source = Path(str(payload.get("resume_path", "")))
            telemetry_sources = sorted(resume_source.parent.glob(resume_source.name + ".telemetry.*.jsonl"))
            legacy_telemetry = Path(str(resume_source) + ".telemetry.jsonl")
            if legacy_telemetry.is_file():
                telemetry_sources.append(legacy_telemetry)
            telemetry_paths: list[str] = []
            for seg_index, telemetry_source in enumerate(telemetry_sources, start=1):
                telemetry_target = staging / f"{branch_id}.telemetry.{seg_index:06d}.jsonl"
                with telemetry_source.open("rb") as src, telemetry_target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                telemetry_paths.append(str((final_dir / telemetry_target.name).resolve()))
            rows.append({
                "branch_id": branch_id,
                "seed": int(payload["seed"]),
                "strategy": str(payload["strategy"]),
                "resume_path": str((final_dir / target.name).resolve()),
                "resume_epoch": actual_epoch,
                "telemetry_path": telemetry_paths[0] if telemetry_paths else "",
                "telemetry_paths": telemetry_paths,
                "assigned_device": str(payload.get("assigned_device", "")),
                "resource_slot": int(payload.get("resource_slot", 0) or 0),
                "cpu_worker_budget": int(payload.get("cpu_worker_budget", 1) or 1),
                "auxiliary_xpu_runtime": str(payload.get("auxiliary_xpu_runtime", "") or ""),
                "auxiliary_xpu_name": str(payload.get("auxiliary_xpu_name", "") or ""),
                "status": "safe_stopped" if payload.get("session_cancelled") else "completed",
            })
        _atomic_json(staging / "generation.json", {"schema_version": 1, "session_id": session_id, "common_resume_epoch": int(common_epoch), "branches": rows})
        os.replace(staging, final_dir)
        # Directory fsync is best effort through a tiny durable marker in the parent.
        marker = generations / f".{session_id}.committed"
        durable_write_bytes(marker, b"committed\n")
        marker.unlink(missing_ok=True)
        return rows
    except Exception:
        # Staging is retained for forensic/recovery use only when it contains useful files.
        raise


def _drain_queue(status_queue, recent_messages: deque, fatal_messages: list[str], terminal_by_branch: dict[str, dict]) -> None:
    while True:
        try:
            message = status_queue.get_nowait()
        except queue.Empty:
            return
        recent_messages.append(message)
        msg_type = str(message.get("type", ""))
        branch = str(message.get("branch_id", ""))
        if msg_type == "fatal":
            fatal_messages.append(f"{branch}: {message.get('error')}")
        if msg_type in {"completed", "cancelled", "fatal"} and branch:
            terminal_by_branch[branch] = message


def _common_evaluate_candidates(previous_payload: dict | None, finalized: list[dict], config) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    evidence: list[dict] = []
    if previous_payload is not None:
        metrics = _evaluate_payload(previous_payload, config)
        candidates.append({"candidate_id": "previous_base", "source_priority": 0, "payload": previous_payload, "metrics": metrics, "branch_id": "", "seed": 0, "champion": None})
        evidence.append({"candidate_id": "previous_base", "metrics": metrics})

    for row in finalized:
        resume_path = Path(row["resume_path"])
        champion = _candidate_from_resume(resume_path)
        if champion:
            state = champion["model_state_dict"]
            epoch = int(champion.get("epoch", row.get("resume_epoch", 0)) or 0)
            source = str(champion.get("source", "branch_champion"))
        else:
            # A branch that never crossed the feasibility gate still has a scientifically useful
            # exact terminal candidate. It may be stored as provisional evidence, but cannot become
            # the logical Base until it passes the common eligibility gate.
            resume_payload = load_trusted_resume(resume_path, map_location="cpu")
            state = resume_payload["model_state_dict"]
            epoch = int(resume_payload.get("next_epoch", row.get("resume_epoch", 0)) or 0)
            source = "terminal_provisional"
            champion = {"model_state_dict": state, "metrics": {}, "epoch": epoch, "source": source}
        payload = {
            "model_state_dict": state,
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": int(config.hidden_dim)},
            "metadata": {},
        }
        metrics = _evaluate_payload(payload, config)
        champion = dict(champion)
        champion["metrics"] = metrics
        champion["epoch"] = epoch
        champion["source"] = source
        candidates.append({"candidate_id": str(row["branch_id"]), "source_priority": 1, "payload": payload, "metrics": metrics, "branch_id": str(row["branch_id"]), "seed": int(row["seed"]), "champion": champion})
        evidence.append({"candidate_id": str(row["branch_id"]), "metrics": metrics, "source": source})
    return candidates, evidence


def recover_competitive_session(output_path: str | Path, session_id: str) -> dict:
    """Recover the last common authenticated safe branch set after an interrupted session.

    Recovery intentionally does not promote an un-finalized branch champion. It publishes a new exact-resume
    generation at the last common safe epoch while retaining the previously committed Base artifact.
    """
    output = Path(output_path).expanduser().resolve()
    index_path = _recovery_directory(output) / f"{session_id}.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    recovery = json.loads(index_path.read_text(encoding="utf-8"))

    # v5.9 authority guard: an interrupted session may only replace the exact
    # authoritative generation from which it started.  If another session has
    # committed meanwhile, recovering the stale session as authoritative would
    # roll the repository backwards or mix branch/Base lineage.
    manifest_path = _manifest_path(output)
    expected_prior_sha = str(recovery.get("prior_manifest_sha256", "") or "")
    current_prior_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else ""
    if current_prior_sha != expected_prior_sha:
        raise RuntimeError(
            "Stale competitive recovery refused: the authoritative branch manifest changed "
            "after this session started. Recover/export the interrupted state as an explicit "
            "fork instead of replacing the newer authoritative generation."
        )
    current_prior = load_branch_manifest(output)
    expected_generation = str(recovery.get("prior_generation_id", "") or "")
    current_generation = str(current_prior.get("generation_id", "") or "")
    if current_generation != expected_generation:
        raise RuntimeError(
            "Stale competitive recovery refused: authoritative generation identity no longer "
            "matches the interrupted session's parent generation."
        )

    common = int(recovery.get("latest_common_safe_epoch", -1))
    if common < 0:
        raise RuntimeError("Recovery session has no common authenticated safe epoch")
    payloads = list(recovery.get("branches", []))
    sources = []
    for payload in payloads:
        source = RollingSafeStore(Path(recovery["scratch_root"]), payload["branch_id"]).path(common)
        if not source.is_file():
            raise RuntimeError(f"Missing recovery safe snapshot for {payload['branch_id']} at epoch {common}")
        load_trusted_resume(source, map_location="cpu")
        sources.append((payload, source))
    branch_dir = output.parent / f"{output.stem}_branches"
    generation_id = f"recovered_{session_id}_{uuid.uuid4().hex[:8]}"
    rows = _commit_branch_generation(branch_dir, generation_id, sources, common)
    prior = load_branch_manifest(output)
    # Recovery never manufactures/promotes a Base.  When a previous committed Base exists it is
    # retained byte-for-byte in the manifest.  A first-ever interrupted session is still recoverable
    # as an exact branch generation with an explicitly empty Base, ready for Exact Resume or a later
    # normal finalization/qualification cycle.
    manifest = dict(prior) if prior else {
        "schema_version": _MANIFEST_SCHEMA,
        "policy_lineage_id": str(recovery.get("policy_lineage_id", "")),
        "policy_lineage_name": str(recovery.get("policy_lineage_name", "")),
        "logical_base_alias": str(output),
        "base_artifact_path": "",
        "provisional_artifact_path": "",
        "base_sha256": "",
        "base_source": "none_recovered_exact_state_only",
        "base_source_branch": "",
        "base_metrics": {},
        "validation_bundle_fingerprint": "",
        "champion_comparator_schema": _COMPARATOR_SCHEMA,
        "base_candidate_ranking": [],
        "common_candidate_evidence": [],
        "previous_training_mode": str(recovery.get("training_mode", "")),
        "previous_session_epochs": 0,
        "seed_plan": [],
    }
    manifest.update({
        "schema_version": _MANIFEST_SCHEMA,
        "generation_id": generation_id,
        "common_resume_epoch": common,
        "branches": rows,
        "session": {"session_id": session_id, "status": "RECOVERED_SAFE_STATE", "recovered": True, "common_resume_epoch": common, "base_retained": bool(prior)},
    })
    _atomic_json(_manifest_path(output), manifest)
    recovery["status"] = "RECOVERED"
    recovery["recovered_generation_id"] = generation_id
    _atomic_json(index_path, recovery)
    shutil.rmtree(Path(recovery["scratch_root"]), ignore_errors=True)
    index_path.unlink(missing_ok=True)
    return manifest


def train_policy_competitive(
    config,
    output_path,
    *,
    parallel_runs: int | None = None,
    progress_callback=None,
    cancel_callback=None,
    session_state_callback=None,
) -> CompetitiveTrainingResult:
    """Train independent branches and transactionally publish one coherent exact-resume generation."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_plan = build_branch_seed_plan(config, parallel_runs)
    start_mode = str(getattr(config, "parallel_start_mode", "new") or "new").strip().lower()
    if start_mode not in {"new", "exact_resume", "base_guided_fork"}:
        raise ValueError(f"Unsupported parallel branch start mode: {start_mode}")

    prior_manifest = load_branch_manifest(output_path)
    if start_mode == "exact_resume" and prior_manifest.get("branches"):
        seed_plan = [BranchSeed(str(row.get("branch_id")), int(row.get("seed", 0)), str(row.get("strategy", "restored"))) for row in prior_manifest["branches"]]
    if start_mode == "exact_resume" and not prior_manifest:
        raise ValueError("Exact multi-branch resume requires an existing .branches.json manifest")
    config.parallel_runs = len(seed_plan)

    base_checkpoint = str(getattr(config, "base_model_checkpoint", "") or "")
    if not base_checkpoint and prior_manifest.get("base_artifact_path"):
        base_checkpoint = str(prior_manifest.get("base_artifact_path") or "")
    elif not base_checkpoint and not prior_manifest and output_path.is_file():
        try:
            payload = load_checkpoint(output_path, map_location="cpu")
            metadata = dict(payload.get("metadata", {}) or {})
            if bool(metadata.get("base_eligible", True)) and metadata.get("checkpoint_role") != "competitive_provisional_candidate":
                base_checkpoint = str(metadata.get("immutable_artifact_path", "") or output_path)
        except Exception:
            _LOG.warning("Could not inspect existing Base checkpoint", exc_info=True)
            base_checkpoint = ""

    previous_payload = None
    base_metrics_current = None
    if base_checkpoint and Path(base_checkpoint).is_file():
        previous_payload = load_checkpoint(base_checkpoint, map_location="cpu")
        base_metrics_current = _evaluate_payload(previous_payload, config)

    session_id = uuid.uuid4().hex
    scratch_base = Path(str(getattr(config, "training_scratch_dir", "") or "").strip() or (Path(tempfile.gettempdir()) / "CALO-RPD" / "policy_training"))
    scratch_root = scratch_base / session_id
    scratch_root.mkdir(parents=True, exist_ok=True)
    branch_dir = output_path.parent / f"{output_path.stem}_branches"
    branch_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir = branch_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    recovery_index = recovery_dir / f"{session_id}.json"

    ctx = mp.get_context("spawn")
    cancel_event = ctx.RawValue("b", 0)
    protection_level = ctx.RawValue("b", 0)
    current_epochs = ctx.RawArray("q", [0] * len(seed_plan))
    last_safe_epochs = ctx.RawArray("q", [-1] * len(seed_plan))
    global_safe_epoch = ctx.RawValue("q", -1)
    last_progress = ctx.RawArray("d", [time.monotonic()] * len(seed_plan))
    status_queue = ctx.Queue()

    prior_by_id = {str(row.get("branch_id")): row for row in prior_manifest.get("branches", [])}
    # v6.2 RC1: one live topology/profile snapshot admits the session, then a stateful adaptive
    # compute/thermal governor continuously controls staged branch admission and protective stop.
    from calo_rpd_studio.compute.topology import ComputeTopologyService, SafeResourceBudgetEngine
    from calo_rpd_studio.compute.training_resources import build_training_resource_plan
    from calo_rpd_studio.compute.governor import AdaptiveComputeGovernor, GovernorConfig, ProtectionState
    from calo_rpd_studio.compute.provenance import ComputeProvenanceRecorder

    topology_service = ComputeTopologyService()
    live_topology = topology_service.scan()
    live_profile = SafeResourceBudgetEngine(allocation_limit_fraction=0.80).calculate(live_topology)
    resource_plan = build_training_resource_plan(
        config, len(seed_plan), topology=live_topology, profile=live_profile
    )
    governor = AdaptiveComputeGovernor(
        live_profile,
        monitor=topology_service.monitor,
        config=GovernorConfig(
            allocation_limit_fraction=float(live_profile.allocation_limit_fraction),
            staged_startup_delay_seconds=max(0.0, float(getattr(config, "staged_startup_delay_seconds", 2.0) or 2.0)),
            sample_interval_seconds=max(0.1, float(getattr(config, "governor_sample_interval_seconds", 1.0) or 1.0)),
            amber_pause_seconds=max(0.0, float(getattr(config, "governor_amber_pause_seconds", 0.25) or 0.25)),
        ),
    )
    concurrency = int(resource_plan.simultaneous_branches)
    config.parallel_concurrency = concurrency
    # Freeze the exact live admission identity that actually governed this session, not merely
    # a potentially older Dashboard snapshot supplied by the caller.
    config.compute_topology_fingerprint = str(resource_plan.topology_fingerprint)
    config.compute_profile_fingerprint = str(resource_plan.protection_profile_fingerprint)
    global_cpu_budget = int(resource_plan.global_cpu_worker_budget)
    quantum = max(10, int(getattr(config, "branch_queue_quantum_epochs", 10) or 10))
    if quantum % 10:
        quantum = int(math.ceil(quantum / 10.0) * 10)
    slot_by_index = {int(slot.slot_index): slot for slot in resource_plan.slots}
    branch_payloads: list[dict] = []
    for index, spec in enumerate(seed_plan):
        prior = prior_by_id.get(spec.branch_id, {}) if start_mode == "exact_resume" else {}
        staged_resume = scratch_root / spec.branch_id / "terminal.resume.pt"
        staged_resume.parent.mkdir(parents=True, exist_ok=True)
        if start_mode == "exact_resume":
            official = Path(str(prior.get("resume_path", "")))
            if not official.is_file():
                raise FileNotFoundError(f"Exact branch resume checkpoint missing: {official}")
            _copy_trusted_resume(official, staged_resume)
            loaded = load_trusted_resume(staged_resume, map_location="cpu")
            start_epoch = int(loaded.get("next_epoch", 0) or 0)
            current_epochs[index] = start_epoch
            last_safe_epochs[index] = start_epoch
            # Materialize the starting exact state as a valid safe point before child launch.
            _copy_trusted_resume(staged_resume, RollingSafeStore(scratch_root, spec.branch_id).path(start_epoch))
        else:
            start_epoch = 0
        slot_index = index % concurrency
        slot = slot_by_index[slot_index]
        scientific_target = (
            start_epoch + max(1, int(getattr(config, "epochs", 1) or 1))
            if str(getattr(config, "training_mode", "cumulative")) != "indefinite"
            else 0
        )
        branch_payloads.append({
            "index": index,
            "branch_id": spec.branch_id,
            "seed": int(prior.get("seed", spec.seed) if start_mode == "exact_resume" else spec.seed),
            "strategy": str(prior.get("strategy", spec.strategy) if start_mode == "exact_resume" else spec.strategy),
            "start_mode": start_mode,
            "resume_path": str(staged_resume),
            "working_output": str(scratch_root / spec.branch_id / "working.pt"),
            "initial_policy_checkpoint": base_checkpoint if start_mode == "base_guided_fork" else "",
            "base_model_checkpoint": base_checkpoint,
            "base_metrics": dict(base_metrics_current or {}),
            "assigned_device": slot.primary_device,
            "resource_slot": slot_index,
            "cpu_worker_budget": int(slot.cpu_worker_budget),
            "auxiliary_xpu_runtime": str(slot.auxiliary_xpu_runtime or ""),
            "auxiliary_xpu_name": str(slot.auxiliary_xpu_name or ""),
            "start_epoch": start_epoch,
            "scientific_session_target_epoch": int(scientific_target),
        })

    if start_mode == "exact_resume" and len(branch_payloads) > 1:
        starts = {int(current_epochs[i]) for i in range(len(branch_payloads))}
        if len(starts) != 1:
            raise ValueError("Exact competitive resume requires all branches to start from one common saved epoch")
        global_safe_epoch.value = min(starts)

    recovery_payload = {
        "schema_version": 1,
        "session_id": session_id,
        "status": "RUNNING",
        "output_path": str(output_path),
        "scratch_root": str(scratch_root),
        "start_mode": start_mode,
        "training_mode": str(getattr(config, "training_mode", "cumulative")),
        "policy_lineage_id": str(getattr(config, "policy_lineage_id", "")),
        "policy_lineage_name": str(getattr(config, "policy_lineage_name", "")),
        "created_unix": time.time(),
        "latest_common_safe_epoch": int(global_safe_epoch.value),
        "branches": [{k: v for k, v in row.items() if k not in {"base_metrics"}} for row in branch_payloads],
        "prior_manifest_sha256": hashlib.sha256(_manifest_path(output_path).read_bytes()).hexdigest() if _manifest_path(output_path).is_file() else "",
        "prior_generation_id": str(prior_manifest.get("generation_id", "") or ""),
        "prior_common_resume_epoch": int(prior_manifest.get("common_resume_epoch", 0) or 0),
        "prior_validation_bundle_fingerprint": str(prior_manifest.get("validation_bundle_fingerprint", "") or ""),
    }
    _atomic_json(recovery_index, recovery_payload)

    provenance_path = branch_dir / "provenance" / f"{session_id}.compute.jsonl"
    provenance = ComputeProvenanceRecorder(
        provenance_path,
        session_id=session_id,
        metadata={
            "software_feature": "v6.2 adaptive compute/thermal governor",
            "topology_fingerprint": live_topology.fingerprint,
            "protection_profile": live_profile.to_dict(),
            "resource_plan": resource_plan.to_dict(),
        },
    )
    governor_decision = governor.sample(active_branches=0)
    protection_level.value = int(governor_decision.throttle_level)
    provenance.append("GOVERNOR_SAMPLE", governor_decision.to_dict())
    last_governor_poll = time.monotonic()
    last_governor_state = governor_decision.state
    protection_stop = False
    protection_reason = ""
    admission_wait_started = time.monotonic()
    startup_admission_timeout = max(1.0, float(getattr(config, "governor_startup_admission_timeout_seconds", 30.0) or 30.0))

    # v6.2 RC1 protected queue scheduler with staged admission.
    config_dict = _config_payload(config)
    recent_messages: deque[dict] = deque(maxlen=max(100, int(getattr(config, "coordinator_message_limit", 2000) or 2000)))
    terminal_by_branch: dict[str, dict] = {}
    fatal_messages: list[str] = []
    cancelled = False
    forced_terminated: list[str] = []
    cancel_started: float | None = None
    cancel_grace = max(1.0, float(getattr(config, "safe_stop_grace_seconds", 30.0) or 30.0))
    last_recovery_common = int(global_safe_epoch.value)
    committed_successfully = False
    started_monotonic = time.monotonic()

    pending: deque[int] = deque(range(len(branch_payloads)))
    active: dict[int, mp.Process] = {}
    active_payloads: dict[int, dict] = {}
    active_started_at: dict[int, float] = {}
    slot_active: dict[int, int] = {}
    started_once: set[int] = set()
    finished: set[int] = set()
    lease_counter = [0] * len(branch_payloads)
    slice_failures: list[str] = []

    def emit_session_state() -> None:
        if session_state_callback is None:
            return
        queued = [
            int(index)
            for index in range(len(branch_payloads))
            if index not in active and index not in finished
        ]
        progress_percent, progress_detail, progress_payload = competitive_progress_snapshot(
            branch_payloads,
            current_epochs,
            active_indices=active.keys(),
            finished_indices=finished,
            concurrency=concurrency,
            common_safe_epoch=int(global_safe_epoch.value),
            training_mode=str(getattr(config, "training_mode", "cumulative")),
        )
        session_state_callback({
            "session_id": session_id,
            "total_branches": len(branch_payloads),
            "simultaneous_limit": concurrency,
            "active_branches": len(active),
            "queued_branches": len(queued),
            "completed_branches": len(finished),
            "active_branch_ids": [branch_payloads[index]["branch_id"] for index in sorted(active)],
            "queued_branch_ids": [branch_payloads[index]["branch_id"] for index in queued],
            "epochs": [int(current_epochs[index]) for index in range(len(branch_payloads))],
            "branch_progress": progress_payload["branches"],
            "overall_percent": int(progress_percent),
            "progress_detail": progress_detail,
            "completed_branch_epochs": int(progress_payload["completed_branch_epochs"]),
            "total_branch_epochs": int(progress_payload["total_branch_epochs"]),
            "common_safe_epoch": int(global_safe_epoch.value),
            "resource_plan": resource_plan.to_dict(),
            "cancel_requested": bool(cancel_event.value),
            "governor": governor_decision.to_dict(),
            "protection_provenance_path": str(provenance_path),
            "protection_stop": bool(protection_stop),
        })

    def launch_branch(index: int) -> bool:
        payload = branch_payloads[index]
        slot = int(payload["resource_slot"])
        if slot in slot_active:
            return False
        current = int(current_epochs[index])
        target = int(payload.get("scientific_session_target_epoch", 0) or 0)
        if bool(cancel_event.value):
            # A never-started queued branch is launched only long enough to materialize its exact
            # starting safe state. The worker observes cancel immediately after the initial snapshot.
            lease_target = 0
        elif target > 0:
            # Fixed/cumulative sessions run each admitted branch to its declared session target in
            # one process lease. This avoids repeated CUDA/XPU context creation and startup power
            # spikes. Queued branches start when a protected slot becomes free.
            lease_target = target
        else:
            # Indefinite sessions must rotate so queued scientific branches are never starved.
            lease_target = current + quantum
        lease_counter[index] += 1
        lease_payload = dict(payload)
        lease_payload["lease_target_epoch"] = int(lease_target)
        lease_payload["lease_number"] = int(lease_counter[index])
        lease_payload["lease_id"] = f"{payload['branch_id']}-L{lease_counter[index]:06d}"
        # Base-Guided Fork is used only to initialize the first lease. Once an exact staged resume
        # exists, all later queue leases must continue that exact optimizer/RNG/curriculum state
        # without reapplying the source Base initialization.
        if Path(str(payload.get("resume_path", ""))).is_file():
            lease_payload["initial_policy_checkpoint"] = ""
        process = ctx.Process(
            target=_branch_worker_main,
            args=(config_dict, lease_payload, str(scratch_root), cancel_event, protection_level, current_epochs, last_safe_epochs, global_safe_epoch, last_progress, status_queue),
            name=f"CALO-Policy-{payload['branch_id']}-L{lease_counter[index]:06d}",
        )
        process.start()
        active[index] = process
        active_payloads[index] = lease_payload
        active_started_at[index] = time.monotonic()
        slot_active[slot] = index
        started_once.add(index)
        governor.note_branch_launch()
        provenance.append(
            "BRANCH_LAUNCHED",
            {
                "branch_id": str(payload["branch_id"]),
                "lease_id": str(lease_payload["lease_id"]),
                "resource_slot": slot,
                "assigned_device": str(payload.get("assigned_device", "")),
                "active_after_launch": len(active),
            },
        )
        return True

    def fill_available_slots() -> None:
        if fatal_messages and not cancelled:
            return
        # v6.2 staged startup/admission: under normal operation only GREEN may admit new work and
        # at most one branch is started per staging interval. User-requested Safe Stop may still
        # initialize a never-started branch solely to materialize a coherent exact start state; a
        # RED protection stop never launches additional work.
        if not bool(cancel_event.value):
            if not bool(governor_decision.allow_new_admission):
                return
            if active and not governor.staged_delay_elapsed():
                return
        elif protection_stop:
            return

        attempts = len(pending)
        for _ in range(attempts):
            if len(active) >= concurrency or not pending:
                break
            index = pending.popleft()
            if index in finished or index in active:
                continue
            if bool(cancel_event.value) and index in started_once:
                # On user Safe Stop, already-started queued branches already own an exact safe
                # state and are not relaunched. Never-started branches may be initialized once.
                continue
            slot = int(branch_payloads[index]["resource_slot"])
            if slot in slot_active:
                pending.append(index)
                continue
            if not launch_branch(index):
                pending.append(index)
                continue
            # One launch per staging interval prevents simultaneous CUDA/XPU/CPU context/process
            # startup spikes. Cancellation-time exact-state initialization is also serialized.
            break

    def finalize_exited_process(index: int, process: mp.Process) -> None:
        process.join(timeout=2)
        _drain_queue(status_queue, recent_messages, fatal_messages, terminal_by_branch)
        payload = branch_payloads[index]
        lease_payload = active_payloads[index]
        branch_id = str(payload["branch_id"])
        lease_id = str(lease_payload["lease_id"])
        terminal = dict(terminal_by_branch.get(branch_id, {}) or {})
        if process.exitcode not in (0, None):
            if not (cancelled and branch_id in forced_terminated):
                slice_failures.append(f"{branch_id}/{lease_id}: exitcode {process.exitcode}")
        elif str(terminal.get("lease_id", "")) != lease_id:
            if branch_id not in forced_terminated:
                slice_failures.append(f"{branch_id}/{lease_id}: missing terminal coordinator message")
        process_exitcode = process.exitcode
        try:
            process.close()
        except (ValueError, OSError):
            _LOG.debug("Could not close competitive child process handle", exc_info=True)
        slot = int(payload["resource_slot"])
        active.pop(index, None)
        active_payloads.pop(index, None)
        active_started_at.pop(index, None)
        slot_active.pop(slot, None)
        provenance.append(
            "BRANCH_EXITED",
            {
                "branch_id": branch_id,
                "lease_id": lease_id,
                "exitcode": process_exitcode,
                "epoch": int(current_epochs[index]),
                "cancel_requested": bool(cancel_event.value),
            },
        )

        if bool(cancel_event.value):
            return
        target = int(payload.get("scientific_session_target_epoch", 0) or 0)
        if target > 0 and int(current_epochs[index]) >= target:
            finished.add(index)
        else:
            # Indefinite sessions and unfinished cumulative branches rotate to the back of their
            # resource-slot queue after every exact lease boundary.
            pending.append(index)

    fill_available_slots()
    emit_session_state()

    try:
        while True:
            _drain_queue(status_queue, recent_messages, fatal_messages, terminal_by_branch)
            now = time.monotonic()
            if now - last_governor_poll >= float(governor.config.sample_interval_seconds):
                previous_state = governor_decision.state
                governor_decision = governor.sample(active_branches=len(active))
                last_governor_poll = now
                protection_level.value = int(governor_decision.throttle_level)
                if governor_decision.state != previous_state or governor_decision.reasons:
                    provenance.append("GOVERNOR_SAMPLE", governor_decision.to_dict())
                if governor_decision.state != previous_state:
                    provenance.append(
                        "PROTECTION_STATE_CHANGED",
                        {
                            "from": previous_state.value,
                            "to": governor_decision.state.value,
                            "reasons": list(governor_decision.reasons),
                            "active_branches": len(active),
                        },
                    )
                last_governor_state = governor_decision.state
                if governor_decision.request_safe_stop and not bool(cancel_event.value):
                    protection_stop = True
                    protection_reason = "; ".join(governor_decision.reasons) or "RED compute protection state"
                    cancelled = True
                    cancel_started = now
                    setattr(cancel_event, "value", 1)
                    # Never start new work after a RED protection trigger. Already-started branches
                    # stop at their latest exact safe state. Unstarted branches remain scientifically
                    # unexecuted and the previous authoritative generation remains untouched.
                    pending.clear()
                    provenance.append(
                        "PROTECTION_SAFE_STOP_REQUESTED",
                        {
                            "reason": protection_reason,
                            "active_branch_ids": [branch_payloads[i]["branch_id"] for i in active],
                            "never_started_branch_ids": [
                                branch_payloads[i]["branch_id"]
                                for i in range(len(branch_payloads))
                                if i not in started_once
                            ],
                        },
                    )
                    if progress_callback:
                        progress_percent, progress_detail, _ = competitive_progress_snapshot(
                            branch_payloads, current_epochs, active_indices=active.keys(), finished_indices=finished,
                            concurrency=concurrency, common_safe_epoch=int(global_safe_epoch.value),
                            training_mode=str(getattr(config, "training_mode", "cumulative")),
                        )
                        progress_callback(
                            progress_percent,
                            f"Compute protection RED · exact Safe Stop requested · {protection_reason} · {progress_detail}",
                        )

            if (
                not started_once
                and not active
                and pending
                and now - admission_wait_started >= startup_admission_timeout
            ):
                fatal_messages.append(
                    "Adaptive compute protection could not obtain a GREEN startup-admission window "
                    f"within {startup_admission_timeout:.1f} s. No policy branch was started."
                )
                provenance.append(
                    "STARTUP_ADMISSION_TIMEOUT",
                    {
                        "timeout_seconds": startup_admission_timeout,
                        "governor": governor_decision.to_dict(),
                    },
                )

            safe_values = [int(last_safe_epochs[i]) for i in range(len(branch_payloads))]
            if safe_values and all(value >= 0 for value in safe_values):
                common = min(safe_values)
                if common > int(global_safe_epoch.value):
                    global_safe_epoch.value = common
                if common != last_recovery_common:
                    last_recovery_common = common
                    recovery_payload["latest_common_safe_epoch"] = common
                    recovery_payload["updated_unix"] = time.time()
                    _atomic_json(recovery_index, recovery_payload)

            if cancel_callback and cancel_callback() and not cancelled:
                cancelled = True
                cancel_started = time.monotonic()
                setattr(cancel_event, "value", 1)
                # Drop already-started queued work. Never-started branches stay queued only for
                # cancellation-time initialization of their exact starting safe state.
                pending = deque(index for index in pending if index not in started_once)
                if progress_callback:
                    progress_percent, progress_detail, _ = competitive_progress_snapshot(
                        branch_payloads, current_epochs, active_indices=active.keys(), finished_indices=finished,
                        concurrency=concurrency, common_safe_epoch=int(global_safe_epoch.value),
                        training_mode=str(getattr(config, "training_mode", "cumulative")),
                    )
                    progress_callback(
                        progress_percent,
                        "Safe Stop requested · preserving one coherent exact checkpoint · " + progress_detail,
                    )

            if slice_failures:
                fatal_messages.extend(slice_failures)
                slice_failures.clear()
            if fatal_messages and not bool(cancel_event.value):
                setattr(cancel_event, "value", 1)
                cancel_started = time.monotonic()

            # Every active process gets its own full grace interval, including a never-started branch
            # launched after Safe Stop solely to materialize epoch-0/start-epoch exact state.
            if bool(cancel_event.value) and cancel_started is not None:
                now = time.monotonic()
                for index, process in list(active.items()):
                    grace_origin = max(float(cancel_started), float(active_started_at.get(index, cancel_started)))
                    if process.is_alive() and now - grace_origin >= cancel_grace:
                        branch_id = str(branch_payloads[index]["branch_id"])
                        process.terminate()
                        if branch_id not in forced_terminated:
                            forced_terminated.append(branch_id)

            for index, process in list(active.items()):
                if not process.is_alive():
                    finalize_exited_process(index, process)

            if fatal_messages and not cancelled:
                if not active:
                    break
            elif cancelled:
                fill_available_slots()
                # Safe Stop completes only when every scientific branch has an exact safe point.
                if not active and all(int(last_safe_epochs[i]) >= 0 for i in range(len(branch_payloads))):
                    break
                if not active and not pending and not all(int(last_safe_epochs[i]) >= 0 for i in range(len(branch_payloads))):
                    break
            else:
                fill_available_slots()
                if str(getattr(config, "training_mode", "cumulative")) != "indefinite" and len(finished) == len(branch_payloads) and not active:
                    break

            if progress_callback:
                progress_percent, progress_detail, _ = competitive_progress_snapshot(
                    branch_payloads,
                    current_epochs,
                    active_indices=active.keys(),
                    finished_indices=finished,
                    concurrency=concurrency,
                    common_safe_epoch=int(global_safe_epoch.value),
                    training_mode=str(getattr(config, "training_mode", "cumulative")),
                )
                progress_callback(progress_percent, progress_detail)
            emit_session_state()
            time.sleep(0.20)

        _drain_queue(status_queue, recent_messages, fatal_messages, terminal_by_branch)
        if fatal_messages and not cancelled:
            recovery_payload["status"] = "FAILED"
            recovery_payload["fatal_messages"] = list(dict.fromkeys(fatal_messages))
            recovery_payload["latest_common_safe_epoch"] = int(global_safe_epoch.value)
            _atomic_json(recovery_index, recovery_payload)
            raise RuntimeError("Competitive policy branch failure: " + "; ".join(dict.fromkeys(fatal_messages)))

        if cancelled:
            safe_epochs = [int(last_safe_epochs[i]) for i in range(len(branch_payloads))]
            if not safe_epochs or any(epoch < 0 for epoch in safe_epochs):
                if protection_stop:
                    # A RED protection event must never launch additional work merely to manufacture
                    # epoch-0 checkpoints for never-started queued branches. Keep the previous
                    # authoritative generation untouched and retain started-branch recovery evidence.
                    prior_epoch = int(prior_manifest.get("common_resume_epoch", 0) or 0)
                    nonnegative = [epoch for epoch in safe_epochs if epoch >= 0]
                    common_safe = min(nonnegative) if nonnegative else prior_epoch
                    never_started = [
                        branch_payloads[i]["branch_id"]
                        for i, epoch in enumerate(safe_epochs)
                        if epoch < 0
                    ]
                    recovery_payload.update({
                        "status": "PROTECTION_STOP_ROLLBACK_SAFE",
                        "protection_reason": protection_reason,
                        "started_branch_safe_epochs": safe_epochs,
                        "never_started_branch_ids": never_started,
                        "authoritative_generation_unchanged": True,
                    })
                    _atomic_json(recovery_index, recovery_payload)
                    provenance.append(
                        "PROTECTION_STOP_ROLLBACK_SAFE",
                        {
                            "reason": protection_reason,
                            "common_started_safe_epoch": common_safe,
                            "never_started_branch_ids": never_started,
                            "authoritative_generation_unchanged": True,
                        },
                    )
                    return CompetitiveTrainingResult(
                        output_path=str(output_path),
                        history=list(recent_messages) + [{
                            "type": "session_status",
                            "status": TrainingSessionStatus.SAFE_STOPPED_PROTECTION.value,
                            "reason": protection_reason,
                            "authoritative_generation_unchanged": True,
                        }],
                        status=TrainingSessionStatus.SAFE_STOPPED_PROTECTION,
                        common_resume_epoch=int(common_safe),
                        manifest_path=(str(_manifest_path(output_path)) if _manifest_path(output_path).is_file() else ""),
                        selected_artifact_path=(str(base_checkpoint) if base_checkpoint and Path(base_checkpoint).is_file() else ""),
                        degraded_branches=tuple(never_started),
                    )
                recovery_payload["status"] = "FAILED_SAFE_STOP_NO_COMMON_CHECKPOINT"
                _atomic_json(recovery_index, recovery_payload)
                raise RuntimeError(
                    "Safe Stop could not establish an exact checkpoint for every active/queued scientific branch; "
                    "interrupted scratch/recovery evidence was retained."
                )
            common_epoch = min(safe_epochs)
            sources: list[tuple[dict, Path]] = []
            for payload in branch_payloads:
                source = RollingSafeStore(scratch_root, payload["branch_id"]).path(common_epoch)
                if not source.is_file():
                    recovery_payload["status"] = "FAILED_SAFE_STOP"
                    _atomic_json(recovery_index, recovery_payload)
                    raise RuntimeError(f"Branch {payload['branch_id']} has no common safe snapshot at epoch {common_epoch}")
                payload["session_cancelled"] = True
                sources.append((payload, source))
        else:
            epochs = [int(current_epochs[i]) for i in range(len(branch_payloads))]
            if len(set(epochs)) != 1:
                recovery_payload["status"] = "FAILED_MIXED_TERMINAL_EPOCHS"
                recovery_payload["terminal_epochs"] = epochs
                _atomic_json(recovery_index, recovery_payload)
                raise RuntimeError(f"Competitive session ended at mixed branch epochs; transaction refused: {epochs}")
            common_epoch = epochs[0]
            sources = []
            for payload in branch_payloads:
                source = Path(payload["resume_path"])
                if not source.is_file():
                    raise RuntimeError(f"Branch {payload['branch_id']} has no staged terminal exact state: {source}")
                sources.append((payload, source))

        generation_id = session_id
        finalized = _commit_branch_generation(branch_dir, generation_id, sources, common_epoch)

        session_meta = {
            "session_id": session_id,
            "status": (
                TrainingSessionStatus.SAFE_STOPPED_PROTECTION.value
                if protection_stop
                else TrainingSessionStatus.SAFE_STOPPED_DEGRADED.value
                if cancelled and (forced_terminated or fatal_messages)
                else TrainingSessionStatus.SAFE_STOPPED.value
                if cancelled
                else TrainingSessionStatus.COMPLETED.value
            ),
            "started_monotonic": started_monotonic,
            "requested_branches": len(branch_payloads),
            "started_branches": len(started_once),
            "successful_branches": len(finalized),
            "failed_branches": (len(set(forced_terminated)) + len(set(fatal_messages))) if cancelled else 0,
            "branch_count": len(finalized),
            "seed_plan": [asdict(item) for item in seed_plan],
            "resource_assignments": {
                payload["branch_id"]: {
                    "primary_device": payload.get("assigned_device", ""),
                    "resource_slot": int(payload.get("resource_slot", 0) or 0),
                    "cpu_worker_budget": int(payload.get("cpu_worker_budget", 1) or 1),
                    "auxiliary_xpu_runtime": payload.get("auxiliary_xpu_runtime", ""),
                    "auxiliary_xpu_name": payload.get("auxiliary_xpu_name", ""),
                }
                for payload in branch_payloads
            },
            "safe_parallel_branches": int(getattr(config, "safe_parallel_branches", 0) or 0),
            "parallel_concurrency": int(concurrency),
            "queued_branch_scheduler": True,
            "branch_queue_quantum_epochs": int(quantum),
            "safe_global_cpu_workers": int(global_cpu_budget),
            "cpu_worker_budgets_by_slot": {str(slot.slot_index): int(slot.cpu_worker_budget) for slot in resource_plan.slots},
            "compute_resource_plan": resource_plan.to_dict(),
            "compute_profile_fingerprint": str(getattr(config, "compute_profile_fingerprint", "") or ""),
            "compute_protection": {
                "final_state": governor_decision.state.value,
                "protection_stop": bool(protection_stop),
                "protection_reason": protection_reason,
                "provenance_path": str(provenance_path),
                "staged_startup_delay_seconds": float(governor.config.staged_startup_delay_seconds),
            },
            "common_resume_epoch": int(common_epoch),
            "training_mode": str(getattr(config, "training_mode", "cumulative")),
            "start_mode": start_mode,
            "cancelled_safe_stop": bool(cancelled),
            "degraded_branches": list(forced_terminated),
            "fatal_branch_diagnostics": list(dict.fromkeys(fatal_messages)),
            "method": "competitive independent PPO branches with protected queued scheduling and exact-resume indefinite rotation; no parameter averaging",
            "persistence": "two-phase immutable branch generation + atomic root manifest commit",
            "safe_stop_semantics": "latest validated common exact checkpoint; rolling snapshots every 10 epochs plus the exact session start state",
        }

        candidates, common_evidence = _common_evaluate_candidates(previous_payload, finalized, config)
        eligible_candidates = [item for item in candidates if _deployable_eligible(item["metrics"])]
        winner = (
            min(eligible_candidates, key=lambda item: _rank_key(item["metrics"], source_priority=item["source_priority"], stable_id=item["candidate_id"]))
            if eligible_candidates
            else None
        )
        # A session without exact real-ORPD development evidence may evolve Training Champions and
        # exact branch state, but it cannot replace a previously committed deployable Base.
        if winner is None and previous_payload is not None:
            winner = next((item for item in candidates if item["candidate_id"] == "previous_base"), None)

        ranking = sorted(
            ({"candidate_id": item["candidate_id"], "metrics": item["metrics"], "rank_key": list(_rank_key(item["metrics"], source_priority=item["source_priority"], stable_id=item["candidate_id"]))} for item in candidates),
            key=lambda row: tuple(row["rank_key"]),
        )

        provisional_artifact = ""
        if winner is None:
            # Transactional exact state is still committed. The best available terminal candidate is
            # saved only as an explicitly provisional artifact and is never labeled/promoted as Base.
            provisional = min(candidates, key=lambda item: _rank_key(item["metrics"], source_priority=item["source_priority"], stable_id=item["candidate_id"])) if candidates else None
            if provisional is not None and provisional.get("champion") is not None:
                champion = provisional["champion"]
                champion["metrics"] = provisional["metrics"]
                provisional_payload = _policy_payload_from_champion(champion, config, branch_id=provisional["branch_id"], seed=int(provisional["seed"]), session=session_meta)
                provisional_payload["metadata"]["checkpoint_role"] = "competitive_provisional_candidate"
                provisional_payload["metadata"]["base_eligible"] = False
                provisional_payload["metadata"]["base_selection_protocol"] = "v5.9 training-champion only; no exact real-ORPD deployable Base evidence"
                provisional_payload["metadata"]["base_candidate_ranking"] = ranking
                artifact, sha = _save_immutable_base(output_path, provisional_payload)
                provisional_artifact = str(artifact)
            else:
                artifact = Path("")
                sha = ""
            best_source = "none_no_eligible_base"
            best_branch = ""
            best_metrics = {}
        elif winner["candidate_id"] == "previous_base":
            artifact = Path(base_checkpoint)
            if not artifact.is_file():
                artifact = output_path
            sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            best_source = "previous_base"
            best_branch = ""
            best_metrics = winner["metrics"]
        else:
            champion = winner["champion"]
            champion["metrics"] = winner["metrics"]
            best_payload = _policy_payload_from_champion(champion, config, branch_id=winner["branch_id"], seed=int(winner["seed"]), session=session_meta)
            best_payload["metadata"]["base_eligible"] = True
            best_payload["metadata"]["base_selection_protocol"] = "v5.9 exact-real-ORPD common-bundle order-independent normalized scientific ranking"
            best_payload["metadata"]["base_candidate_ranking"] = ranking
            best_payload["metadata"]["branch_manifest"] = str(_manifest_path(output_path))
            artifact, sha = _save_immutable_base(output_path, best_payload)
            best_source = "branch_champion"
            best_branch = winner["branch_id"]
            best_metrics = winner["metrics"]

        manifest = {
            "schema_version": _MANIFEST_SCHEMA,
            "generation_id": generation_id,
            "policy_lineage_id": str(getattr(config, "policy_lineage_id", "")),
            "policy_lineage_name": str(getattr(config, "policy_lineage_name", "")),
            "logical_base_alias": str(output_path),
            "base_artifact_path": (str(artifact) if winner is not None else ""),
            "provisional_artifact_path": provisional_artifact,
            "base_sha256": (sha if winner is not None else ""),
            "base_source": best_source,
            "base_source_branch": best_branch,
            "base_metrics": dict(best_metrics or {}),
            "validation_bundle_fingerprint": validation_bundle_fingerprint(config),
            "champion_comparator_schema": _COMPARATOR_SCHEMA,
            "base_candidate_ranking": ranking,
            "common_candidate_evidence": common_evidence,
            "common_resume_epoch": int(common_epoch),
            "previous_training_mode": str(getattr(config, "training_mode", "cumulative")),
            "previous_session_epochs": int(getattr(config, "epochs", 0) or 0),
            "branches": finalized,
            "seed_plan": [asdict(item) for item in seed_plan],
            "session": session_meta,
        }
        # This is the authoritative commit point. Prior manifest/generation and logical Base alias
        # remain untouched until now.
        _atomic_json(_manifest_path(output_path), manifest)
        committed_successfully = True

        # The logical Base alias is non-authoritative. Refresh it only after the authoritative
        # manifest commits, and never from a provisional/non-eligible candidate. A failure here
        # leaves the manifest's immutable Base artifact as the source of truth.
        if winner is not None and Path(artifact).is_file():
            try:
                if Path(artifact).resolve() != output_path.resolve():
                    durable_write_bytes(output_path, Path(artifact).read_bytes())
                base_payload_for_alias = load_checkpoint(Path(artifact), map_location="cpu")
                _atomic_json(
                    output_path.with_suffix(".json"),
                    {**dict(base_payload_for_alias.get("metadata", {}) or {}), "sha256": sha},
                )
            except (OSError, ValueError, RuntimeError):
                _LOG.error(
                    "Authoritative competitive manifest committed but logical Base alias refresh failed; "
                    "the immutable manifest artifact remains authoritative",
                    exc_info=True,
                )

        # Compatibility branch aliases are non-authoritative and updated only after authoritative manifest commit.
        for row in finalized:
            try:
                alias = branch_dir / f"{row['branch_id']}.resume.pt"
                durable_write_bytes(alias, Path(row["resume_path"]).read_bytes())
                sidecar = Path(row["resume_path"] + ".sha256")
                if sidecar.is_file():
                    durable_write_bytes(alias.with_suffix(alias.suffix + ".sha256"), sidecar.read_bytes())
            except OSError:
                _LOG.warning("Could not refresh non-authoritative branch convenience alias for %s", row["branch_id"], exc_info=True)

        recovery_payload["status"] = "COMMITTED"
        recovery_payload["generation_id"] = generation_id
        recovery_payload["latest_common_safe_epoch"] = int(common_epoch)
        _atomic_json(recovery_index, recovery_payload)
        recovery_index.unlink(missing_ok=True)
        shutil.rmtree(scratch_root, ignore_errors=True)

        status = (
            TrainingSessionStatus.SAFE_STOPPED_PROTECTION
            if protection_stop
            else TrainingSessionStatus.SAFE_STOPPED_DEGRADED
            if cancelled and (forced_terminated or fatal_messages)
            else TrainingSessionStatus.SAFE_STOPPED
            if cancelled
            else TrainingSessionStatus.COMPLETED
        )
        provenance.append(
            "SESSION_TERMINAL",
            {
                "status": status.value,
                "common_resume_epoch": int(common_epoch),
                "generation_id": generation_id,
                "protection_reason": protection_reason,
            },
        )
        if progress_callback:
            label = "safe-stopped" if cancelled else "complete"
            artifact_label = Path(artifact).name if winner is not None else (Path(provisional_artifact).name if provisional_artifact else "no eligible Base")
            progress_callback(100 if not cancelled else 0, f"Competitive training {label} · {artifact_label} · {len(finalized)} branches · exact resume epoch {common_epoch}")
        history = list(recent_messages) + [{"type": "base_selection", **manifest}, {"type": "session_status", "status": status.value, "common_resume_epoch": int(common_epoch)}]
        selected_artifact = str(artifact) if winner is not None and Path(artifact).is_file() else str(provisional_artifact or "")
        return CompetitiveTrainingResult(
            output_path=str(output_path),
            history=history,
            status=status,
            common_resume_epoch=int(common_epoch),
            manifest_path=str(_manifest_path(output_path)),
            selected_artifact_path=selected_artifact,
            degraded_branches=tuple(forced_terminated),
        )
    finally:
        setattr(cancel_event, "value", 1)
        for index, process in list(active.items()):
            if process.is_alive():
                process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            try:
                process.close()
            except (ValueError, OSError):
                _LOG.debug("Could not close competitive child process handle", exc_info=True)
        active.clear()
        active_payloads.clear()
        active_started_at.clear()
        slot_active.clear()
        _drain_queue(status_queue, recent_messages, fatal_messages, terminal_by_branch)
        try:
            status_queue.close()
            status_queue.join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        if committed_successfully:
            shutil.rmtree(scratch_root, ignore_errors=True)
        else:
            # Preserve interrupted-session scratch and durable recovery index for explicit Recover/Discard.
            if recovery_index.exists():
                try:
                    recovery_payload["latest_common_safe_epoch"] = int(global_safe_epoch.value)
                    recovery_payload.setdefault("status", "INTERRUPTED")
                    if recovery_payload["status"] == "RUNNING":
                        recovery_payload["status"] = "INTERRUPTED"
                    _atomic_json(recovery_index, recovery_payload)
                except OSError:
                    _LOG.error("Failed to update competitive-session recovery index", exc_info=True)
