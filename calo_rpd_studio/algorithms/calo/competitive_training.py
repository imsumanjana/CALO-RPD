"""CALO v5.6 competitive multi-branch policy evolution.

Parallel branches are independent PPO trajectories.  They are never weight-averaged.
Each branch preserves an exact resumable working state and a separate best-so-far champion.
The logical base policy is promoted only when a branch champion is scientifically superior under
one fixed deterministic multi-metric validation bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import tempfile
import time
import uuid
from typing import Any

import numpy as np
import torch

from calo_rpd_studio.ai.model_io import (
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

@dataclass(frozen=True, slots=True)
class BranchSeed:
    branch_id: str
    seed: int
    strategy: str


@dataclass(frozen=True, slots=True)
class ChampionDecision:
    superior: bool
    wins: int
    losses: int
    ties: int
    critical_wins: int
    critical_losses: int
    reason: str


_METRIC_DIRECTIONS: dict[str, str] = {
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
    "policy_inference_ms": "min",
}
_CRITICAL_METRICS = (
    "feasible_episode_rate",
    "median_final_feasible_objective",
    "convergence_auc",
    "median_constraint_violation",
)


def build_branch_seed_plan(config, parallel_runs: int | None = None) -> list[BranchSeed]:
    """Build the explicit user-controlled same/increment/decrement/custom seed plan."""

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
        # Backward-compatible default: first branch uses the base seed, remaining branches use +1,+2...
        same = 1
        inc = max(0, requested - 1)
    total = same + inc + dec + len(custom)
    if total <= 0:
        raise ValueError("At least one policy-training branch is required")
    if requested > 0 and total != requested:
        # Explicit seed counts are authoritative; keep config/metadata honest rather than silently dropping branches.
        requested = total

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
    value = metrics.get(key)
    if value is None:
        return math.inf if _METRIC_DIRECTIONS.get(key) == "min" else -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf if _METRIC_DIRECTIONS.get(key) == "min" else -math.inf
    if math.isnan(number):
        return math.inf if _METRIC_DIRECTIONS.get(key) == "min" else -math.inf
    return number


def _compare_one(candidate: float, incumbent: float, direction: str) -> int:
    if not math.isfinite(candidate) and not math.isfinite(incumbent):
        return 0
    if direction == "min":
        if math.isfinite(candidate) and not math.isfinite(incumbent):
            return 1
        if not math.isfinite(candidate) and math.isfinite(incumbent):
            return -1
    else:
        if math.isfinite(candidate) and not math.isfinite(incumbent):
            return 1
        if not math.isfinite(candidate) and math.isfinite(incumbent):
            return -1
    scale = max(abs(candidate), abs(incumbent), 1.0)
    tol = 1e-7 * scale
    if abs(candidate - incumbent) <= tol:
        return 0
    if direction == "min":
        return 1 if candidate < incumbent else -1
    return 1 if candidate > incumbent else -1


def compare_champion_metrics(candidate: dict, incumbent: dict | None) -> ChampionDecision:
    """Hierarchical multi-metric comparator used for branch/base promotion.

    Mandatory validity and feasibility safeguards are applied before a majority-of-evidence decision.
    Critical scientific metrics are Pareto-checked first.  Runtime can break close ties but cannot
    compensate for a material loss in feasibility or final feasible objective.
    """

    if not bool(candidate.get("valid", False)):
        return ChampionDecision(False, 0, 1, 0, 0, 1, "candidate failed mandatory validity gates")
    if incumbent is None or not bool(incumbent.get("valid", False)):
        return ChampionDecision(True, 1, 0, 0, 1, 0, "first valid champion")

    cand_feas = _metric_value(candidate, "feasible_episode_rate")
    base_feas = _metric_value(incumbent, "feasible_episode_rate")
    if cand_feas + 0.05 < base_feas:
        return ChampionDecision(
            False, 0, 1, 0, 0, 1, "candidate materially reduces feasible-episode probability"
        )

    wins = losses = ties = critical_wins = critical_losses = 0
    for key, direction in _METRIC_DIRECTIONS.items():
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

    # Critical Pareto dominance is an immediate scientifically strong promotion signal.
    if critical_wins > 0 and critical_losses == 0:
        return ChampionDecision(
            True, wins, losses, ties, critical_wins, critical_losses,
            "candidate Pareto-improves the critical scientific metric set",
        )
    if critical_losses > 0 and critical_wins == 0:
        return ChampionDecision(
            False, wins, losses, ties, critical_wins, critical_losses,
            "candidate is dominated on the critical scientific metric set",
        )

    # Guard against trading a materially worse final objective for several minor efficiency wins.
    cand_obj = _metric_value(candidate, "median_final_feasible_objective")
    base_obj = _metric_value(incumbent, "median_final_feasible_objective")
    if math.isfinite(cand_obj) and math.isfinite(base_obj):
        if cand_obj > base_obj + max(abs(base_obj), 1.0) * 0.01:
            return ChampionDecision(
                False, wins, losses, ties, critical_wins, critical_losses,
                "candidate worsens median final feasible objective by more than 1%",
            )

    superior = wins > losses and critical_wins >= critical_losses
    reason = (
        f"multi-metric majority supports candidate ({wins} superior, {losses} inferior, {ties} tied; "
        f"critical {critical_wins}-{critical_losses})"
        if superior
        else f"multi-metric evidence does not support promotion ({wins} superior, {losses} inferior, {ties} tied; "
        f"critical {critical_wins}-{critical_losses})"
    )
    return ChampionDecision(superior, wins, losses, ties, critical_wins, critical_losses, reason)


def _deterministic_action(network, state: np.ndarray, device: torch.device):
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    with torch.inference_mode():
        regime_logits, operator_logits, alpha, beta, _value = network(tensor)
        regime = int(torch.argmax(regime_logits).item())
        operator = int(torch.argmax(operator_logits).item())
        parameter = (alpha / torch.clamp(alpha + beta, min=1e-8)).detach().cpu().numpy()
    return regime, operator, parameter


def evaluate_policy_multimetric(network, config, *, validation_seed: int | None = None) -> dict:
    """Evaluate one policy on a fixed held-out lightweight bundle using many outcome metrics.

    This is a branch-champion comparator, not formal Policy Qualification.  It deliberately uses a
    fixed deterministic validation bundle so every epoch/branch is compared on identical evidence.
    """

    from .training import SyntheticCALOEnvironment

    base_seed = int(
        validation_seed
        if validation_seed is not None
        else getattr(config, "champion_validation_seed", 918_273)
    )
    episodes_per_stage = max(1, int(getattr(config, "champion_validation_episodes", 1) or 1))
    horizon = max(2, min(int(getattr(config, "horizon", 28)), int(getattr(config, "champion_validation_horizon", 12) or 12)))
    stages = [0, 1, 2, 3]
    if getattr(config, "development_cases", ()):  # development systems remain development-only
        stages.append(4)

    device = next(network.parameters()).device
    was_training = bool(network.training)
    network.eval()
    returns: list[float] = []
    final_ratios: list[float] = []
    final_objs: list[float] = []
    final_violations: list[float] = []
    first_feasible_steps: list[float] = []
    aucs: list[float] = []
    inference_seconds = 0.0
    decisions = 0
    try:
        for stage in stages:
            for rep in range(episodes_per_stage):
                seed = base_seed + stage * 100_003 + rep * 10_007
                rng = np.random.default_rng(seed)
                if stage == 4:
                    from calo_rpd_studio.orpd.problem import ORPDProblem
                    from calo_rpd_studio.power_system.case_loader import CaseLoader

                    source = config.development_cases[rep % len(config.development_cases)]
                    env = SyntheticCALOEnvironment(
                        rng,
                        stage,
                        int(config.population_size),
                        problem=ORPDProblem(CaseLoader.load(source)),
                    )
                else:
                    env = SyntheticCALOEnvironment(rng, stage, int(config.population_size))
                episode_return = 0.0
                first_feasible = horizon + 1
                quality_curve: list[float] = []
                for step in range(horizon):
                    state = env.policy_state(horizon)
                    started = time.perf_counter()
                    regime, operator, parameter = _deterministic_action(network, state, device)
                    inference_seconds += time.perf_counter() - started
                    decisions += 1
                    episode_return += float(env.step(regime, operator, parameter, horizon))
                    violation, objective, feasible_ratio = env._diagnostics(env.evaluations)
                    if feasible_ratio > 0.0 and first_feasible == horizon + 1:
                        first_feasible = step + 1
                    # One finite quality trajectory gives infeasibility a large explicit penalty.
                    quality = (
                        float(objective)
                        if math.isfinite(float(objective))
                        else 1.0e9 + 1.0e6 * max(float(violation), 0.0)
                    )
                    quality_curve.append(quality)
                violation, objective, feasible_ratio = env._diagnostics(env.evaluations)
                returns.append(float(episode_return))
                final_ratios.append(float(feasible_ratio))
                final_violations.append(float(violation) if math.isfinite(float(violation)) else 1.0e12)
                if math.isfinite(float(objective)):
                    final_objs.append(float(objective))
                first_feasible_steps.append(float(first_feasible))
                aucs.append(float(np.mean(quality_curve)) if quality_curve else 1.0e12)
    finally:
        network.train(was_training)

    feasible_episode_rate = float(np.mean([ratio > 0.0 for ratio in final_ratios])) if final_ratios else 0.0
    objective_values = np.asarray(final_objs, dtype=float)
    objective_fallback = 1.0e12
    metrics = {
        "valid": bool(returns) and all(math.isfinite(v) for v in returns),
        "validation_seed": base_seed,
        "validation_episodes": len(returns),
        "feasible_episode_rate": feasible_episode_rate,
        "mean_final_feasible_ratio": float(np.mean(final_ratios)) if final_ratios else 0.0,
        "median_final_feasible_objective": float(np.median(objective_values)) if len(objective_values) else objective_fallback,
        "mean_final_feasible_objective": float(np.mean(objective_values)) if len(objective_values) else objective_fallback,
        "best_final_feasible_objective": float(np.min(objective_values)) if len(objective_values) else objective_fallback,
        "worst_final_feasible_objective": float(np.max(objective_values)) if len(objective_values) else objective_fallback,
        "convergence_auc": float(np.mean(aucs)) if aucs else objective_fallback,
        "median_constraint_violation": float(np.median(final_violations)) if final_violations else objective_fallback,
        "median_steps_to_first_feasibility": float(np.median(first_feasible_steps)) if first_feasible_steps else float(horizon + 1),
        "mean_validation_return": float(np.mean(returns)) if returns else -objective_fallback,
        "median_validation_return": float(np.median(returns)) if returns else -objective_fallback,
        "worst_validation_return": float(np.min(returns)) if returns else -objective_fallback,
        "objective_iqr": float(np.percentile(objective_values, 75) - np.percentile(objective_values, 25)) if len(objective_values) >= 2 else 0.0,
        "policy_inference_ms": float(1000.0 * inference_seconds / max(decisions, 1)),
    }
    return metrics


class BranchChampionTracker:
    def __init__(self, *, base_payload: dict | None = None, base_metrics: dict | None = None):
        self.state_dict: dict[str, torch.Tensor] | None = None
        self.metrics: dict | None = None
        self.epoch = 0
        self.source = "none"
        self.decisions: list[dict] = []
        if base_payload is not None and base_metrics is not None:
            self.state_dict = {
                k: v.detach().cpu().clone()
                for k, v in dict(base_payload.get("model_state_dict", base_payload)).items()
            }
            self.metrics = dict(base_metrics)
            self.epoch = int(dict(base_payload.get("metadata", {}) or {}).get("cumulative_epoch", 0) or 0)
            self.source = "base_threshold"

    def restore_from_extra(self, extra: dict) -> None:
        champion = dict(extra.get("branch_champion", {}) or {})
        state = champion.get("model_state_dict")
        metrics = champion.get("metrics")
        if isinstance(state, dict) and isinstance(metrics, dict):
            candidate = {k: v.detach().cpu().clone() for k, v in state.items() if torch.is_tensor(v)}
            decision = compare_champion_metrics(metrics, self.metrics)
            if decision.superior or self.metrics is None:
                self.state_dict = candidate
                self.metrics = dict(metrics)
                self.epoch = int(champion.get("epoch", 0) or 0)
                self.source = str(champion.get("source", "restored_branch"))

    def consider(self, network, metrics: dict, epoch: int, *, source: str) -> ChampionDecision:
        decision = compare_champion_metrics(metrics, self.metrics)
        self.decisions.append({
            "epoch": int(epoch),
            "source": str(source),
            "decision": asdict(decision),
            "metrics": dict(metrics),
        })
        if decision.superior:
            self.state_dict = {
                k: v.detach().cpu().clone() for k, v in network.state_dict().items()
            }
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
                "decision_history_tail": self.decisions[-50:],
            }
        }


class RollingSafeStore:
    """Disk-backed exact-state snapshots used only during an active training session."""

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
            except Exception:
                continue
        return sorted(output)

    def cleanup_before(self, epoch: int) -> None:
        for old_epoch in self.epochs():
            if old_epoch < int(epoch):
                path = self.path(old_epoch)
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)


def _copy_trusted_resume(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
        write_trusted_resume_hash(target)
    finally:
        temporary.unlink(missing_ok=True)


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
    if "development_cases" in payload:
        payload["development_cases"] = tuple(payload["development_cases"])
    if "parallel_custom_seeds" in payload and not isinstance(payload["parallel_custom_seeds"], tuple):
        payload["parallel_custom_seeds"] = tuple(payload["parallel_custom_seeds"] or ())
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


def _branch_worker_main(
    config_dict: dict,
    branch_payload: dict,
    scratch_root: str,
    cancel_event,
    current_epochs,
    last_safe_epochs,
    global_safe_epoch,
    status_queue,
) -> None:
    """One independent branch process.  No network parameters are merged with another branch."""

    from .training import TrainingCancelled, save_training_resume, train_policy
    from .heterogeneous_training import train_policy_heterogeneous

    index = int(branch_payload["index"])
    branch_id = str(branch_payload["branch_id"])
    config = _rebuild_config(config_dict)
    config.seed = int(branch_payload["seed"])
    config.parallel_runs = 1
    config.checkpoint_each_epoch = False
    config.resume_checkpoint = str(branch_payload.get("resume_path", "") or "")
    config.initial_policy_checkpoint = str(branch_payload.get("initial_policy_checkpoint", "") or "")
    output_path = Path(branch_payload["working_output"])
    scratch = RollingSafeStore(Path(scratch_root), branch_id)
    base_payload, base_metrics = _load_base_payload(branch_payload.get("base_model_checkpoint"))
    tracker = BranchChampionTracker(base_payload=base_payload, base_metrics=base_metrics)

    resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    branch_initial_epoch = 0
    if resume_path is not None and resume_path.is_file():
        try:
            resume_payload = load_trusted_resume(resume_path, map_location="cpu")
            branch_initial_epoch = int(resume_payload.get("next_epoch", 0) or 0)
            tracker.restore_from_extra(dict(resume_payload.get("extra", {}) or {}))
        except Exception as exc:
            status_queue.put({"type": "fatal", "branch_id": branch_id, "error": f"resume inspect failed: {exc}"})
            raise

    validation_interval = max(1, int(getattr(config, "champion_validation_interval_epochs", 10) or 10))
    safe_interval = max(1, int(getattr(config, "safe_snapshot_interval_epochs", 10) or 10))
    max_lead = max(safe_interval, int(getattr(config, "max_branch_lead_epochs", 30) or 30))
    screening_best_by_stage: dict[int, float] = {}
    session_target = (
        branch_initial_epoch + max(1, int(getattr(config, "epochs", 1) or 1))
        if str(getattr(config, "training_mode", "cumulative")) != "indefinite"
        else None
    )

    def extra_provider() -> dict:
        return {
            **tracker.extra_payload(),
            "branch_id": branch_id,
            "branch_seed": int(config.seed),
            "branch_seed_strategy": str(branch_payload.get("strategy", "")),
            "branch_start_mode": str(branch_payload.get("start_mode", "new")),
        }

    def observer(state: dict) -> None:
        completed_epoch = int(state["epoch"])
        current_epochs[index] = completed_epoch
        # Tier 1 runs every epoch using the already-produced training return. Tier 2 performs the
        # complete fixed multi-metric champion validation only when that cheap screen improves, at
        # the periodic deep-validation boundary, at the initial state, or at cumulative completion.
        # This preserves "always compare" semantics without multiplying a 10M-epoch campaign by a
        # full validation suite on every non-promising epoch.
        stage = int(state.get("stage", 0))
        returns = [float(value) for value in state.get("episode_returns", []) if math.isfinite(float(value))]
        screen_value = float(np.mean(returns)) if returns else -1.0e12
        previous_screen = screening_best_by_stage.get(stage, -math.inf)
        screen_promising = screen_value > previous_screen + 1e-12
        if screen_promising:
            screening_best_by_stage[stage] = screen_value
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
            decision = tracker.consider(
                state["network"], metrics, completed_epoch, source=f"{branch_id}@{completed_epoch}"
            )
            status_queue.put({
                "type": "champion",
                "branch_id": branch_id,
                "epoch": completed_epoch,
                "promoted": bool(decision.superior),
                "reason": decision.reason,
                "metrics": metrics,
            })
        else:
            status_queue.put({
                "type": "screen",
                "branch_id": branch_id,
                "epoch": completed_epoch,
                "screening_mean_episode_return": screen_value,
                "deep_validation": False,
            })
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
                extra={
                    **extra_provider(),
                    "temporary_safe_snapshot": True,
                    "curriculum_encoding": "zero_based_0_4",
                },
            )
            last_safe_epochs[index] = completed_epoch
            committed = int(global_safe_epoch.value)
            if committed >= 0:
                scratch.cleanup_before(committed)
            status_queue.put({"type": "safe", "branch_id": branch_id, "epoch": completed_epoch})

            # Bound asynchronous lead so the rolling disk window stays small and comparable.
            while (
                completed_epoch - int(global_safe_epoch.value) > max_lead
                and not cancel_event.is_set()
            ):
                time.sleep(0.05)

    def cancelled() -> bool:
        return bool(cancel_event.is_set())

    try:
        status_queue.put({"type": "started", "branch_id": branch_id, "seed": int(config.seed)})
        trainer = (
            train_policy_heterogeneous
            if bool(getattr(config, "heterogeneous_rollouts", False))
            else train_policy
        )
        trainer(
            config,
            output_path,
            progress_callback=None,
            cancel_callback=cancelled,
            epoch_observer=observer,
            resume_extra_provider=extra_provider,
            cancel_during_rollout=False,
            suppress_cancel_persistence=True,
        )
        status_queue.put({
            "type": "completed",
            "branch_id": branch_id,
            "epoch": int(current_epochs[index]),
            "terminal_resume": str(Path(config.resume_checkpoint) if config.resume_checkpoint else output_path.with_suffix(".resume.pt")),
        })
    except TrainingCancelled:
        status_queue.put({"type": "cancelled", "branch_id": branch_id, "epoch": int(current_epochs[index])})
    except BaseException as exc:
        status_queue.put({"type": "fatal", "branch_id": branch_id, "error": f"{type(exc).__name__}: {exc}"})
        raise


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".branches.json")


def load_branch_manifest(output_path: str | Path) -> dict:
    path = _manifest_path(Path(output_path))
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    temporary.replace(path)


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
            "policy_training_architecture": "v5.6",
            "training_method": "competitive multi-branch PPO; no neural weight averaging",
            "training_config": _config_payload(config),
            "training_seed": int(seed),
            "cumulative_epoch": epoch,
            "champion_epoch": epoch,
            "champion_metrics": metrics,
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
    artifact_dir = output_path.parent / f"{output_path.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"base_{int(payload['metadata'].get('champion_epoch', 0)):012d}_{uuid.uuid4().hex[:10]}.pt"
    payload = {**payload, "metadata": dict(payload.get("metadata", {}))}
    payload["metadata"]["immutable_artifact_path"] = str(artifact_path.resolve())
    payload["metadata"]["immutable_terminal_checkpoint"] = str(artifact_path.resolve())
    with tempfile.NamedTemporaryFile(delete=False, dir=artifact_dir, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(artifact_path)
    finally:
        temporary.unlink(missing_ok=True)
    sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps({**payload["metadata"], "sha256": sha}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return artifact_path, sha


def _candidate_from_resume(path: Path) -> dict | None:
    payload = load_trusted_resume(path, map_location="cpu")
    extra = dict(payload.get("extra", {}) or {})
    champion = dict(extra.get("branch_champion", {}) or {})
    if not champion.get("model_state_dict") or not champion.get("metrics"):
        return None
    # A branch may carry the frozen Base only as its promotion threshold. Do not report that
    # inherited threshold as if the branch independently discovered a new champion.
    if str(champion.get("source", "")) == "base_threshold":
        return None
    return champion


def train_policy_competitive(
    config,
    output_path,
    *,
    parallel_runs: int | None = None,
    progress_callback=None,
    cancel_callback=None,
) -> tuple[str, list]:
    """Train independent policy branches competitively and promote only the best policy.

    No model averaging is performed.  Exact branch resume states are preserved separately from the
    logical base policy.  Safe-stop rolls all branches back to a common available 10-epoch boundary.
    """

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_plan = build_branch_seed_plan(config, parallel_runs)
    start_mode = str(getattr(config, "parallel_start_mode", "new") or "new").strip().lower()
    if start_mode not in {"new", "exact_resume", "base_guided_fork"}:
        raise ValueError(f"Unsupported parallel branch start mode: {start_mode}")

    prior_manifest = load_branch_manifest(output_path)
    if start_mode == "exact_resume" and prior_manifest.get("branches"):
        seed_plan = [
            BranchSeed(
                branch_id=str(row.get("branch_id")),
                seed=int(row.get("seed", 0)),
                strategy=str(row.get("strategy", "restored")),
            )
            for row in prior_manifest["branches"]
        ]
    config.parallel_runs = len(seed_plan)
    base_checkpoint = str(getattr(config, "base_model_checkpoint", "") or "")
    if not base_checkpoint and output_path.is_file():
        try:
            payload = load_checkpoint(output_path, map_location="cpu")
            base_checkpoint = str(dict(payload.get("metadata", {}) or {}).get("immutable_artifact_path", "") or output_path)
        except Exception:
            base_checkpoint = ""
    if start_mode == "exact_resume" and not prior_manifest:
        raise ValueError("Exact multi-branch resume requires an existing .branches.json manifest")

    session_id = uuid.uuid4().hex
    scratch_base = Path(str(getattr(config, "training_scratch_dir", "") or "").strip() or (Path(tempfile.gettempdir()) / "CALO-RPD" / "policy_training"))
    scratch_root = scratch_base / session_id
    scratch_root.mkdir(parents=True, exist_ok=True)
    branch_dir = output_path.parent / f"{output_path.stem}_branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    cancel_event = ctx.Event()
    current_epochs = ctx.Array("q", [0] * len(seed_plan), lock=True)
    last_safe_epochs = ctx.Array("q", [0] * len(seed_plan), lock=True)
    global_safe_epoch = ctx.Value("q", 0)
    status_queue = ctx.Queue()

    branch_payloads: list[dict] = []
    prior_by_id = {str(row.get("branch_id")): row for row in prior_manifest.get("branches", [])}
    for index, spec in enumerate(seed_plan):
        prior = prior_by_id.get(spec.branch_id, {}) if start_mode == "exact_resume" else {}
        permanent_resume = Path(str(prior.get("resume_path", "") or (branch_dir / f"{spec.branch_id}.resume.pt")))
        if start_mode == "exact_resume" and not permanent_resume.is_file():
            raise FileNotFoundError(f"Exact branch resume checkpoint missing: {permanent_resume}")
        branch_payloads.append({
            "index": index,
            "branch_id": spec.branch_id,
            "seed": int(prior.get("seed", spec.seed) if start_mode == "exact_resume" else spec.seed),
            "strategy": str(prior.get("strategy", spec.strategy) if start_mode == "exact_resume" else spec.strategy),
            "start_mode": start_mode,
            "resume_path": str(permanent_resume) if start_mode == "exact_resume" else str(scratch_root / spec.branch_id / "terminal.resume.pt"),
            "working_output": str(scratch_root / spec.branch_id / "working.pt"),
            "initial_policy_checkpoint": base_checkpoint if start_mode == "base_guided_fork" else "",
            "base_model_checkpoint": base_checkpoint,
            "permanent_resume": str(branch_dir / f"{spec.branch_id}.resume.pt"),
        })
        if start_mode == "exact_resume":
            try:
                p = load_trusted_resume(permanent_resume, map_location="cpu")
                start_epoch = int(p.get("next_epoch", 0) or 0)
            except Exception:
                start_epoch = 0
            current_epochs[index] = start_epoch
            last_safe_epochs[index] = start_epoch

    if len(branch_payloads) > 1 and start_mode == "exact_resume":
        starts = {int(current_epochs[i]) for i in range(len(branch_payloads))}
        if len(starts) != 1:
            raise ValueError(
                "Exact competitive resume requires all branches to start from one common saved epoch"
            )
        global_safe_epoch.value = min(starts)

    processes: list[mp.Process] = []
    config_dict = _config_payload(config)
    for payload in branch_payloads:
        process = ctx.Process(
            target=_branch_worker_main,
            args=(
                config_dict,
                payload,
                str(scratch_root),
                cancel_event,
                current_epochs,
                last_safe_epochs,
                global_safe_epoch,
                status_queue,
            ),
            name=f"CALO-Policy-{payload['branch_id']}",
        )
        process.start()
        processes.append(process)

    messages: list[dict] = []
    cancelled = False
    fatal_messages: list[str] = []
    started = time.monotonic()
    try:
        while True:
            try:
                while True:
                    message = status_queue.get_nowait()
                    messages.append(message)
                    if message.get("type") == "fatal":
                        fatal_messages.append(f"{message.get('branch_id')}: {message.get('error')}")
                    if progress_callback and message.get("type") == "champion":
                        progress_callback(
                            0,
                            f"{message['branch_id']} epoch {message['epoch']} · "
                            + ("new branch champion" if message.get("promoted") else "evaluated"),
                        )
            except queue.Empty:
                pass

            safe_values = [int(last_safe_epochs[i]) for i in range(len(processes))]
            if safe_values and all(value >= 0 for value in safe_values):
                common = min(safe_values)
                if common > int(global_safe_epoch.value):
                    global_safe_epoch.value = common

            if cancel_callback and cancel_callback() and not cancelled:
                cancelled = True
                cancel_event.set()
                if progress_callback:
                    progress_callback(0, "Safe Stop requested · preserving the common previous 10-epoch state")

            if fatal_messages and not cancel_event.is_set():
                cancel_event.set()

            if all(not process.is_alive() for process in processes):
                break
            if progress_callback:
                alive = sum(1 for process in processes if process.is_alive())
                epoch_values = [int(current_epochs[i]) for i in range(len(processes))]
                progress_callback(
                    0,
                    f"Competitive branches · {alive}/{len(processes)} active · epochs {epoch_values} · common safe {int(global_safe_epoch.value)}",
                )
            time.sleep(0.25)

        for process in processes:
            process.join(timeout=1)
        exit_failures = [
            f"{branch_payloads[i]['branch_id']}: exitcode {p.exitcode}"
            for i, p in enumerate(processes)
            if p.exitcode not in (0, None)
        ]
        if exit_failures:
            fatal_messages.extend(exit_failures)
        if fatal_messages:
            raise RuntimeError("Competitive policy branch failure: " + "; ".join(dict.fromkeys(fatal_messages)))

        # Safe-stop selects one common exact boundary.  Normal cumulative completion uses each
        # branch terminal state, which should be the same requested epoch.
        if cancelled:
            common_epoch = min(int(last_safe_epochs[i]) for i in range(len(processes)))
        else:
            common_epoch = min(int(current_epochs[i]) for i in range(len(processes)))
        finalized: list[dict] = []
        for payload in branch_payloads:
            branch_id = payload["branch_id"]
            if cancelled:
                source = RollingSafeStore(scratch_root, branch_id).path(common_epoch)
            else:
                source = Path(payload["resume_path"])
                # Exact-resume branches write terminal state back to their permanent path; new/fork
                # branches write to session scratch and are committed below.
            if not source.is_file():
                raise RuntimeError(f"Branch {branch_id} has no exact resume state at epoch {common_epoch}: {source}")
            permanent = Path(payload["permanent_resume"])
            if source.resolve() != permanent.resolve():
                _copy_trusted_resume(source, permanent)
            else:
                # Exact-resume branches atomically advanced their own trusted resume file in place.
                write_trusted_resume_hash(permanent)
            resume_payload = load_trusted_resume(permanent, map_location="cpu")
            actual_epoch = int(resume_payload.get("next_epoch", 0) or 0)
            if cancelled and actual_epoch != common_epoch:
                raise RuntimeError(
                    f"Branch {branch_id} safe-stop epoch mismatch: expected {common_epoch}, got {actual_epoch}"
                )
            champion = _candidate_from_resume(permanent)
            finalized.append({
                "branch_id": branch_id,
                "seed": int(payload["seed"]),
                "strategy": payload["strategy"],
                "resume_path": str(permanent),
                "resume_epoch": actual_epoch,
                "champion": champion,
                "status": "completed",
            })

        session_meta = {
            "session_id": session_id,
            "started_monotonic": started,
            "requested_branches": len(branch_payloads),
            "started_branches": len(processes),
            "successful_branches": len(finalized),
            "failed_branches": 0,
            "branch_count": len(finalized),
            "seed_plan": [asdict(item) for item in seed_plan],
            "common_resume_epoch": int(common_epoch),
            "training_mode": str(getattr(config, "training_mode", "cumulative")),
            "start_mode": start_mode,
            "cancelled_safe_stop": bool(cancelled),
            "method": "competitive independent PPO branches; no parameter averaging",
        }

        previous_payload = previous_metrics = None
        if base_checkpoint and Path(base_checkpoint).is_file():
            previous_payload = load_checkpoint(base_checkpoint, map_location="cpu")
            previous_metrics = dict(dict(previous_payload.get("metadata", {}) or {}).get("champion_metrics", {}) or {})
            if not previous_metrics:
                # Evaluate the previous base once under the same fixed branch comparator.
                from .policy_network import CALOPolicyNetwork
                from .policy_schema import POLICY_STATE_DIM

                arch = dict(previous_payload.get("architecture", {}) or {})
                net = CALOPolicyNetwork(POLICY_STATE_DIM, int(arch.get("hidden_dim", config.hidden_dim)))
                net.load_state_dict(previous_payload["model_state_dict"])
                previous_metrics = evaluate_policy_multimetric(net, config)

        best_payload = previous_payload
        best_metrics = previous_metrics
        best_source = "previous_base" if previous_payload is not None else ""
        best_branch = ""
        promotion_decisions: list[dict] = []
        for row in finalized:
            champion = row.get("champion")
            if not champion:
                continue
            metrics = dict(champion.get("metrics", {}) or {})
            decision = compare_champion_metrics(metrics, best_metrics)
            promotion_decisions.append({
                "branch_id": row["branch_id"],
                "decision": asdict(decision),
                "metrics": metrics,
            })
            if decision.superior:
                best_metrics = metrics
                best_payload = _policy_payload_from_champion(
                    champion,
                    config,
                    branch_id=row["branch_id"],
                    seed=int(row["seed"]),
                    session=session_meta,
                )
                best_source = "branch_champion"
                best_branch = row["branch_id"]

        if best_payload is None:
            raise RuntimeError("No valid policy champion was produced by any branch")
        if best_source == "previous_base":
            artifact = Path(base_checkpoint)
            sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if output_path != artifact:
                shutil.copy2(artifact, output_path)
        else:
            best_payload["metadata"]["base_promotion_decisions"] = promotion_decisions
            best_payload["metadata"]["branch_manifest"] = str(_manifest_path(output_path))
            artifact, sha = _save_immutable_base(output_path, best_payload)

        manifest = {
            "schema_version": 2,
            "policy_lineage_id": str(getattr(config, "policy_lineage_id", "")),
            "policy_lineage_name": str(getattr(config, "policy_lineage_name", "")),
            "logical_base_alias": str(output_path),
            "base_artifact_path": str(artifact),
            "base_sha256": sha,
            "base_source": best_source,
            "base_source_branch": best_branch,
            "base_metrics": dict(best_metrics or {}),
            "common_resume_epoch": int(common_epoch),
            "previous_training_mode": str(getattr(config, "training_mode", "cumulative")),
            "previous_session_epochs": int(getattr(config, "epochs", 0) or 0),
            "branches": [
                {key: value for key, value in row.items() if key != "champion"}
                for row in finalized
            ],
            "seed_plan": [asdict(item) for item in seed_plan],
            "promotion_decisions": promotion_decisions,
            "session": session_meta,
        }
        _atomic_json(_manifest_path(output_path), manifest)
        if progress_callback:
            progress_callback(
                100,
                f"Competitive training complete · base {Path(artifact).name} · {len(finalized)} branches · resume epoch {common_epoch}",
            )
        # Return compact coordinator history rather than the old empty history contract.
        return str(output_path), messages + [{"type": "base_selection", **manifest}]
    finally:
        cancel_event.set()
        for process in processes:
            if process.is_alive():
                process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        try:
            status_queue.close()
            status_queue.join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            shutil.rmtree(scratch_root, ignore_errors=True)
        except OSError:
            pass
