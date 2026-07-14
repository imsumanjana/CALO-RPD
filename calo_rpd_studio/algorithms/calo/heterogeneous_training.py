"""Weighted heterogeneous actor collection for CALO policy training.

This module intentionally sits outside the frozen CALO v2 implementation.  It changes only the
training orchestration used to create a *candidate* policy checkpoint.  The frozen benchmark policy,
operators, state vector, constraint handling, and optimizer equations remain untouched until the
user explicitly validates and re-freezes a newly trained checkpoint.

Each PPO epoch follows a synchronous actor--learner protocol:

1. snapshot one policy version on the CPU;
2. allocate complete rollout episodes to CUDA, Intel XPU, and CPU actor lanes;
3. collect all lanes in parallel using the same policy snapshot;
4. reject any stale/mismatched actor payload;
5. combine the fresh trajectories into one on-policy buffer;
6. update the policy on one primary learner device;
7. broadcast the updated snapshot at the next epoch.

The requested default transition split is 50% CUDA, 30% XPU, and 20% CPU.  Since every episode has
one common horizon, an episode split is also a transition split.  Accelerator lanes batch policy
inference on their assigned device.  The synthetic/ORPD environments, PYPOWER, and most physical
constraint calculations remain host-CPU workloads, so hardware-utilization percentages will not
numerically equal the configured rollout shares.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .cognitive_state import STATE_DIM
from .policy_network import CALOPolicyNetwork
from .training import (
    SyntheticCALOEnvironment,
    TrainingCancelled,
    TrainingConfig,
    _compute_gae,
    _cpu_state_dict,
    _curriculum_stage,
    _historical_pretrain,
    _parameter_action_distribution,
    _resolve_training_device,
    available_training_devices,
    recommended_rollout_workers,
)


ROLLOUT_KEYS = (
    "state",
    "regime",
    "operator",
    "parameter",
    "logp",
    "value",
    "reward",
    "done",
)
LANE_ORDER = ("cuda", "xpu", "cpu")


@dataclass(slots=True)
class HeterogeneousTrainingConfig(TrainingConfig):
    """Training configuration with weighted synchronous actor lanes."""

    heterogeneous_rollouts: bool = True
    cuda_rollout_share: int = 50
    xpu_rollout_share: int = 30
    cpu_rollout_share: int = 20
    actor_batch_size: int = 0


@dataclass(frozen=True, slots=True)
class TrainingLanePlan:
    requested_shares: dict[str, int]
    available_lanes: dict[str, bool]
    episode_counts: dict[str, int]
    effective_shares: dict[str, float]
    devices: dict[str, str]
    xpu_runtime: str
    warnings: tuple[str, ...]

    @property
    def total_episodes(self) -> int:
        return int(sum(self.episode_counts.values()))

    def summary(self) -> str:
        counts = self.episode_counts
        effective = self.effective_shares
        return (
            f"CUDA {counts['cuda']} ({effective['cuda']:.1f}%) · "
            f"XPU {counts['xpu']} ({effective['xpu']:.1f}%) · "
            f"CPU {counts['cpu']} ({effective['cpu']:.1f}%)"
        )


def _validate_shares(cuda_share: int, xpu_share: int, cpu_share: int) -> dict[str, int]:
    shares = {
        "cuda": int(cuda_share),
        "xpu": int(xpu_share),
        "cpu": int(cpu_share),
    }
    if any(value < 0 or value > 100 for value in shares.values()):
        raise ValueError("CUDA, XPU, and CPU rollout shares must each be between 0 and 100.")
    if sum(shares.values()) != 100:
        raise ValueError("CUDA, XPU, and CPU rollout shares must total exactly 100%.")
    return shares


def _largest_remainder_allocation(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total < 1:
        raise ValueError("At least one rollout episode is required.")
    positive = {name: float(value) for name, value in weights.items() if float(value) > 0.0}
    if not positive:
        raise ValueError("At least one available rollout lane must have a positive share.")
    weight_sum = sum(positive.values())
    exact = {name: total * value / weight_sum for name, value in positive.items()}
    allocation = {name: int(np.floor(exact[name])) for name in positive}
    remainder = total - sum(allocation.values())
    priority = {name: index for index, name in enumerate(LANE_ORDER)}
    ranked = sorted(
        positive,
        key=lambda name: (-(exact[name] - allocation[name]), priority.get(name, 999)),
    )
    for name in ranked[:remainder]:
        allocation[name] += 1
    return {name: int(allocation.get(name, 0)) for name in LANE_ORDER}


def plan_training_lanes(
    episodes_per_epoch: int,
    *,
    cuda_share: int = 50,
    xpu_share: int = 30,
    cpu_share: int = 20,
    cuda_available: bool | None = None,
    xpu_available: bool | None = None,
    xpu_sidecar_available: bool | None = None,
) -> TrainingLanePlan:
    """Create a deterministic per-epoch actor allocation.

    Unavailable accelerator shares are redistributed proportionally across available requested
    lanes.  CPU is always available.  Allocation uses the largest-remainder method so 12 episodes
    with 50/30/20 become 6 CUDA, 4 XPU, and 2 CPU episodes.
    """

    requested = _validate_shares(cuda_share, xpu_share, cpu_share)
    info = available_training_devices()
    cuda_ok = bool(info["cuda_available"] if cuda_available is None else cuda_available)
    direct_xpu = bool(info["xpu_available"] if xpu_available is None else xpu_available)
    sidecar_xpu = bool(
        info["xpu_sidecar_available"]
        if xpu_sidecar_available is None
        else xpu_sidecar_available
    )
    xpu_ok = direct_xpu or sidecar_xpu
    available = {"cuda": cuda_ok, "xpu": xpu_ok, "cpu": True}

    usable_weights = {
        name: requested[name] if available[name] else 0
        for name in LANE_ORDER
    }
    warnings: list[str] = []
    for name in ("cuda", "xpu"):
        if requested[name] and not available[name]:
            warnings.append(
                f"Requested {name.upper()} share is unavailable and will be redistributed."
            )
    if not any(usable_weights.values()):
        usable_weights["cpu"] = 100
        warnings.append("All requested accelerator lanes are unavailable; using CPU actors.")

    counts = _largest_remainder_allocation(int(episodes_per_epoch), usable_weights)
    total = max(1, int(episodes_per_epoch))
    effective = {name: 100.0 * counts[name] / total for name in LANE_ORDER}
    xpu_runtime = "primary" if direct_xpu else ("sidecar" if sidecar_xpu else "unavailable")
    devices = {
        "cuda": "cuda:0" if cuda_ok else "unavailable",
        "xpu": "xpu:0" if xpu_ok else "unavailable",
        "cpu": "cpu",
    }
    return TrainingLanePlan(
        requested_shares=requested,
        available_lanes=available,
        episode_counts=counts,
        effective_shares=effective,
        devices=devices,
        xpu_runtime=xpu_runtime,
        warnings=tuple(warnings),
    )


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _new_rollout() -> dict[str, list]:
    return {key: [] for key in ROLLOUT_KEYS}


def _environment_for_episode(
    config: HeterogeneousTrainingConfig,
    *,
    epoch: int,
    stage: int,
    episode: int,
):
    episode_seed = int(config.seed + 1_000_003 * epoch + 10_007 * int(episode))
    random.seed(episode_seed)
    np.random.seed(episode_seed % (2**32 - 1))
    rng = np.random.default_rng(episode_seed)
    if stage == 4:
        from calo_rpd_studio.orpd.problem import ORPDProblem
        from calo_rpd_studio.power_system.case_loader import CaseLoader

        source = config.development_cases[
            (epoch * config.episodes_per_epoch + int(episode)) % len(config.development_cases)
        ]
        problem = ORPDProblem(CaseLoader.load(source))
        environment = SyntheticCALOEnvironment(
            rng,
            stage,
            config.population_size,
            problem=problem,
        )
    else:
        environment = SyntheticCALOEnvironment(rng, stage, config.population_size)
    return episode_seed, environment


def _sample_actions(regime_logits, operator_logits, alpha, beta):
    """Sample actions on the actor device, with a CPU fallback for unsupported XPU kernels."""
    try:
        regime_dist = torch.distributions.Categorical(logits=regime_logits)
        operator_dist = torch.distributions.Categorical(logits=operator_logits)
        parameter_dist = _parameter_action_distribution(alpha, beta)
        regime = regime_dist.sample()
        operator = operator_dist.sample()
        parameter = parameter_dist.sample()
        logp = (
            regime_dist.log_prob(regime)
            + operator_dist.log_prob(operator)
            + parameter_dist.log_prob(parameter).sum(-1)
        )
        return regime, operator, parameter, logp
    except Exception:
        regime_logits_cpu = regime_logits.detach().cpu()
        operator_logits_cpu = operator_logits.detach().cpu()
        alpha_cpu = alpha.detach().cpu()
        beta_cpu = beta.detach().cpu()
        regime_dist = torch.distributions.Categorical(logits=regime_logits_cpu)
        operator_dist = torch.distributions.Categorical(logits=operator_logits_cpu)
        parameter_dist = _parameter_action_distribution(alpha_cpu, beta_cpu)
        regime = regime_dist.sample()
        operator = operator_dist.sample()
        parameter = parameter_dist.sample()
        logp = (
            regime_dist.log_prob(regime)
            + operator_dist.log_prob(operator)
            + parameter_dist.log_prob(parameter).sum(-1)
        )
        return regime, operator, parameter, logp


def collect_actor_lane_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect one complete actor lane using batched policy inference.

    This function is module-level and pickle-safe so it can run in a Windows ``spawn`` process or
    in the isolated Intel-XPU Python environment.
    """

    config = HeterogeneousTrainingConfig(**payload["config"])
    network_state = payload["network_state"]
    epoch = int(payload["epoch"])
    stage = int(payload["stage"])
    episode_indices = [int(item) for item in payload["episode_indices"]]
    device_name = str(payload["device"])
    lane = str(payload["lane"])
    expected_snapshot = str(payload["policy_snapshot_sha256"])

    torch.set_num_threads(max(1, int(config.cpu_threads_per_worker)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device(device_name)
    network = CALOPolicyNetwork(STATE_DIM, config.hidden_dim).to(device)
    network.load_state_dict(network_state)
    network.eval()
    actual_snapshot = _state_dict_sha256(_cpu_state_dict(network))
    if actual_snapshot != expected_snapshot:
        raise RuntimeError("Actor received a policy snapshot that does not match the learner.")

    environments = []
    for episode in episode_indices:
        _, environment = _environment_for_episode(
            config,
            epoch=epoch,
            stage=stage,
            episode=episode,
        )
        environments.append((episode, environment))

    lane_seed = int(config.seed + 7_919 * epoch + 1_000_000 * (LANE_ORDER.index(lane) + 1))
    torch.manual_seed(lane_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(lane_seed)

    episode_rollouts = {episode: _new_rollout() for episode, _ in environments}
    episode_returns = {episode: 0.0 for episode, _ in environments}

    for step in range(config.horizon):
        states = np.stack(
            [environment.state(config.horizon).vector() for _, environment in environments],
            axis=0,
        )
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        with torch.inference_mode():
            regime_logits, operator_logits, alpha, beta, values = network(state_tensor)
            regimes, operators, parameters, logps = _sample_actions(
                regime_logits,
                operator_logits,
                alpha,
                beta,
            )
        regimes_cpu = regimes.detach().cpu().numpy()
        operators_cpu = operators.detach().cpu().numpy()
        parameters_cpu = parameters.detach().cpu().numpy()
        logps_cpu = logps.detach().cpu().numpy()
        values_cpu = values.detach().cpu().numpy()

        for index, (episode, environment) in enumerate(environments):
            reward = environment.step(
                int(regimes_cpu[index]),
                int(operators_cpu[index]),
                np.asarray(parameters_cpu[index], dtype=float),
                config.horizon,
            )
            done = step == config.horizon - 1
            record = episode_rollouts[episode]
            record["state"].append(states[index])
            record["regime"].append(int(regimes_cpu[index]))
            record["operator"].append(int(operators_cpu[index]))
            record["parameter"].append(np.asarray(parameters_cpu[index], dtype=float))
            record["logp"].append(float(logps_cpu[index]))
            record["value"].append(float(values_cpu[index]))
            record["reward"].append(float(reward))
            record["done"].append(bool(done))
            episode_returns[episode] += float(reward)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            torch.xpu.synchronize(device)
        except Exception:
            pass

    return {
        "lane": lane,
        "device": device_name,
        "policy_snapshot_sha256": actual_snapshot,
        "episodes": [
            {
                "episode_index": int(episode),
                "rollout": episode_rollouts[episode],
                "episode_return": float(episode_returns[episode]),
            }
            for episode in sorted(episode_rollouts)
        ],
    }


def _collect_cpu_lane(
    config: HeterogeneousTrainingConfig,
    network_state: dict[str, torch.Tensor],
    *,
    epoch: int,
    stage: int,
    episode_indices: list[int],
    policy_snapshot_sha256: str,
) -> list[dict[str, Any]]:
    if not episode_indices:
        return []
    workers = max(1, min(int(config.rollout_workers), len(episode_indices)))
    chunks = [chunk.tolist() for chunk in np.array_split(episode_indices, workers) if len(chunk)]
    payloads = [
        {
            "config": asdict(config),
            "network_state": network_state,
            "epoch": epoch,
            "stage": stage,
            "episode_indices": chunk,
            "device": "cpu",
            "lane": "cpu",
            "policy_snapshot_sha256": policy_snapshot_sha256,
        }
        for chunk in chunks
    ]
    if len(payloads) == 1:
        return [collect_actor_lane_payload(payloads[0])]
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(collect_actor_lane_payload, payload) for payload in payloads]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _collect_accelerator_subprocess(
    interpreter: str,
    payload: dict[str, Any],
    *,
    cancel_callback=None,
) -> dict[str, Any]:
    """Run one accelerator lane in its own interpreter and return its rollout payload."""
    with tempfile.TemporaryDirectory(prefix=f"calo_{payload['lane']}_actor_") as tmp:
        root = Path(tmp)
        input_path = root / "actor_input.pt"
        output_path = root / "actor_output.pt"
        torch.save(payload, input_path)
        process = subprocess.Popen(
            [
                interpreter,
                "-m",
                "calo_rpd_studio.compute.training_actor_worker",
                str(input_path),
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
        while process.poll() is None:
            if cancel_callback and cancel_callback():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TrainingCancelled("CALO policy training was cancelled safely.")
            time.sleep(0.1)
        stdout, _ = process.communicate()
        output_lines = [line.rstrip() for line in stdout.splitlines()]
        if process.returncode != 0 or not output_path.exists():
            detail = "\n".join(output_lines[-20:])
            raise RuntimeError(
                f"{payload['lane'].upper()} actor worker exited with code {process.returncode}."
                + (f"\n{detail}" if detail else "")
            )
        return torch.load(output_path, map_location="cpu", weights_only=False)


def _xpu_interpreter_for_plan(plan: TrainingLanePlan) -> str:
    if plan.xpu_runtime == "primary":
        return sys.executable
    if plan.xpu_runtime == "sidecar":
        from calo_rpd_studio.compute.resource_scheduler import configured_xpu_interpreter

        interpreter = configured_xpu_interpreter()
        if interpreter:
            return interpreter
    raise RuntimeError("No verified Intel XPU interpreter is available for the XPU actor lane.")


def _flatten_actor_results(
    results: list[dict[str, Any]],
    *,
    expected_snapshot: str,
    expected_episodes: int,
):
    by_episode: dict[int, dict[str, Any]] = {}
    lane_records: list[dict[str, Any]] = []
    stale_payloads = 0
    for result in results:
        if str(result.get("policy_snapshot_sha256", "")) != expected_snapshot:
            stale_payloads += 1
            continue
        lane_records.append(
            {
                "lane": str(result.get("lane", "")),
                "device": str(result.get("device", "")),
                "episodes": len(result.get("episodes", [])),
            }
        )
        for episode in result.get("episodes", []):
            index = int(episode["episode_index"])
            if index in by_episode:
                raise RuntimeError(
                    f"Duplicate rollout episode {index} was returned by actor lanes."
                )
            by_episode[index] = episode
    if stale_payloads:
        raise RuntimeError(f"Rejected {stale_payloads} stale actor payload(s).")
    if len(by_episode) != expected_episodes:
        missing = sorted(set(range(expected_episodes)) - set(by_episode))
        raise RuntimeError(
            "Actor collection returned "
            f"{len(by_episode)}/{expected_episodes} episodes; missing {missing}."
        )

    rollout = _new_rollout()
    episode_returns: list[float] = []
    for episode_index in sorted(by_episode):
        episode = by_episode[episode_index]
        episode_rollout = episode["rollout"]
        for key in ROLLOUT_KEYS:
            rollout[key].extend(episode_rollout[key])
        episode_returns.append(float(episode["episode_return"]))
    return rollout, episode_returns, lane_records


def collect_weighted_epoch_rollouts(
    config: HeterogeneousTrainingConfig,
    network: nn.Module,
    *,
    epoch: int,
    stage: int,
    progress_callback=None,
    cancel_callback=None,
):
    """Collect one synchronous weighted epoch from CUDA, XPU, and CPU actors in parallel."""
    plan = plan_training_lanes(
        config.episodes_per_epoch,
        cuda_share=config.cuda_rollout_share,
        xpu_share=config.xpu_rollout_share,
        cpu_share=config.cpu_rollout_share,
    )
    network_state = _cpu_state_dict(network)
    snapshot = _state_dict_sha256(network_state)

    episode_order = list(range(config.episodes_per_epoch))
    cursor = 0
    lane_episodes: dict[str, list[int]] = {}
    for lane in LANE_ORDER:
        count = plan.episode_counts[lane]
        lane_episodes[lane] = episode_order[cursor : cursor + count]
        cursor += count

    futures = {}
    results: list[dict[str, Any]] = []
    active_lanes = [lane for lane in LANE_ORDER if lane_episodes[lane]]
    with ThreadPoolExecutor(max_workers=max(1, len(active_lanes))) as executor:
        if lane_episodes["cuda"]:
            payload = {
                "config": asdict(config),
                "network_state": network_state,
                "epoch": epoch,
                "stage": stage,
                "episode_indices": lane_episodes["cuda"],
                "device": "cuda:0",
                "lane": "cuda",
                "policy_snapshot_sha256": snapshot,
            }
            futures[
                executor.submit(
                    _collect_accelerator_subprocess,
                    sys.executable,
                    payload,
                    cancel_callback=cancel_callback,
                )
            ] = "cuda"
        if lane_episodes["xpu"]:
            payload = {
                "config": asdict(config),
                "network_state": network_state,
                "epoch": epoch,
                "stage": stage,
                "episode_indices": lane_episodes["xpu"],
                "device": "xpu:0",
                "lane": "xpu",
                "policy_snapshot_sha256": snapshot,
            }
            futures[
                executor.submit(
                    _collect_accelerator_subprocess,
                    _xpu_interpreter_for_plan(plan),
                    payload,
                    cancel_callback=cancel_callback,
                )
            ] = "xpu"
        if lane_episodes["cpu"]:
            futures[
                executor.submit(
                    _collect_cpu_lane,
                    config,
                    network_state,
                    epoch=epoch,
                    stage=stage,
                    episode_indices=lane_episodes["cpu"],
                    policy_snapshot_sha256=snapshot,
                )
            ] = "cpu"

        completed = 0
        for future in as_completed(futures):
            if cancel_callback and cancel_callback():
                for pending in futures:
                    pending.cancel()
                raise TrainingCancelled("CALO policy training was cancelled safely.")
            lane = futures[future]
            value = future.result()
            if isinstance(value, list):
                results.extend(value)
                completed += sum(len(item.get("episodes", [])) for item in value)
            else:
                results.append(value)
                completed += len(value.get("episodes", []))
            if progress_callback:
                progress_callback(
                    completed,
                    f"{lane.upper()} actor lane completed · "
                    f"{completed}/{config.episodes_per_epoch} episodes",
                )

    rollout, episode_returns, lane_records = _flatten_actor_results(
        results,
        expected_snapshot=snapshot,
        expected_episodes=config.episodes_per_epoch,
    )
    return rollout, episode_returns, plan, lane_records, snapshot


def train_policy_heterogeneous(
    config: HeterogeneousTrainingConfig,
    output_path,
    progress_callback=None,
    cancel_callback=None,
):
    """Train a candidate CALO policy with synchronous weighted heterogeneous actors."""
    final_benchmark_names = {"case30", "case57", "case118"}
    development_names = {Path(item).stem.lower() for item in config.development_cases}
    leaked = sorted(final_benchmark_names & development_names)
    if leaked and not config.allow_final_benchmark_training:
        raise ValueError(
            "Final publication benchmark cases cannot be used for CALO policy training by default: "
            + ", ".join(leaked)
            + ". Supply separate development systems or explicitly enable the override "
            "in a documented non-final study."
        )
    if not config.heterogeneous_rollouts:
        from .training import train_policy

        base_payload = {
            key: value
            for key, value in asdict(config).items()
            if key in TrainingConfig.__dataclass_fields__
        }
        return train_policy(
            TrainingConfig(**base_payload),
            output_path,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    _validate_shares(
        config.cuda_rollout_share,
        config.xpu_rollout_share,
        config.cpu_rollout_share,
    )
    if str(config.ppo_device).lower() == "xpu_sidecar":
        raise ValueError(
            "Weighted multi-device training requires the PPO learner in the primary runtime. "
            "Select Automatic, CUDA, direct XPU, or CPU. The secondary XPU runtime "
            "remains available as an actor lane."
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    learner_device = _resolve_training_device(config.ppo_device)
    if str(config.ppo_device).lower() == "auto" and not torch.cuda.is_available():
        # Direct XPU can be selected by _resolve_training_device.  A sidecar XPU cannot host the
        # centralized learner in this process, so CPU is the correct fallback in that configuration.
        learner_device = _resolve_training_device("auto")

    if int(config.rollout_workers) <= 0:
        config.rollout_workers = recommended_rollout_workers(config.episodes_per_epoch)
    else:
        config.rollout_workers = min(
            int(config.rollout_workers),
            max(1, int(config.episodes_per_epoch)),
        )

    network = CALOPolicyNetwork(STATE_DIM, config.hidden_dim).to(learner_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    historical_pretraining = _historical_pretrain(
        network,
        optimizer,
        learner_device,
        config,
        rng,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    history: list[dict[str, Any]] = []
    total_units = config.epochs * config.episodes_per_epoch
    completed_units = 0

    for epoch in range(config.epochs):
        if cancel_callback and cancel_callback():
            raise TrainingCancelled("CALO policy training was cancelled safely.")
        stage = _curriculum_stage(epoch, config.epochs, bool(config.development_cases))

        def actor_progress(completed_in_epoch: int, detail: str) -> None:
            if progress_callback:
                absolute = completed_units + completed_in_epoch
                progress_callback(
                    int(100 * absolute / max(total_units, 1)),
                    f"Epoch {epoch + 1}/{config.epochs} · {detail}",
                )

        rollout, episode_returns, plan, lane_records, snapshot = collect_weighted_epoch_rollouts(
            config,
            network,
            epoch=epoch,
            stage=stage,
            progress_callback=actor_progress,
            cancel_callback=cancel_callback,
        )
        completed_units += config.episodes_per_epoch
        if progress_callback:
            progress_callback(
                int(100 * completed_units / max(total_units, 1)),
                f"Epoch {epoch + 1}/{config.epochs} · PPO update on {learner_device} · "
                f"{len(rollout['state'])} fresh transitions · {plan.summary()}",
            )

        advantages, returns = _compute_gae(
            rollout["reward"],
            rollout["value"],
            rollout["done"],
            config.gamma,
            config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = torch.as_tensor(
            np.asarray(rollout["state"]),
            dtype=torch.float32,
            device=learner_device,
        )
        regimes = torch.as_tensor(rollout["regime"], dtype=torch.long, device=learner_device)
        operators = torch.as_tensor(rollout["operator"], dtype=torch.long, device=learner_device)
        parameters = torch.as_tensor(
            np.asarray(rollout["parameter"]),
            dtype=torch.float32,
            device=learner_device,
        )
        old_logp = torch.as_tensor(rollout["logp"], dtype=torch.float32, device=learner_device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=learner_device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=learner_device)

        epoch_losses: list[float] = []
        indices = np.arange(len(states))
        network.train()
        for _ in range(config.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), config.minibatch_size):
                batch = indices[start : start + config.minibatch_size]
                batch_t = torch.as_tensor(batch, dtype=torch.long, device=learner_device)
                regime_logits, operator_logits, alpha, beta, values = network(states[batch_t])
                regime_dist = torch.distributions.Categorical(logits=regime_logits)
                operator_dist = torch.distributions.Categorical(logits=operator_logits)
                parameter_dist = _parameter_action_distribution(alpha, beta)
                new_logp = (
                    regime_dist.log_prob(regimes[batch_t])
                    + operator_dist.log_prob(operators[batch_t])
                    + parameter_dist.log_prob(parameters[batch_t].clamp(1e-5, 1 - 1e-5)).sum(-1)
                )
                ratio = torch.exp(new_logp - old_logp[batch_t])
                unclipped = ratio * advantages_t[batch_t]
                clipped = torch.clamp(
                    ratio,
                    1.0 - config.clip_ratio,
                    1.0 + config.clip_ratio,
                ) * advantages_t[batch_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * ((values - returns_t[batch_t]) ** 2).mean()
                entropy = (
                    regime_dist.entropy().mean()
                    + operator_dist.entropy().mean()
                    + parameter_dist.entropy().sum(-1).mean()
                )
                loss = (
                    policy_loss
                    + config.value_weight * value_loss
                    - config.entropy_weight * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
        network.eval()
        history.append(
            {
                "epoch": epoch + 1,
                "curriculum_stage": stage + 1,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "mean_episode_return": float(np.mean(episode_returns)),
                "transitions": len(rollout["state"]),
                "ppo_learner_device": str(learner_device),
                "policy_snapshot_sha256": snapshot,
                "requested_rollout_shares": plan.requested_shares,
                "effective_rollout_shares": plan.effective_shares,
                "episode_allocation": plan.episode_counts,
                "actor_lanes": lane_records,
                "warnings": list(plan.warnings),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_info = available_training_devices()
    final_plan = plan_training_lanes(
        config.episodes_per_epoch,
        cuda_share=config.cuda_rollout_share,
        xpu_share=config.xpu_rollout_share,
        cpu_share=config.cpu_rollout_share,
    )
    metadata = {
        "algorithm": "CALO",
        "calo_core": "v2",
        "training_method": "synchronous weighted heterogeneous PPO",
        "candidate_checkpoint": True,
        "benchmark_freeze_status": (
            "Not automatically part of the frozen TEST benchmark. Validate this candidate "
            "and create a new freeze manifest before benchmark use."
        ),
        "training_config": asdict(config),
        "training_seed": config.seed,
        "state_dimension": STATE_DIM,
        "execution": {
            "architecture": (
                "same-policy synchronous CUDA/XPU/CPU actor lanes followed by one "
                "centralized PPO learner update"
            ),
            "ppo_learner_device": str(learner_device),
            "requested_rollout_shares": final_plan.requested_shares,
            "effective_episode_shares": final_plan.effective_shares,
            "episodes_per_epoch_by_lane": final_plan.episode_counts,
            "cuda_available": bool(device_info["cuda_available"]),
            "cuda_name": str(device_info["cuda_name"]),
            "xpu_available": bool(device_info["xpu_available"]),
            "xpu_name": str(device_info["xpu_name"]),
            "xpu_sidecar_available": bool(device_info["xpu_sidecar_available"]),
            "xpu_actor_runtime": final_plan.xpu_runtime,
            "cpu_rollout_workers": int(config.rollout_workers),
            "on_policy_synchronization": (
                "All actor lanes use one policy snapshot per epoch; PPO starts only after "
                "all matching trajectories arrive."
            ),
            "hardware_scope_note": (
                "Configured shares refer to rollout episodes/transitions, not measured device "
                "utilization. Policy inference runs on each actor device, while environment and "
                "power-flow calculations remain primarily CPU-based."
            ),
        },
        "curriculum": [
            "continuous unconstrained",
            "constrained continuous",
            "mixed discrete-continuous",
            "narrow feasible region",
            *(["explicit ORPD development systems"] if config.development_cases else []),
        ],
        "development_cases": list(config.development_cases),
        "final_publication_benchmarks_used_for_training": bool(leaked),
        "historical_pretraining": historical_pretraining,
        "historical_data_policy": (
            "eligible TRAIN experiments only; validation/test experiments excluded; old "
            "trajectories used only for offline pretraining, never as PPO on-policy rollouts"
        ),
        "history": history,
    }
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(network),
            "architecture": {"input_dim": STATE_DIM, "hidden_dim": config.hidden_dim},
            "metadata": metadata,
        },
        output_path,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return str(output_path), history
