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

The requested 80% CUDA, 10% XPU, and 10% CPU split is retained as a deterministic fallback.
Version 3.3 can first time complete discarded calibration episodes on each verified lane and then
allocate fresh on-policy episodes by measured transitions per second.  CUDA/XPU actor interpreters,
policy modules and ORPD tensors remain resident for the full training session.  Compatible ORPD
population requests from simultaneous episodes are combined by the same FP64 cross-run batching
engine used during comparative evaluation.  Synthetic curriculum stages still contain host-side
logic, so Task Manager percentages need not equal the episode allocation.
"""

from __future__ import annotations

import logging

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
import atexit
from typing import Any

import numpy as np
import torch
from torch import nn

from .policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_DIM,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
)
from .policy_network import CALOPolicyNetwork
from calo_rpd_studio.accelerated.runtime_context import set_cross_run_broker
from calo_rpd_studio.accelerated.throughput_engine import (
    CrossRunBatchBroker,
    largest_remainder_counts,
)
from calo_rpd_studio.compute.persistent_training_actor import PersistentTrainingActorClient
from calo_rpd_studio.ai.model_io import load_checkpoint

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
    _resolve_training_target,
    _stage_floor_from_history,
    _write_policy_alias,
    available_training_devices,
    recommended_rollout_workers,
    load_training_resume,
    save_training_resume,
    save_deployable_policy_snapshot,
    training_resume_path,
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


_LOG = logging.getLogger(__name__)

@dataclass(slots=True)
class HeterogeneousTrainingConfig(TrainingConfig):
    """Training configuration with weighted synchronous actor lanes."""

    heterogeneous_rollouts: bool = True
    cuda_rollout_share: int = 100
    xpu_rollout_share: int = 0
    cpu_rollout_share: int = 0
    actor_batch_size: int = 0
    throughput_adaptive_rollouts: bool = False
    persistent_actor_workers: bool = True
    actor_calibration_episodes: int = 1
    use_accelerated_orpd_rollouts: bool = True
    training_cross_episode_batching: bool = True
    training_batch_window_ms: float = 4.0
    training_max_cross_batch: int = 2048
    training_tensor_batch_size: int = 64


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
    cuda_share: int = 100,
    xpu_share: int = 0,
    cpu_share: int = 0,
    cuda_available: bool | None = None,
    xpu_available: bool | None = None,
    xpu_sidecar_available: bool | None = None,
) -> TrainingLanePlan:
    """Create a deterministic per-epoch actor allocation.

    The default is GPU maximum: all rollout episodes use CUDA when CUDA is available.
    Unavailable requested accelerator shares fall back to the next available numerical lane; CPU
    is always available. Explicit non-default shares use the largest-remainder method.
    """

    requested = _validate_shares(cuda_share, xpu_share, cpu_share)
    info = available_training_devices()
    cuda_ok = bool(info["cuda_available"] if cuda_available is None else cuda_available)
    direct_xpu = bool(info["xpu_available"] if xpu_available is None else xpu_available)
    sidecar_xpu = bool(
        info["xpu_sidecar_available"] if xpu_sidecar_available is None else xpu_sidecar_available
    )
    xpu_ok = direct_xpu or sidecar_xpu
    available = {"cuda": cuda_ok, "xpu": xpu_ok, "cpu": True}

    usable_weights = {name: requested[name] if available[name] else 0 for name in LANE_ORDER}
    warnings: list[str] = []
    gpu_maximum_request = requested == {"cuda": 100, "xpu": 0, "cpu": 0}
    if gpu_maximum_request and not cuda_ok:
        if xpu_ok:
            usable_weights = {"cuda": 0, "xpu": 100, "cpu": 0}
            warnings.append("CUDA is unavailable; GPU-maximum training fell back to Intel XPU.")
        else:
            usable_weights = {"cuda": 0, "xpu": 0, "cpu": 100}
            warnings.append("CUDA and Intel XPU are unavailable; training fell back to CPU actors.")
    else:
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


def plan_training_lanes_from_throughput(
    episodes_per_epoch: int,
    measured_transitions_per_second: dict[str, float],
    *,
    base_plan: TrainingLanePlan | None = None,
) -> TrainingLanePlan:
    """Allocate rollout episodes by measured actor transition throughput."""
    if base_plan is None:
        base_plan = plan_training_lanes(int(episodes_per_epoch))
    weights = {
        lane: (
            max(0.0, float(measured_transitions_per_second.get(lane, 0.0)))
            if base_plan.available_lanes.get(lane, False)
            else 0.0
        )
        for lane in LANE_ORDER
    }
    if sum(weights.values()) <= 0:
        return base_plan
    counts = largest_remainder_counts(int(episodes_per_epoch), weights)
    total = max(1, int(episodes_per_epoch))
    effective = {lane: 100.0 * counts.get(lane, 0) / total for lane in LANE_ORDER}
    normalized_sum = sum(weights.values())
    requested = {lane: int(round(100.0 * weights[lane] / normalized_sum)) for lane in LANE_ORDER}
    # Correct rounding drift deterministically.
    requested["cpu"] += 100 - sum(requested.values())
    warnings = list(base_plan.warnings)
    warnings.append(
        "Episode shares were auto-tuned from measured transitions per second rather than fixed percentages."
    )
    return TrainingLanePlan(
        requested_shares=requested,
        available_lanes=dict(base_plan.available_lanes),
        episode_counts={lane: int(counts.get(lane, 0)) for lane in LANE_ORDER},
        effective_shares=effective,
        devices=dict(base_plan.devices),
        xpu_runtime=base_plan.xpu_runtime,
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
    compute_device: str = "cpu",
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
        case = CaseLoader.load(source)
        if bool(config.use_accelerated_orpd_rollouts):
            from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem

            problem = AcceleratedORPDProblem(
                case,
                device=compute_device,
                dtype_name="float64",
                batch_size=max(1, int(config.training_tensor_batch_size)),
            )
        else:
            problem = ORPDProblem(case)
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


_ACTOR_NETWORK_CACHE: dict[tuple[str, int], nn.Module] = {}
_ACTOR_BROKER_CACHE: dict[tuple[str, float, int], CrossRunBatchBroker] = {}


def _close_actor_runtime_caches() -> None:
    for broker in list(_ACTOR_BROKER_CACHE.values()):
        try:
            broker.close()
        except Exception:
            _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)
    _ACTOR_BROKER_CACHE.clear()
    _ACTOR_NETWORK_CACHE.clear()


atexit.register(_close_actor_runtime_caches)


def _persistent_actor_network(device, hidden_dim: int):
    key = (str(device), int(hidden_dim))
    network = _ACTOR_NETWORK_CACHE.get(key)
    if network is None:
        network = CALOPolicyNetwork(POLICY_STATE_DIM, int(hidden_dim)).to(device)
        _ACTOR_NETWORK_CACHE[key] = network
    return network


def _persistent_actor_broker(config: HeterogeneousTrainingConfig, device_name: str):
    if not (config.training_cross_episode_batching and config.use_accelerated_orpd_rollouts):
        return None
    key = (
        str(device_name),
        float(config.training_batch_window_ms),
        int(config.training_max_cross_batch),
    )
    broker = _ACTOR_BROKER_CACHE.get(key)
    if broker is None:
        broker = CrossRunBatchBroker(
            batch_window_ms=float(config.training_batch_window_ms),
            max_candidates=int(config.training_max_cross_batch),
        )
        _ACTOR_BROKER_CACHE[key] = broker
    return broker


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
    network = (
        _persistent_actor_network(device, config.hidden_dim)
        if config.persistent_actor_workers
        else CALOPolicyNetwork(POLICY_STATE_DIM, config.hidden_dim).to(device)
    )
    network.load_state_dict(network_state)
    network.eval()
    actual_snapshot = _state_dict_sha256(_cpu_state_dict(network))
    if actual_snapshot != expected_snapshot:
        raise RuntimeError("Actor received a policy snapshot that does not match the learner.")

    broker = _persistent_actor_broker(config, device_name)
    if broker is not None:
        set_cross_run_broker(broker)
    environments = []
    for episode in episode_indices:
        _, environment = _environment_for_episode(
            config,
            epoch=epoch,
            stage=stage,
            episode=episode,
            compute_device=device_name,
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
            [environment.policy_state(config.horizon) for _, environment in environments],
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

        def _step_one(index_environment):
            index, (episode, environment) = index_environment
            reward = environment.step(
                int(regimes_cpu[index]),
                int(operators_cpu[index]),
                np.asarray(parameters_cpu[index], dtype=float),
                config.horizon,
            )
            return index, episode, reward

        if broker is not None and len(environments) > 1:
            with ThreadPoolExecutor(max_workers=len(environments)) as step_executor:
                step_results = list(step_executor.map(_step_one, enumerate(environments)))
        else:
            step_results = [_step_one(item) for item in enumerate(environments)]
        for index, episode, reward in step_results:
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
            _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)

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
    persistent_executor=None,
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
    results: list[dict[str, Any]] = []
    if persistent_executor is not None:
        futures = [
            persistent_executor.submit(collect_actor_lane_payload, payload) for payload in payloads
        ]
        for future in as_completed(futures):
            results.append(future.result())
        return results
    if len(payloads) == 1:
        return [collect_actor_lane_payload(payloads[0])]
    context = mp.get_context("spawn")
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
                    process.wait()
                process.communicate()
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
    plan_override: TrainingLanePlan | None = None,
    actor_clients: dict[str, PersistentTrainingActorClient] | None = None,
    cpu_executor=None,
):
    """Collect one synchronous weighted epoch from CUDA, XPU, and CPU actors in parallel."""
    plan = plan_override or plan_training_lanes(
        config.episodes_per_epoch,
        cuda_share=config.cuda_rollout_share,
        xpu_share=config.xpu_rollout_share,
        cpu_share=config.cpu_rollout_share,
    )
    actor_clients = actor_clients or {}
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
            if "cuda" in actor_clients:
                futures[executor.submit(actor_clients["cuda"].request, payload, None)] = "cuda"
            else:
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
            if "xpu" in actor_clients:
                futures[executor.submit(actor_clients["xpu"].request, payload, None)] = "xpu"
            else:
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
                    persistent_executor=cpu_executor,
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


def calibrate_training_actor_throughput(
    config: HeterogeneousTrainingConfig,
    network: nn.Module,
    *,
    actor_clients: dict[str, PersistentTrainingActorClient] | None = None,
    cpu_executor=None,
    progress_callback=None,
) -> dict[str, float]:
    """Measure complete actor transition throughput on each available lane.

    The probe uses the current policy snapshot and one or more complete episodes.  Probe
    transitions are discarded and never enter PPO's on-policy buffer.
    """
    actor_clients = actor_clients or {}
    base_plan = plan_training_lanes(
        max(1, int(config.actor_calibration_episodes)),
        cuda_share=config.cuda_rollout_share,
        xpu_share=config.xpu_rollout_share,
        cpu_share=config.cpu_rollout_share,
    )
    network_state = _cpu_state_dict(network)
    snapshot = _state_dict_sha256(network_state)
    stage = 4 if config.development_cases else 3
    throughputs: dict[str, float] = {}
    for lane in LANE_ORDER:
        if not base_plan.available_lanes.get(lane, False):
            continue
        device = base_plan.devices[lane]
        payload = {
            "config": asdict(config),
            "network_state": network_state,
            "epoch": 0,
            "stage": stage,
            "episode_indices": list(range(max(1, int(config.actor_calibration_episodes)))),
            "device": device,
            "lane": lane,
            "policy_snapshot_sha256": snapshot,
        }
        started = time.perf_counter()
        try:
            if lane in actor_clients:
                result = actor_clients[lane].request(payload)
                episodes = len(result.get("episodes", []))
            elif lane == "cpu":
                results = _collect_cpu_lane(
                    config,
                    network_state,
                    epoch=0,
                    stage=stage,
                    episode_indices=payload["episode_indices"],
                    policy_snapshot_sha256=snapshot,
                    persistent_executor=cpu_executor,
                )
                episodes = sum(len(item.get("episodes", [])) for item in results)
            else:
                result = _collect_accelerator_subprocess(
                    _xpu_interpreter_for_plan(base_plan) if lane == "xpu" else sys.executable,
                    payload,
                )
                episodes = len(result.get("episodes", []))
            seconds = max(time.perf_counter() - started, 1e-12)
            throughputs[lane] = float(episodes * config.horizon / seconds)
            if progress_callback:
                progress_callback(
                    0, f"Calibrated {lane.upper()} actor: {throughputs[lane]:,.1f} transitions/s"
                )
        except Exception as exc:
            throughputs[lane] = 0.0
            if progress_callback:
                progress_callback(0, f"{lane.upper()} actor calibration unavailable: {exc}")
    return throughputs


def _train_policy_heterogeneous_impl(
    config: HeterogeneousTrainingConfig,
    output_path,
    progress_callback=None,
    cancel_callback=None,
    *,
    epoch_observer=None,
    resume_extra_provider=None,
    cancel_during_rollout: bool = True,
    suppress_cancel_persistence: bool = False,
):
    """Train a candidate CALO policy with synchronous weighted heterogeneous actors."""
    final_benchmark_names = {"case118", "case300"}
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
            epoch_observer=epoch_observer,
            resume_extra_provider=resume_extra_provider,
            cancel_during_rollout=cancel_during_rollout,
            suppress_cancel_persistence=suppress_cancel_persistence,
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

    if int(config.rollout_workers) <= 0:
        config.rollout_workers = recommended_rollout_workers(config.episodes_per_epoch)
    else:
        config.rollout_workers = min(
            int(config.rollout_workers),
            max(1, int(config.episodes_per_epoch)),
        )

    network = CALOPolicyNetwork(POLICY_STATE_DIM, config.hidden_dim).to(learner_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    resume_path = training_resume_path(config, output_path)
    initial_policy = str(getattr(config, "initial_policy_checkpoint", "") or "").strip()
    if initial_policy and not resume_path.is_file():
        payload = load_checkpoint(initial_policy, map_location="cpu")
        architecture = dict(payload.get("architecture", {}) or {})
        input_dim = int(architecture.get("input_dim", POLICY_STATE_DIM) or POLICY_STATE_DIM)
        hidden_dim = int(architecture.get("hidden_dim", config.hidden_dim) or config.hidden_dim)
        if input_dim != POLICY_STATE_DIM or hidden_dim != int(config.hidden_dim):
            raise ValueError(
                "Fine-tune/fork checkpoint architecture is incompatible with native CALO policy training"
            )
        network.load_state_dict(payload.get("model_state_dict", payload))
    start_epoch = 0
    history: list[dict[str, Any]] = []
    historical_pretraining: dict[str, Any] = {}
    resume_extra: dict[str, Any] = {}
    if resume_path.is_file():
        start_epoch, history, historical_pretraining, resume_extra = load_training_resume(
            resume_path, network, optimizer, learner_device, rng, current_config=config
        )
        if progress_callback:
            progress_callback(
                int(100 * start_epoch / max(config.epochs, 1)),
                f"Resumed heterogeneous CALO policy training from completed epoch {start_epoch}/{config.epochs}",
            )
    else:
        historical_pretraining = _historical_pretrain(
            network,
            optimizer,
            learner_device,
            config,
            rng,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    target_epoch, training_mode = _resolve_training_target(config, start_epoch)
    stage_floor = _stage_floor_from_history(history, resume_extra)
    nominal_target = (
        target_epoch if target_epoch is not None else max(start_epoch + 1, int(config.epochs), 1)
    )
    total_units = nominal_target * config.episodes_per_epoch
    completed_units = start_epoch * config.episodes_per_epoch

    # v3.4 keeps one actor interpreter/process alive per device for the complete training
    # session.  CUDA/XPU contexts, policy modules, ORPD tensors and cross-episode batch
    # brokers are therefore reused instead of being reconstructed every epoch.
    base_plan = plan_training_lanes(
        config.episodes_per_epoch,
        cuda_share=config.cuda_rollout_share,
        xpu_share=config.xpu_rollout_share,
        cpu_share=config.cpu_rollout_share,
    )
    actor_clients: dict[str, PersistentTrainingActorClient] = {}
    cpu_executor = None
    measured_actor_throughput: dict[str, float] = dict(
        resume_extra.get("measured_actor_throughput", {})
    )

    def _current_resume_extra(**updates):
        extra = {
            "learner_device": str(learner_device),
            "measured_actor_throughput": dict(measured_actor_throughput),
            "training_mode": training_mode,
            "curriculum_encoding": "zero_based_0_4",
            "exact_resume": bool(resume_path.is_file() and start_epoch > 0),
        }
        if resume_extra_provider is not None:
            supplied = resume_extra_provider()
            if supplied:
                extra.update(dict(supplied))
        extra.update(updates)
        return extra

    def _notify_epoch(completed_epoch: int, stage_value: int, episode_returns_value=None, epoch_losses_value=None):
        if epoch_observer is None:
            return
        epoch_observer({
            "epoch": int(completed_epoch),
            "stage": int(stage_value),
            "network": network,
            "optimizer": optimizer,
            "rng": rng,
            "history": history,
            "historical_pretraining": historical_pretraining,
            "config": config,
            "device": learner_device,
            "rollout_workers": int(config.rollout_workers),
            "episode_returns": list(episode_returns_value or []),
            "epoch_losses": list(epoch_losses_value or []),
        })

    _notify_epoch(start_epoch, stage_floor, [], [])
    try:
        if config.persistent_actor_workers:
            if (
                base_plan.available_lanes.get("cuda", False)
                and base_plan.episode_counts.get("cuda", 0) > 0
            ):
                actor_clients["cuda"] = PersistentTrainingActorClient(
                    sys.executable,
                    base_plan.devices["cuda"],
                    "cuda",
                )
            if (
                base_plan.available_lanes.get("xpu", False)
                and base_plan.episode_counts.get("xpu", 0) > 0
            ):
                actor_clients["xpu"] = PersistentTrainingActorClient(
                    _xpu_interpreter_for_plan(base_plan),
                    base_plan.devices["xpu"],
                    "xpu",
                )
            if base_plan.episode_counts.get("cpu", 0) > 0:
                context = mp.get_context("spawn")
                cpu_executor = ProcessPoolExecutor(
                    max_workers=max(1, int(config.rollout_workers)),
                    mp_context=context,
                )

        if config.throughput_adaptive_rollouts and not measured_actor_throughput:
            if progress_callback:
                progress_callback(
                    0,
                    "Calibrating persistent CUDA/XPU/CPU policy actors; probe trajectories are discarded",
                )
            measured_actor_throughput = calibrate_training_actor_throughput(
                config,
                network,
                actor_clients=actor_clients,
                cpu_executor=cpu_executor,
                progress_callback=progress_callback,
            )

        epoch = start_epoch
        while target_epoch is None or epoch < target_epoch:
            cancel = cancel_callback() if cancel_callback else False
            if cancel:
                if isinstance(cancel, (int, float)) and not isinstance(cancel, bool) and cancel > epoch:
                    target_epoch = int(cancel)
                    if progress_callback:
                        progress_callback(
                            0, f"Rounding to epoch {target_epoch} before stop..."
                        )
                else:
                    if suppress_cancel_persistence:
                        raise TrainingCancelled(
                            f"CALO policy training stop requested after completed epoch {epoch}."
                        )
                    save_training_resume(
                        resume_path, network=network, optimizer=optimizer, next_epoch=epoch,
                        history=history, rng=rng, historical_pretraining=historical_pretraining,
                        config=config, extra=_current_resume_extra(safe_stop=True),
                    )
                    terminal = save_deployable_policy_snapshot(
                        output_path, network, config, history, historical_pretraining, epoch,
                        device=str(learner_device), rollout_workers=int(config.rollout_workers),
                    )
                    _write_policy_alias(output_path, terminal)
                    raise TrainingCancelled(
                        f"CALO policy training stopped safely after cumulative epoch {epoch}."
                    )
            proposed_stage = _curriculum_stage(
                epoch, max(nominal_target, epoch + 1), bool(config.development_cases)
            )
            stage = max(stage_floor, proposed_stage)
            stage_floor = max(stage_floor, stage)
            epoch_plan = (
                plan_training_lanes_from_throughput(
                    config.episodes_per_epoch,
                    measured_actor_throughput,
                    base_plan=base_plan,
                )
                if config.throughput_adaptive_rollouts and measured_actor_throughput
                else base_plan
            )

            def actor_progress(completed_in_epoch: int, detail: str) -> None:
                if progress_callback:
                    absolute = completed_units + completed_in_epoch
                    progress_callback(
                        (
                            int(100 * absolute / max(total_units, 1))
                            if target_epoch is not None
                            else 0
                        ),
                        f"Epoch {epoch + 1}/{target_epoch if target_epoch is not None else '∞'} · {detail}",
                    )

            try:
                rollout, episode_returns, plan, lane_records, snapshot = (
                    collect_weighted_epoch_rollouts(
                        config,
                        network,
                        epoch=epoch,
                        stage=stage,
                        progress_callback=actor_progress,
                        cancel_callback=(cancel_callback if cancel_during_rollout else None),
                        plan_override=epoch_plan,
                        actor_clients=actor_clients,
                        cpu_executor=cpu_executor,
                    )
                )
            except TrainingCancelled:
                if suppress_cancel_persistence:
                    raise
                save_training_resume(
                    resume_path, network=network, optimizer=optimizer, next_epoch=epoch,
                    history=history, rng=rng, historical_pretraining=historical_pretraining,
                    config=config, extra=_current_resume_extra(safe_stop=True),
                )
                terminal = save_deployable_policy_snapshot(
                    output_path, network, config, history, historical_pretraining, epoch,
                    device=str(learner_device), rollout_workers=int(config.rollout_workers),
                )
                _write_policy_alias(output_path, terminal)
                raise
            completed_units += config.episodes_per_epoch
            if progress_callback:
                progress_callback(
                    (
                        int(100 * completed_units / max(total_units, 1))
                        if target_epoch is not None
                        else 0
                    ),
                    f"Epoch {epoch + 1}/{target_epoch if target_epoch is not None else '∞'} · PPO update on {learner_device} · "
                    f"{len(rollout['state'])} fresh transitions · {plan.summary()}",
                )

            advantages, returns = _compute_gae(
                rollout["reward"],
                rollout["value"],
                rollout["done"],
                config.gamma,
                config.gae_lambda,
            )
            if len(advantages) == 0:
                epoch += 1
                continue
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            states = torch.as_tensor(
                np.asarray(rollout["state"]),
                dtype=torch.float32,
                device=learner_device,
            )
            regimes = torch.as_tensor(rollout["regime"], dtype=torch.long, device=learner_device)
            operators = torch.as_tensor(
                rollout["operator"], dtype=torch.long, device=learner_device
            )
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
                    clipped = (
                        torch.clamp(
                            ratio,
                            1.0 - config.clip_ratio,
                            1.0 + config.clip_ratio,
                        )
                        * advantages_t[batch_t]
                    )
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
                    "curriculum_stage": stage,
                    "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    "mean_episode_return": float(np.mean(episode_returns)),
                    "transitions": len(rollout["state"]),
                    "ppo_learner_device": str(learner_device),
                    "policy_snapshot_sha256": snapshot,
                    "requested_rollout_shares": plan.requested_shares,
                    "effective_rollout_shares": plan.effective_shares,
                    "episode_allocation": plan.episode_counts,
                    "actor_lanes": lane_records,
                    "measured_actor_transitions_per_second": dict(measured_actor_throughput),
                    "persistent_actor_workers": bool(config.persistent_actor_workers),
                    "cross_episode_batching": bool(config.training_cross_episode_batching),
                    "warnings": list(plan.warnings),
                }
            )
            completed_epoch = epoch + 1
            _notify_epoch(completed_epoch, stage, episode_returns, epoch_losses)
            checkpoint_interval = max(1, int(getattr(config, "checkpoint_interval_epochs", 1) or 1))
            if (
                bool(getattr(config, "checkpoint_each_epoch", True))
                and completed_epoch % checkpoint_interval == 0
            ):
                save_training_resume(
                    resume_path,
                    network=network,
                    optimizer=optimizer,
                    next_epoch=completed_epoch,
                    history=history,
                    rng=rng,
                    historical_pretraining=historical_pretraining,
                    config=config,
                    extra=_current_resume_extra(),
                )
            epoch += 1
            if target_epoch is None and int(getattr(config, "max_session_epochs", 0) or 0) > 0:
                if epoch - start_epoch >= int(config.max_session_epochs):
                    break
        # Always preserve an exact trusted resume point at the terminal safe boundary.
        save_training_resume(
            resume_path,
            network=network,
            optimizer=optimizer,
            next_epoch=epoch,
            history=history,
            rng=rng,
            historical_pretraining=historical_pretraining,
            config=config,
            extra=_current_resume_extra(completed_target=target_epoch),
        )
        terminal_snapshot = save_deployable_policy_snapshot(
            output_path,
            network,
            config,
            history,
            historical_pretraining,
            int(epoch),
            device=str(learner_device),
            rollout_workers=int(config.rollout_workers),
        )
    finally:
        if cpu_executor is not None:
            cpu_executor.shutdown(wait=True, cancel_futures=True)
        for client in actor_clients.values():
            try:
                client.close()
            except Exception:
                _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_info = available_training_devices()
    final_plan = (
        plan_training_lanes_from_throughput(
            config.episodes_per_epoch,
            measured_actor_throughput,
            base_plan=base_plan,
        )
        if config.throughput_adaptive_rollouts and measured_actor_throughput
        else base_plan
    )
    metadata = {
        "algorithm": "CALO",
        "calo_core": "v5.0",
        "policy_training_architecture": "v5.8",
        "training_method": "persistent auto-tuned batched heterogeneous PPO",
        "candidate_checkpoint": True,
        "benchmark_freeze_status": (
            "Not automatically part of the frozen TEST benchmark. Validate this candidate "
            "and create a new freeze manifest before benchmark use."
        ),
        "training_config": asdict(config),
        "training_seed": config.seed,
        "training_mode": training_mode,
        "cumulative_epoch": int(epoch),
        "policy_lineage_id": str(getattr(config, "policy_lineage_id", "") or ""),
        "policy_lineage_name": str(getattr(config, "policy_lineage_name", "") or ""),
        "policy_phase_index": int(getattr(config, "policy_phase_index", 0) or 0),
        "immutable_terminal_checkpoint": str(terminal_snapshot),
        "state_dimension": POLICY_STATE_DIM,
        "state_schema_version": POLICY_STATE_SCHEMA,
        "action_schema_version": POLICY_ACTION_SCHEMA,
        "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
        "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
        "execution": {
            "architecture": (
                "same-policy synchronous persistent CUDA/XPU/CPU actor lanes with cross-episode "
                "ORPD batching, measured-throughput allocation, and one centralized PPO learner update"
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
            "persistent_actor_workers": bool(config.persistent_actor_workers),
            "throughput_adaptive_rollouts": bool(config.throughput_adaptive_rollouts),
            "measured_actor_transitions_per_second": dict(measured_actor_throughput),
            "accelerated_orpd_rollouts": bool(config.use_accelerated_orpd_rollouts),
            "cross_episode_batching": bool(config.training_cross_episode_batching),
            "cross_episode_batch_window_ms": float(config.training_batch_window_ms),
            "maximum_cross_episode_candidate_batch": int(config.training_max_cross_batch),
            "training_tensor_batch_size": int(config.training_tensor_batch_size),
            "on_policy_synchronization": (
                "All actor lanes use one policy snapshot per epoch; PPO starts only after "
                "all matching trajectories arrive."
            ),
            "hardware_scope_note": (
                "Auto-tuned shares are based on measured complete actor transitions per second, "
                "not Task Manager utilization percentages. Explicit ORPD development rollouts use "
                "the FP64 accelerator evaluator on the actor device; synthetic curriculum stages "
                "still contain host-side environment logic."
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
    _write_policy_alias(output_path, terminal_snapshot)
    metadata["immutable_artifact_path"] = str(terminal_snapshot)
    metadata["immutable_terminal_checkpoint"] = str(terminal_snapshot)
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    if not bool(getattr(config, "keep_resume_after_completion", True)):
        try:
            resume_path.unlink(missing_ok=True)
            resume_path.with_suffix(resume_path.suffix + ".sha256").unlink(missing_ok=True)
        except OSError:
            pass
    return str(output_path), history

def train_policy_heterogeneous(
    config: HeterogeneousTrainingConfig,
    output_path,
    progress_callback=None,
    cancel_callback=None,
    *,
    epoch_observer=None,
    resume_extra_provider=None,
    cancel_during_rollout: bool = True,
    suppress_cancel_persistence: bool = False,
):
    """Isolated wrapper that prevents GUI-thread policy training from perturbing global RNG users."""
    caller_python = random.getstate()
    caller_numpy = np.random.get_state()
    caller_torch = torch.random.get_rng_state()
    caller_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        return _train_policy_heterogeneous_impl(
            config, output_path, progress_callback, cancel_callback,
            epoch_observer=epoch_observer,
            resume_extra_provider=resume_extra_provider,
            cancel_during_rollout=cancel_during_rollout,
            suppress_cancel_persistence=suppress_cancel_persistence,
        )
    finally:
        random.setstate(caller_python)
        np.random.set_state(caller_numpy)
        torch.random.set_rng_state(caller_torch)
        if torch.cuda.is_available() and caller_cuda:
            torch.cuda.set_rng_state_all(caller_cuda)

