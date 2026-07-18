"""Reproducible PPO training for the CALO Core v2 hierarchical policy.

The training environment uses the same CALO Core v2 operator implementations, epsilon-feasible
selection, dual archives, cognitive state builder, and mixed-variable moves as runtime. The
curriculum progresses through unconstrained, constrained, mixed-variable, and narrow-feasible
problems. Final publication benchmark systems are not used by this module unless a user explicitly
adds separate development systems to an external training workflow.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import multiprocessing as mp
import os
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from calo_rpd_studio.ai.model_io import load_trusted_resume, write_trusted_resume_hash
from torch import nn

from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import STATE_DIM, build_cognitive_state, population_diversity
from .environmental_selection import environmental_select, epsilon_better
from .learning_operators import (
    cognitive_teacher_learning,
    constraint_boundary_differential,
    diversity_recovery,
    feasible_elite_learning,
    mixed_variable_neighbourhood,
    success_distribution_memory,
)
from .operator_credit import OperatorCredit
from .policy_network import CALOPolicyNetwork
from .success_memory import SuccessMemory


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 24
    episodes_per_epoch: int = 12
    horizon: int = 28
    seed: int = 2026
    learning_rate: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 128
    hidden_dim: int = 96
    population_size: int = 20
    rollout_workers: int = 0
    ppo_device: str = "auto"
    cpu_threads_per_worker: int = 1
    development_cases: tuple[str, ...] = ()
    allow_final_benchmark_training: bool = False
    historical_repository: str = ""
    use_historical_trajectories: bool = False
    historical_pretraining_epochs: int = 4
    resume_checkpoint: str = ""
    checkpoint_each_epoch: bool = True
    resume_task_id: str = ""


class TrainingCancelled(RuntimeError):
    """Raised when the user requests a safe stop between training units."""


@dataclass(slots=True)
class SyntheticEvaluation:
    value: float
    feasible: bool
    violation: float
    metadata: dict


class _Variable:
    def __init__(self, discrete: bool, levels: int = 0) -> None:
        self.kind = SimpleNamespace(value="discrete" if discrete else "continuous")
        self.values = tuple(np.linspace(0.0, 1.0, levels)) if discrete and levels > 1 else ()


class CurriculumProblem:
    """Procedurally generated task with explicit constraint-component decomposition."""

    def __init__(self, rng: np.random.Generator, stage: int) -> None:
        self.stage = int(stage)
        self.dimension = int(rng.integers(8, 21))
        self.shift = rng.uniform(0.20, 0.80, self.dimension)
        self.rotation = rng.normal(size=(self.dimension, self.dimension))
        q, _ = np.linalg.qr(self.rotation)
        self.rotation = q
        discrete_fraction = 0.0 if stage < 2 else (0.25 if stage == 2 else 0.40)
        self.variables = [
            _Variable(rng.random() < discrete_fraction, int(rng.integers(5, 17)))
            for _ in range(self.dimension)
        ]
        self.decoder = SimpleNamespace(variables=self.variables)
        self.narrowness = 0.22 if stage < 3 else 0.08
        self.constraint_centres = rng.uniform(0.25, 0.75, (4, self.dimension))
        self.constraint_normals = rng.normal(size=(4, self.dimension))
        self.constraint_normals /= np.maximum(
            np.linalg.norm(self.constraint_normals, axis=1, keepdims=True), 1e-12
        )

    def _objective(self, x: np.ndarray) -> float:
        y = self.rotation @ (x - self.shift)
        rastrigin = 10 * self.dimension + np.sum(25 * y**2 - 10 * np.cos(2 * np.pi * 5 * y))
        bowl = np.sum((x - self.shift) ** 2)
        return float(0.35 * rastrigin / max(self.dimension, 1) + 0.65 * bowl)

    def evaluate(self, x) -> SyntheticEvaluation:
        x = np.clip(np.asarray(x, float), 0, 1)
        objective = self._objective(x)
        if self.stage == 0:
            components = {
                "bus_voltage": 0.0,
                "generator_q": 0.0,
                "generator_p": 0.0,
                "branch_thermal": 0.0,
                "power_flow": 0.0,
            }
        else:
            # Four differently oriented narrow half-space/slab constraints emulate distinct ORPD
            # feasibility mechanisms without using final benchmark systems.
            projections = np.sum((x - self.constraint_centres) * self.constraint_normals, axis=1)
            limits = np.asarray([
                self.narrowness,
                self.narrowness * 0.8,
                self.narrowness * 1.2,
                self.narrowness * 0.9,
            ])
            raw = np.maximum(np.abs(projections) - limits, 0.0)
            if self.stage >= 2:
                # Mixed-variable consistency pressure: decoded discrete coordinates should remain
                # close to an admissible lattice point.
                lattice_penalty = 0.0
                for value, variable in zip(x, self.variables):
                    if variable.values:
                        nearest = min(variable.values, key=lambda level: abs(level - value))
                        lattice_penalty += max(abs(value - nearest) - 0.035, 0.0)
                raw[1] += lattice_penalty / max(self.dimension, 1)
            components = {
                "bus_voltage": float(raw[0]),
                "generator_q": float(raw[1]),
                "generator_p": float(raw[2]),
                "branch_thermal": float(raw[3]),
                "power_flow": 0.0,
            }
        violation = float(sum(components.values()))
        return SyntheticEvaluation(
            value=objective,
            feasible=violation <= 1e-12,
            violation=violation,
            metadata={"constraint_components": components},
        )


class SyntheticCALOEnvironment:
    def __init__(self, rng: np.random.Generator, stage: int, population_size: int, problem=None) -> None:
        self.rng = rng
        self.problem = problem if problem is not None else CurriculumProblem(rng, stage)
        self.population_size = int(population_size)
        self.population = rng.random((self.population_size, self.problem.dimension))
        batch_evaluator = getattr(self.problem, "evaluate_population", None)
        self.evaluations = (
            list(batch_evaluator(self.population))
            if callable(batch_evaluator)
            else [self.problem.evaluate(x) for x in self.population]
        )
        self.feasible_archive = FeasibleEliteArchive(24)
        self.boundary_archive = ConstraintBoundaryArchive(36)
        self.feasible_archive.update(self.population, self.evaluations)
        self.boundary_archive.update(self.population, self.evaluations)
        self.memory = SuccessMemory(192, 0.97)
        self.credit = OperatorCredit(6, 0.90)
        self.previous_violation = float("inf")
        self.previous_objective = float("inf")
        self.constraint_stagnation = 0
        self.objective_stagnation = 0
        violations = [e.violation for e in self.evaluations]
        self.epsilon0 = float(np.quantile(violations, 0.75)) if any(violations) else 0.0
        self.step_index = 0

    @staticmethod
    def _diagnostics(evaluations):
        violations = np.asarray([e.violation for e in evaluations], dtype=float)
        feasible_values = [e.value for e in evaluations if e.feasible]
        return (
            float(np.min(violations)),
            min(feasible_values) if feasible_values else float("inf"),
            float(np.mean([e.feasible for e in evaluations])),
        )

    def state(self, horizon: int):
        epsilon = self.epsilon0 * max(0.0, 1.0 - self.step_index / max(0.7 * horizon, 1.0)) ** 2
        return build_cognitive_state(
            self.population,
            self.evaluations,
            epsilon=epsilon,
            previous_best_violation=self.previous_violation,
            previous_best_objective=self.previous_objective,
            constraint_stagnation=min(self.constraint_stagnation / 8.0, 1.0),
            objective_stagnation=min(self.objective_stagnation / 8.0, 1.0),
            remaining_budget=max(0.0, 1.0 - self.step_index / max(horizon, 1)),
            operator_credit=self.credit.probabilities(),
            feasible_archive_size=len(self.feasible_archive),
            feasible_archive_capacity=self.feasible_archive.capacity,
            boundary_archive_size=len(self.boundary_archive),
            boundary_archive_capacity=self.boundary_archive.capacity,
        )

    def step(self, regime: int, operator: int, raw_parameters: np.ndarray, horizon: int) -> float:
        low = np.asarray([0.15, 0.05, 0.005, 0.05, 0.05, 0.05])
        high = np.asarray([1.40, 0.95, 0.30, 1.00, 0.45, 0.45])
        params = low + np.asarray(raw_parameters, float) * (high - low)
        attraction, differential, sigma, memory_weight, diversity_weight, _ = params
        epsilon = self.epsilon0 * max(0.0, 1.0 - self.step_index / max(0.7 * horizon, 1.0)) ** 2
        old_violation, old_objective, old_feasible = self._diagnostics(self.evaluations)
        old_diversity = population_diversity(self.population)
        mean = self.population.mean(axis=0)
        quality = sorted(
            range(len(self.evaluations)),
            key=lambda i: (0 if self.evaluations[i].feasible else 1,
                           self.evaluations[i].value if self.evaluations[i].feasible else self.evaluations[i].violation),
        )
        best = self.population[quality[0]]
        offspring = []
        for index, x in enumerate(self.population):
            candidates = [i for i in range(len(self.population)) if i != index]
            r1_i, r2_i = self.rng.choice(candidates, size=2, replace=False)
            r1, r2 = self.population[int(r1_i)], self.population[int(r2_i)]
            feasible_teacher = self.feasible_archive.sample(self.rng, best)
            boundary_teacher = self.boundary_archive.sample(self.rng, best)
            if operator == 0:
                teacher = feasible_teacher if len(self.feasible_archive) else boundary_teacher
                candidate = feasible_elite_learning(x, teacher, r1, r2, self.rng, attraction, differential)
            elif operator == 1:
                candidate = constraint_boundary_differential(
                    x, boundary_teacher, r1, r2, self.rng, attraction, differential
                )
            elif operator == 2:
                teacher = feasible_teacher if regime >= 2 and len(self.feasible_archive) else boundary_teacher
                candidate = cognitive_teacher_learning(x, teacher, mean, self.rng, attraction, 0.35 * sigma)
            elif operator == 3:
                direction = self.memory.sample_direction(
                    self.problem.dimension, self.rng, prefer_feasibility=regime <= 1
                )
                candidate = success_distribution_memory(
                    x, x, direction, self.rng, 0.55, memory_weight
                )
            elif operator == 4:
                variables = getattr(self.problem, "variables", None)
                if variables is None:
                    variables = getattr(getattr(self.problem, "decoder", None), "variables", [])
                candidate = mixed_variable_neighbourhood(
                    x, variables, self.rng, max(0.35 * sigma, 0.006), 1
                )
            else:
                reference = boundary_teacher if regime <= 1 else feasible_teacher
                candidate = diversity_recovery(reference, self.population, self.rng, max(sigma, 0.06))
            offspring.append(candidate)
        offspring = np.asarray(offspring)
        batch_evaluator = getattr(self.problem, "evaluate_population", None)
        offspring_evaluations = (
            list(batch_evaluator(offspring))
            if callable(batch_evaluator)
            else [self.problem.evaluate(x) for x in offspring]
        )
        for index, (child, child_ev) in enumerate(zip(offspring, offspring_evaluations)):
            parent_ev = self.evaluations[index]
            successful = epsilon_better(child_ev, parent_ev, epsilon)
            feasibility_gain = max(parent_ev.violation - child_ev.violation, 0.0)
            objective_gain = (
                max((parent_ev.value - child_ev.value) / max(abs(parent_ev.value), 1.0), 0.0)
                if parent_ev.feasible and child_ev.feasible
                else 0.0
            )
            self.credit.update(operator, objective_gain + feasibility_gain, successful)
            if successful:
                self.memory.add(child - self.population[index], operator, objective_gain, feasibility_gain)
        combined_population = np.vstack([self.population, offspring])
        combined_evaluations = list(self.evaluations) + list(offspring_evaluations)
        self.population, self.evaluations = environmental_select(
            combined_population,
            combined_evaluations,
            self.population_size,
            epsilon,
            diversity_weight=float(diversity_weight),
        )
        self.feasible_archive.update(combined_population, combined_evaluations)
        self.boundary_archive.update(combined_population, combined_evaluations)
        new_violation, new_objective, new_feasible = self._diagnostics(self.evaluations)
        new_diversity = population_diversity(self.population)
        violation_gain = 0.0 if not np.isfinite(old_violation) else np.clip(
            (old_violation - new_violation) / max(abs(old_violation), 1e-12), -1, 1
        )
        objective_gain = 0.0
        if np.isfinite(old_objective) and np.isfinite(new_objective):
            objective_gain = np.clip(
                (old_objective - new_objective) / max(abs(old_objective), 1e-12), -1, 1
            )
        reward = float(
            1.2 * violation_gain
            + 0.85 * objective_gain
            + 0.75 * (new_feasible - old_feasible)
            + 0.10 * np.clip(new_diversity - old_diversity, -0.5, 0.5)
        )
        self.constraint_stagnation = 0 if new_violation < old_violation - 1e-12 else self.constraint_stagnation + 1
        if np.isfinite(new_objective):
            self.objective_stagnation = 0 if new_objective < old_objective - 1e-12 else self.objective_stagnation + 1
        self.previous_violation = new_violation
        self.previous_objective = new_objective
        self.step_index += 1
        return reward


def _curriculum_stage(epoch: int, epochs: int, has_development_cases: bool = False) -> int:
    fraction = epoch / max(epochs, 1)
    if fraction < 0.18:
        return 0  # continuous unconstrained
    if fraction < 0.40:
        return 1  # constrained continuous
    if fraction < 0.64:
        return 2  # mixed-variable constrained
    if not has_development_cases or fraction < 0.82:
        return 3  # narrow feasible region
    return 4      # explicitly configured ORPD development system


def _parameter_action_distribution(alpha, beta):
    return torch.distributions.Beta(alpha, beta)


def _compute_gae(rewards, values, dones, gamma: float, gae_lambda: float):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns



def recommended_rollout_workers(episodes_per_epoch: int | None = None) -> int:
    """Return a conservative CPU rollout-worker count that avoids oversubscription."""
    logical = max(1, os.cpu_count() or 1)
    reserve = 1 if logical <= 4 else max(1, logical // 8)
    workers = max(1, logical - reserve)
    if episodes_per_epoch is not None:
        workers = min(workers, max(1, int(episodes_per_epoch)))
    return workers


def available_training_devices() -> dict[str, str | bool]:
    """Describe accelerator availability without allocating large training tensors."""
    cuda = bool(torch.cuda.is_available())
    cuda_name = torch.cuda.get_device_name(0) if cuda else ""
    xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    xpu_name = ""
    if xpu:
        try:
            xpu_name = str(torch.xpu.get_device_properties(0).name)
        except Exception:
            xpu_name = "Intel XPU"
    try:
        from calo_rpd_studio.compute.resource_scheduler import configured_xpu_interpreter

        xpu_sidecar = bool(configured_xpu_interpreter())
    except Exception:
        xpu_sidecar = False
    recommended = "cuda" if cuda else ("xpu" if xpu else ("xpu_sidecar" if xpu_sidecar else "cpu"))
    return {
        "cuda_available": cuda,
        "cuda_name": cuda_name,
        "xpu_available": xpu,
        "xpu_name": xpu_name,
        "xpu_sidecar_available": xpu_sidecar,
        "recommended_device": recommended,
    }


def _resolve_training_device(requested: str) -> torch.device:
    choice = str(requested or "auto").strip().lower()
    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    if choice == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else ("xpu:0" if xpu_available else "cpu")
        )
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training was requested, but this PyTorch installation cannot access a CUDA GPU. "
                "Install a CUDA-enabled PyTorch build or select CPU/Auto."
            )
        return torch.device(choice if ":" in choice else "cuda:0")
    if choice.startswith("xpu"):
        if not xpu_available:
            raise RuntimeError(
                "Intel XPU training was requested, but this PyTorch runtime cannot access an XPU device. "
                "Use the verified secondary XPU runtime option or select CPU/Auto."
            )
        return torch.device(choice if ":" in choice else "xpu:0")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "xpu_sidecar":
        raise RuntimeError(
            "The secondary XPU runtime must be launched through train_policy_in_xpu_sidecar()."
        )
    raise ValueError(f"Unsupported CALO training device: {requested}")


def _cpu_state_dict(network: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in network.state_dict().items()}


def _merge_rollout(target: dict[str, list], source: dict[str, list]) -> None:
    for key in target:
        target[key].extend(source[key])


def _collect_rollout_chunk(payload):
    """Collect on-policy episodes in an isolated CPU process.

    Rollout processes intentionally keep tensors on CPU. The parent process owns the accelerator and
    performs PPO updates there, avoiding CUDA tensor sharing between worker processes on Windows.
    """
    config_dict, network_state, epoch, stage, episode_indices = payload
    config = TrainingConfig(**config_dict)
    torch.set_num_threads(max(1, int(config.cpu_threads_per_worker)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    network = CALOPolicyNetwork(STATE_DIM, config.hidden_dim).cpu()
    network.load_state_dict(network_state)
    network.eval()
    rollout = {key: [] for key in (
        "state", "regime", "operator", "parameter", "logp", "value", "reward", "done"
    )}
    episode_returns: list[float] = []
    for episode in episode_indices:
        episode_seed = int(config.seed + 1_000_003 * epoch + 10_007 * int(episode))
        random.seed(episode_seed)
        np.random.seed(episode_seed % (2**32 - 1))
        torch.manual_seed(episode_seed)
        rng = np.random.default_rng(episode_seed)
        if stage == 4:
            from calo_rpd_studio.orpd.problem import ORPDProblem
            from calo_rpd_studio.power_system.case_loader import CaseLoader

            source = config.development_cases[
                (epoch * config.episodes_per_epoch + int(episode)) % len(config.development_cases)
            ]
            development_problem = ORPDProblem(CaseLoader.load(source))
            environment = SyntheticCALOEnvironment(
                rng, stage, config.population_size, problem=development_problem
            )
        else:
            environment = SyntheticCALOEnvironment(rng, stage, config.population_size)
        episode_return = 0.0
        for step in range(config.horizon):
            state = environment.state(config.horizon).vector()
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            with torch.inference_mode():
                regime_logits, operator_logits, alpha, beta, value = network(state_tensor)
                regime_dist = torch.distributions.Categorical(logits=regime_logits)
                operator_dist = torch.distributions.Categorical(logits=operator_logits)
                parameter_dist = _parameter_action_distribution(alpha, beta)
                regime = regime_dist.sample()
                operator = operator_dist.sample()
                parameter = parameter_dist.sample()
                logp = (
                    regime_dist.log_prob(regime)
                    + operator_dist.log_prob(operator)
                    + parameter_dist.log_prob(parameter).sum()
                )
            reward = environment.step(
                int(regime.item()),
                int(operator.item()),
                parameter.cpu().numpy(),
                config.horizon,
            )
            done = step == config.horizon - 1
            rollout["state"].append(state)
            rollout["regime"].append(int(regime.item()))
            rollout["operator"].append(int(operator.item()))
            rollout["parameter"].append(parameter.cpu().numpy())
            rollout["logp"].append(float(logp.item()))
            rollout["value"].append(float(value.item()))
            rollout["reward"].append(float(reward))
            rollout["done"].append(bool(done))
            episode_return += reward
        episode_returns.append(float(episode_return))
    return rollout, episode_returns, list(episode_indices)


def _collect_epoch_rollouts(
    config: TrainingConfig,
    network: nn.Module,
    epoch: int,
    stage: int,
    workers: int,
    progress_callback=None,
    cancel_callback=None,
):
    rollout = {key: [] for key in (
        "state", "regime", "operator", "parameter", "logp", "value", "reward", "done"
    )}
    episode_returns: list[float] = []
    episode_indices = list(range(config.episodes_per_epoch))
    workers = max(1, min(int(workers), len(episode_indices)))
    config_dict = asdict(config)
    network_state = _cpu_state_dict(network)
    chunks = [chunk.tolist() for chunk in np.array_split(episode_indices, workers) if len(chunk)]

    if workers == 1:
        chunk_rollout, chunk_returns, completed = _collect_rollout_chunk(
            (config_dict, network_state, epoch, stage, chunks[0])
        )
        _merge_rollout(rollout, chunk_rollout)
        episode_returns.extend(chunk_returns)
        return rollout, episode_returns, completed

    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    futures = [
        executor.submit(
            _collect_rollout_chunk,
            (config_dict, network_state, epoch, stage, chunk),
        )
        for chunk in chunks
    ]
    completed_episodes: list[int] = []
    try:
        for future in as_completed(futures):
            if cancel_callback and cancel_callback():
                for pending in futures:
                    pending.cancel()
                raise TrainingCancelled("CALO policy training was cancelled safely.")
            chunk_rollout, chunk_returns, completed = future.result()
            _merge_rollout(rollout, chunk_rollout)
            episode_returns.extend(chunk_returns)
            completed_episodes.extend(completed)
            if progress_callback:
                progress_callback(len(completed_episodes), sorted(completed_episodes))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return rollout, episode_returns, completed_episodes


def _historical_pretrain(
    network: nn.Module,
    optimizer,
    device,
    config: TrainingConfig,
    rng: np.random.Generator,
    progress_callback=None,
    cancel_callback=None,
) -> dict:
    """Reward-weighted offline pretraining followed later by fresh on-policy PPO.

    Historical trajectories are never inserted into PPO's on-policy rollout buffer. They are used
    only for an explicit supervised pretraining stage, preventing old replay data from being
    misrepresented as current-policy PPO experience.
    """
    if not config.use_historical_trajectories or not str(config.historical_repository).strip():
        return {"enabled": False, "samples": 0, "epochs": 0, "mean_loss": None}

    from calo_rpd_studio.learning.experience_repository import load_experience_repository

    repository = load_experience_repository(config.historical_repository)
    records: list[dict] = []
    for trajectory in repository.policy_trajectories:
        transitions = list(trajectory.get("transitions") or [])
        returns = [0.0] * len(transitions)
        running = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            reward = float(transitions[index].get("reward", 0.0))
            running = reward + float(config.gamma) * running
            returns[index] = running
        for transition, return_value in zip(transitions, returns):
            state = np.asarray(transition.get("state") or [], dtype=float)
            parameter = np.asarray(transition.get("parameter") or [], dtype=float)
            regime = int(transition.get("regime", -1))
            operator = int(transition.get("operator", -1))
            if state.shape != (STATE_DIM,) or parameter.shape != (6,):
                continue
            if not (0 <= regime < 4 and 0 <= operator < 6):
                continue
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(parameter)):
                continue
            records.append(
                {
                    "state": np.clip(state, -1.0, 1.0),
                    "regime": regime,
                    "operator": operator,
                    "parameter": np.clip(parameter, 1e-5, 1 - 1e-5),
                    "reward": float(transition.get("reward", 0.0)),
                    "return": float(return_value),
                    "parameter_supervision": bool(transition.get("parameter_supervision", True)),
                    "quality_weight": float(np.clip(transition.get("quality_weight", 1.0), 0.05, 1.0)),
                }
            )

    if not records:
        return {"enabled": True, "samples": 0, "epochs": 0, "mean_loss": None}

    states = torch.as_tensor(np.asarray([r["state"] for r in records]), dtype=torch.float32, device=device)
    regimes = torch.as_tensor([r["regime"] for r in records], dtype=torch.long, device=device)
    operators = torch.as_tensor([r["operator"] for r in records], dtype=torch.long, device=device)
    parameters = torch.as_tensor(np.asarray([r["parameter"] for r in records]), dtype=torch.float32, device=device)
    returns = torch.as_tensor([r["return"] for r in records], dtype=torch.float32, device=device)
    parameter_supervision = torch.as_tensor(
        [1.0 if r["parameter_supervision"] else 0.0 for r in records],
        dtype=torch.float32,
        device=device,
    )
    quality_weights = np.asarray([r["quality_weight"] for r in records], dtype=float)
    rewards = np.asarray([r["reward"] for r in records], dtype=float)
    reward_z = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    # Keep every transition, but give successful and exactly recorded decisions more influence.
    weights = torch.as_tensor(
        quality_weights * (0.25 + 1.75 / (1.0 + np.exp(-reward_z))),
        dtype=torch.float32,
        device=device,
    )

    losses: list[float] = []
    indices = np.arange(len(records))
    epochs = max(0, int(config.historical_pretraining_epochs))
    network.train()
    for epoch in range(epochs):
        if cancel_callback and cancel_callback():
            raise TrainingCancelled("CALO policy training was cancelled safely.")
        rng.shuffle(indices)
        for start in range(0, len(indices), max(1, int(config.minibatch_size))):
            batch = indices[start : start + max(1, int(config.minibatch_size))]
            batch_t = torch.as_tensor(batch, dtype=torch.long, device=device)
            regime_logits, operator_logits, alpha, beta, values = network(states[batch_t])
            regime_loss = torch.nn.functional.cross_entropy(
                regime_logits, regimes[batch_t], reduction="none"
            )
            operator_loss = torch.nn.functional.cross_entropy(
                operator_logits, operators[batch_t], reduction="none"
            )
            parameter_mean = alpha / (alpha + beta)
            parameter_loss = (
                ((parameter_mean - parameters[batch_t]) ** 2).mean(dim=-1)
                * parameter_supervision[batch_t]
            )
            value_loss = (values - returns[batch_t]) ** 2
            sample_loss = regime_loss + operator_loss + parameter_loss + 0.25 * value_loss
            loss = (weights[batch_t] * sample_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if progress_callback:
            progress_callback(
                0,
                f"Historical policy pretraining · epoch {epoch + 1}/{epochs} · {len(records)} eligible transitions · device {device}",
            )
    network.eval()
    return {
        "enabled": True,
        "repository": str(Path(config.historical_repository).expanduser().resolve()),
        "repository_sha256": repository.payload.get("repository_sha256", ""),
        "samples": len(records),
        "exact_parameter_supervision_samples": int(sum(r["parameter_supervision"] for r in records)),
        "epochs": epochs,
        "method": "quality- and reward-weighted behavior/value pretraining before fresh on-policy PPO",
        "mean_loss": float(np.mean(losses)) if losses else None,
    }


def training_resume_path(config: TrainingConfig, output_path) -> Path:
    if str(getattr(config, "resume_checkpoint", "")).strip():
        return Path(str(config.resume_checkpoint)).expanduser()
    return Path(output_path).with_suffix(".resume.pt")


def _optimizer_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_resume(
    path: Path,
    *,
    network,
    optimizer,
    next_epoch: int,
    history: list,
    rng,
    historical_pretraining: dict,
    config,
    extra: dict | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "calo_policy_training_resume_v32",
        "next_epoch": int(next_epoch),
        "model_state_dict": _cpu_state_dict(network),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history),
        "historical_pretraining": dict(historical_pretraining or {}),
        "python_random_state": random.getstate(),
        "numpy_global_state": np.random.get_state(),
        "numpy_generator_state": rng.bit_generator.state,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "training_config": asdict(config),
        "extra": dict(extra or {}),
    }
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    temporary.replace(path)
    write_trusted_resume_hash(path)
    return path


def load_training_resume(path: Path, network, optimizer, device, rng) -> tuple[int, list, dict, dict]:
    payload = load_trusted_resume(path, map_location=device)
    if payload.get("format") != "calo_policy_training_resume_v32":
        raise ValueError("Unsupported CALO policy-training resume format")
    network.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_global_state"])
    rng.bit_generator.state = payload["numpy_generator_state"]
    torch.random.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return (
        int(payload.get("next_epoch", 0)),
        list(payload.get("history", [])),
        dict(payload.get("historical_pretraining", {})),
        dict(payload.get("extra", {})),
    )


def train_policy(config: TrainingConfig, output_path, progress_callback=None, cancel_callback=None):
    final_benchmark_names = {"case30", "case57", "case118"}
    development_names = {Path(item).stem.lower() for item in config.development_cases}
    leaked = sorted(final_benchmark_names & development_names)
    if leaked and not config.allow_final_benchmark_training:
        raise ValueError(
            "Final publication benchmark cases cannot be used for CALO policy training by default: "
            + ", ".join(leaked)
            + ". Supply separate development systems or explicitly enable the override in a documented non-final study."
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    device = _resolve_training_device(config.ppo_device)
    workers = (
        recommended_rollout_workers(config.episodes_per_epoch)
        if int(config.rollout_workers) <= 0
        else min(int(config.rollout_workers), max(1, config.episodes_per_epoch))
    )
    network = CALOPolicyNetwork(STATE_DIM, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    resume_path = training_resume_path(config, output_path)
    start_epoch = 0
    history = []
    historical_pretraining = {}
    if resume_path.is_file():
        start_epoch, history, historical_pretraining, _extra = load_training_resume(
            resume_path, network, optimizer, device, rng
        )
        if progress_callback:
            progress_callback(
                int(100 * start_epoch / max(config.epochs, 1)),
                f"Resumed CALO policy training from completed epoch {start_epoch}/{config.epochs}",
            )
    else:
        historical_pretraining = _historical_pretrain(
            network,
            optimizer,
            device,
            config,
            rng,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    total_units = config.epochs * config.episodes_per_epoch
    completed_units = start_epoch * config.episodes_per_epoch
    for epoch in range(start_epoch, config.epochs):
        if cancel_callback and cancel_callback():
            raise TrainingCancelled("CALO policy training was cancelled safely.")
        stage = _curriculum_stage(epoch, config.epochs, bool(config.development_cases))

        def epoch_progress(completed_in_epoch: int, _completed_indices) -> None:
            if progress_callback:
                absolute = completed_units + completed_in_epoch
                progress_callback(
                    int(100 * absolute / max(total_units, 1)),
                    f"Epoch {epoch + 1}/{config.epochs} · {completed_in_epoch}/{config.episodes_per_epoch} rollout episodes · "
                    f"{workers} CPU worker{'s' if workers != 1 else ''} · PPO device {device}",
                )

        rollout, episode_returns, completed = _collect_epoch_rollouts(
            config,
            network,
            epoch,
            stage,
            workers,
            progress_callback=epoch_progress,
            cancel_callback=cancel_callback,
        )
        completed_units += len(completed)
        if progress_callback:
            progress_callback(
                int(100 * completed_units / max(total_units, 1)),
                f"Epoch {epoch + 1}/{config.epochs} · PPO update on {device} · {len(rollout['state'])} transitions",
            )

        advantages, returns = _compute_gae(
            rollout["reward"], rollout["value"], rollout["done"], config.gamma, config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = torch.as_tensor(np.asarray(rollout["state"]), dtype=torch.float32, device=device)
        regimes = torch.as_tensor(rollout["regime"], dtype=torch.long, device=device)
        operators = torch.as_tensor(rollout["operator"], dtype=torch.long, device=device)
        parameters = torch.as_tensor(
            np.asarray(rollout["parameter"]), dtype=torch.float32, device=device
        )
        old_logp = torch.as_tensor(rollout["logp"], dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        epoch_losses = []
        indices = np.arange(len(states))
        network.train()
        for _ in range(config.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), config.minibatch_size):
                batch = indices[start : start + config.minibatch_size]
                batch_t = torch.as_tensor(batch, dtype=torch.long, device=device)
                regime_logits, operator_logits, alpha, beta, values = network(states[batch_t])
                regime_dist = torch.distributions.Categorical(logits=regime_logits)
                operator_dist = torch.distributions.Categorical(logits=operator_logits)
                parameter_dist = _parameter_action_distribution(alpha, beta)
                new_logp = (
                    regime_dist.log_prob(regimes[batch_t])
                    + operator_dist.log_prob(operators[batch_t])
                    + parameter_dist.log_prob(
                        parameters[batch_t].clamp(1e-5, 1 - 1e-5)
                    ).sum(-1)
                )
                ratio = torch.exp(new_logp - old_logp[batch_t])
                unclipped = ratio * advantages_t[batch_t]
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
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
                "rollout_workers": workers,
                "ppo_device": str(device),
                "transitions": len(rollout["state"]),
            }
        )
        if bool(getattr(config, "checkpoint_each_epoch", True)):
            save_training_resume(
                resume_path,
                network=network,
                optimizer=optimizer,
                next_epoch=epoch + 1,
                history=history,
                rng=rng,
                historical_pretraining=historical_pretraining,
                config=config,
                extra={"device": str(device), "rollout_workers": workers},
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_info = available_training_devices()
    metadata = {
        "algorithm": "CALO",
        "calo_core": "v2",
        "training_method": "PPO",
        "training_config": asdict(config),
        "training_seed": config.seed,
        "state_dimension": STATE_DIM,
        "execution": {
            "rollout_workers": workers,
            "ppo_device": str(device),
            "cuda_available": bool(device_info["cuda_available"]),
            "cuda_name": str(device_info["cuda_name"]),
            "xpu_available": bool(device_info["xpu_available"]),
            "xpu_name": str(device_info["xpu_name"]),
            "xpu_sidecar_available": bool(device_info["xpu_sidecar_available"]),
            "architecture": "parallel CPU rollout collection plus centralized PPO update on the selected accelerator",
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
            "eligible TRAIN experiments only; validation/test experiments excluded; old trajectories used only for offline pretraining, never as PPO on-policy rollouts"
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
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    try:
        resume_path.unlink(missing_ok=True)
        resume_path.with_suffix(resume_path.suffix + ".sha256").unlink(missing_ok=True)
    except OSError:
        pass
    return str(output_path), history
