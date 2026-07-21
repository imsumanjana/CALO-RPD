"""Reproducible PPO training for the CALO v4.1 hierarchical policy.

The training environment uses the CALO v4.1 cognitive state/action schema and reuses the same core
operator, persistent-memory, HPEM, contextual-credit, adaptive-epsilon, dual-lane, and mixed-variable
components used by runtime. It is a lightweight rollout environment rather than a bit-identical copy
of the complete runtime transition loop; Policy Qualification on the real optimizer is therefore the
mandatory promotion gate. The
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

from calo_rpd_studio.ai.model_io import (
    load_trusted_resume,
    write_trusted_resume_hash,
    load_checkpoint,
)
from torch import nn

from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import (
    STATE_DIM,
    build_cognitive_state,
    population_diversity,
    rule_based_regime_prior,
)
from .environmental_selection import environmental_select, epsilon_better
from .learning_operators import (
    cognitive_teacher_learning,
    constraint_boundary_differential,
    diversity_recovery,
    feasible_elite_learning,
    mixed_variable_neighbourhood,
    success_distribution_memory,
)
from .contextual_credit import ContextualCredit, classify_contexts
from .hierarchical_memory import HierarchicalPrefixEliteMemory
from .variable_intelligence import VariableGroupIntelligence
from .dual_lane_controller import DualLaneController
from .precision_engine import CognitivePrecisionEngine
from .adaptive_epsilon import AdaptiveEpsilonController
from .ai_controller import PARAMETER_LOW, PARAMETER_HIGH
from .policy_schema import (
    POLICY_STATE_DIM,
    POLICY_STATE_SCHEMA,
    POLICY_ACTION_SCHEMA,
    CALO_RUNTIME_ARCHITECTURE,
    TRAINING_ENVIRONMENT_VERSION,
    PolicyRuntimeContext,
    build_policy_vector,
    variable_group_concentration,
)
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
    initial_policy_checkpoint: str = (
        ""  # weights-only fine-tune/fork start; not an exact optimizer resume
    )
    # v5 continuation semantics. ``epochs`` remains the requested target/additional amount for
    # backward compatibility; ``training_mode`` defines how it is interpreted.
    training_mode: str = "cumulative"  # cumulative | additional | indefinite
    checkpoint_interval_epochs: int = 1
    deployable_checkpoint_interval_epochs: int = 1000
    qualification_interval_epochs: int = 10000
    policy_lineage_id: str = ""
    policy_lineage_name: str = ""
    policy_phase_index: int = 1
    keep_resume_after_completion: bool = True
    # Primarily for controlled tests/automation. Zero means no session cap in indefinite mode.
    max_session_epochs: int = 0


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
            limits = np.asarray(
                [
                    self.narrowness,
                    self.narrowness * 0.8,
                    self.narrowness * 1.2,
                    self.narrowness * 0.9,
                ]
            )
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
    """Compact v4.1 training environment sharing runtime cognition semantics.

    The environment remains deliberately lightweight enough for PPO rollouts, but it uses the same
    persistent personal memory, HPEM, contextual credit, variable-group intelligence, adaptive
    epsilon, dual-lane readiness, and recovery semantics exposed to the v4.1 runtime policy.
    """

    def __init__(
        self, rng: np.random.Generator, stage: int, population_size: int, problem=None
    ) -> None:
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
        self.personal_best = self.population.copy()
        self.personal_best_evaluations = list(self.evaluations)
        self.feasible_archive = FeasibleEliteArchive(24)
        self.boundary_archive = ConstraintBoundaryArchive(36)
        self.feasible_archive.update(self.population, self.evaluations)
        self.boundary_archive.update(self.population, self.evaluations)
        variables = getattr(getattr(self.problem, "decoder", None), "variables", None) or getattr(
            self.problem, "variables", []
        )
        self.hpem = HierarchicalPrefixEliteMemory(self.problem.dimension, variables=variables)
        self.hpem.update(self.population, self.evaluations)
        self.memory = SuccessMemory(192, 0.97, n_operators=7)
        self.credit = ContextualCredit(4, 6, 4, 4, decay=0.90, floor=0.02)
        self.group_intelligence = VariableGroupIntelligence(variables, decay=0.90)
        self.lane_controller = DualLaneController(max_learning=0.92)
        self.precision = CognitivePrecisionEngine(
            initial_radius=0.04, min_radius=5e-4, max_radius=0.15
        )
        self.previous_violation = float("inf")
        self.previous_objective = float("inf")
        self.constraint_stagnation = 0
        self.objective_stagnation = 0
        violations = [
            float(e.violation) for e in self.evaluations if np.isfinite(float(e.violation))
        ]
        epsilon0 = float(np.quantile(violations, 0.75)) if violations else 0.0
        self.epsilon_controller = AdaptiveEpsilonController(epsilon0, 0.65, 2.0)
        self.step_index = 0
        self._last_cognitive = None
        self._last_context = PolicyRuntimeContext()

    @staticmethod
    def _diagnostics(evaluations):
        violations = np.asarray([e.violation for e in evaluations], dtype=float)
        feasible_values = [e.value for e in evaluations if e.feasible]
        return (
            float(np.min(violations)),
            min(feasible_values) if feasible_values else float("inf"),
            float(np.mean([e.feasible for e in evaluations])),
        )

    def _epsilon(self, horizon: int) -> float:
        best_violation, _best_obj, feasible_ratio = self._diagnostics(self.evaluations)
        improving = best_violation < self.previous_violation - 1e-12
        return self.epsilon_controller.value(
            self.step_index,
            max(int(horizon), 1),
            feasible_ratio,
            improving,
            min(self.constraint_stagnation / 8.0, 1.0),
        )

    def state(self, horizon: int):
        epsilon = self._epsilon(horizon)
        cognitive = build_cognitive_state(
            self.population,
            self.evaluations,
            epsilon=epsilon,
            previous_best_violation=self.previous_violation,
            previous_best_objective=self.previous_objective,
            constraint_stagnation=min(self.constraint_stagnation / 8.0, 1.0),
            objective_stagnation=min(self.objective_stagnation / 8.0, 1.0),
            remaining_budget=max(0.0, 1.0 - self.step_index / max(horizon, 1)),
            operator_credit=self.credit.global_operator_probabilities(),
            feasible_archive_size=len(self.feasible_archive),
            feasible_archive_capacity=self.feasible_archive.capacity,
            boundary_archive_size=len(self.boundary_archive),
            boundary_archive_capacity=self.boundary_archive.capacity,
        )
        _best_violation, _best_obj, feasible_ratio = self._diagnostics(self.evaluations)
        diversity = population_diversity(self.population)
        consensus = self.hpem.consensus(self.population.mean(axis=0)) if len(self.hpem) else 0.0
        readiness = self.lane_controller.memory_readiness(
            feasible_ratio,
            self.hpem.occupancy,
            self.memory.density,
            min(self.step_index / 6.0, 1.0),
            consensus,
        )
        progress = self.step_index / max(horizon, 1)
        severe = max(self.constraint_stagnation, self.objective_stagnation) >= 8
        learning_fraction = self.lane_controller.learning_fraction(
            readiness, progress, diversity, severe
        )
        precision_active = self.precision.active(
            feasible_ratio,
            min(self.objective_stagnation / 8.0, 1.0),
            progress,
            len(self.hpem),
        )
        provisional_regime = int(np.argmax(rule_based_regime_prior(cognitive)))
        context = PolicyRuntimeContext(
            hpem_occupancy=float(self.hpem.occupancy),
            memory_consensus=float(consensus),
            memory_readiness=float(readiness),
            success_memory_density=float(self.memory.density),
            learning_lane_fraction=float(learning_fraction),
            precision_active=float(bool(precision_active)),
            precision_radius=float(
                np.clip(self.precision.radius / max(self.precision.max_radius, 1e-12), 0.0, 1.0)
            ),
            variable_group_concentration=variable_group_concentration(
                self.group_intelligence.probabilities(provisional_regime)
            ),
        )
        self._last_cognitive = cognitive
        self._last_context = context
        return cognitive

    def policy_state(self, horizon: int) -> np.ndarray:
        cognitive = self.state(horizon)
        return build_policy_vector(cognitive, self._last_context, input_dim=POLICY_STATE_DIM)

    def step(self, regime: int, operator: int, raw_parameters: np.ndarray, horizon: int) -> float:
        low = PARAMETER_LOW
        high = PARAMETER_HIGH
        params = low + np.asarray(raw_parameters, float) * (high - low)
        attraction, differential, sigma, memory_weight, diversity_weight, recovery_fraction = params
        epsilon = self._epsilon(horizon)
        old_violation, old_objective, old_feasible = self._diagnostics(self.evaluations)
        old_diversity = population_diversity(self.population)
        mean = self.population.mean(axis=0)
        quality = sorted(
            range(len(self.evaluations)),
            key=lambda i: (
                0 if self.evaluations[i].feasible else 1,
                self.evaluations[i].value
                if self.evaluations[i].feasible
                else self.evaluations[i].violation,
            ),
        )
        best = self.population[quality[0]]
        consensus = self.hpem.consensus(mean) if len(self.hpem) else 0.0
        readiness = self.lane_controller.memory_readiness(
            old_feasible,
            self.hpem.occupancy,
            self.memory.density,
            min(self.step_index / 6.0, 1.0),
            consensus,
        )
        progress = self.step_index / max(horizon, 1)
        severe = max(self.constraint_stagnation, self.objective_stagnation) >= 8
        learning_fraction = self.lane_controller.learning_fraction(
            readiness, progress, old_diversity, severe
        )
        learned_lanes = self.lane_controller.assign(
            self.population_size, learning_fraction, self.rng, False
        )
        contexts = classify_contexts(
            self.population, self.evaluations, old_violation < self.previous_violation - 1e-12
        )
        group_probs = self.group_intelligence.probabilities(regime)
        groups = self.rng.choice(len(group_probs), size=self.population_size, p=group_probs)
        variables = getattr(self.problem, "variables", None)
        if variables is None:
            variables = getattr(getattr(self.problem, "decoder", None), "variables", [])
        offspring = []
        assigned_memory = np.zeros(self.population_size, dtype=int)
        for index, x in enumerate(self.population):
            candidates = np.delete(np.arange(self.population_size), index)
            if candidates.size >= 2:
                r1_i, r2_i = self.rng.choice(candidates, size=2, replace=False)
                r1, r2 = self.population[int(r1_i)], self.population[int(r2_i)]
            else:
                r1 = r2 = x
            feasible_teacher = self.feasible_archive.sample(self.rng, best)
            boundary_teacher = self.boundary_archive.sample(self.rng, best)
            mem_probs = self.credit.memory_probabilities(regime, int(contexts[index]))
            level = int(self.rng.choice(4, p=mem_probs)) if len(self.hpem) else 0
            assigned_memory[index] = level
            memory_teacher = (
                self.hpem.summary(level, feasible_teacher) if len(self.hpem) else feasible_teacher
            )
            teacher = (
                memory_teacher
                if learned_lanes[index]
                else (feasible_teacher if len(self.feasible_archive) else boundary_teacher)
            )
            if operator == 0:
                candidate = feasible_elite_learning(
                    x, teacher, r1, r2, self.rng, attraction, differential
                )
            elif operator == 1:
                candidate = constraint_boundary_differential(
                    x, boundary_teacher, r1, r2, self.rng, attraction, differential
                )
            elif operator == 2:
                candidate = cognitive_teacher_learning(
                    x, teacher, mean, self.rng, attraction, 0.35 * sigma
                )
            elif operator == 3:
                direction = self.memory.sample_direction(
                    self.problem.dimension,
                    self.rng,
                    prefer_feasibility=regime <= 1,
                    regime=regime,
                    context=int(contexts[index]),
                    group=int(groups[index]),
                )
                candidate = success_distribution_memory(
                    x, self.personal_best[index], direction, self.rng, 0.55, memory_weight
                )
            elif operator == 4:
                candidate = mixed_variable_neighbourhood(
                    x, variables, self.rng, max(0.35 * sigma, 0.006), 1
                )
            else:
                reference = (
                    boundary_teacher
                    if regime <= 1
                    else (
                        self.hpem.summary(3, feasible_teacher)
                        if len(self.hpem)
                        else feasible_teacher
                    )
                )
                candidate = diversity_recovery(
                    reference, self.population, self.rng, max(sigma, 0.06)
                )
            mask = self.group_intelligence.mask(int(groups[index]), self.problem.dimension)
            if operator != 5 and np.any(mask):
                focused = x.copy()
                focused[mask] = candidate[mask]
                candidate = focused
            offspring.append(np.clip(candidate, 0.0, 1.0))
        offspring = np.asarray(offspring)

        # Make recovery_fraction operational under the same stagnation/diversity condition as runtime.
        if severe and old_diversity < 0.06:
            count = max(
                1,
                min(
                    self.population_size - 1,
                    int(
                        round(self.population_size * float(np.clip(recovery_fraction, 0.05, 0.45)))
                    ),
                ),
            )
            worst = quality[-count:]
            reference = self.hpem.summary(3, best) if len(self.hpem) else best
            for index in worst:
                offspring[index] = diversity_recovery(
                    reference, self.population, self.rng, max(sigma, 0.06)
                )

        batch_evaluator = getattr(self.problem, "evaluate_population", None)
        offspring_evaluations = (
            list(batch_evaluator(offspring))
            if callable(batch_evaluator)
            else [self.problem.evaluate(x) for x in offspring]
        )
        successful = np.zeros(self.population_size, dtype=bool)
        objective_gains = np.zeros(self.population_size)
        feasibility_gains = np.zeros(self.population_size)
        transitions = np.zeros(self.population_size)
        step_norms = np.linalg.norm(offspring - self.population, axis=1) / max(
            np.sqrt(self.problem.dimension), 1.0
        )
        for index, (child, child_ev) in enumerate(zip(offspring, offspring_evaluations)):
            parent_ev = self.evaluations[index]
            ok = epsilon_better(child_ev, parent_ev, epsilon)
            successful[index] = ok
            feasibility_gain = max(parent_ev.violation - child_ev.violation, 0.0)
            objective_gain = (
                max((parent_ev.value - child_ev.value) / max(abs(parent_ev.value), 1.0), 0.0)
                if parent_ev.feasible and child_ev.feasible
                else 0.0
            )
            objective_gains[index] = objective_gain
            feasibility_gains[index] = feasibility_gain
            transitions[index] = float((not parent_ev.feasible) and child_ev.feasible)
            if ok:
                self.memory.add(
                    child - self.population[index],
                    operator,
                    objective_gain,
                    feasibility_gain,
                    regime=regime,
                    context=int(contexts[index]),
                    group=int(groups[index]),
                )
                if epsilon_better(child_ev, self.personal_best_evaluations[index], 0.0):
                    self.personal_best[index] = child
                    self.personal_best_evaluations[index] = child_ev
        self.credit.batch_update(
            regime=np.full(self.population_size, regime),
            contexts=contexts,
            operators=np.full(self.population_size, operator),
            memory_levels=assigned_memory,
            successful=successful,
            objective_gain=objective_gains,
            feasibility_gain=feasibility_gains,
            feasibility_transition=transitions,
        )
        self.group_intelligence.batch_update(
            np.full(self.population_size, regime),
            groups,
            successful,
            objective_gains,
            feasibility_gains,
            step_norms,
        )
        combined_population = np.vstack([self.population, offspring])
        combined_evaluations = list(self.evaluations) + list(offspring_evaluations)
        # Preserve lineage-aware personal memory across environmental selection.
        selected_population, selected_evaluations = environmental_select(
            combined_population,
            combined_evaluations,
            self.population_size,
            epsilon,
            diversity_weight=float(diversity_weight),
        )
        # Recover selected source indices without changing scientific selection semantics.
        used = set()
        selected_indices = []
        for vec, ev in zip(selected_population, selected_evaluations):
            matches = [
                i
                for i, (src, sev) in enumerate(zip(combined_population, combined_evaluations))
                if i not in used and np.array_equal(src, vec) and sev is ev
            ]
            if not matches:
                matches = [
                    i
                    for i, src in enumerate(combined_population)
                    if i not in used and np.array_equal(src, vec)
                ]
            idx = matches[0] if matches else 0
            used.add(idx)
            selected_indices.append(idx)
        parent_pb = self.personal_best.copy()
        parent_pb_ev = list(self.personal_best_evaluations)
        combined_pb = np.vstack([parent_pb, parent_pb])
        combined_pb_ev = parent_pb_ev + parent_pb_ev
        for i in range(self.population_size):
            if successful[i] and epsilon_better(offspring_evaluations[i], parent_pb_ev[i], 0.0):
                combined_pb[self.population_size + i] = offspring[i]
                combined_pb_ev[self.population_size + i] = offspring_evaluations[i]
        self.population = np.asarray(selected_population)
        self.evaluations = list(selected_evaluations)
        self.personal_best = combined_pb[np.asarray(selected_indices)].copy()
        self.personal_best_evaluations = [combined_pb_ev[i] for i in selected_indices]
        self.feasible_archive.update(combined_population, combined_evaluations)
        self.boundary_archive.update(combined_population, combined_evaluations)
        self.hpem.update(combined_population, combined_evaluations)
        new_violation, new_objective, new_feasible = self._diagnostics(self.evaluations)
        new_diversity = population_diversity(self.population)
        violation_gain = (
            0.0
            if not np.isfinite(old_violation)
            else np.clip((old_violation - new_violation) / max(abs(old_violation), 1e-12), -1, 1)
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
        self.constraint_stagnation = (
            0 if new_violation < old_violation - 1e-12 else self.constraint_stagnation + 1
        )
        if np.isfinite(new_objective):
            self.objective_stagnation = (
                0 if new_objective < old_objective - 1e-12 else self.objective_stagnation + 1
            )
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
    return 4  # explicitly configured ORPD development system


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
    network = CALOPolicyNetwork(POLICY_STATE_DIM, config.hidden_dim).cpu()
    network.load_state_dict(network_state)
    network.eval()
    rollout = {
        key: []
        for key in ("state", "regime", "operator", "parameter", "logp", "value", "reward", "done")
    }
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
            state = environment.policy_state(config.horizon)
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
    rollout = {
        key: []
        for key in ("state", "regime", "operator", "parameter", "logp", "value", "reward", "done")
    }
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
    chunk_results = []
    try:
        for future in as_completed(futures):
            if cancel_callback and cancel_callback():
                for pending in futures:
                    pending.cancel()
                raise TrainingCancelled("CALO policy training was cancelled safely.")
            chunk_rollout, chunk_returns, completed = future.result()
            chunk_results.append((list(completed), chunk_rollout, list(chunk_returns)))
            completed_episodes.extend(completed)
            if progress_callback:
                progress_callback(len(completed_episodes), sorted(completed_episodes))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    # Merge in deterministic episode order, not worker-completion order. This makes changing the
    # number/speed of rollout workers an execution concern rather than a hidden training semantic.
    for completed, chunk_rollout, chunk_returns in sorted(
        chunk_results, key=lambda row: min(row[0]) if row[0] else 10**18
    ):
        _merge_rollout(rollout, chunk_rollout)
        episode_returns.extend(chunk_returns)
    return rollout, episode_returns, sorted(completed_episodes)


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
            if state.shape not in {(STATE_DIM,), (POLICY_STATE_DIM,)} or parameter.shape != (6,):
                continue
            if state.shape == (STATE_DIM,):
                state = np.concatenate((state, np.zeros(POLICY_STATE_DIM - STATE_DIM, dtype=float)))
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
                    "quality_weight": float(
                        np.clip(transition.get("quality_weight", 1.0), 0.05, 1.0)
                    ),
                }
            )

    if not records:
        return {"enabled": True, "samples": 0, "epochs": 0, "mean_loss": None}

    states = torch.as_tensor(
        np.asarray([r["state"] for r in records]), dtype=torch.float32, device=device
    )
    regimes = torch.as_tensor([r["regime"] for r in records], dtype=torch.long, device=device)
    operators = torch.as_tensor([r["operator"] for r in records], dtype=torch.long, device=device)
    parameters = torch.as_tensor(
        np.asarray([r["parameter"] for r in records]), dtype=torch.float32, device=device
    )
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
            parameter_loss = ((parameter_mean - parameters[batch_t]) ** 2).mean(
                dim=-1
            ) * parameter_supervision[batch_t]
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
        "exact_parameter_supervision_samples": int(
            sum(r["parameter_supervision"] for r in records)
        ),
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


_EXACT_RESUME_MUTABLE_FIELDS = {
    # Continuation target/checkpoint controls may change without changing the learned process.
    "epochs",
    "training_mode",
    "checkpoint_interval_epochs",
    "deployable_checkpoint_interval_epochs",
    "qualification_interval_epochs",
    "keep_resume_after_completion",
    "max_session_epochs",
    "resume_checkpoint",
    "resume_task_id",
    "initial_policy_checkpoint",
    # Execution placement may change between sessions. This is recorded in provenance; exact
    # numerical bit-replay across different hardware is not claimed.
    "ppo_device",
    "rollout_workers",
    "cpu_threads_per_worker",
}


def _normalize_config_value(value):
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_config_value(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _normalize_config_value(v) for k, v in sorted(value.items())}
    return value


def _validate_exact_resume_config(stored: dict, current: TrainingConfig) -> None:
    """Reject silent scientific hyperparameter drift during an exact training continuation."""
    now = asdict(current)
    mismatches = []
    for key, old_value in stored.items():
        if key in _EXACT_RESUME_MUTABLE_FIELDS or key not in now:
            continue
        if _normalize_config_value(old_value) != _normalize_config_value(now[key]):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "Exact policy-training resume is blocked because scientific/training hyperparameters changed: "
            + ", ".join(sorted(mismatches))
            + ". Use Continue/Fine-Tune (weights-only new phase) to intentionally change training semantics."
        )


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
        "format": "calo_policy_training_resume_v5",
        "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
        "state_schema_version": POLICY_STATE_SCHEMA,
        "action_schema_version": POLICY_ACTION_SCHEMA,
        "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
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


def load_training_resume(
    path: Path, network, optimizer, device, rng, current_config: TrainingConfig | None = None
) -> tuple[int, list, dict, dict]:
    payload = load_trusted_resume(path, map_location=device)
    resume_format = str(payload.get("format", ""))
    if resume_format not in {"calo_policy_training_resume_v5", "calo_policy_training_resume_v41"}:
        if resume_format == "calo_policy_training_resume_v32":
            raise ValueError(
                "This resume checkpoint belongs to the legacy 24-feature CALO policy architecture and "
                "cannot be resumed exactly by the native 32-feature v4.1 trainer. Finish it with the "
                "matching legacy release or start a new v4.1 training candidate from a documented workflow."
            )
        raise ValueError("Unsupported CALO policy-training resume format")
    if str(payload.get("state_schema_version", "")) != POLICY_STATE_SCHEMA:
        raise ValueError("CALO policy-training resume state schema is incompatible with v4.1")
    if str(payload.get("action_schema_version", "")) != POLICY_ACTION_SCHEMA:
        raise ValueError("CALO policy-training resume action schema is incompatible with v4.1")
    if current_config is not None:
        _validate_exact_resume_config(
            dict(payload.get("training_config", {}) or {}), current_config
        )
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


def _resolve_training_target(config: TrainingConfig, start_epoch: int) -> tuple[int | None, str]:
    mode = str(getattr(config, "training_mode", "cumulative") or "cumulative").strip().lower()
    if mode not in {"cumulative", "additional", "indefinite"}:
        raise ValueError(f"Unsupported CALO policy training mode: {mode}")
    if mode == "indefinite":
        cap = int(getattr(config, "max_session_epochs", 0) or 0)
        return ((start_epoch + cap) if cap > 0 else None), mode
    requested = max(1, int(config.epochs))
    if mode == "additional":
        return start_epoch + requested, mode
    if requested < start_epoch:
        raise ValueError(
            f"Cumulative target {requested} is below already completed epoch {start_epoch}. "
            "Choose additional or indefinite training, or increase the cumulative target."
        )
    return requested, mode


def _deployable_policy_payload(
    network,
    config: TrainingConfig,
    history: list,
    historical_pretraining: dict,
    cumulative_epoch: int,
    *,
    device: str,
    rollout_workers: int,
) -> dict:
    return {
        "model_state_dict": _cpu_state_dict(network),
        "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": config.hidden_dim},
        "metadata": {
            "algorithm": "CALO",
            "calo_core": "v5.0",
            "training_method": "PPO",
            "training_config": asdict(config),
            "training_seed": config.seed,
            "cumulative_epoch": int(cumulative_epoch),
            "policy_lineage_id": str(getattr(config, "policy_lineage_id", "")),
            "policy_lineage_name": str(getattr(config, "policy_lineage_name", "")),
            "policy_phase_index": int(getattr(config, "policy_phase_index", 1)),
            "state_dimension": POLICY_STATE_DIM,
            "state_schema_version": POLICY_STATE_SCHEMA,
            "action_schema_version": POLICY_ACTION_SCHEMA,
            "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
            "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
            "execution": {"rollout_workers": int(rollout_workers), "ppo_device": str(device)},
            "development_cases": list(config.development_cases),
            "historical_pretraining": historical_pretraining,
            "history": list(history),
            "checkpoint_role": "usable_policy_snapshot",
        },
    }


def save_deployable_policy_snapshot(
    output_path,
    network,
    config: TrainingConfig,
    history: list,
    historical_pretraining: dict,
    cumulative_epoch: int,
    *,
    device: str,
    rollout_workers: int,
) -> Path:
    """Write a policy artifact that is immediately usable for evaluation/qualification."""
    from calo_rpd_studio.ai.checkpoint_manager import CheckpointManager

    output_path = Path(output_path)
    lineage_dir = output_path.parent / f"{output_path.stem}_lineage"
    manager = CheckpointManager(lineage_dir)
    snapshot = lineage_dir / f"epoch_{int(cumulative_epoch):012d}.pt"
    # A deployable lineage checkpoint is immutable evidence. Repeated safe-stops at the same
    # cumulative epoch must never overwrite an earlier artifact; allocate a deterministic suffix.
    if snapshot.exists():
        counter = 1
        while True:
            candidate = (
                lineage_dir / f"epoch_{int(cumulative_epoch):012d}_snapshot_{counter:03d}.pt"
            )
            if not candidate.exists():
                snapshot = candidate
                break
            counter += 1
    manager.write_torch(
        snapshot,
        _deployable_policy_payload(
            network,
            config,
            history,
            historical_pretraining,
            cumulative_epoch,
            device=str(device),
            rollout_workers=int(rollout_workers),
        ),
    )
    return snapshot


def train_policy(config: TrainingConfig, output_path, progress_callback=None, cancel_callback=None):
    final_benchmark_names = {"case118", "case300"}
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
    network = CALOPolicyNetwork(POLICY_STATE_DIM, config.hidden_dim).to(device)
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
                "Fine-tune/fork checkpoint architecture does not match the configured native CALO policy network"
            )
        network.load_state_dict(payload.get("model_state_dict", payload))
    start_epoch = 0
    history = []
    historical_pretraining = {}
    if resume_path.is_file():
        start_epoch, history, historical_pretraining, _extra = load_training_resume(
            resume_path, network, optimizer, device, rng, current_config=config
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

    target_epoch, training_mode = _resolve_training_target(config, start_epoch)
    # Never regress the curriculum when a completed/partial lineage is extended to a larger target.
    # A policy that already reached a harder stage remains at least at that stage on continuation.
    stage_floor = 0
    if history:
        try:
            stage_floor = max(0, int(history[-1].get("curriculum_stage", 1)) - 1)
        except (TypeError, ValueError):
            stage_floor = 0
    # In indefinite mode progress is epoch based rather than percent-to-target.
    nominal_target = (
        target_epoch if target_epoch is not None else max(start_epoch + 1, int(config.epochs), 1)
    )
    total_units = nominal_target * config.episodes_per_epoch
    completed_units = start_epoch * config.episodes_per_epoch
    epoch = start_epoch
    while target_epoch is None or epoch < target_epoch:
        if cancel_callback and cancel_callback():
            save_training_resume(
                resume_path,
                network=network,
                optimizer=optimizer,
                next_epoch=epoch,
                history=history,
                rng=rng,
                historical_pretraining=historical_pretraining,
                config=config,
                extra={
                    "device": str(device),
                    "rollout_workers": workers,
                    "training_mode": training_mode,
                    "safe_stop": True,
                },
            )
            # A safe-stop boundary is also a deployable immutable policy checkpoint. The exact
            # resume file remains separate because it additionally contains optimizer/RNG state.
            save_deployable_policy_snapshot(
                output_path,
                network,
                config,
                history,
                historical_pretraining,
                epoch,
                device=str(device),
                rollout_workers=workers,
            )
            raise TrainingCancelled(
                f"CALO policy training stopped safely after cumulative epoch {epoch}."
            )
        proposed_stage = _curriculum_stage(
            epoch, max(nominal_target, epoch + 1), bool(config.development_cases)
        )
        stage = max(stage_floor, proposed_stage)
        stage_floor = max(stage_floor, stage)

        def epoch_progress(completed_in_epoch: int, _completed_indices) -> None:
            if progress_callback:
                absolute = completed_units + completed_in_epoch
                progress_callback(
                    (int(100 * absolute / max(total_units, 1)) if target_epoch is not None else 0),
                    f"Epoch {epoch + 1}/{target_epoch if target_epoch is not None else '∞'} · {completed_in_epoch}/{config.episodes_per_epoch} rollout episodes · "
                    f"{workers} CPU worker{'s' if workers != 1 else ''} · PPO device {device}",
                )

        try:
            rollout, episode_returns, completed = _collect_epoch_rollouts(
                config,
                network,
                epoch,
                stage,
                workers,
                progress_callback=epoch_progress,
                cancel_callback=cancel_callback,
            )
        except TrainingCancelled:
            # No PPO update for the incomplete epoch has been accepted. Persist the exact state at
            # the last completed epoch even when the normal checkpoint interval has not elapsed.
            save_training_resume(
                resume_path,
                network=network,
                optimizer=optimizer,
                next_epoch=epoch,
                history=history,
                rng=rng,
                historical_pretraining=historical_pretraining,
                config=config,
                extra={
                    "device": str(device),
                    "rollout_workers": workers,
                    "training_mode": training_mode,
                    "safe_stop": True,
                },
            )
            save_deployable_policy_snapshot(
                output_path,
                network,
                config,
                history,
                historical_pretraining,
                epoch,
                device=str(device),
                rollout_workers=workers,
            )
            raise
        completed_units += len(completed)
        if progress_callback:
            progress_callback(
                (
                    int(100 * completed_units / max(total_units, 1))
                    if target_epoch is not None
                    else 0
                ),
                f"Epoch {epoch + 1}/{target_epoch if target_epoch is not None else '∞'} · PPO update on {device} · {len(rollout['state'])} transitions",
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
                    + parameter_dist.log_prob(parameters[batch_t].clamp(1e-5, 1 - 1e-5)).sum(-1)
                )
                ratio = torch.exp(new_logp - old_logp[batch_t])
                unclipped = ratio * advantages_t[batch_t]
                clipped = (
                    torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
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
                    policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy
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
        completed_epoch = epoch + 1
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
                extra={
                    "device": str(device),
                    "rollout_workers": workers,
                    "training_mode": training_mode,
                },
            )
        deploy_interval = max(
            1, int(getattr(config, "deployable_checkpoint_interval_epochs", 1000) or 1000)
        )
        if completed_epoch % deploy_interval == 0:
            save_deployable_policy_snapshot(
                output_path,
                network,
                config,
                history,
                historical_pretraining,
                completed_epoch,
                device=str(device),
                rollout_workers=workers,
            )
        epoch += 1

    # Always persist the exact terminal state, including completed fixed targets. This allows a
    # completed policy to be continued later without degrading to weights-only fine tuning.
    save_training_resume(
        resume_path,
        network=network,
        optimizer=optimizer,
        next_epoch=epoch,
        history=history,
        rng=rng,
        historical_pretraining=historical_pretraining,
        config=config,
        extra={
            "device": str(device),
            "rollout_workers": workers,
            "training_mode": training_mode,
            "completed_target": target_epoch,
        },
    )
    # The working output alias may be replaced in a later continuation session. Always create an
    # immutable, immediately deployable terminal checkpoint for experiment binding/provenance.
    terminal_snapshot = save_deployable_policy_snapshot(
        output_path,
        network,
        config,
        history,
        historical_pretraining,
        int(epoch),
        device=str(device),
        rollout_workers=workers,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_info = available_training_devices()
    metadata = {
        "algorithm": "CALO",
        "calo_core": "v5.0",
        "training_method": "PPO",
        "training_config": asdict(config),
        "training_seed": config.seed,
        "state_dimension": POLICY_STATE_DIM,
        "state_schema_version": POLICY_STATE_SCHEMA,
        "action_schema_version": POLICY_ACTION_SCHEMA,
        "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
        "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
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
        "cumulative_epoch": int(epoch),
        "policy_lineage_id": str(getattr(config, "policy_lineage_id", "")),
        "policy_lineage_name": str(getattr(config, "policy_lineage_name", "")),
        "policy_phase_index": int(getattr(config, "policy_phase_index", 1)),
        "training_mode": training_mode,
        "immutable_terminal_checkpoint": str(terminal_snapshot),
    }
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(network),
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": config.hidden_dim},
            "metadata": metadata,
        },
        output_path,
    )
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # v5 deliberately keeps the trusted resume checkpoint after a completed target. The final
    # deployable policy remains immutable for experiments already bound to its SHA; future training
    # may continue the same lineage from the preserved optimizer/RNG/curriculum state.
    if not bool(getattr(config, "keep_resume_after_completion", True)):
        try:
            resume_path.unlink(missing_ok=True)
            resume_path.with_suffix(resume_path.suffix + ".sha256").unlink(missing_ok=True)
        except OSError:
            pass
    return str(output_path), history
