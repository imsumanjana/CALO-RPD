"""Reproducible PPO training for the native CALO v5.9 hierarchical policy.

The v5.9 rollout environment follows the deployed CALO controller transition semantics for raw policy
actions, per-learner regime adaptation, memory/group selection, precision/recovery intervention,
operator credit and reward decomposition. Policy Qualification on real ORPD development systems remains
a separate mandatory gate for deployable Scientific Base promotion. The
curriculum progresses through unconstrained, constrained, mixed-variable, and narrow-feasible
problems. Final publication benchmark systems are not used by this module unless a user explicitly
adds separate development systems to an external training workflow.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import logging
import multiprocessing as mp
import os
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from calo_rpd_studio.ai.model_io import (
    durable_torch_save,
    load_trusted_resume,
    write_trusted_resume_hash,
    load_checkpoint,
)
from torch import nn

_LOG = logging.getLogger(__name__)

from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import (
    STATE_DIM,
    build_cognitive_state,
    population_diversity,
    rule_based_regime_prior,
)
from .environmental_selection import environmental_select, epsilon_better, epsilon_sort_key
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
from .operator_credit import blend_probabilities
from .reward import calculate_reward
from calo_rpd_studio.orpd.feasibility_rules import better
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
    # Optional full ExperimentConfig JSON/YAML used to reproduce the exact ORPD formulation
    # (objective, variables, PF options, robust scenarios and constraint tolerances) on every
    # development case. Without this, real-case results are screening evidence only and cannot
    # qualify a deployable Scientific Base.
    development_experiment_config_path: str = ""
    allow_final_benchmark_training: bool = False
    historical_repository: str = ""
    use_historical_trajectories: bool = False
    historical_pretraining_epochs: int = 4
    resume_checkpoint: str = ""
    checkpoint_each_epoch: bool = False
    resume_task_id: str = ""
    initial_policy_checkpoint: str = (
        ""  # weights-only fine-tune/fork start; not an exact optimizer resume
    )
    # v5.9 continuation semantics. ``epochs`` is the fixed length of the current cumulative session.
    # ``additional`` remains accepted only as a backward-compatible synonym for old saved configs.
    training_mode: str = "cumulative"  # cumulative | additional(legacy synonym) | indefinite
    checkpoint_interval_epochs: int = 1
    qualification_interval_epochs: int = 0  # v5.9: retired; formal qualification applies only to saved Base artifacts
    policy_lineage_id: str = ""
    policy_lineage_name: str = ""
    policy_phase_index: int = 1
    keep_resume_after_completion: bool = True
    # Primarily for controlled tests/automation. Zero means no session cap in indefinite mode.
    max_session_epochs: int = 0
    parallel_runs: int = 1
    # v6.1: total scientific branches and simultaneous execution concurrency are separate.
    # Zero concurrency means "use the current Dashboard Safe-80 maximum".
    parallel_concurrency: int = 0
    # Queue scheduler time-slices branches at exact-resume boundaries so indefinite training can
    # fairly rotate more scientific branches than may safely run simultaneously.
    branch_queue_quantum_epochs: int = 10
    # Internal process-lease target. This is orchestration-only and never changes curriculum semantics.
    lease_target_epoch: int = 0
    # v5.9 competitive multi-branch policy evolution. Explicit seed counts are authoritative.
    parallel_same_seed_branches: int = 0
    parallel_incremental_branches: int = 0
    parallel_decremental_branches: int = 0
    parallel_custom_seeds: tuple[int, ...] = ()
    parallel_start_mode: str = "new"  # new | exact_resume | base_guided_fork
    base_model_checkpoint: str = ""
    training_scratch_dir: str = ""
    safe_snapshot_interval_epochs: int = 10
    max_branch_lead_epochs: int = 30
    champion_validation_interval_epochs: int = 10
    champion_validation_episodes: int = 5
    champion_validation_horizon: int = 12
    champion_validation_seed: int = 918273
    champion_min_feasible_rate: float = 0.80
    # Scientific curriculum is absolute and immutable across continuation-session duration changes.
    curriculum_stage_milestones: tuple[int, int, int, int] = (5, 10, 16, 20)
    # Infinite-session state is deliberately bounded in RAM/checkpoints; full telemetry belongs outside resume state.
    resume_history_limit: int = 256
    coordinator_message_limit: int = 2000
    champion_decision_history_limit: int = 200
    safe_stop_grace_seconds: float = 30.0
    max_branches_per_accelerator: int = 1
    accelerator_memory_reserve_fraction: float = 0.20
    estimated_branch_memory_mb: int = 1024
    # v6.1 Safe-80 protection provenance. A zero parallel limit means a legacy/programmatic
    # configuration that must be validated by the caller before competitive launch.
    safe_parallel_branches: int = 0
    safe_global_cpu_workers: int = 0
    compute_profile_fingerprint: str = ""
    compute_topology_fingerprint: str = ""
    # v6.2 adaptive compute/thermal governor. These fields are orchestration/safety controls and do
    # not alter the scientific PPO objective/curriculum semantics.
    adaptive_compute_governor: bool = True
    staged_startup_delay_seconds: float = 2.0
    governor_sample_interval_seconds: float = 1.0
    governor_amber_pause_seconds: float = 0.25
    governor_startup_admission_timeout_seconds: float = 30.0
    telemetry_enabled: bool = True
    telemetry_segment_max_bytes: int = 8 * 1024 * 1024
    telemetry_max_segments: int = 64


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
    """Native v5.9 training environment sharing runtime cognition semantics.

    The environment remains deliberately lightweight enough for PPO rollouts, but it uses the same
    persistent personal memory, HPEM, contextual credit, variable-group intelligence, adaptive
    epsilon, dual-lane readiness, and recovery semantics exposed to the v5.9 runtime policy.
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
        self.feasible_archive = FeasibleEliteArchive(32)
        self.boundary_archive = ConstraintBoundaryArchive(48)
        self.feasible_archive.update(self.population, self.evaluations)
        self.boundary_archive.update(self.population, self.evaluations)
        variables = getattr(getattr(self.problem, "decoder", None), "variables", None) or getattr(
            self.problem, "variables", []
        )
        self.hpem = HierarchicalPrefixEliteMemory(self.problem.dimension, variables=variables)
        self.hpem.update(self.population, self.evaluations)
        self.memory = SuccessMemory(256, 0.97, n_operators=7)
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

    def _runtime_clock(self, horizon: int) -> tuple[int, int, float, int]:
        """Mirror deployed CALO's FE/batch clock for one training transition.

        Deployed CALO evaluates the initial population before iteration 1. Therefore, immediately
        before training transition ``step_index == 0``, the equivalent requested-FE count is one
        complete population batch, not zero. ``horizon`` denotes the number of offspring
        transitions in the rollout, so the exact equivalent run budget is ``horizon + 1``
        population batches (initial population + offspring batches).
        """
        transitions = max(int(horizon), 1)
        batch_count = int(self.step_index) + 1
        evaluations = int(self.population_size) * batch_count
        max_evaluations = int(self.population_size) * (transitions + 1)
        progress = float(np.clip(evaluations / max(max_evaluations, 1), 0.0, 1.0))
        return evaluations, max_evaluations, progress, batch_count

    def _epsilon(self, horizon: int) -> float:
        best_violation, _best_obj, feasible_ratio = self._diagnostics(self.evaluations)
        improving = best_violation < self.previous_violation - 1e-12
        evaluations, max_evaluations, _progress, _batch_count = self._runtime_clock(horizon)
        return self.epsilon_controller.value(
            evaluations,
            max_evaluations,
            feasible_ratio,
            improving,
            min(self.constraint_stagnation / 12.0, 1.0),
        )

    def state(self, horizon: int):
        epsilon = self._epsilon(horizon)
        _evaluations, _max_evaluations, progress, batch_count = self._runtime_clock(horizon)
        cognitive = build_cognitive_state(
            self.population,
            self.evaluations,
            epsilon=epsilon,
            previous_best_violation=self.previous_violation,
            previous_best_objective=self.previous_objective,
            constraint_stagnation=min(self.constraint_stagnation / 12.0, 1.0),
            objective_stagnation=min(self.objective_stagnation / 12.0, 1.0),
            remaining_budget=max(0.0, 1.0 - progress),
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
            min(batch_count / 6.0, 1.0),
            consensus,
        )
        severe = max(self.constraint_stagnation, self.objective_stagnation) >= 12
        learning_fraction = self.lane_controller.learning_fraction(
            readiness, progress, diversity, severe
        )
        precision_active = self.precision.active(
            feasible_ratio,
            min(self.objective_stagnation / 12.0, 1.0),
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

    @staticmethod
    def _individual_regime(global_regime: int, context: int) -> int:
        if int(context) == 3:
            return 3
        if int(context) == 2 and int(global_regime) >= 2:
            return 1
        if int(context) <= 1 and int(global_regime) == 0:
            return 1
        return int(global_regime)

    @staticmethod
    def _memory_prior(regime: int) -> np.ndarray:
        priors = np.asarray(
            [
                [0.05, 0.15, 0.30, 0.50],
                [0.10, 0.25, 0.40, 0.25],
                [0.40, 0.35, 0.20, 0.05],
                [0.05, 0.10, 0.20, 0.65],
            ],
            dtype=float,
        )
        return priors[int(regime)].copy()

    def step(self, regime: int, operator: int, raw_parameters: np.ndarray, horizon: int) -> float:
        """Execute one native-v5.9 CALO controller transition.

        ``regime``/``operator``/``raw_parameters`` are the *raw neural-policy action*.  Per-learner
        regime adaptation, memory/group choices, precision and forced recovery are environmental
        controller interventions matching deployed CALO semantics.  ``last_step_trace`` records both
        layers so PPO credit is never assigned to an overridden action as if it were executed.
        """
        low = PARAMETER_LOW
        high = PARAMETER_HIGH
        raw_parameters = np.clip(np.asarray(raw_parameters, float), 0.0, 1.0)
        params = low + raw_parameters * (high - low)
        attraction, differential, sigma, memory_weight, diversity_weight, recovery_fraction = params
        adaptive = {
            "attraction": float(attraction),
            "differential": float(differential),
            "exploration_sigma": float(sigma),
            "memory_weight": float(memory_weight),
            "diversity_weight": float(diversity_weight),
            "recovery_fraction": float(recovery_fraction),
        }
        epsilon = self._epsilon(horizon)
        old_violation, old_objective, old_feasible = self._diagnostics(self.evaluations)
        old_diversity = population_diversity(self.population)
        mean = self.population.mean(axis=0)
        quality_order = sorted(
            range(self.population_size),
            key=lambda i: epsilon_sort_key(self.evaluations[i], epsilon),
        )
        best = self.population[quality_order[0]]
        consensus = self.hpem.consensus(mean) if len(self.hpem) else 0.0
        _evaluations, _max_evaluations, progress, batch_count = self._runtime_clock(horizon)
        readiness = self.lane_controller.memory_readiness(
            old_feasible,
            self.hpem.occupancy,
            self.memory.density,
            min(batch_count / 6.0, 1.0),
            consensus,
        )
        severe = max(self.constraint_stagnation, self.objective_stagnation) >= 12
        learning_fraction = self.lane_controller.learning_fraction(
            readiness, progress, old_diversity, severe
        )
        # Native deterministic-policy mode applies only to network sampling.  Controller/environment
        # stochasticity remains in the transition kernel exactly as in deployed CALO.
        learned_lanes = self.lane_controller.assign(
            self.population_size, learning_fraction, self.rng, False
        )
        contexts = classify_contexts(
            self.population,
            self.evaluations,
            old_violation < self.previous_violation - 1e-12,
        )
        precision_active = self.precision.active(
            old_feasible,
            min(self.objective_stagnation / 12.0, 1.0),
            progress,
            len(self.hpem),
        )
        precision_fraction = 0.0
        if precision_active:
            precision_fraction = float(
                np.clip(
                    0.12
                    + 0.28 * min(self.objective_stagnation / 12.0, 1.0)
                    + 0.15 * max(progress - 0.70, 0.0) / 0.30,
                    0.12,
                    0.55,
                )
            )

        sigma = float(adaptive["exploration_sigma"])
        if severe and old_diversity < 0.05:
            sigma *= 1.35
        elif old_feasible >= 0.65 and self.objective_stagnation > 0:
            sigma *= 0.75
        adaptive["exploration_sigma"] = float(np.clip(sigma, PARAMETER_LOW[2], PARAMETER_HIGH[2]))

        forced_recovery: set[int] = set()
        if severe and old_diversity < 0.06:
            fraction = float(np.clip(adaptive["recovery_fraction"], 0.05, 0.45))
            count = max(1, min(self.population_size - 1, int(round(self.population_size * fraction))))
            worst_first = sorted(
                range(self.population_size),
                key=lambda i: epsilon_sort_key(self.evaluations[i], epsilon),
                reverse=True,
            )
            forced_recovery = set(worst_first[:count])

        variables = getattr(getattr(self.problem, "decoder", None), "variables", None) or getattr(
            self.problem, "variables", []
        )
        hierarchy = self.hpem.hierarchy() if len(self.hpem) else np.zeros((4, self.problem.dimension))
        offspring = np.empty_like(self.population)
        assigned_memory = np.zeros(self.population_size, dtype=np.int8)
        assigned_groups = np.zeros(self.population_size, dtype=np.int8)
        assigned_operators = np.full(self.population_size, -1, dtype=np.int8)
        individual_regimes = np.zeros(self.population_size, dtype=np.int8)
        precision_mask = np.zeros(self.population_size, dtype=bool)
        discovery_memory_prior = np.asarray([0.03, 0.07, 0.25, 0.65], dtype=float)

        for index, x in enumerate(self.population):
            context = int(contexts[index])
            local_regime = self._individual_regime(int(regime), context)
            individual_regimes[index] = local_regime
            learned_lane = bool(learned_lanes[index])

            memory_prior = self._memory_prior(local_regime)
            if not learned_lane:
                memory_prior = discovery_memory_prior.copy()
            memory_online = self.credit.memory_probabilities(local_regime, context)
            memory_probabilities = blend_probabilities(memory_prior, memory_online, alpha=0.65)
            memory_level = int(self.rng.choice(4, p=memory_probabilities)) if len(self.hpem) else 0
            assigned_memory[index] = memory_level

            group = self.group_intelligence.choose(local_regime, self.rng, False)
            assigned_groups[index] = group

            should_precision = (
                precision_active
                and learned_lane
                and index not in forced_recovery
                and self.rng.random() < precision_fraction
            )
            if should_precision and len(self.hpem):
                success_direction = self.memory.mean_direction(
                    self.problem.dimension,
                    regime=local_regime,
                    context=context,
                    group=group,
                )
                group_mask = self.group_intelligence.mask(group, self.problem.dimension)
                offspring[index] = self.precision.propose(
                    self.hpem.best_vector,
                    hierarchy,
                    success_direction,
                    variables,
                    group_mask,
                    self.rng,
                    consensus,
                )
                precision_mask[index] = True
                continue

            executed_operator = 5 if index in forced_recovery else int(operator)
            if index in forced_recovery:
                learned_lanes[index] = 0
                assigned_memory[index] = 3
            assigned_operators[index] = executed_operator

            candidates = [i for i in range(self.population_size) if i != index]
            if len(candidates) >= 2:
                r1_i, r2_i = self.rng.choice(candidates, size=2, replace=False)
                r1, r2 = self.population[int(r1_i)], self.population[int(r2_i)]
            else:
                r1 = r2 = x
            feasible_teacher = self.feasible_archive.sample(self.rng, best)
            boundary_teacher = self.boundary_archive.sample(self.rng, best)
            memory_teacher = self.hpem.summary(int(assigned_memory[index]), feasible_teacher) if len(self.hpem) else feasible_teacher
            group_mask = self.group_intelligence.mask(group, self.problem.dimension)

            if executed_operator == 0:
                teacher = memory_teacher if learned_lanes[index] and len(self.hpem) else (feasible_teacher if len(self.feasible_archive) else boundary_teacher)
                candidate = feasible_elite_learning(
                    x, teacher, r1, r2, self.rng, adaptive["attraction"], adaptive["differential"]
                )
            elif executed_operator == 1:
                candidate = constraint_boundary_differential(
                    x, boundary_teacher, r1, r2, self.rng, adaptive["attraction"], adaptive["differential"]
                )
            elif executed_operator == 2:
                if learned_lanes[index] and len(self.hpem):
                    teacher = memory_teacher
                else:
                    teacher = feasible_teacher if local_regime >= 2 and len(self.feasible_archive) else boundary_teacher
                candidate = cognitive_teacher_learning(
                    x, teacher, mean, self.rng, adaptive["attraction"], 0.35 * adaptive["exploration_sigma"]
                )
            elif executed_operator == 3:
                direction = self.memory.sample_direction(
                    self.problem.dimension,
                    self.rng,
                    prefer_feasibility=local_regime <= 1,
                    regime=local_regime,
                    context=context,
                    group=group,
                )
                candidate = success_distribution_memory(
                    x, self.personal_best[index], direction, self.rng, 0.55, adaptive["memory_weight"]
                )
                if learned_lanes[index] and len(self.hpem):
                    candidate = np.clip(
                        candidate + 0.12 * adaptive["attraction"] * (memory_teacher - candidate),
                        0.0,
                        1.0,
                    )
            elif executed_operator == 4:
                candidate = mixed_variable_neighbourhood(
                    x,
                    variables,
                    self.rng,
                    max(adaptive["exploration_sigma"] * 0.35, 0.004),
                    2 if local_regime == 3 else 1,
                )
            else:
                reference = boundary_teacher if local_regime <= 1 else (self.hpem.summary(3, feasible_teacher) if len(self.hpem) else feasible_teacher)
                candidate = diversity_recovery(
                    reference, self.population, self.rng, max(adaptive["exploration_sigma"], 0.05)
                )
            if executed_operator != 5 and np.any(group_mask):
                focused = x.copy()
                focused[group_mask] = np.asarray(candidate)[group_mask]
                candidate = focused
            offspring[index] = np.clip(candidate, 0.0, 1.0)

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
        step_norms = np.linalg.norm(offspring - self.population, axis=1)
        offspring_pb = self.personal_best.copy()
        offspring_pb_ev = list(self.personal_best_evaluations)
        precision_successes = 0
        for index, (child, child_ev) in enumerate(zip(offspring, offspring_evaluations)):
            parent_ev = self.evaluations[index]
            ok = epsilon_better(child_ev, parent_ev, epsilon)
            successful[index] = ok
            if parent_ev.feasible and child_ev.feasible and np.isfinite(parent_ev.value):
                objective_gains[index] = max(
                    (float(parent_ev.value) - float(child_ev.value)) / max(abs(float(parent_ev.value)), 1.0),
                    0.0,
                )
            pv = float(parent_ev.violation)
            cv = float(child_ev.violation)
            if np.isposinf(pv) and np.isfinite(cv):
                feasibility_gains[index] = np.inf
            elif np.isfinite(pv) and np.isfinite(cv):
                feasibility_gains[index] = max(pv - cv, 0.0)
            transitions[index] = float((not parent_ev.feasible) and child_ev.feasible)
            if better(child_ev, offspring_pb_ev[index]):
                offspring_pb[index] = child.copy()
                offspring_pb_ev[index] = child_ev
            if ok:
                memory_operator = 6 if precision_mask[index] else int(assigned_operators[index])
                self.memory.add(
                    child - self.population[index],
                    memory_operator,
                    objective_gains[index],
                    feasibility_gains[index],
                    regime=int(individual_regimes[index]),
                    context=int(contexts[index]),
                    group=int(assigned_groups[index]),
                )
            if precision_mask[index] and ok:
                precision_successes += 1

        self.credit.batch_update(
            individual_regimes,
            contexts,
            assigned_operators,
            assigned_memory,
            successful,
            objective_gains,
            feasibility_gains,
            transitions,
        )
        self.group_intelligence.batch_update(
            individual_regimes,
            assigned_groups,
            successful,
            objective_gains,
            feasibility_gains,
            step_norms,
        )
        self.precision.update(int(np.count_nonzero(precision_mask)), precision_successes)

        combined_population = np.vstack([self.population, offspring])
        combined_evaluations = list(self.evaluations) + list(offspring_evaluations)
        selected_population, selected_evaluations, selected_indices = environmental_select(
            combined_population,
            combined_evaluations,
            self.population_size,
            epsilon,
            diversity_weight=float(adaptive["diversity_weight"]),
            return_indices=True,
        )
        parent_pb = self.personal_best.copy()
        parent_pb_ev = list(self.personal_best_evaluations)
        combined_pb = np.vstack([parent_pb, offspring_pb])
        combined_pb_ev = parent_pb_ev + offspring_pb_ev
        self.population = np.asarray(selected_population)
        self.evaluations = list(selected_evaluations)
        self.personal_best = combined_pb[np.asarray(selected_indices)].copy()
        self.personal_best_evaluations = [combined_pb_ev[int(i)] for i in selected_indices]
        self.feasible_archive.update(combined_population, combined_evaluations)
        self.boundary_archive.update(combined_population, combined_evaluations)
        self.hpem.update(combined_population, combined_evaluations)

        new_violation, new_objective, new_feasible = self._diagnostics(self.evaluations)
        new_diversity = population_diversity(self.population)
        reward_components = calculate_reward(
            old_objective,
            new_objective,
            old_violation,
            new_violation,
            old_feasible,
            new_feasible,
            old_diversity,
            new_diversity,
            overhead=0.0,
        )
        violation_improving = new_violation < old_violation - 1e-12
        objective_improving = np.isfinite(new_objective) and new_objective < old_objective - 1e-12
        self.constraint_stagnation = 0 if violation_improving else self.constraint_stagnation + 1
        self.objective_stagnation = 0 if objective_improving else self.objective_stagnation + 1
        self.previous_violation = new_violation
        self.previous_objective = new_objective
        self.step_index += 1
        raw_operator_executed = (assigned_operators == int(operator)) & (~precision_mask)
        self.last_step_trace = {
            "schema_version": "calo-policy-transition-v5.9",
            "raw_regime": int(regime),
            "raw_operator": int(operator),
            "individual_regimes": individual_regimes.astype(int).tolist(),
            "executed_operators": assigned_operators.astype(int).tolist(),
            "precision_mask": precision_mask.astype(bool).tolist(),
            "forced_recovery_indices": sorted(int(i) for i in forced_recovery),
            "operator_policy_active_fraction": float(np.mean(raw_operator_executed)) if self.population_size else 0.0,
            "reward_components": {
                "objective_improvement": reward_components.objective_improvement,
                "constraint_improvement": reward_components.constraint_improvement,
                "feasible_ratio_improvement": reward_components.feasible_ratio_improvement,
                "diversity_recovery": reward_components.diversity_recovery,
                "overhead_penalty": reward_components.overhead_penalty,
            },
        }
        return float(reward_components.total)


def _curriculum_stage(
    epoch: int,
    epochs: int | None = None,
    has_development_cases: bool = False,
    *,
    milestones: tuple[int, int, int, int] | None = None,
) -> int:
    """Return the curriculum stage from absolute persisted milestones.

    ``epochs`` is retained only for source/API compatibility and is intentionally ignored when
    explicit milestones are supplied. v5.9 training always supplies immutable absolute milestones,
    so changing Cumulative/Infinite session duration cannot silently change future learning dynamics.
    """
    if milestones is not None:
        if len(milestones) != 4:
            raise ValueError("curriculum_stage_milestones must contain exactly four increasing epochs")
        values = tuple(int(v) for v in milestones)
        if values[0] < 0 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("curriculum_stage_milestones must be non-negative and strictly increasing")
        m0, m1, m2, m3 = values
        if epoch < m0:
            return 0
        if epoch < m1:
            return 1
        if epoch < m2:
            return 2
        if not has_development_cases or epoch < m3:
            return 3
        return 4

    # Legacy compatibility for direct callers only. New training paths never use duration fractions.
    fraction = epoch / max(int(epochs or 1), 1)
    if fraction < 0.18:
        return 0
    if fraction < 0.40:
        return 1
    if fraction < 0.64:
        return 2
    if not has_development_cases or fraction < 0.82:
        return 3
    return 4


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
            _LOG.debug("Could not query Intel XPU properties; using generic device name", exc_info=True)
            xpu_name = "Intel XPU"
    try:
        from calo_rpd_studio.compute.resource_scheduler import configured_xpu_interpreter

        xpu_sidecar = bool(configured_xpu_interpreter())
    except Exception:
        _LOG.debug("Could not probe configured XPU sidecar interpreter", exc_info=True)
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


def recommended_worker_distribution(
    total_workers: int,
    *,
    device_info: dict[str, str | bool] | None = None,
) -> dict[str, int]:
    """Return a recommended CUDA/XPU/CPU worker split for the given total worker count.

    The heuristic assigns workers proportional to relative device throughput:
    CUDA (10x) > XPU (4x) > CPU (1x).  When only a subset of accelerators is
    available the pool is redistributed across the remaining devices.
    """
    total_workers = int(total_workers)
    if total_workers <= 0:
        raise ValueError("total_workers must be a positive integer")
    if device_info is None:
        device_info = available_training_devices()
    cuda = bool(device_info.get("cuda_available", False))
    # Treat the verified secondary XPU runtime as a real scheduling capability even when the
    # primary PyTorch build has no direct torch.xpu backend.
    xpu = bool(device_info.get("xpu_available", False) or device_info.get("xpu_sidecar_available", False))
    weights: dict[str, float] = {}
    if cuda:
        weights["cuda"] = 10.0
    if xpu:
        weights["xpu"] = 4.0
    weights["cpu"] = 1.0
    total_weight = sum(weights.values()) or 1.0
    raw: dict[str, float] = {k: (v / total_weight) * total_workers for k, v in weights.items()}
    result: dict[str, int] = {}
    assigned = 0
    for lane in ("cuda", "xpu", "cpu"):
        if lane in raw:
            count = max(0, round(raw[lane]))
            result[lane] = count
            assigned += count
    # Adjust rounding drift onto the fastest available lane
    diff = total_workers - assigned
    if diff != 0:
        for lane in ("cuda", "xpu", "cpu"):
            if lane in result:
                result[lane] += diff
                break
    return result


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
            from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
            from calo_rpd_studio.experiments.experiment_runner import build_scenarios
            from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
            from calo_rpd_studio.power_system.case_loader import CaseLoader

            config_path = str(config.development_experiment_config_path or "").strip()
            if not config_path:
                raise RuntimeError(
                    "Real ORPD policy rollouts require development_experiment_config_path so the "
                    "training formulation exactly matches the declared objective/controls/PF/scenarios."
                )
            source = config.development_cases[
                (epoch * config.episodes_per_epoch + int(episode)) % len(config.development_cases)
            ]
            experiment = ExperimentConfig.load(config_path)
            experiment.case_name = str(source)
            experiment.validate()
            case = CaseLoader.load(experiment.case_name)
            scenarios = build_scenarios(experiment, episode_seed, case)
            problem_config = ORPDProblemConfig(
                objective=experiment.objective,
                variables=experiment.variables,
                robust=experiment.robust_objective,
                power_flow=experiment.power_flow,
                constraint_tolerances=experiment.constraint_tolerances,
            )
            development_problem = ORPDProblem(case, problem_config, scenarios)
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
            schema_version = str(transition.get("schema_version", "") or "")
            raw_policy = dict(transition.get("raw_policy") or {})
            if schema_version == "calo-policy-trajectory-v5.9" and raw_policy:
                state = np.asarray(transition.get("policy_state") or [], dtype=float)
                parameter = np.asarray(raw_policy.get("parameter") or [], dtype=float)
                regime = int(raw_policy.get("regime", -1))
                operator = int(raw_policy.get("operator", -1))
                exact_native = True
            else:
                # Legacy records may contain only the old 24-D cognitive vector and a composite
                # post-controller action. They remain importable only as explicitly down-weighted
                # compatibility evidence; they are never represented as exact native supervision.
                state = np.asarray(transition.get("state") or [], dtype=float)
                parameter = np.asarray(transition.get("parameter") or [], dtype=float)
                regime = int(transition.get("regime", -1))
                operator = int(transition.get("operator", -1))
                exact_native = state.shape == (POLICY_STATE_DIM,)
                if state.shape == (STATE_DIM,):
                    state = np.concatenate((state, np.zeros(POLICY_STATE_DIM - STATE_DIM, dtype=float)))
            if state.shape != (POLICY_STATE_DIM,) or parameter.shape != (6,):
                continue
            if not (0 <= regime < 4 and 0 <= operator < 6):
                continue
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(parameter)):
                continue
            quality = float(np.clip(transition.get("quality_weight", 1.0), 0.05, 1.0))
            if not exact_native:
                quality *= 0.20
            records.append(
                {
                    "state": np.clip(state, -1.0, 1.0),
                    "regime": regime,
                    "operator": operator,
                    "parameter": np.clip(parameter, 1e-5, 1 - 1e-5),
                    "reward": float(transition.get("reward", 0.0)),
                    "return": float(return_value),
                    "parameter_supervision": bool(exact_native and transition.get("parameter_supervision", True)),
                    "quality_weight": quality,
                    "exact_native_supervision": exact_native,
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




def _append_training_telemetry(
    resume_path: Path,
    record: dict,
    *,
    enabled: bool = True,
    segment_max_bytes: int = 8 * 1024 * 1024,
    max_segments: int = 64,
) -> None:
    """Append bounded, segmented non-critical telemetry outside exact-resume state.

    Infinite training must be bounded in both RAM *and disk*. Telemetry therefore rotates immutable-ish
    JSONL segments under a fixed retention cap. Losing an old segment never changes optimizer state.
    """
    if not enabled:
        return
    resume_path = Path(resume_path)
    max_bytes = max(64 * 1024, int(segment_max_bytes or 0))
    keep = max(1, int(max_segments or 1))
    prefix = resume_path.name + ".telemetry."
    parent = resume_path.parent
    manifest_path = Path(str(resume_path) + ".telemetry.manifest.json")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        segments = sorted(parent.glob(prefix + "*.jsonl"))
        if segments:
            current = segments[-1]
            try:
                index = int(current.name.split(".telemetry.", 1)[1].split(".jsonl", 1)[0])
            except (ValueError, IndexError):
                index = len(segments)
        else:
            index = 1
            current = parent / f"{prefix}{index:06d}.jsonl"
        line = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if current.exists() and current.stat().st_size + len(line) > max_bytes:
            index += 1
            current = parent / f"{prefix}{index:06d}.jsonl"
        with current.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

        segments = sorted(parent.glob(prefix + "*.jsonl"))
        while len(segments) > keep:
            oldest = segments.pop(0)
            oldest.unlink(missing_ok=True)
        manifest = {
            "schema_version": "calo-policy-telemetry-v5.9",
            "resume_path": str(resume_path),
            "segment_max_bytes": max_bytes,
            "max_segments": keep,
            "segments": [
                {"name": item.name, "size_bytes": int(item.stat().st_size)} for item in segments
            ],
            "retention": "bounded_ring_noncritical_telemetry",
        }
        tmp = manifest_path.with_name(manifest_path.name + f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, manifest_path)
        try:
            fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    except (OSError, TypeError, ValueError):
        _LOG.warning("Could not append non-critical policy-training telemetry for %s", resume_path, exc_info=True)


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
    "qualification_interval_epochs",
    "keep_resume_after_completion",
    "max_session_epochs",
    "resume_checkpoint",
    "resume_task_id",
    "initial_policy_checkpoint",
    # Competitive-branch orchestration may change at the session boundary without changing the
    # exact saved optimizer/RNG trajectory of an existing branch. Exact resume restores the branch
    # identities/seeds from the branch manifest, not from these GUI planning fields.
    "parallel_runs",
    "parallel_concurrency",
    "branch_queue_quantum_epochs",
    "lease_target_epoch",
    "parallel_same_seed_branches",
    "parallel_incremental_branches",
    "parallel_decremental_branches",
    "parallel_custom_seeds",
    "parallel_start_mode",
    "base_model_checkpoint",
    "training_scratch_dir",
    "max_branch_lead_epochs",
    "safe_snapshot_interval_epochs",  # compatibility field; competitive v5.9 cadence is fixed at 10
    "resume_history_limit",
    "coordinator_message_limit",
    "champion_decision_history_limit",
    "safe_stop_grace_seconds",
    "max_branches_per_accelerator",
    "safe_parallel_branches",
    "safe_global_cpu_workers",
    "compute_profile_fingerprint",
    "compute_topology_fingerprint",
    "adaptive_compute_governor",
    "staged_startup_delay_seconds",
    "governor_sample_interval_seconds",
    "governor_amber_pause_seconds",
    "governor_startup_admission_timeout_seconds",
    "accelerator_memory_reserve_fraction",
    "estimated_branch_memory_mb",
    "telemetry_enabled",
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
        "format": "calo_policy_training_resume_v58",
        "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
        "state_schema_version": POLICY_STATE_SCHEMA,
        "action_schema_version": POLICY_ACTION_SCHEMA,
        "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
        "curriculum_encoding": "zero_based_0_4",
        "next_epoch": int(next_epoch),
        "model_state_dict": _cpu_state_dict(network),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history)[-max(1, int(getattr(config, "resume_history_limit", 256) or 256)):],
        "historical_pretraining": dict(historical_pretraining or {}),
        "python_random_state": random.getstate(),
        "numpy_global_state": np.random.get_state(),
        "numpy_generator_state": rng.bit_generator.state,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "xpu_rng_state_all": (
            torch.xpu.get_rng_state_all()
            if hasattr(torch, "xpu")
            and torch.xpu.is_available()
            and hasattr(torch.xpu, "get_rng_state_all")
            else []
        ),
        "training_config": asdict(config),
        "extra": dict(extra or {}),
    }
    durable_torch_save(payload, path)
    write_trusted_resume_hash(path)
    return path


def load_training_resume(
    path: Path, network, optimizer, device, rng, current_config: TrainingConfig | None = None
) -> tuple[int, list, dict, dict]:
    payload = load_trusted_resume(path, map_location=device)
    resume_format = str(payload.get("format", ""))
    if resume_format not in {"calo_policy_training_resume_v58", "calo_policy_training_resume_v56", "calo_policy_training_resume_v5", "calo_policy_training_resume_v41"}:
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
    if (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and payload.get("xpu_rng_state_all")
        and hasattr(torch.xpu, "set_rng_state_all")
    ):
        torch.xpu.set_rng_state_all(payload["xpu_rng_state_all"])
    extra = dict(payload.get("extra", {}))
    extra.setdefault("_resume_format", resume_format)
    extra.setdefault("_curriculum_encoding", str(payload.get("curriculum_encoding", "")))
    return (
        int(payload.get("next_epoch", 0)),
        list(payload.get("history", [])),
        dict(payload.get("historical_pretraining", {})),
        extra,
    )


def _resolve_training_target(config: TrainingConfig, start_epoch: int) -> tuple[int | None, str]:
    mode = str(getattr(config, "training_mode", "cumulative") or "cumulative").strip().lower()
    if mode not in {"cumulative", "additional", "indefinite"}:
        raise ValueError(f"Unsupported CALO policy training mode: {mode}")
    # v6.1 protected queue leases cap one child process at an absolute exact-resume epoch while
    # preserving the original scientific training_mode. This makes process rotation orchestration-
    # only: curriculum, reward, optimizer and RNG semantics remain those of the parent session.
    lease_target = int(getattr(config, "lease_target_epoch", 0) or 0)
    if lease_target > int(start_epoch):
        return lease_target, mode
    if mode == "indefinite":
        cap = int(getattr(config, "max_session_epochs", 0) or 0)
        return ((start_epoch + cap) if cap > 0 else None), mode
    requested = max(1, int(config.epochs))
    # v5.9 user semantics: Cumulative is a fixed-length training session that accumulates on the
    # exact saved state. ``additional`` is retained as a backward-compatible synonym.
    return start_epoch + requested, mode


def _stage_floor_from_history(history: list, resume_extra: dict | None = None) -> int:
    """Restore curriculum stage without guessing old/new encodings from the numeric value alone."""
    if not history:
        return 0
    try:
        raw = int(history[-1].get("curriculum_stage", 0))
    except (TypeError, ValueError):
        return 0
    extra = dict(resume_extra or {})
    encoding = str(extra.get("_curriculum_encoding", "") or extra.get("curriculum_encoding", ""))
    resume_format = str(extra.get("_resume_format", ""))
    if encoding == "one_based_1_5" or resume_format == "calo_policy_training_resume_v41":
        return int(np.clip(raw - 1, 0, 4))
    # v5/v5.6 and all new checkpoints use the native zero-based 0..4 encoding.
    return int(np.clip(raw, 0, 4))


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
            "policy_training_architecture": "v5.9",
            "training_method": "PPO",
            "candidate_checkpoint": bool(getattr(config, "heterogeneous_rollouts", False)),
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
            "execution": {
                "rollout_workers": int(rollout_workers),
                "ppo_device": str(device),
                "architecture": (
                    "same-policy synchronous persistent CUDA/XPU/CPU actor lanes with one centralized PPO learner"
                    if bool(getattr(config, "heterogeneous_rollouts", False))
                    else "parallel rollout collection with one centralized PPO learner"
                ),
            },
            "development_cases": list(config.development_cases),
            "final_publication_benchmarks_used_for_training": bool(
                {Path(item).stem.lower() for item in config.development_cases} & {"case118", "case300"}
            ),
            "historical_pretraining": historical_pretraining,
            "historical_data_policy": (
                "eligible TRAIN experiments only; validation/test experiments excluded; old trajectories "
                "used only for offline pretraining, never as PPO on-policy rollouts"
            ),
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
    """Create one immutable deployable artifact; the logical output path is only an alias.

    v5.9 never reuses an immutable artifact path for later training.  Older experiments may remain
    bound to the original SHA while the logical policy lineage continues improving.
    """
    output_path = Path(output_path)
    artifact_dir = output_path.parent / f"{output_path.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / (
        f"policy_e{int(cumulative_epoch):012d}_{int(time.time_ns()):x}.pt"
    )
    payload = _deployable_policy_payload(
        network, config, history, historical_pretraining, cumulative_epoch,
        device=str(device), rollout_workers=int(rollout_workers),
    )
    payload["metadata"]["immutable_artifact_path"] = str(artifact_path.resolve())
    payload["metadata"]["immutable_terminal_checkpoint"] = str(artifact_path.resolve())
    with tempfile.NamedTemporaryFile(delete=False, dir=artifact_dir, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(artifact_path)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact_path


def _write_policy_alias(output_path, artifact_path: Path) -> None:
    """Atomically refresh the mutable convenience alias from an immutable artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=output_path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        import shutil
        shutil.copy2(artifact_path, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _apply_protection_control(config: TrainingConfig, protection_callback, progress_callback=None) -> None:
    """Apply v6.2 governor throttling between scientific training units.

    Level 1 (AMBER) introduces a small duty-cycle pause without changing RNG or optimization
    semantics. Level 2 (RED) is treated as a Safe-Stop request; competitive orchestration also sets
    the shared cancellation event and commits the latest common exact checkpoint.
    """
    if protection_callback is None:
        return
    try:
        level = int(protection_callback() or 0)
    except Exception:
        level = 0
    if level >= 2:
        raise TrainingCancelled("CALO policy training stopped by the compute/thermal protection governor.")
    if level == 1:
        pause = max(0.0, float(getattr(config, "governor_amber_pause_seconds", 0.25) or 0.25))
        if progress_callback:
            progress_callback(0, "Compute protection AMBER · pausing briefly before the next training unit")
        if pause > 0:
            time.sleep(pause)


def _train_policy_impl(
    config: TrainingConfig,
    output_path,
    progress_callback=None,
    cancel_callback=None,
    *,
    epoch_observer=None,
    resume_extra_provider=None,
    cancel_during_rollout: bool = True,
    suppress_cancel_persistence: bool = False,
    protection_callback=None,
):
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
        _initial_meta = dict(payload.get("metadata", {}) or {})
        fork_source_epoch = int(_initial_meta.get("cumulative_epoch", 0) or 0)
        start_epoch = 0
    else:
        fork_source_epoch = 0
        start_epoch = 0
    history = []
    historical_pretraining = {}
    resume_extra = {}
    if resume_path.is_file():
        start_epoch, history, historical_pretraining, resume_extra = load_training_resume(
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
    stage_floor = _stage_floor_from_history(history, resume_extra)
    # In indefinite mode progress is epoch based rather than percent-to-target. ``nominal_target``
    # is presentation/accounting only; curriculum progression is decoupled from session duration.
    nominal_target = target_epoch if target_epoch is not None else max(start_epoch + 1, 1)
    total_units = nominal_target * config.episodes_per_epoch
    completed_units = start_epoch * config.episodes_per_epoch
    epoch = start_epoch

    def _current_resume_extra(**updates):
        extra = {
            "device": str(device),
            "rollout_workers": workers,
            "training_mode": training_mode,
            "curriculum_encoding": "zero_based_0_4",
            "fork_source_cumulative_epoch": int(fork_source_epoch),
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
            "device": device,
            "rollout_workers": workers,
            "episode_returns": list(episode_returns_value or []),
            "epoch_losses": list(epoch_losses_value or []),
        })

    _notify_epoch(start_epoch, stage_floor, [], [])
    while target_epoch is None or epoch < target_epoch:
        _apply_protection_control(config, protection_callback, progress_callback)
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
                    resume_path,
                    network=network,
                    optimizer=optimizer,
                    next_epoch=epoch,
                    history=history,
                    rng=rng,
                    historical_pretraining=historical_pretraining,
                    config=config,
                    extra=_current_resume_extra(safe_stop=True),
                )
                terminal = save_deployable_policy_snapshot(
                    output_path, network, config, history, historical_pretraining, epoch,
                    device=str(device), rollout_workers=workers,
                )
                _write_policy_alias(output_path, terminal)
                raise TrainingCancelled(
                    f"CALO policy training stopped safely after cumulative epoch {epoch}."
                )
        proposed_stage = _curriculum_stage(
            epoch,
            None,
            bool(config.development_cases and str(config.development_experiment_config_path or "").strip()),
            milestones=tuple(config.curriculum_stage_milestones),
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
                cancel_callback=(cancel_callback if cancel_during_rollout else None),
            )
        except TrainingCancelled:
            # The incomplete epoch is never accepted. Competitive branch orchestration owns its
            # own rolling safe snapshots; standalone training persists the last completed state.
            if suppress_cancel_persistence:
                raise
            save_training_resume(
                resume_path, network=network, optimizer=optimizer, next_epoch=epoch, history=history,
                rng=rng, historical_pretraining=historical_pretraining, config=config,
                extra=_current_resume_extra(safe_stop=True),
            )
            terminal = save_deployable_policy_snapshot(
                output_path, network, config, history, historical_pretraining, epoch,
                device=str(device), rollout_workers=workers,
            )
            _write_policy_alias(output_path, terminal)
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
        if len(advantages) == 0:
            epoch += 1
            continue
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
        epoch_record = {
            "epoch": epoch + 1,
            "curriculum_stage": stage,
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "mean_episode_return": float(np.mean(episode_returns)),
            "rollout_workers": workers,
            "ppo_device": str(device),
            "transitions": len(rollout["state"]),
        }
        history.append(epoch_record)
        _append_training_telemetry(
            resume_path,
            epoch_record,
            enabled=bool(getattr(config, "telemetry_enabled", True)),
            segment_max_bytes=int(getattr(config, "telemetry_segment_max_bytes", 8 * 1024 * 1024)),
            max_segments=int(getattr(config, "telemetry_max_segments", 64)),
        )
        history_limit = max(1, int(getattr(config, "resume_history_limit", 256) or 256))
        if len(history) > history_limit:
            del history[:-history_limit]
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
        extra=_current_resume_extra(completed_target=target_epoch),
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
            "policy_training_architecture": "v5.9",
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
    # The mutable convenience alias always mirrors the immutable artifact byte-for-byte.
    _write_policy_alias(output_path, terminal_snapshot)
    metadata["immutable_artifact_path"] = str(terminal_snapshot)
    metadata["immutable_terminal_checkpoint"] = str(terminal_snapshot)
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


def train_policy(
    config: TrainingConfig,
    output_path,
    progress_callback=None,
    cancel_callback=None,
    *,
    epoch_observer=None,
    resume_extra_provider=None,
    cancel_during_rollout: bool = True,
    suppress_cancel_persistence: bool = False,
    protection_callback=None,
):
    """Train one coherent PPO policy while isolating process-global RNG state from GUI callers."""
    caller_python = random.getstate()
    caller_numpy = np.random.get_state()
    caller_torch = torch.random.get_rng_state()
    caller_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        return _train_policy_impl(
            config, output_path, progress_callback, cancel_callback,
            epoch_observer=epoch_observer,
            resume_extra_provider=resume_extra_provider,
            cancel_during_rollout=cancel_during_rollout,
            suppress_cancel_persistence=suppress_cancel_persistence,
            protection_callback=protection_callback,
        )
    finally:
        random.setstate(caller_python)
        np.random.set_state(caller_numpy)
        torch.random.set_rng_state(caller_torch)
        if torch.cuda.is_available() and caller_cuda:
            torch.cuda.set_rng_state_all(caller_cuda)



def train_policy_parallel(
    config: TrainingConfig,
    output_path,
    *,
    parallel_runs: int = 2,
    progress_callback=None,
    cancel_callback=None,
    session_state_callback=None,
) -> tuple[str, list]:
    """v6.1 protected queued transactional competitive independent-branch training.

    Branches retain separate exact optimizer/RNG states and compete through a fixed multi-metric
    champion comparator.  Independent neural-network parameters are never arithmetically averaged.
    """
    from .competitive_training import train_policy_competitive

    return train_policy_competitive(
        config,
        output_path,
        parallel_runs=parallel_runs,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        session_state_callback=session_state_callback,
    )
