"""Counted development-only ORPD environment for independent TSH-CALO training.

The environment has no experiment-runner, policy-registry, activation, GUI, or production
inference authority.  It executes raw single-member hierarchical actions against the same
versioned transition kernel used by production TSH-CALO.  Production ensemble uncertainty,
bandit residuals, admission, and fallback remain the responsibility of the inference boundary.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
from calo_rpd_studio.power_system.case_identity import (
    PROTECTED_HOLDOUT_BUS_COUNTS,
    canonical_protected_holdout_checksums,
)

from .adaptive_epsilon import AdaptiveEpsilonController
from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import (
    build_cognitive_state,
    population_diversity,
    rule_based_regime_prior,
)
from .contextual_credit import ContextualCredit, classify_contexts
from .diagnostics import PopulationDiagnostics, population_diagnostics
from .dual_lane_controller import DualLaneController
from .environmental_selection import epsilon_sort_key
from .hierarchical_memory import HierarchicalPrefixEliteMemory
from .policy_schema import POLICY_STATE_DIM, PolicyRuntimeContext, build_policy_vector
from .precision_engine import CognitivePrecisionEngine
from .success_memory import SuccessMemory
from .tensor_state import CALOTensorState
from .transition_kernel import TransitionResult, evaluate_candidates
from .tsh_calo_policy import GroupActionMask
from .tsh_calo_physics_repair import (
    PhysicsRepairConfig,
    PhysicsRepairOperator,
    physics_repair_context_from_counted_evaluation,
    physics_repair_context_is_usable,
)
from .tsh_calo_runtime_context import build_runtime_topology_policy_context
from .tsh_calo_schema import N_OPERATORS, TSH_CALO_TRAINING_ENVIRONMENT
from .tsh_calo_training import TSHCALOTrainingAction, TSHCALOTrainingConfig
from .tsh_calo_transition_kernel import (
    complete_tsh_transition,
    effective_recovery_fraction,
    generate_tsh_offspring,
)
from .variable_intelligence import VariableGroupIntelligence


def _normalise_first_six(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)[:6]
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full(6, 1.0 / 6.0)


def _problem_protected_identity(problem) -> str:
    case = getattr(problem, "case", None)
    checksum_fn = getattr(case, "checksum", None)
    checksum = str(checksum_fn()).lower() if callable(checksum_fn) else ""
    references = canonical_protected_holdout_checksums()
    for name, reference in references.items():
        if checksum and checksum == str(reference).lower():
            return name
    bus_count = int(getattr(case, "n_bus", -1))
    for name, protected_count in PROTECTED_HOLDOUT_BUS_COUNTS.items():
        if name not in references and bus_count == int(protected_count):
            return name
    return ""


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _update_array_digest(digest, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingEnvironmentConfig:
    """Fixed, hash-bound scientific controls for one development rollout environment."""

    case_identity: str
    population_size: int = 20
    max_evaluations: int = 200
    seed: int = 0
    environment_deterministic: bool = False
    feasible_archive_capacity: int = 32
    boundary_archive_capacity: int = 48
    memory_capacity: int = 256
    memory_decay: float = 0.97
    credit_decay: float = 0.90
    credit_floor: float = 0.02
    group_credit_decay: float = 0.90
    max_learning_lane_fraction: float = 0.92
    precision_start_radius: float = 0.04
    precision_min_radius: float = 5e-4
    precision_max_radius: float = 0.15
    epsilon_quantile: float = 0.75
    epsilon_control_fraction: float = 0.65
    epsilon_exponent: float = 2.0
    stagnation_window: int = 12
    memory_evidence_batches: int = 6
    recovery_diversity_threshold: float = 0.06
    recovery_fraction: float = 0.18

    def validate(self, training_config: TSHCALOTrainingConfig) -> None:
        training_config.validate()
        identity = self.case_identity.strip()
        if not identity:
            raise ValueError("TSH-CALO training environment requires a case identity")
        if identity not in training_config.development_cases:
            raise ValueError(
                "TSH-CALO environment case must be declared in independent development_cases"
            )
        if self.population_size < 2:
            raise ValueError("TSH-CALO training population must contain at least two learners")
        if self.population_size > training_config.resource_envelope.maximum_population_size:
            raise ValueError(
                "TSH-CALO environment population exceeds the frozen training resource envelope"
            )
        if self.max_evaluations < 2 * self.population_size:
            raise ValueError("TSH-CALO training requires an initial and one offspring population")
        if self.max_evaluations % self.population_size != 0:
            raise ValueError("TSH-CALO training FE budget must be divisible by population size")
        if self.feasible_archive_capacity < 2 or self.boundary_archive_capacity < 4:
            raise ValueError("TSH-CALO training archive capacities are invalid")
        if self.memory_capacity < 1 or not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("TSH-CALO training success-memory controls are invalid")
        if not 0.0 <= self.credit_decay <= 1.0 or self.credit_floor <= 0.0:
            raise ValueError("TSH-CALO training credit controls are invalid")
        if not 0.0 <= self.group_credit_decay <= 1.0:
            raise ValueError("TSH-CALO training group-credit decay is invalid")
        if not 0.0 <= self.max_learning_lane_fraction <= 1.0:
            raise ValueError("TSH-CALO training learning-lane fraction is invalid")
        if not (
            0.0
            < self.precision_min_radius
            <= self.precision_start_radius
            <= self.precision_max_radius
        ):
            raise ValueError("TSH-CALO training precision radii are invalid")
        if not 0.0 <= self.epsilon_quantile <= 1.0:
            raise ValueError("TSH-CALO training epsilon quantile is invalid")
        if not 0.0 < self.epsilon_control_fraction <= 1.0 or self.epsilon_exponent <= 0.0:
            raise ValueError("TSH-CALO training epsilon schedule is invalid")
        if self.stagnation_window < 4 or self.memory_evidence_batches < 1:
            raise ValueError("TSH-CALO training evidence windows are invalid")
        if self.recovery_diversity_threshold < 0.0 or not 0.0 <= self.recovery_fraction <= 1.0:
            raise ValueError("TSH-CALO training recovery controls are invalid")
        if training_config.feature_flags.population_schedule:
            raise ValueError(
                "Experimental Change F is not admitted by the fixed-population training environment"
            )

    def scientific_design_hash(self, training_config: TSHCALOTrainingConfig) -> str:
        self.validate(training_config)
        return _json_hash(
            {
                "schema_version": TSH_CALO_TRAINING_ENVIRONMENT,
                "training_design_sha256": training_config.scientific_design_hash(),
                "environment": asdict(self),
            }
        )


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingObservation:
    policy_state: object
    action_mask: GroupActionMask
    learner_groups: np.ndarray
    learner_contexts: np.ndarray
    candidate_evaluations: int
    remaining_candidate_evaluations: int
    scenario_power_flow_calls: int
    reference_scenario: str
    physics_repair_status: str

    def validate(self, population_size: int) -> None:
        self.policy_state.validate()
        self.action_mask.validate()
        groups = np.asarray(self.learner_groups, dtype=int)
        contexts = np.asarray(self.learner_contexts, dtype=int)
        if groups.shape != (population_size,) or contexts.shape != (population_size,):
            raise ValueError("TSH-CALO observation learner vectors must align with population")
        if np.any((groups < 0) | (groups >= 3)) or np.any((contexts < 0) | (contexts >= 4)):
            raise ValueError("TSH-CALO observation contains an invalid learner assignment")
        if self.candidate_evaluations < 0 or self.remaining_candidate_evaluations < 0:
            raise ValueError("TSH-CALO observation evaluation counts cannot be negative")
        if self.scenario_power_flow_calls < 0 or not self.reference_scenario:
            raise ValueError("TSH-CALO observation counted context is invalid")
        physics_exposed = bool(np.asarray(self.action_mask.allowed, dtype=bool)[:, 6].any())
        if physics_exposed and self.physics_repair_status != "available_counted_proposal_only":
            raise ValueError("Physics repair cannot be exposed without a counted usable context")


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingStep:
    transition: TransitionResult
    next_observation: TSHCALOTrainingObservation | None
    terminal: bool
    candidate_evaluations: int
    scenario_power_flow_calls: int
    executed_operators: np.ndarray


@dataclass(slots=True)
class _PreparedGeneration:
    observation: TSHCALOTrainingObservation
    epsilon: float
    diagnostics: PopulationDiagnostics
    diversity: float
    quality_order: np.ndarray
    learned_lanes: np.ndarray
    precision_active: bool
    precision_fraction: float
    recovery_required: bool
    consensus: float
    physics_contexts: tuple[object | None, ...]


class IndependentTSHCALOTrainingEnvironment:
    """A counted ORPD rollout environment with no policy lifecycle authority."""

    CHECKPOINT_SCHEMA = "tsh-calo-independent-environment-v3-batched-device-context"

    def __init__(
        self,
        problem,
        training_config: TSHCALOTrainingConfig,
        config: TSHCALOTrainingEnvironmentConfig,
    ) -> None:
        config.validate(training_config)
        evaluator = getattr(problem, "evaluate_with_context", None)
        if not callable(evaluator):
            raise TypeError("TSH-CALO training requires the counted ORPD evaluate_with_context API")
        protected = _problem_protected_identity(problem)
        if protected:
            raise ValueError(f"Protected holdout {protected} cannot enter TSH-CALO policy training")
        if int(getattr(problem, "dimension", 0)) < 1:
            raise ValueError("TSH-CALO training problem must have a positive dimension")
        envelope = training_config.resource_envelope
        static_counts = {
            "topology nodes": int(problem.case.n_bus),
            "topology edges": 2 * int(problem.case.n_branch),
            "topology controls": len(
                list(getattr(getattr(problem, "decoder", None), "variables", None) or [])
            ),
            "scenarios": len(list(getattr(problem, "scenarios", None) or [])),
        }
        declared = {
            "topology nodes": envelope.maximum_topology_nodes,
            "topology edges": envelope.maximum_topology_edges,
            "topology controls": envelope.maximum_topology_controls,
            "scenarios": envelope.maximum_scenarios,
        }
        exceeded = [
            f"{name}={static_counts[name]}>{declared[name]}"
            for name in static_counts
            if static_counts[name] > declared[name]
        ]
        if exceeded:
            raise MemoryError(
                "TSH-CALO problem exceeds the frozen training resource envelope before solve: "
                + ", ".join(exceeded)
            )

        self.problem = problem
        self.training_config = training_config
        self.config = config
        self.problem_fingerprint = scientific_problem_fingerprint(problem)
        self.case_checksum = str(problem.case.checksum())
        self.rng = np.random.default_rng(config.seed)
        self.initialized = False
        self.failed = False
        self.accounting_complete = True
        self.candidate_evaluations = 0
        self.scenario_power_flow_calls = 0
        self.physics_repair_available_generations = 0
        self.physics_repair_proposal_count = 0
        self.physics_repair_linear_algebra_seconds = 0.0
        self.iteration = 0
        self.state: CALOTensorState | None = None
        self.evaluation_contexts: list[object] = []
        self._prepared: _PreparedGeneration | None = None
        self._pending_digest = ""
        self._initialize_components()

    def _initialize_components(self) -> None:
        cfg = self.config
        variables = list(getattr(getattr(self.problem, "decoder", None), "variables", None) or [])
        self.variables = variables
        self.physics_repair_operator = PhysicsRepairOperator(
            PhysicsRepairConfig(enabled=self.training_config.feature_flags.physics_repair)
        )
        self.feasible_archive = FeasibleEliteArchive(cfg.feasible_archive_capacity)
        self.boundary_archive = ConstraintBoundaryArchive(cfg.boundary_archive_capacity)
        self.hpem = HierarchicalPrefixEliteMemory(self.problem.dimension, variables=variables)
        self.memory = SuccessMemory(
            cfg.memory_capacity, cfg.memory_decay, n_operators=N_OPERATORS + 1
        )
        self.credit = ContextualCredit(
            4,
            N_OPERATORS,
            4,
            4,
            decay=cfg.credit_decay,
            floor=cfg.credit_floor,
        )
        self.groups = VariableGroupIntelligence(variables, decay=cfg.group_credit_decay)
        self.lanes = DualLaneController(max_learning=cfg.max_learning_lane_fraction)
        self.precision = CognitivePrecisionEngine(
            cfg.precision_start_radius,
            cfg.precision_min_radius,
            cfg.precision_max_radius,
        )
        self.epsilon_controller: AdaptiveEpsilonController | None = None
        self.previous_best_violation = float("inf")
        self.previous_best_objective = float("inf")
        self.constraint_stagnation = 0
        self.objective_stagnation = 0
        self.violation_improving = False

    @property
    def terminal(self) -> bool:
        return self.initialized and (
            self.candidate_evaluations + self.config.population_size > self.config.max_evaluations
        )

    def _assert_usable(self) -> None:
        if self.failed:
            raise RuntimeError(
                "TSH-CALO training environment failed with incomplete solver accounting; retain "
                "its failure provenance and create a new environment"
            )

    def _evaluate_population(self, population) -> tuple[list[object], list[object]]:
        rows = np.clip(np.asarray(population, dtype=float), 0.0, 1.0)
        batch_evaluator = getattr(self.problem, "evaluate_population_with_context", None)
        if callable(batch_evaluator):
            self.candidate_evaluations += int(len(rows))
            try:
                records = list(
                    batch_evaluator(
                        rows,
                        retain_control_linearization=bool(
                            self.training_config.feature_flags.physics_repair
                        ),
                    )
                )
                if len(records) != len(rows):
                    raise RuntimeError(
                        "Counted ORPD batch-context evaluator returned an incomplete population"
                    )
                batch_evaluations = [record[0] for record in records]
                batch_contexts = [record[1] for record in records]
                self.scenario_power_flow_calls += sum(
                    len(context.scenarios) for context in batch_contexts
                )
                return batch_evaluations, batch_contexts
            except Exception:
                self.failed = True
                self.accounting_complete = False
                raise

        evaluations: list[object] = []
        contexts: list[object] = []
        for row in rows:
            self.candidate_evaluations += 1
            try:
                evaluation, context = self.problem.evaluate_with_context(
                    row,
                    retain_control_linearization=bool(
                        self.training_config.feature_flags.physics_repair
                    ),
                )
            except Exception:
                self.failed = True
                self.accounting_complete = False
                raise
            evaluations.append(evaluation)
            contexts.append(context)
            self.scenario_power_flow_calls += len(context.scenarios)
        return evaluations, contexts

    def reset(self) -> TSHCALOTrainingObservation:
        """Start this instance's one hash-bound episode and spend one population batch."""

        if self.failed:
            self._assert_usable()
        if self.initialized:
            raise RuntimeError(
                "A TSH-CALO training environment represents one hash-bound episode; create a "
                "new instance instead of resetting and discarding its trajectory"
            )
        self.config.validate(self.training_config)
        self.rng = np.random.default_rng(self.config.seed)
        self.initialized = False
        self.failed = False
        self.accounting_complete = True
        self.candidate_evaluations = 0
        self.scenario_power_flow_calls = 0
        self.physics_repair_available_generations = 0
        self.physics_repair_proposal_count = 0
        self.physics_repair_linear_algebra_seconds = 0.0
        self.iteration = 0
        self.state = None
        self.evaluation_contexts = []
        self._prepared = None
        self._pending_digest = ""
        self._initialize_components()

        population = self.rng.random(
            (self.config.population_size, int(self.problem.dimension)), dtype=np.float64
        )
        evaluations, contexts = self._evaluate_population(population)
        self.state = CALOTensorState.initialize(population, evaluations)
        self.evaluation_contexts = contexts
        self.feasible_archive.update(population, evaluations)
        self.boundary_archive.update(population, evaluations)
        self.hpem.update(population, evaluations)
        finite = [
            float(item.violation) for item in evaluations if np.isfinite(float(item.violation))
        ]
        initial_epsilon = (
            float(np.quantile(finite, self.config.epsilon_quantile)) if finite else 0.0
        )
        self.epsilon_controller = AdaptiveEpsilonController(
            initial_epsilon,
            self.config.epsilon_control_fraction,
            self.config.epsilon_exponent,
        )
        self.initialized = True
        return self.observe()

    def _observation_digest(self, observation: TSHCALOTrainingObservation) -> str:
        digest = hashlib.sha256()
        state = observation.policy_state
        for values in (
            state.aggregate_features,
            state.topology.node_features,
            state.topology.edge_index,
            state.topology.edge_features,
            state.topology.control_features,
            state.topology.control_bus_index,
            state.topology.control_groups,
            state.topology.scenario_features,
            observation.action_mask.allowed,
            observation.action_mask.available_groups,
            observation.learner_groups,
            observation.learner_contexts,
        ):
            _update_array_digest(digest, values)
        digest.update(
            json.dumps(
                {
                    "evaluations": observation.candidate_evaluations,
                    "remaining": observation.remaining_candidate_evaluations,
                    "scenario_calls": observation.scenario_power_flow_calls,
                    "reference": observation.reference_scenario,
                    "physics": observation.physics_repair_status,
                    "bus_numbers": state.topology.bus_numbers,
                    "branch_indices": state.topology.branch_indices,
                    "control_labels": state.topology.control_labels,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def observe(self) -> TSHCALOTrainingObservation:
        self._assert_usable()
        if not self.initialized or self.state is None or self.epsilon_controller is None:
            raise RuntimeError("TSH-CALO training environment must be reset before observation")
        if self.terminal:
            raise RuntimeError("TSH-CALO training episode has exhausted its exact FE budget")
        if self._prepared is not None:
            return self._prepared.observation

        cfg = self.config
        state = self.state
        progress = float(self.candidate_evaluations / cfg.max_evaluations)
        rough = population_diagnostics(state.evaluations, self.epsilon_controller.current)
        epsilon = self.epsilon_controller.value(
            self.candidate_evaluations,
            cfg.max_evaluations,
            rough.feasible_ratio,
            self.violation_improving,
            min(self.constraint_stagnation / cfg.stagnation_window, 1.0),
        )
        diagnostics = population_diagnostics(state.evaluations, epsilon)
        diversity = population_diversity(state.population)
        cognition = build_cognitive_state(
            state.population,
            state.evaluations,
            epsilon=epsilon,
            previous_best_violation=self.previous_best_violation,
            previous_best_objective=self.previous_best_objective,
            constraint_stagnation=min(self.constraint_stagnation / cfg.stagnation_window, 1.0),
            objective_stagnation=min(self.objective_stagnation / cfg.stagnation_window, 1.0),
            remaining_budget=max(0.0, 1.0 - progress),
            operator_credit=_normalise_first_six(self.credit.global_operator_probabilities()),
            feasible_archive_size=len(self.feasible_archive),
            feasible_archive_capacity=self.feasible_archive.capacity,
            boundary_archive_size=len(self.boundary_archive),
            boundary_archive_capacity=self.boundary_archive.capacity,
        )
        consensus = self.hpem.consensus(state.population.mean(axis=0)) if len(self.hpem) else 0.0
        readiness = self.lanes.memory_readiness(
            diagnostics.feasible_ratio,
            self.hpem.occupancy,
            self.memory.density,
            min(self.iteration / cfg.memory_evidence_batches, 1.0),
            consensus,
        )
        severe_stagnation = (
            max(self.constraint_stagnation, self.objective_stagnation) >= cfg.stagnation_window
        )
        learning_fraction = self.lanes.learning_fraction(
            readiness, progress, diversity, severe_stagnation
        )
        learned_lanes = self.lanes.assign(
            cfg.population_size,
            learning_fraction,
            self.rng,
            cfg.environment_deterministic,
        )
        precision_active = self.precision.active(
            diagnostics.feasible_ratio,
            min(self.objective_stagnation / cfg.stagnation_window, 1.0),
            progress,
            len(self.hpem),
        )
        precision_fraction = (
            float(
                np.clip(
                    0.12 + 0.28 * self.objective_stagnation / cfg.stagnation_window,
                    0.12,
                    0.55,
                )
            )
            if precision_active
            else 0.0
        )
        prior = rule_based_regime_prior(cognition)
        provisional_regime = int(np.argmax(prior))
        policy_context = PolicyRuntimeContext(
            hpem_occupancy=float(self.hpem.occupancy),
            memory_consensus=float(consensus),
            memory_readiness=float(readiness),
            success_memory_density=float(self.memory.density),
            learning_lane_fraction=float(learning_fraction),
            precision_active=float(precision_active),
            precision_radius=float(
                np.clip(
                    self.precision.radius / max(self.precision.max_radius, 1e-12),
                    0.0,
                    1.0,
                )
            ),
            variable_group_concentration=float(
                np.max(self.groups.probabilities(provisional_regime))
            ),
        )
        aggregate = build_policy_vector(cognition, policy_context, input_dim=POLICY_STATE_DIM)
        quality_order = np.asarray(
            sorted(
                range(cfg.population_size),
                key=lambda index: epsilon_sort_key(state.evaluations[index], epsilon),
            ),
            dtype=int,
        )
        reference_index = int(quality_order[0])
        runtime_context = build_runtime_topology_policy_context(
            aggregate, self.problem, self.evaluation_contexts[reference_index]
        )
        learner_contexts = classify_contexts(
            state.population, state.evaluations, self.violation_improving
        )
        learner_groups = np.asarray(
            [
                self.groups.choose(provisional_regime, self.rng, cfg.environment_deterministic)
                for _ in range(cfg.population_size)
            ],
            dtype=np.int8,
        )
        physics_contexts = tuple(
            physics_repair_context_from_counted_evaluation(context)
            for context in self.evaluation_contexts
        )
        physics_repair_available = bool(
            self.training_config.feature_flags.physics_repair
            and all(
                physics_repair_context_is_usable(
                    context,
                    maximum_condition_number=self.physics_repair_operator.config.maximum_condition_number,
                )
                for context in physics_contexts
            )
        )
        if physics_repair_available:
            self.physics_repair_available_generations += 1
        action_mask = GroupActionMask.from_control_groups(
            self.groups.variable_groups,
            mixed_variable_enabled=True,
            diversity_recovery_enabled=True,
            physics_repair_enabled=physics_repair_available,
        )
        recovery_required = bool(
            severe_stagnation and diversity < cfg.recovery_diversity_threshold
        )
        observation = TSHCALOTrainingObservation(
            runtime_context.policy_state,
            action_mask,
            learner_groups.copy(),
            learner_contexts.copy(),
            self.candidate_evaluations,
            cfg.max_evaluations - self.candidate_evaluations,
            self.scenario_power_flow_calls,
            runtime_context.reference_scenario,
            (
                "available_counted_proposal_only"
                if physics_repair_available
                else (
                    "masked_counted_context_unavailable"
                    if self.training_config.feature_flags.physics_repair
                    else "disabled_by_immutable_training_feature_flags"
                )
            ),
        )
        observation.validate(cfg.population_size)
        self._prepared = _PreparedGeneration(
            observation,
            epsilon,
            diagnostics,
            diversity,
            quality_order,
            learned_lanes,
            precision_active,
            precision_fraction,
            recovery_required,
            consensus,
            physics_contexts,
        )
        self._pending_digest = self._observation_digest(observation)
        return observation

    def _validate_action(self, action: TSHCALOTrainingAction) -> None:
        if self._prepared is None:
            raise RuntimeError("TSH-CALO training action requires a pending observation")
        action.validate()
        observation = self._prepared.observation
        if self._observation_digest(observation) != self._pending_digest:
            raise RuntimeError("TSH-CALO pending observation was mutated after issuance")
        if not np.array_equal(action.learner_groups, observation.learner_groups):
            raise ValueError("TSH-CALO action learner groups do not match the observation")
        if not np.array_equal(action.learner_contexts, observation.learner_contexts):
            raise ValueError("TSH-CALO action learner contexts do not match the observation")
        if not np.array_equal(
            action.action_mask.allowed, observation.action_mask.allowed
        ) or not np.array_equal(
            action.action_mask.available_groups,
            observation.action_mask.available_groups,
        ):
            raise ValueError("TSH-CALO action mask does not match the counted observation")

    def step(self, action: TSHCALOTrainingAction) -> TSHCALOTrainingStep:
        """Evaluate one full offspring batch and return only the canonical transition reward."""

        self._assert_usable()
        if not self.initialized or self.state is None:
            raise RuntimeError("TSH-CALO training environment must be reset before stepping")
        if self.terminal:
            raise RuntimeError("TSH-CALO training episode has exhausted its exact FE budget")
        self._validate_action(action)
        assert self._prepared is not None
        prepared = self._prepared
        state = self.state
        forced_recovery: set[int] = set()
        if prepared.recovery_required:
            recovery_fraction = effective_recovery_fraction(
                np.asarray(action.group_parameters, dtype=float),
                prepared.observation.learner_groups,
                maximum_fraction=float(self.config.recovery_fraction),
            )
            count = max(
                1,
                min(
                    self.config.population_size - 1,
                    int(round(self.config.population_size * recovery_fraction)),
                ),
            )
            weakest = sorted(
                range(self.config.population_size),
                key=lambda index: epsilon_sort_key(state.evaluations[index], prepared.epsilon),
                reverse=True,
            )
            forced_recovery = set(weakest[:count])
        batch = generate_tsh_offspring(
            population=state.population,
            evaluations=state.evaluations,
            personal_best=state.personal_best,
            rng=self.rng,
            dimension=self.problem.dimension,
            variables=self.variables,
            quality_order=prepared.quality_order,
            contexts=prepared.observation.learner_contexts,
            learner_groups=prepared.observation.learner_groups,
            learned_lanes=prepared.learned_lanes,
            global_regime=int(action.regime),
            learner_operators=np.asarray(action.learner_operators, dtype=int),
            group_parameter_actions=np.asarray(action.group_parameters, dtype=float),
            memory=self.memory,
            hpem=self.hpem,
            feasible_archive=self.feasible_archive,
            boundary_archive=self.boundary_archive,
            credit=self.credit,
            group_intelligence=self.groups,
            precision=self.precision,
            precision_active=prepared.precision_active,
            precision_fraction=prepared.precision_fraction,
            forced_recovery=forced_recovery,
            consensus=prepared.consensus,
            environment_deterministic=self.config.environment_deterministic,
            physics_repair_operator=(
                self.physics_repair_operator
                if prepared.observation.physics_repair_status == "available_counted_proposal_only"
                else None
            ),
            physics_contexts=prepared.physics_contexts,
        )
        repair_traces = [
            proposal for proposal in batch.physics_repair_proposals if proposal is not None
        ]
        self.physics_repair_proposal_count += len(repair_traces)
        self.physics_repair_linear_algebra_seconds += sum(
            float(proposal.linear_algebra_seconds) for proposal in repair_traces
        )
        offspring_contexts: list[object] = []

        def counted_evaluator(values):
            evaluated, counted = self._evaluate_population(values)
            offspring_contexts.extend(counted)
            return evaluated

        evaluated = evaluate_candidates(batch.candidates.offspring, counted_evaluator)
        if not evaluated.complete or len(offspring_contexts) != self.config.population_size:
            self.failed = True
            self.accounting_complete = False
            raise RuntimeError("TSH-CALO training environment did not evaluate a complete batch")
        selected_diversity = batch.group_parameter_values[prepared.observation.learner_groups, 4]
        transition = complete_tsh_transition(
            population=state.population,
            evaluations=state.evaluations,
            personal_best=state.personal_best,
            personal_best_evaluations=state.personal_best_evaluations,
            offspring=batch.candidates.offspring,
            offspring_evaluations=evaluated.evaluations,
            epsilon=prepared.epsilon,
            assigned_operators=batch.candidates.assigned_operators,
            assigned_memory=batch.candidates.assigned_memory,
            assigned_groups=batch.candidates.assigned_groups,
            individual_regimes=batch.candidates.individual_regimes,
            contexts=prepared.observation.learner_contexts,
            precision_mask=batch.candidates.precision_mask,
            memory=self.memory,
            credit=self.credit,
            group_intelligence=self.groups,
            precision=self.precision,
            feasible_archive=self.feasible_archive,
            boundary_archive=self.boundary_archive,
            hpem=self.hpem,
            old_diagnostics=prepared.diagnostics,
            old_diversity=prepared.diversity,
            diversity_weight=float(np.mean(selected_diversity)),
            population_size=self.config.population_size,
        )
        combined_contexts = self.evaluation_contexts + offspring_contexts
        self.evaluation_contexts = [
            combined_contexts[int(index)] for index in transition.selected_indices
        ]
        state.select_from_combined(
            transition.combined_population,
            transition.combined_evaluations,
            transition.selected_indices,
            transition.offspring_personal_best,
            transition.offspring_personal_best_evaluations,
        )
        new_diagnostics = transition.new_diagnostics
        self.violation_improving = (
            new_diagnostics.best_violation < prepared.diagnostics.best_violation - 1e-12
        )
        self.constraint_stagnation = (
            0 if self.violation_improving else self.constraint_stagnation + 1
        )
        objective_improving = (
            np.isfinite(new_diagnostics.best_feasible_objective)
            and new_diagnostics.best_feasible_objective
            < prepared.diagnostics.best_feasible_objective - 1e-12
        )
        self.objective_stagnation = 0 if objective_improving else self.objective_stagnation + 1
        self.previous_best_violation = new_diagnostics.best_violation
        self.previous_best_objective = new_diagnostics.best_feasible_objective
        self.iteration += 1
        self._prepared = None
        self._pending_digest = ""
        terminal = self.terminal
        next_observation = None if terminal else self.observe()
        return TSHCALOTrainingStep(
            transition,
            next_observation,
            terminal,
            self.candidate_evaluations,
            self.scenario_power_flow_calls,
            batch.candidates.assigned_operators.copy(),
        )

    def scientific_provenance(self) -> dict:
        evaluator_device = str(getattr(self.problem, "device", "cpu"))
        batch_context_available = callable(
            getattr(self.problem, "evaluate_population_with_context", None)
        )
        return {
            "schema_version": self.CHECKPOINT_SCHEMA,
            "training_environment_version": TSH_CALO_TRAINING_ENVIRONMENT,
            "training_run_id": self.training_config.training_run_id,
            "training_design_sha256": self.training_config.scientific_design_hash(),
            "environment_design_sha256": self.config.scientific_design_hash(self.training_config),
            "case_identity": self.config.case_identity,
            "case_checksum": self.case_checksum,
            "problem_fingerprint": self.problem_fingerprint,
            "candidate_evaluations": self.candidate_evaluations,
            "scenario_power_flow_calls": self.scenario_power_flow_calls,
            "accounting_complete": self.accounting_complete,
            "trusted_orpd_evaluator_computation": (
                "nvidia_gpu" if evaluator_device.startswith("cuda") else "cpu"
            ),
            "counted_orpd_execution": {
                "selected_device": evaluator_device,
                "batch_context_api": batch_context_available,
                "target_evaluations_per_host_boundary": (100 if batch_context_available else 1),
                "cpu_cuda_inner_loop_transfers": 0 if batch_context_available else None,
                "context_power_flow_reruns": 0 if batch_context_available else None,
                "greater_than_95_percent_cuda_claim": False,
                "claim_status": "requires_physical_candidate-bound_timing_evidence",
            },
            "policy_member_execution": "raw hierarchical action; production shield not invoked",
            "physics_repair": {
                "status": (
                    "enabled_counted_proposal_only"
                    if self.training_config.feature_flags.physics_repair
                    else "disabled_by_immutable_training_feature_flags"
                ),
                "available_generations": self.physics_repair_available_generations,
                "proposal_count": self.physics_repair_proposal_count,
                "linear_algebra_seconds": self.physics_repair_linear_algebra_seconds,
                "hidden_solver_calls": 0,
                "feasibility_authority": False,
            },
            "population_schedule": "disabled",
            "lifecycle_authority": "none",
        }

    def state_dict(self) -> dict:
        self._assert_usable()
        if not self.initialized or self.state is None:
            raise RuntimeError("TSH-CALO training environment has no initialized state")
        runtime = {
            "rng_state": self.rng.bit_generator.state,
            "candidate_evaluations": self.candidate_evaluations,
            "scenario_power_flow_calls": self.scenario_power_flow_calls,
            "physics_repair_available_generations": self.physics_repair_available_generations,
            "physics_repair_proposal_count": self.physics_repair_proposal_count,
            "physics_repair_linear_algebra_seconds": self.physics_repair_linear_algebra_seconds,
            "iteration": self.iteration,
            "state": self.state,
            "evaluation_contexts": self.evaluation_contexts,
            "feasible_archive": self.feasible_archive,
            "boundary_archive": self.boundary_archive,
            "hpem": self.hpem,
            "memory": self.memory,
            "credit": self.credit,
            "groups": self.groups,
            "lanes": self.lanes,
            "precision": self.precision,
            "epsilon_controller": self.epsilon_controller,
            "previous_best_violation": self.previous_best_violation,
            "previous_best_objective": self.previous_best_objective,
            "constraint_stagnation": self.constraint_stagnation,
            "objective_stagnation": self.objective_stagnation,
            "violation_improving": self.violation_improving,
            "prepared": self._prepared,
            "pending_digest": self._pending_digest,
        }
        return {
            "schema_version": self.CHECKPOINT_SCHEMA,
            "training_design_sha256": self.training_config.scientific_design_hash(),
            "environment_design_sha256": self.config.scientific_design_hash(self.training_config),
            "problem_fingerprint": self.problem_fingerprint,
            "case_checksum": self.case_checksum,
            "runtime": copy.deepcopy(runtime),
        }

    @classmethod
    def from_state_dict(
        cls,
        problem,
        training_config: TSHCALOTrainingConfig,
        config: TSHCALOTrainingEnvironmentConfig,
        payload: dict,
    ) -> "IndependentTSHCALOTrainingEnvironment":
        environment = cls(problem, training_config, config)
        expected = {
            "schema_version": cls.CHECKPOINT_SCHEMA,
            "training_design_sha256": training_config.scientific_design_hash(),
            "environment_design_sha256": config.scientific_design_hash(training_config),
            "problem_fingerprint": environment.problem_fingerprint,
            "case_checksum": environment.case_checksum,
        }
        for key, value in expected.items():
            if str(payload.get(key, "")) != str(value):
                raise ValueError(f"TSH-CALO training environment checkpoint {key} changed")
        runtime = copy.deepcopy(dict(payload.get("runtime", {})))
        required = {
            "rng_state",
            "candidate_evaluations",
            "scenario_power_flow_calls",
            "state",
            "evaluation_contexts",
            "epsilon_controller",
        }
        if not required.issubset(runtime):
            raise ValueError("TSH-CALO training environment checkpoint is incomplete")
        environment.rng.bit_generator.state = runtime["rng_state"]
        environment.candidate_evaluations = int(runtime["candidate_evaluations"])
        environment.scenario_power_flow_calls = int(runtime["scenario_power_flow_calls"])
        environment.physics_repair_available_generations = int(
            runtime.get("physics_repair_available_generations", 0)
        )
        environment.physics_repair_proposal_count = int(
            runtime.get("physics_repair_proposal_count", 0)
        )
        environment.physics_repair_linear_algebra_seconds = float(
            runtime.get("physics_repair_linear_algebra_seconds", 0.0)
        )
        environment.iteration = int(runtime.get("iteration", 0))
        environment.state = runtime["state"]
        environment.evaluation_contexts = list(runtime["evaluation_contexts"])
        for name in (
            "feasible_archive",
            "boundary_archive",
            "hpem",
            "memory",
            "credit",
            "groups",
            "lanes",
            "precision",
            "epsilon_controller",
        ):
            setattr(environment, name, runtime[name])
        environment.previous_best_violation = float(runtime["previous_best_violation"])
        environment.previous_best_objective = float(runtime["previous_best_objective"])
        environment.constraint_stagnation = int(runtime["constraint_stagnation"])
        environment.objective_stagnation = int(runtime["objective_stagnation"])
        environment.violation_improving = bool(runtime["violation_improving"])
        environment._prepared = runtime.get("prepared")
        environment._pending_digest = str(runtime.get("pending_digest", ""))
        environment.initialized = True
        environment.failed = False
        environment.accounting_complete = True
        if environment.candidate_evaluations < config.population_size or (
            environment.candidate_evaluations % config.population_size
        ):
            raise ValueError("TSH-CALO training checkpoint FE count is not batch-aligned")
        if environment.candidate_evaluations > config.max_evaluations:
            raise ValueError("TSH-CALO training checkpoint exceeds the declared FE budget")
        if (
            environment.state.population.shape
            != (
                config.population_size,
                int(problem.dimension),
            )
            or len(environment.evaluation_contexts) != config.population_size
        ):
            raise ValueError("TSH-CALO training checkpoint population state is incompatible")
        expected_iteration = environment.candidate_evaluations // config.population_size - 1
        if environment.iteration != expected_iteration:
            raise ValueError("TSH-CALO training checkpoint iteration and FE count disagree")
        if environment.scenario_power_flow_calls < environment.candidate_evaluations:
            raise ValueError("TSH-CALO training checkpoint scenario-call count is impossible")
        if not np.all(np.isfinite(environment.state.population)) or np.any(
            (environment.state.population < 0.0) | (environment.state.population > 1.0)
        ):
            raise ValueError("TSH-CALO training checkpoint population is not finite and bounded")
        if environment.terminal and environment._prepared is not None:
            raise ValueError("Terminal TSH-CALO training checkpoint cannot hold an observation")
        if environment._prepared is not None:
            environment._prepared.observation.validate(config.population_size)
            if (
                environment._observation_digest(environment._prepared.observation)
                != environment._pending_digest
            ):
                raise ValueError("TSH-CALO training checkpoint observation integrity failed")
        elif not environment.terminal:
            raise ValueError("Non-terminal TSH-CALO training checkpoint lacks an observation")
        return environment
