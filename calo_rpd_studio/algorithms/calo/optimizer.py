"""Cognitive Adaptive Learning Optimizer — CALO v5.0.

CALO v5.0 is a single-budget, constraint-cognitive optimizer with persistent
individual memory, Hierarchical Prefix Elite Memory (Best-1/3/5/7), contextual
batch credit, bounded 3D success history, mixed-variable group intelligence,
behavior-driven epsilon control, dual discovery/learning lanes, partial recovery,
and a counted cognitive precision engine.

All repeated benchmark runs start from fresh runtime memory.  Historical
cross-experiment learning remains explicit and is blocked by strict benchmark
mode unless the caller deliberately disables that guard.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer
from calo_rpd_studio.accelerated.scratch_pool import ScratchPool
from calo_rpd_studio.orpd.feasibility_rules import better
from .adaptive_epsilon import AdaptiveEpsilonController
from .ai_controller import AIController, PARAMETER_HIGH, PARAMETER_LOW, PARAMETER_NAMES
from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import (
    REGIME_NAMES,
    build_cognitive_state,
    population_diversity,
    rule_based_regime_prior,
)
from .contextual_credit import ContextualCredit, classify_contexts
from .diagnostics import CONSTRAINT_COMPONENTS, diagnostic_history_template, population_diagnostics
from .dual_lane_controller import DualLaneController
from .environmental_selection import environmental_select, epsilon_better, epsilon_sort_key
from .evaluation_cache import ExactEvaluationCache
from .hierarchical_memory import HierarchicalPrefixEliteMemory
from .learning_operators import (
    OPERATOR_NAMES,
    cognitive_teacher_learning,
    constraint_boundary_differential,
    diversity_recovery,
    feasible_elite_learning,
    mixed_variable_neighbourhood,
    success_distribution_memory,
)
from .operator_credit import blend_probabilities
from .precision_engine import CognitivePrecisionEngine
from .policy_schema import PolicyRuntimeContext, variable_group_concentration
from .reward import calculate_reward
from .success_memory import SuccessMemory
from .tensor_state import CALOTensorState
from .run_checkpoint import save_exact_run_checkpoint, load_exact_run_checkpoint
from .variable_intelligence import VariableGroupIntelligence


REGIME_OPERATOR_PRIORS = np.asarray(
    [
        [0.05, 0.33, 0.12, 0.08, 0.30, 0.12],  # feasibility
        [0.18, 0.24, 0.18, 0.14, 0.18, 0.08],  # transition
        [0.34, 0.08, 0.22, 0.20, 0.12, 0.04],  # objective refinement
        [0.08, 0.15, 0.10, 0.10, 0.12, 0.45],  # recovery
    ],
    dtype=float,
)

REGIME_MEMORY_PRIORS = np.asarray(
    [
        [0.05, 0.15, 0.30, 0.50],  # feasibility: preserve broad routes
        [0.10, 0.25, 0.40, 0.25],  # transition: structural memory dominates
        [0.40, 0.35, 0.20, 0.05],  # objective: anchor + local elite geometry
        [0.05, 0.10, 0.20, 0.65],  # recovery: diverse Best-7 knowledge
    ],
    dtype=float,
)
DISCOVERY_OPERATOR_PRIOR = np.asarray([0.05, 0.28, 0.08, 0.05, 0.22, 0.32], dtype=float)
DISCOVERY_MEMORY_PRIOR = np.asarray([0.03, 0.07, 0.25, 0.65], dtype=float)


class CALOOptimizer(BaseOptimizer):
    name = "CALO"
    supports_exact_resume = True

    def _default_checkpoint(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "trained_models" / "calo_policy_v2.pt"

    @staticmethod
    def _rule_operator_probabilities(regime: int) -> np.ndarray:
        values = REGIME_OPERATOR_PRIORS[int(regime)].copy()
        return values / values.sum()

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
        total = float(values.sum())
        return values / total if total > 0.0 else np.full(values.shape, 1.0 / len(values))

    def _select_distinct(
        self, population: np.ndarray, index: int, count: int = 2
    ) -> list[np.ndarray]:
        candidates = [i for i in range(len(population)) if i != index]
        if len(candidates) < count:
            return [population[index].copy() for _ in range(count)]
        chosen = self.rng.choice(candidates, size=count, replace=False)
        return [population[int(i)].copy() for i in chosen]

    @staticmethod
    def _individual_regime(global_regime: int, context: int) -> int:
        # Global policy remains authoritative unless a learner's feasibility state makes
        # that regime inappropriate.  This is compact per-individual cognition, not a
        # second independent policy network.
        if context == 3:
            return 3  # infeasible and stagnated -> recovery
        if context == 2 and global_regime >= 2:
            return 1  # infeasible but improving -> transition learning
        if context <= 1 and global_regime == 0:
            return 1  # already feasible -> do not keep treating learner as infeasible
        return int(global_regime)

    @staticmethod
    def _focus_to_group(
        x: np.ndarray, candidate: np.ndarray, mask: np.ndarray, operator: int
    ) -> np.ndarray:
        if operator == 5 or not np.any(mask):
            return np.clip(candidate, 0.0, 1.0)
        focused = np.asarray(x, float).copy()
        focused[mask] = np.asarray(candidate, float)[mask]
        return np.clip(focused, 0.0, 1.0)

    def _candidate(
        self,
        operator: int,
        index: int,
        state: CALOTensorState,
        memory: SuccessMemory,
        hpem: HierarchicalPrefixEliteMemory,
        feasible_archive: FeasibleEliteArchive,
        boundary_archive: ConstraintBoundaryArchive,
        parameters: dict[str, float],
        regime: int,
        context: int,
        memory_level: int,
        memory_direction: np.ndarray,
        group: int,
        group_intelligence: VariableGroupIntelligence,
        learned_lane: bool,
        *,
        r1: np.ndarray | None = None,
        r2: np.ndarray | None = None,
        best: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        variables=None,
    ) -> np.ndarray:
        population = state.population
        evaluations = state.evaluations
        x = population[index]
        if r1 is None or r2 is None:
            r1, r2 = self._select_distinct(population, index, 2)
        if best is None:
            best = population[self.order(evaluations)[0]]
        if mean is None:
            mean = population.mean(axis=0)
        feasible_teacher = feasible_archive.sample(self.rng, best)
        boundary_teacher = boundary_archive.sample(self.rng, best)
        memory_teacher = (
            np.clip(x + np.asarray(memory_direction, dtype=float), 0.0, 1.0)
            if len(hpem)
            else feasible_teacher
        )
        if variables is None:
            variables = getattr(getattr(self.problem, "decoder", None), "variables", None)
        group_mask = group_intelligence.mask(group, self.problem.dimension)

        if operator == 0:
            teacher = (
                memory_teacher
                if learned_lane and len(hpem)
                else (feasible_teacher if len(feasible_archive) else boundary_teacher)
            )
            candidate = feasible_elite_learning(
                x,
                teacher,
                r1,
                r2,
                self.rng,
                parameters["attraction"],
                parameters["differential"],
            )
        elif operator == 1:
            candidate = constraint_boundary_differential(
                x,
                boundary_teacher,
                r1,
                r2,
                self.rng,
                parameters["attraction"],
                parameters["differential"],
            )
        elif operator == 2:
            if learned_lane and len(hpem):
                teacher = memory_teacher
            else:
                teacher = (
                    feasible_teacher if regime >= 2 and len(feasible_archive) else boundary_teacher
                )
            candidate = cognitive_teacher_learning(
                x,
                teacher,
                mean,
                self.rng,
                parameters["attraction"],
                0.35 * parameters["exploration_sigma"],
            )
        elif operator == 3:
            direction = memory.sample_direction(
                self.problem.dimension,
                self.rng,
                prefer_feasibility=regime <= 1,
                regime=regime,
                context=context,
                group=group,
            )
            candidate = success_distribution_memory(
                x,
                state.personal_best[index],
                direction,
                self.rng,
                0.55,
                parameters["memory_weight"],
            )
            if learned_lane and len(hpem):
                candidate = np.clip(
                    candidate + 0.12 * parameters["attraction"] * (memory_teacher - candidate),
                    0.0,
                    1.0,
                )
        elif operator == 4:
            candidate = mixed_variable_neighbourhood(
                x,
                variables,
                self.rng,
                continuous_sigma=max(parameters["exploration_sigma"] * 0.35, 0.004),
                discrete_radius=2 if regime == 3 else 1,
            )
        else:
            reference = (
                boundary_teacher
                if regime <= 1
                else (hpem.summary(3, feasible_teacher) if len(hpem) else feasible_teacher)
            )
            candidate = diversity_recovery(
                reference,
                population,
                self.rng,
                sigma=max(parameters["exploration_sigma"], 0.05),
            )
        return self._focus_to_group(x, candidate, group_mask, operator)

    def _historical_learning_setup(
        self, parameters: dict
    ) -> tuple[object | None, dict[str, float], str]:
        repository = None
        applied: dict[str, float] = {}
        path = str(parameters.get("historical_repository", "") or "").strip()
        requested = bool(parameters.get("use_historical_parameter_priors", False)) or bool(
            parameters.get("use_cross_algorithm_warm_start", False)
        )
        if bool(parameters.get("strict_benchmark_mode", True)) and requested:
            raise ValueError(
                "Strict benchmark mode forbids historical CALO priors/warm starts. "
                "Disable strict_benchmark_mode only for an explicitly declared transfer-learning study."
            )
        if not path or not requested:
            return None, applied, path

        from calo_rpd_studio.learning.experience_repository import load_experience_repository

        repository = load_experience_repository(path)
        if bool(parameters.get("use_historical_parameter_priors", False)):
            prior = repository.calo_parameter_prior(
                case_checksum=self.problem.case.checksum(),
                case_name=self.problem.case.name,
                dimension=self.problem.dimension,
            )
            blend = float(np.clip(parameters.get("historical_prior_blend", 0.35), 0.0, 1.0))
            tunable = {
                "epsilon_quantile",
                "epsilon_control_fraction",
                "epsilon_exponent",
                "stagnation_window",
                "ai_credit_blend",
                "ai_policy_weight",
                "credit_decay",
                "memory_decay",
                "precision_start_radius",
            }
            for name, prior_value in prior.items():
                if name not in tunable or not isinstance(prior_value, (int, float)):
                    continue
                current = parameters.get(name, prior_value)
                if isinstance(current, (int, float)):
                    blended = (1.0 - blend) * float(current) + blend * float(prior_value)
                    parameters[name] = (
                        int(round(blended)) if name == "stagnation_window" else blended
                    )
                    applied[name] = parameters[name]
        return repository, applied, path

    @staticmethod
    def _compatibility_jsonable(value):
        """Canonicalize scientific problem state for exact-resume compatibility hashing."""
        if is_dataclass(value):
            return CALOOptimizer._compatibility_jsonable(asdict(value))
        if isinstance(value, Enum):
            return CALOOptimizer._compatibility_jsonable(value.value)
        if isinstance(value, np.ndarray):
            return CALOOptimizer._compatibility_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(k): CALOOptimizer._compatibility_jsonable(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [CALOOptimizer._compatibility_jsonable(v) for v in value]
        if callable(value):
            defaults = getattr(value, "__defaults__", None)
            return {
                "callable_module": str(getattr(value, "__module__", "")),
                "callable_qualname": str(getattr(value, "__qualname__", type(value).__qualname__)),
                "defaults": CALOOptimizer._compatibility_jsonable(defaults or ()),
            }
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    def _problem_compatibility_fingerprint(self) -> str:
        decoder = getattr(self.problem, "decoder", None)
        manifest = None
        if decoder is not None and callable(getattr(decoder, "formulation_manifest", None)):
            manifest = decoder.formulation_manifest()
        scenarios = []
        for scenario in list(getattr(self.problem, "scenarios", []) or []):
            scenarios.append(
                {
                    "name": str(getattr(scenario, "name", "")),
                    "weight": float(getattr(scenario, "weight", 1.0)),
                    "transform": self._compatibility_jsonable(getattr(scenario, "transform", None)),
                }
            )
        payload = {
            "case_checksum": str(self.problem.case.checksum()),
            "dimension": int(self.problem.dimension),
            "formulation_manifest": manifest,
            "problem_config": self._compatibility_jsonable(getattr(self.problem, "config", None)),
            "scenarios": scenarios,
        }
        encoded = json.dumps(
            self._compatibility_jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _checkpoint_compatibility(self, parameters: dict, controller) -> dict:
        ignored = {
            "run_checkpoint_path",
            "resume_run_checkpoint",
            "checkpoint_interval_evaluations",
            "extended_evaluation_target",
            "continuation_segment_index",
        }
        stable_parameters = {str(k): v for k, v in parameters.items() if str(k) not in ignored}
        return {
            "algorithm": self.name,
            "seed": int(self.seed),
            "dimension": int(self.problem.dimension),
            "population_size": int(self.config.population_size),
            "case_checksum": str(self.problem.case.checksum()),
            "problem_fingerprint": self._problem_compatibility_fingerprint(),
            "policy_checksum": str(getattr(controller, "checksum", "")),
            "parameters": stable_parameters,
        }

    def _save_run_checkpoint(
        self, path: str, *, parameters: dict, controller, locals_payload: dict
    ) -> str:
        base_state = {
            "evaluations": int(self.evaluations),
            "iteration": int(self.iteration),
            "best_evaluation": self.best_evaluation,
            "best_vector": self.best_vector,
            "history": list(self.history),
            "evaluation_history": list(self.evaluation_history),
            "best_feasible_objective_history": list(self.best_feasible_objective_history),
            "best_constraint_violation_history": list(self.best_constraint_violation_history),
            "best_feasible_objective": float(self._best_feasible_objective),
            "best_constraint_violation": float(self._best_constraint_violation),
            "best_constraint_evaluation": self._best_constraint_evaluation,
            "first_feasible_evaluation": self.first_feasible_evaluation,
            "constraint_component_histories": {
                k: list(v) for k, v in self.constraint_component_histories.items()
            },
            "rng_state": self.rng.bit_generator.state,
            "controller_rng_state": controller.rng.bit_generator.state,
        }
        return save_exact_run_checkpoint(
            path,
            {
                "compatibility": self._checkpoint_compatibility(parameters, controller),
                "base_state": base_state,
                "runtime_state": locals_payload,
            },
        )

    def _restore_base_checkpoint_state(
        self, payload: dict, *, parameters: dict, controller
    ) -> dict:
        expected = self._checkpoint_compatibility(parameters, controller)
        actual = dict(payload.get("compatibility", {}))
        # Horizon may grow, but scientific formulation, seed, policy and algorithm controls may not.
        if actual != expected:
            raise RuntimeError(
                "CALO run checkpoint is incompatible with the current scientific configuration"
            )
        base = dict(payload.get("base_state", {}))
        self.evaluations = int(base["evaluations"])
        self.iteration = int(base["iteration"])
        self.best_evaluation = base.get("best_evaluation")
        self.best_vector = base.get("best_vector")
        self.history = list(base.get("history", []))
        self.evaluation_history = list(base.get("evaluation_history", []))
        self.best_feasible_objective_history = list(base.get("best_feasible_objective_history", []))
        self.best_constraint_violation_history = list(
            base.get("best_constraint_violation_history", [])
        )
        self._best_feasible_objective = float(base.get("best_feasible_objective", float("inf")))
        self._best_constraint_violation = float(base.get("best_constraint_violation", float("inf")))
        self._best_constraint_evaluation = base.get("best_constraint_evaluation")
        self.first_feasible_evaluation = base.get("first_feasible_evaluation")
        self.constraint_component_histories = {
            str(k): list(v) for k, v in dict(base.get("constraint_component_histories", {})).items()
        }
        self.rng.bit_generator.state = base["rng_state"]
        controller.rng.bit_generator.state = base["controller_rng_state"]
        return dict(payload.get("runtime_state", {}))

    def run(self):
        started = time.perf_counter()
        parameters = dict(self.config.parameters)
        population_size = int(self.config.population_size)
        run_checkpoint_path = str(parameters.get("run_checkpoint_path", "") or "").strip()
        resume_run_checkpoint = str(parameters.get("resume_run_checkpoint", "") or "").strip()
        checkpoint_interval = max(
            1, int(parameters.get("checkpoint_interval_evaluations", 500) or 500)
        )
        next_checkpoint_evaluation = checkpoint_interval
        historical_repository, historical_prior_applied, historical_repository_path = (
            self._historical_learning_setup(parameters)
        )

        use_ai = bool(parameters.get("use_ai", True))
        use_memory = bool(parameters.get("use_memory", True))
        use_dual_archives = bool(parameters.get("use_dual_archives", True))
        use_epsilon = bool(parameters.get("use_epsilon", True))
        use_mixed_variable = bool(parameters.get("use_mixed_variable", True))
        use_diversity_recovery = bool(parameters.get("use_diversity_recovery", True))
        use_hpem = bool(parameters.get("use_hpem", True))
        use_contextual_credit = bool(parameters.get("use_contextual_credit", True))
        use_variable_intelligence = bool(parameters.get("use_variable_intelligence", True))
        use_dual_lane = bool(parameters.get("use_dual_lane", True))
        use_precision = bool(parameters.get("use_cognitive_precision", True))
        use_evaluation_cache = bool(parameters.get("use_exact_evaluation_cache", True))
        deterministic_policy = bool(parameters.get("deterministic_policy", False))

        cache = ExactEvaluationCache(
            self.problem,
            capacity=int(parameters.get("evaluation_cache_capacity", 4096))
            if use_evaluation_cache
            else 0,
        )

        # Load an exact continuation payload before any population evaluation. A resumed run must
        # not perform hidden warm-up/scientific solves that are outside its requested FE accounting.
        checkpoint_payload_preloaded = (
            load_exact_run_checkpoint(resume_run_checkpoint) if resume_run_checkpoint else None
        )
        historical_warm_start_count = 0
        if checkpoint_payload_preloaded is not None:
            checkpoint_runtime = dict(checkpoint_payload_preloaded.get("runtime_state", {}))
            checkpoint_state = checkpoint_runtime.get("state")
            if checkpoint_state is None:
                raise RuntimeError(
                    "CALO exact run checkpoint does not contain optimizer tensor state"
                )
            population = np.asarray(checkpoint_state.population, dtype=float).copy()
            evaluations = list(checkpoint_state.evaluations)
        else:
            population = self.random_population()
            if historical_repository is not None and bool(
                parameters.get("use_cross_algorithm_warm_start", False)
            ):
                solutions = historical_repository.compatible_solutions(
                    case_checksum=self.problem.case.checksum(),
                    case_name=self.problem.case.name,
                    dimension=self.problem.dimension,
                )
                fraction = float(
                    np.clip(parameters.get("historical_warm_start_fraction", 0.15), 0.0, 0.50)
                )
                count = min(int(round(population_size * fraction)), len(solutions), population_size)
                for index, item in enumerate(solutions[:count]):
                    vector = np.asarray(item.get("best_vector") or [], dtype=float)
                    if vector.shape == (self.problem.dimension,):
                        population[index] = np.clip(vector, 0.0, 1.0)
                        historical_warm_start_count += 1

            evaluations = (
                cache.evaluate_requests(self, population)
                if use_evaluation_cache
                else self.evaluate_population(population)
            )
            if len(evaluations) < len(population):
                return self.finalize(population[: len(evaluations)], started=started)

        state = CALOTensorState.initialize(population, list(evaluations))
        variables = getattr(getattr(self.problem, "decoder", None), "variables", None) or []
        feasible_archive = FeasibleEliteArchive(
            int(parameters.get("feasible_archive_capacity", 32))
        )
        boundary_archive = ConstraintBoundaryArchive(
            int(parameters.get("boundary_archive_capacity", 48))
        )
        feasible_archive.update(state.population, state.evaluations)
        boundary_archive.update(state.population, state.evaluations)

        hpem = HierarchicalPrefixEliteMemory(self.problem.dimension, variables=variables)
        if use_hpem:
            hpem.update(state.population, state.evaluations)
        memory = SuccessMemory(
            int(parameters.get("memory_capacity", 256)),
            float(parameters.get("memory_decay", 0.97)),
            n_operators=7,  # six portfolio operators + precision-success channel
        )
        credit = ContextualCredit(
            4,
            6,
            4,
            4,
            decay=float(parameters.get("credit_decay", 0.90)),
            floor=float(parameters.get("credit_floor", 0.02)),
        )
        group_intelligence = VariableGroupIntelligence(
            variables,
            decay=float(parameters.get("group_credit_decay", 0.90)),
        )
        lane_controller = DualLaneController(
            max_learning=float(parameters.get("max_learning_lane_fraction", 0.92))
        )
        precision = CognitivePrecisionEngine(
            initial_radius=float(parameters.get("precision_start_radius", 0.04)),
            min_radius=float(parameters.get("precision_min_radius", 5e-4)),
            max_radius=float(parameters.get("precision_max_radius", 0.15)),
        )

        initial_violations = [
            ev.violation for ev in state.evaluations if np.isfinite(float(ev.violation))
        ]
        initial_epsilon = (
            float(np.quantile(initial_violations, float(parameters.get("epsilon_quantile", 0.75))))
            if initial_violations
            else 0.0
        )
        if not use_epsilon:
            initial_epsilon = 0.0
        epsilon_controller = AdaptiveEpsilonController(
            initial_epsilon,
            float(parameters.get("epsilon_control_fraction", 0.65)),
            float(parameters.get("epsilon_exponent", 2.0)),
        )

        checkpoint = str(parameters.get("policy_checkpoint", "") or "").strip()
        if use_ai and not checkpoint:
            if bool(parameters.get("strict_policy_binding", False)):
                raise ValueError(
                    "CALO AI is enabled but no immutable policy checkpoint is bound to this experiment"
                )
            checkpoint = str(self._default_checkpoint())
        controller = AIController(
            checkpoint if use_ai else None,
            seed=int(parameters.get("ai_inference_seed", self.seed + 7919)),
            deterministic=deterministic_policy,
            device=str(parameters.get("inference_device", "auto")),
            expected_checksum=str(parameters.get("policy_sha256", "")) if use_ai else "",
            expected_state_schema=str(parameters.get("policy_state_schema_version", ""))
            if use_ai
            else "",
            expected_action_schema=str(parameters.get("policy_action_schema_version", ""))
            if use_ai
            else "",
        )

        diagnostics_history = diagnostic_history_template()
        operator_usage_history: list[dict[str, int]] = []
        operator_success_history: list[dict[str, float]] = []
        regime_history: list[str] = []
        reward_history: list[float] = []
        memory_readiness_history: list[float] = []
        learning_lane_history: list[float] = []
        memory_consensus_history: list[float] = []
        precision_radius_history: list[float] = []
        previous_best_violation = float("inf")
        previous_best_objective = float("inf")
        constraint_stagnation = 0
        objective_stagnation = 0
        stagnation_window = max(4, int(parameters.get("stagnation_window", 12)))
        violation_improving = False
        policy_trajectory: list[dict] = []
        scratch = ScratchPool()
        precision_evaluations = 0
        precision_successes = 0
        forced_recovery_evaluations = 0
        batch_count = 0
        policy_inference_seconds = 0.0
        candidate_generation_seconds = 0.0
        evaluator_seconds = 0.0
        learning_update_seconds = 0.0

        if resume_run_checkpoint:
            checkpoint_payload = checkpoint_payload_preloaded
            restored = self._restore_base_checkpoint_state(
                checkpoint_payload, parameters=parameters, controller=controller
            )
            state = restored["state"]
            feasible_archive = restored["feasible_archive"]
            boundary_archive = restored["boundary_archive"]
            hpem = restored["hpem"]
            memory = restored["memory"]
            credit = restored["credit"]
            group_intelligence = restored["group_intelligence"]
            lane_controller = restored["lane_controller"]
            precision = restored["precision"]
            epsilon_controller = restored["epsilon_controller"]
            diagnostics_history = restored["diagnostics_history"]
            operator_usage_history = restored["operator_usage_history"]
            operator_success_history = restored["operator_success_history"]
            regime_history = restored["regime_history"]
            reward_history = restored["reward_history"]
            memory_readiness_history = restored["memory_readiness_history"]
            learning_lane_history = restored["learning_lane_history"]
            memory_consensus_history = restored["memory_consensus_history"]
            precision_radius_history = restored["precision_radius_history"]
            previous_best_violation = float(restored["previous_best_violation"])
            previous_best_objective = float(restored["previous_best_objective"])
            constraint_stagnation = int(restored["constraint_stagnation"])
            objective_stagnation = int(restored["objective_stagnation"])
            violation_improving = bool(restored["violation_improving"])
            policy_trajectory = list(restored["policy_trajectory"])
            precision_evaluations = int(restored["precision_evaluations"])
            precision_successes = int(restored["precision_successes"])
            forced_recovery_evaluations = int(restored["forced_recovery_evaluations"])
            batch_count = int(restored["batch_count"])
            policy_inference_seconds = float(restored.get("policy_inference_seconds", 0.0))
            candidate_generation_seconds = float(restored.get("candidate_generation_seconds", 0.0))
            evaluator_seconds = float(restored.get("evaluator_seconds", 0.0))
            learning_update_seconds = float(restored.get("learning_update_seconds", 0.0))
            historical_warm_start_count = int(
                restored.get("historical_warm_start_count", historical_warm_start_count)
            )
            next_checkpoint_evaluation = (
                (int(self.evaluations) // checkpoint_interval) + 1
            ) * checkpoint_interval

        while self.iteration < self.config.max_iterations and self.can_evaluate(population_size):
            self.iteration += 1
            batch_count += 1
            progress = float(
                np.clip(self.evaluations / max(self.config.max_evaluations, 1), 0.0, 1.0)
            )
            rough_diag = population_diagnostics(state.evaluations, epsilon_controller.current)
            epsilon = (
                epsilon_controller.value(
                    self.evaluations,
                    self.config.max_evaluations,
                    rough_diag.feasible_ratio,
                    violation_improving,
                    constraint_stagnation / stagnation_window,
                )
                if use_epsilon
                else 0.0
            )
            current_diag = population_diagnostics(state.evaluations, epsilon)
            current_diversity = population_diversity(state.population)
            remaining_budget = 1.0 - progress

            cognitive = build_cognitive_state(
                state.population,
                state.evaluations,
                epsilon=epsilon,
                previous_best_violation=previous_best_violation,
                previous_best_objective=previous_best_objective,
                constraint_stagnation=min(constraint_stagnation / stagnation_window, 1.0),
                objective_stagnation=min(objective_stagnation / stagnation_window, 1.0),
                remaining_budget=remaining_budget,
                operator_credit=credit.global_operator_probabilities(),
                feasible_archive_size=len(feasible_archive),
                feasible_archive_capacity=feasible_archive.capacity,
                boundary_archive_size=len(boundary_archive),
                boundary_archive_capacity=boundary_archive.capacity,
            )

            # Native v4.1 policies observe the same 24-D cognitive base plus compact HPEM,
            # dual-lane, success-memory, precision, and variable-intelligence signals.  Legacy
            # 24-D checkpoints remain explicitly supported through the checkpoint schema adapter.
            hpem_reference = state.population.mean(axis=0)
            pre_consensus = hpem.consensus(hpem_reference) if use_hpem else 0.0
            pre_readiness = lane_controller.memory_readiness(
                current_diag.feasible_ratio,
                hpem.occupancy if use_hpem else 0.0,
                memory.density if use_memory else 0.0,
                min(batch_count / max(int(parameters.get("memory_evidence_batches", 6)), 1), 1.0),
                pre_consensus,
            )
            pre_learning_fraction = (
                lane_controller.learning_fraction(pre_readiness, progress, current_diversity, False)
                if use_dual_lane
                else 1.0
            )
            pre_precision_active = use_precision and precision.active(
                current_diag.feasible_ratio,
                min(objective_stagnation / stagnation_window, 1.0),
                progress,
                len(hpem),
            )
            provisional_regime = int(np.argmax(rule_based_regime_prior(cognitive)))
            policy_context = PolicyRuntimeContext(
                hpem_occupancy=float(hpem.occupancy if use_hpem else 0.0),
                memory_consensus=float(pre_consensus),
                memory_readiness=float(pre_readiness),
                success_memory_density=float(memory.density if use_memory else 0.0),
                learning_lane_fraction=float(pre_learning_fraction),
                precision_active=float(bool(pre_precision_active)),
                precision_radius=float(
                    np.clip(precision.radius / max(precision.max_radius, 1e-12), 0.0, 1.0)
                ),
                variable_group_concentration=variable_group_concentration(
                    group_intelligence.probabilities(provisional_regime)
                ),
            )
            if use_ai:
                _policy_started = time.perf_counter()
                decision = controller.decide(cognitive, policy_context)
                policy_inference_seconds += time.perf_counter() - _policy_started
                regime_probabilities = decision.regime_probabilities.copy()
                ai_operator_probabilities = decision.operator_probabilities.copy()
                adaptive = dict(decision.parameters)
            else:
                regime_probabilities = rule_based_regime_prior(cognitive)
                adaptive = {
                    "attraction": 0.65,
                    "differential": 0.35,
                    "exploration_sigma": 0.08,
                    "memory_weight": 0.35,
                    "diversity_weight": 0.18,
                    "recovery_fraction": 0.18,
                }
                ai_operator_probabilities = np.full(6, 1.0 / 6.0)

            severe_stagnation = (
                max(constraint_stagnation, objective_stagnation) >= stagnation_window
            )
            if severe_stagnation:
                recovery_prior = np.asarray([0.05, 0.10, 0.10, 0.75])
                regime_probabilities = self._normalise(
                    0.50 * regime_probabilities + 0.50 * recovery_prior
                )
            global_regime = (
                int(np.argmax(regime_probabilities))
                if deterministic_policy
                else int(self.rng.choice(4, p=self._normalise(regime_probabilities)))
            )

            # Behavior-driven search scale. The learned policy supplies the base scale;
            # cognition only modulates it within its declared training bounds.
            sigma = float(adaptive["exploration_sigma"])
            if severe_stagnation and current_diversity < 0.05:
                sigma *= 1.35
            elif current_diag.feasible_ratio >= 0.65 and objective_stagnation > 0:
                sigma *= 0.75
            adaptive["exploration_sigma"] = float(
                np.clip(sigma, PARAMETER_LOW[2], PARAMETER_HIGH[2])
            )

            contexts = classify_contexts(state.population, state.evaluations, violation_improving)
            consensus = pre_consensus
            readiness = pre_readiness
            learning_fraction = (
                lane_controller.learning_fraction(
                    readiness,
                    progress,
                    current_diversity,
                    severe_stagnation,
                )
                if use_dual_lane
                else 1.0
            )
            lanes = lane_controller.assign(
                population_size, learning_fraction, self.rng, deterministic_policy
            )

            precision_active = use_precision and precision.active(
                current_diag.feasible_ratio,
                min(objective_stagnation / stagnation_window, 1.0),
                progress,
                len(hpem),
            )
            precision_fraction = 0.0
            if precision_active:
                precision_fraction = float(
                    np.clip(
                        0.12
                        + 0.28 * min(objective_stagnation / stagnation_window, 1.0)
                        + 0.15 * max(progress - 0.70, 0.0) / 0.30,
                        0.12,
                        0.55,
                    )
                )

            # Operational recovery_fraction: under genuine stagnation/diversity collapse,
            # a bounded fraction of the weakest learners is assigned recovery proposals.
            forced_recovery: set[int] = set()
            if (
                use_diversity_recovery
                and severe_stagnation
                and current_diversity < float(parameters.get("recovery_diversity_threshold", 0.06))
            ):
                fraction = float(np.clip(adaptive["recovery_fraction"], 0.05, 0.45))
                count = max(1, min(population_size - 1, int(round(population_size * fraction))))
                quality = sorted(
                    range(population_size),
                    key=lambda i: epsilon_sort_key(state.evaluations[i], epsilon),
                    reverse=True,
                )
                forced_recovery = set(quality[:count])
                forced_recovery_evaluations += count

            _candidate_started = time.perf_counter()
            offspring = scratch.get("offspring", state.population.shape, np.float64)
            assigned_operators = np.full(population_size, -1, dtype=np.int8)
            assigned_memory = np.zeros(population_size, dtype=np.int8)
            assigned_groups = np.zeros(population_size, dtype=np.int8)
            individual_regimes = np.zeros(population_size, dtype=np.int8)
            precision_mask = np.zeros(population_size, dtype=bool)

            ai_policy_weight = float(np.clip(parameters.get("ai_policy_weight", 0.35), 0.0, 1.0))
            ai_credit_blend = float(np.clip(parameters.get("ai_credit_blend", 0.65), 0.0, 1.0))
            hierarchy = hpem.hierarchy() if len(hpem) else np.zeros((4, self.problem.dimension))
            # One temporary 3D broadcast fuses all learner-to-memory directions [P,4,D].
            # The scratch buffer is reused every batch and is never historized.
            memory_directions = scratch.get(
                "memory_directions",
                (population_size, 4, self.problem.dimension),
                np.float64,
            )
            np.subtract(
                hierarchy[None, :, :],
                state.population[:, None, :],
                out=memory_directions,
            )
            quality_order = self.order(state.evaluations)
            batch_best = state.population[quality_order[0]]
            batch_mean = state.population.mean(axis=0)
            batch_variables = getattr(getattr(self.problem, "decoder", None), "variables", None)

            for index in range(population_size):
                context = int(contexts[index])
                regime = self._individual_regime(global_regime, context)
                individual_regimes[index] = regime
                learned_lane = bool(lanes[index])

                memory_prior = REGIME_MEMORY_PRIORS[regime].copy()
                if not learned_lane:
                    memory_prior = DISCOVERY_MEMORY_PRIOR.copy()
                memory_online = credit.memory_probabilities(regime, context)
                memory_probabilities = blend_probabilities(memory_prior, memory_online, alpha=0.65)
                memory_level = (
                    int(np.argmax(memory_probabilities))
                    if deterministic_policy
                    else int(self.rng.choice(4, p=memory_probabilities))
                )
                assigned_memory[index] = memory_level

                group = (
                    group_intelligence.choose(regime, self.rng, deterministic_policy)
                    if use_variable_intelligence
                    else -1
                )
                assigned_groups[index] = group

                should_precision = (
                    precision_active
                    and learned_lane
                    and index not in forced_recovery
                    and (
                        deterministic_policy
                        and index < int(round(population_size * precision_fraction))
                        or (not deterministic_policy and self.rng.random() < precision_fraction)
                    )
                )
                if should_precision and len(hpem):
                    success_direction = memory.mean_direction(
                        self.problem.dimension,
                        regime=regime,
                        context=context,
                        group=group,
                    )
                    group_mask = group_intelligence.mask(group, self.problem.dimension)
                    offspring[index] = precision.propose(
                        hpem.best_vector,
                        hierarchy,
                        success_direction,
                        variables,
                        group_mask,
                        self.rng,
                        consensus,
                    )
                    precision_mask[index] = True
                    continue

                base_prior = self._rule_operator_probabilities(regime)
                learned_policy = self._normalise(
                    ai_policy_weight * ai_operator_probabilities
                    + (1.0 - ai_policy_weight) * base_prior
                )
                online = (
                    credit.operator_probabilities(regime, context)
                    if use_contextual_credit
                    else np.full(6, 1.0 / 6.0)
                )
                operator_probabilities = blend_probabilities(
                    learned_policy,
                    online,
                    alpha=ai_credit_blend,
                )
                if not learned_lane:
                    operator_probabilities = self._normalise(
                        0.45 * operator_probabilities + 0.55 * DISCOVERY_OPERATOR_PRIOR
                    )
                if not use_mixed_variable:
                    operator_probabilities[4] = 0.0
                if not use_diversity_recovery:
                    operator_probabilities[5] = 0.0
                operator_probabilities = self._normalise(operator_probabilities)

                if index in forced_recovery:
                    operator = 5
                    lanes[index] = 0
                    assigned_memory[index] = 3
                else:
                    operator = (
                        int(np.argmax(operator_probabilities))
                        if deterministic_policy
                        else int(self.rng.choice(6, p=operator_probabilities))
                    )
                assigned_operators[index] = operator
                offspring[index] = self._candidate(
                    operator,
                    index,
                    state,
                    memory,
                    hpem,
                    feasible_archive,
                    boundary_archive,
                    adaptive,
                    regime,
                    context,
                    int(assigned_memory[index]),
                    memory_directions[index, int(assigned_memory[index])],
                    group,
                    group_intelligence,
                    bool(lanes[index]),
                    best=batch_best,
                    mean=batch_mean,
                    variables=batch_variables,
                )
            candidate_generation_seconds += time.perf_counter() - _candidate_started

            _evaluation_started = time.perf_counter()
            offspring_evaluations = (
                cache.evaluate_requests(self, offspring)
                if use_evaluation_cache
                else self.evaluate_population(offspring)
            )
            evaluator_seconds += time.perf_counter() - _evaluation_started
            if len(offspring_evaluations) != len(offspring):
                break

            _learning_started = time.perf_counter()
            successful = np.zeros(population_size, dtype=bool)
            objective_gain = np.zeros(population_size, dtype=float)
            feasibility_gain = np.zeros(population_size, dtype=float)
            feasibility_transition = np.zeros(population_size, dtype=float)
            step_norm = np.linalg.norm(offspring - state.population, axis=1)
            offspring_pb = state.personal_best.copy()
            offspring_pb_ev = list(state.personal_best_evaluations)

            precision_batch_success = 0
            precision_batch_attempts = int(np.count_nonzero(precision_mask))
            precision_evaluations += precision_batch_attempts

            for index, (child, child_ev) in enumerate(zip(offspring, offspring_evaluations)):
                parent_ev = state.evaluations[index]
                successful[index] = epsilon_better(child_ev, parent_ev, epsilon)
                if parent_ev.feasible and child_ev.feasible and np.isfinite(parent_ev.value):
                    objective_gain[index] = max(
                        (float(parent_ev.value) - float(child_ev.value))
                        / max(abs(float(parent_ev.value)), 1.0),
                        0.0,
                    )
                parent_violation = float(parent_ev.violation)
                child_violation = float(child_ev.violation)
                if np.isposinf(parent_violation) and np.isfinite(child_violation):
                    feasibility_gain[index] = np.inf
                elif np.isfinite(parent_violation) and np.isfinite(child_violation):
                    feasibility_gain[index] = max(parent_violation - child_violation, 0.0)
                feasibility_transition[index] = float(not parent_ev.feasible and child_ev.feasible)

                # Persistent pbest uses exact common feasibility-first dominance, not temporary
                # epsilon-feasibility, so a feasible personal record cannot be replaced by a merely
                # epsilon-feasible point with a lower raw objective.
                if better(child_ev, offspring_pb_ev[index]):
                    offspring_pb[index] = child.copy()
                    offspring_pb_ev[index] = child_ev

                if successful[index] and use_memory:
                    memory_operator = 6 if precision_mask[index] else int(assigned_operators[index])
                    memory.add(
                        child - state.population[index],
                        memory_operator,
                        objective_gain[index],
                        feasibility_gain[index],
                        regime=int(individual_regimes[index]),
                        context=int(contexts[index]),
                        group=int(assigned_groups[index]),
                    )
                if precision_mask[index] and successful[index]:
                    precision_batch_success += 1

            if use_contextual_credit:
                credit.batch_update(
                    individual_regimes,
                    contexts,
                    assigned_operators,
                    assigned_memory,
                    successful,
                    objective_gain,
                    feasibility_gain,
                    feasibility_transition,
                )
            if use_variable_intelligence:
                group_intelligence.batch_update(
                    individual_regimes,
                    assigned_groups,
                    successful,
                    objective_gain,
                    feasibility_gain,
                    step_norm,
                )
            precision.update(precision_batch_attempts, precision_batch_success)
            precision_successes += precision_batch_success

            combined_population = np.vstack([state.population, offspring])
            combined_evaluations = list(state.evaluations) + list(offspring_evaluations)
            _, _, selected_indices = environmental_select(
                combined_population,
                combined_evaluations,
                population_size,
                epsilon,
                diversity_weight=float(adaptive["diversity_weight"]),
                return_indices=True,
            )
            state.select_from_combined(
                combined_population,
                combined_evaluations,
                selected_indices,
                offspring_pb,
                offspring_pb_ev,
            )

            if use_dual_archives:
                feasible_archive.update(combined_population, combined_evaluations)
                boundary_archive.update(combined_population, combined_evaluations)
            else:
                feasible_archive.entries = []
                boundary_archive.entries = []
                feasible_archive.update(state.population, state.evaluations)
                boundary_archive.update(state.population, state.evaluations)
            if use_hpem:
                hpem.update(combined_population, combined_evaluations)

            new_diag = population_diagnostics(state.evaluations, epsilon)
            new_diversity = population_diversity(state.population)
            reward = calculate_reward(
                current_diag.best_feasible_objective,
                new_diag.best_feasible_objective,
                current_diag.best_violation,
                new_diag.best_violation,
                current_diag.feasible_ratio,
                new_diag.feasible_ratio,
                current_diversity,
                new_diversity,
            )
            reward_history.append(float(reward.total))

            violation_improving = new_diag.best_violation < current_diag.best_violation - 1e-12
            constraint_stagnation = 0 if violation_improving else constraint_stagnation + 1
            objective_improving = (
                np.isfinite(new_diag.best_feasible_objective)
                and new_diag.best_feasible_objective < current_diag.best_feasible_objective - 1e-12
            )
            if objective_improving:
                objective_stagnation = 0
            elif np.isfinite(new_diag.best_feasible_objective):
                objective_stagnation += 1

            previous_best_violation = new_diag.best_violation
            previous_best_objective = new_diag.best_feasible_objective
            usage = Counter(int(op) for op in assigned_operators if op >= 0)
            operator_usage_history.append(
                {OPERATOR_NAMES[k]: int(usage.get(k, 0)) for k in range(6)}
            )
            rates = credit.success_rates()
            operator_success_history.append({OPERATOR_NAMES[k]: float(rates[k]) for k in range(6)})
            regime_history.append(REGIME_NAMES[global_regime])
            memory_readiness_history.append(readiness)
            learning_lane_history.append(float(np.mean(lanes)))
            memory_consensus_history.append(consensus)
            precision_radius_history.append(float(precision.radius))

            diagnostics_history["best_total_violation"].append(new_diag.best_violation)
            diagnostics_history["mean_total_violation"].append(new_diag.mean_violation)
            diagnostics_history["feasible_ratio"].append(new_diag.feasible_ratio)
            diagnostics_history["epsilon_feasible_ratio"].append(new_diag.epsilon_feasible_ratio)
            diagnostics_history["population_diversity"].append(new_diversity)
            diagnostics_history["elite_diversity"].append(cognitive.elite_spread)
            diagnostics_history["epsilon"].append(epsilon)
            for key in CONSTRAINT_COMPONENTS:
                diagnostics_history[f"best_{key}"].append(new_diag.component_best.get(key, 0.0))
                diagnostics_history[f"mean_{key}"].append(new_diag.component_mean.get(key, 0.0))

            dominant_operator = (
                int(
                    np.argmax(np.bincount(assigned_operators[assigned_operators >= 0], minlength=6))
                )
                if np.any(assigned_operators >= 0)
                else 4
            )
            adaptive_vector = np.asarray([adaptive[name] for name in PARAMETER_NAMES], dtype=float)
            raw_parameter_action = np.clip(
                (adaptive_vector - PARAMETER_LOW)
                / np.maximum(PARAMETER_HIGH - PARAMETER_LOW, 1e-12),
                0.0,
                1.0,
            )
            if bool(parameters.get("record_policy_trajectory", True)):
                policy_trajectory.append(
                    {
                        "state": cognitive.vector().tolist(),
                        "regime": int(global_regime),
                        "operator": int(dominant_operator),
                        "parameter": raw_parameter_action.tolist(),
                        "reward": float(reward.total),
                        "evaluations": int(self.evaluations),
                        "source_policy": "ai" if use_ai else "rule_based",
                    }
                )

            learning_update_seconds += time.perf_counter() - _learning_started
            self.record(
                {
                    "calo_operator": OPERATOR_NAMES[dominant_operator],
                    "calo_regime": REGIME_NAMES[global_regime],
                    "operator_success_rates": {
                        OPERATOR_NAMES[k]: float(rates[k]) for k in range(6)
                    },
                    "diversity": new_diversity,
                    "elite_diversity": cognitive.elite_spread,
                    "feasible_ratio": new_diag.feasible_ratio,
                    "epsilon_feasible_ratio": new_diag.epsilon_feasible_ratio,
                    "epsilon": epsilon,
                    "constraint_components": dict(new_diag.component_best),
                    "reward": float(reward.total),
                    "feasible_archive_size": len(feasible_archive),
                    "boundary_archive_size": len(boundary_archive),
                    "hpem_size": len(hpem),
                    "memory_readiness": readiness,
                    "memory_consensus": consensus,
                    "learning_lane_fraction": float(np.mean(lanes)),
                    "precision_active": bool(precision_active),
                    "precision_radius": float(precision.radius),
                    "forced_recovery_candidates": int(len(forced_recovery)),
                }
            )

            if run_checkpoint_path and self.evaluations >= next_checkpoint_evaluation:
                self._save_run_checkpoint(
                    run_checkpoint_path,
                    parameters=parameters,
                    controller=controller,
                    locals_payload={
                        "state": state,
                        "feasible_archive": feasible_archive,
                        "boundary_archive": boundary_archive,
                        "hpem": hpem,
                        "memory": memory,
                        "credit": credit,
                        "group_intelligence": group_intelligence,
                        "lane_controller": lane_controller,
                        "precision": precision,
                        "epsilon_controller": epsilon_controller,
                        "diagnostics_history": diagnostics_history,
                        "operator_usage_history": operator_usage_history,
                        "operator_success_history": operator_success_history,
                        "regime_history": regime_history,
                        "reward_history": reward_history,
                        "memory_readiness_history": memory_readiness_history,
                        "learning_lane_history": learning_lane_history,
                        "memory_consensus_history": memory_consensus_history,
                        "precision_radius_history": precision_radius_history,
                        "previous_best_violation": previous_best_violation,
                        "previous_best_objective": previous_best_objective,
                        "constraint_stagnation": constraint_stagnation,
                        "objective_stagnation": objective_stagnation,
                        "violation_improving": violation_improving,
                        "policy_trajectory": policy_trajectory,
                        "precision_evaluations": precision_evaluations,
                        "precision_successes": precision_successes,
                        "forced_recovery_evaluations": forced_recovery_evaluations,
                        "batch_count": batch_count,
                        "policy_inference_seconds": policy_inference_seconds,
                        "candidate_generation_seconds": candidate_generation_seconds,
                        "evaluator_seconds": evaluator_seconds,
                        "learning_update_seconds": learning_update_seconds,
                        "historical_warm_start_count": historical_warm_start_count,
                    },
                )
                next_checkpoint_evaluation = (
                    (int(self.evaluations) // checkpoint_interval) + 1
                ) * checkpoint_interval

        # Persist the terminal state too, including a run that stopped at its original FE horizon.
        # This is the exact state used for later v5 horizon extension.
        if run_checkpoint_path:
            self._save_run_checkpoint(
                run_checkpoint_path,
                parameters=parameters,
                controller=controller,
                locals_payload={
                    "state": state,
                    "feasible_archive": feasible_archive,
                    "boundary_archive": boundary_archive,
                    "hpem": hpem,
                    "memory": memory,
                    "credit": credit,
                    "group_intelligence": group_intelligence,
                    "lane_controller": lane_controller,
                    "precision": precision,
                    "epsilon_controller": epsilon_controller,
                    "diagnostics_history": diagnostics_history,
                    "operator_usage_history": operator_usage_history,
                    "operator_success_history": operator_success_history,
                    "regime_history": regime_history,
                    "reward_history": reward_history,
                    "memory_readiness_history": memory_readiness_history,
                    "learning_lane_history": learning_lane_history,
                    "memory_consensus_history": memory_consensus_history,
                    "precision_radius_history": precision_radius_history,
                    "previous_best_violation": previous_best_violation,
                    "previous_best_objective": previous_best_objective,
                    "constraint_stagnation": constraint_stagnation,
                    "objective_stagnation": objective_stagnation,
                    "violation_improving": violation_improving,
                    "policy_trajectory": policy_trajectory,
                    "precision_evaluations": precision_evaluations,
                    "precision_successes": precision_successes,
                    "forced_recovery_evaluations": forced_recovery_evaluations,
                    "batch_count": batch_count,
                    "policy_inference_seconds": policy_inference_seconds,
                    "candidate_generation_seconds": candidate_generation_seconds,
                    "evaluator_seconds": evaluator_seconds,
                    "learning_update_seconds": learning_update_seconds,
                    "historical_warm_start_count": historical_warm_start_count,
                },
            )

        hpem_snapshot = hpem.snapshot()
        metadata = {
            "calo_version": "v5.0",
            "architecture": "constraint-cognitive tensor-native HPEM dual-lane precision",
            "operator_names": list(OPERATOR_NAMES),
            "operator_attempts": credit.attempts.tolist(),
            "operator_successes": credit.successes.tolist(),
            "operator_credit": credit.global_operator_probabilities().tolist(),
            "contextual_operator_credit_shape": list(credit.operator_credit.shape),
            "contextual_memory_credit_shape": list(credit.memory_credit.shape),
            "group_stats_shape": list(group_intelligence.stats.shape),
            "success_memory_shape": (
                list(memory.directions.shape)
                if memory.directions is not None
                else [7, memory.slots, 0]
            ),
            "mean_reward": float(np.mean(reward_history)) if reward_history else 0.0,
            "reward_history": reward_history,
            "success_memory_size": len(memory),
            "feasible_archive_size": len(feasible_archive),
            "boundary_archive_size": len(boundary_archive),
            "hpem": {
                "canonical_shape": list(hpem_snapshot.vectors.shape),
                "hierarchy_shape": list(hpem_snapshot.hierarchy.shape),
                "objectives": hpem_snapshot.objectives.tolist(),
                "occupancy": int(hpem_snapshot.occupancy),
            },
            "memory_readiness_history": memory_readiness_history,
            "learning_lane_history": learning_lane_history,
            "memory_consensus_history": memory_consensus_history,
            "precision_radius_history": precision_radius_history,
            "precision_evaluations": int(precision_evaluations),
            "precision_successes": int(precision_successes),
            "forced_recovery_evaluations": int(forced_recovery_evaluations),
            "physical_solver_calls": int(cache.physical_solver_calls)
            if use_evaluation_cache
            else int(self.evaluations),
            "scratch_pool_bytes": int(scratch.allocated_bytes),
            "exact_cache_hits": int(cache.cache_hits) if use_evaluation_cache else 0,
            "exact_cache_hit_rate": float(cache.hit_rate) if use_evaluation_cache else 0.0,
            "exact_cache_persistent_enabled": bool(cache.persistent_enabled)
            if use_evaluation_cache
            else False,
            "runtime_profile": {
                "policy_inference_seconds": float(policy_inference_seconds),
                "candidate_generation_seconds": float(candidate_generation_seconds),
                "evaluator_seconds": float(evaluator_seconds),
                "learning_update_seconds": float(learning_update_seconds),
                "control_seconds": float(
                    candidate_generation_seconds
                    + learning_update_seconds
                    + policy_inference_seconds
                ),
            },
            "diagnostics_history": diagnostics_history,
            "operator_usage_history": operator_usage_history,
            "operator_success_history": operator_success_history,
            "regime_history": regime_history,
            "policy_checkpoint": controller.checkpoint_path,
            "policy_checksum": controller.checksum,
            "policy_metadata": controller.metadata,
            "policy_inference_device": str(controller.device),
            "policy_state_schema": dict(getattr(controller, "schema", {}) or {}),
            "policy_binding": {
                "policy_id": str(parameters.get("policy_id", "")),
                "sha256": str(parameters.get("policy_sha256", controller.checksum)),
                "state_schema_version": str(
                    parameters.get(
                        "policy_state_schema_version",
                        getattr(controller, "schema", {}).get("state_schema_version", ""),
                    )
                ),
                "action_schema_version": str(
                    parameters.get(
                        "policy_action_schema_version",
                        getattr(controller, "schema", {}).get("action_schema_version", ""),
                    )
                ),
            },
            "policy_cross_run_batched_inference": bool(controller.batched_inference),
            "policy_trajectory": policy_trajectory,
            "historical_learning": {
                "strict_benchmark_mode": bool(parameters.get("strict_benchmark_mode", True)),
                "repository": historical_repository_path,
                "repository_sha256": (
                    historical_repository.payload.get("repository_sha256", "")
                    if historical_repository is not None
                    else ""
                ),
                "parameter_priors_enabled": bool(
                    parameters.get("use_historical_parameter_priors", False)
                ),
                "parameter_priors_applied": historical_prior_applied,
                "cross_algorithm_warm_start_enabled": bool(
                    parameters.get("use_cross_algorithm_warm_start", False)
                ),
                "warm_start_count": historical_warm_start_count,
            },
            "run_continuation": {
                "supports_exact_resume": True,
                "resumed_from": resume_run_checkpoint,
                "checkpoint_path": run_checkpoint_path,
                "checkpoint_interval_evaluations": checkpoint_interval,
            },
            "ablation": {
                "use_ai": use_ai,
                "use_memory": use_memory,
                "use_dual_archives": use_dual_archives,
                "use_epsilon": use_epsilon,
                "use_mixed_variable": use_mixed_variable,
                "use_diversity_recovery": use_diversity_recovery,
                "use_hpem": use_hpem,
                "use_contextual_credit": use_contextual_credit,
                "use_variable_intelligence": use_variable_intelligence,
                "use_dual_lane": use_dual_lane,
                "use_cognitive_precision": use_precision,
                "use_exact_evaluation_cache": use_evaluation_cache,
            },
        }
        return self.finalize(state.population, metadata=metadata, started=started)
