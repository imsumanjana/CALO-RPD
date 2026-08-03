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
import functools
import inspect
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer
from calo_rpd_studio.accelerated.scratch_pool import ScratchPool
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
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
from .environmental_selection import epsilon_sort_key
from .evaluation_cache import ExactEvaluationCache
from .hierarchical_memory import HierarchicalPrefixEliteMemory
from .learning_operators import OPERATOR_NAMES
from .precision_engine import CognitivePrecisionEngine
from .policy_schema import (
    PolicyRuntimeContext,
    variable_group_concentration,
    build_policy_vector,
    POLICY_STATE_DIM,
)
from .success_memory import SuccessMemory
from .tensor_state import CALOTensorState
from .run_checkpoint import save_exact_run_checkpoint, load_exact_run_checkpoint
from .transition_kernel import (
    complete_transition,
    evaluate_candidates,
    generate_offspring,
    normalise_probabilities,
)
from .variable_intelligence import VariableGroupIntelligence


class CALOOptimizer(BaseOptimizer):
    name = "CALO"
    supports_exact_resume = True

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        return normalise_probabilities(values)

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
                scientific_problem_fingerprint=self._problem_compatibility_fingerprint(),
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
        if isinstance(value, (set, frozenset)):
            items = [CALOOptimizer._compatibility_jsonable(v) for v in value]
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        if isinstance(value, Path):
            return {"path": str(value)}
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
        if isinstance(value, functools.partial):
            return {
                "callable_kind": "functools.partial",
                "func": CALOOptimizer._compatibility_jsonable(value.func),
                "args": CALOOptimizer._compatibility_jsonable(value.args),
                "keywords": CALOOptimizer._compatibility_jsonable(value.keywords or {}),
            }
        if inspect.ismethod(value):
            owner = getattr(value, "__self__", None)
            owner_state = {}
            if owner is not None and not inspect.isclass(owner):
                try:
                    owner_state = CALOOptimizer._compatibility_jsonable(vars(owner))
                except TypeError:
                    owner_state = {"type": f"{type(owner).__module__}.{type(owner).__qualname__}"}
            return {
                "callable_kind": "bound_method",
                "function": CALOOptimizer._compatibility_jsonable(value.__func__),
                "owner_type": f"{type(owner).__module__}.{type(owner).__qualname__}"
                if owner is not None
                else "",
                "owner_state": owner_state,
            }
        if callable(value):
            defaults = getattr(value, "__defaults__", None)
            kwdefaults = getattr(value, "__kwdefaults__", None)
            closure = getattr(value, "__closure__", None) or ()
            closure_values = []
            for cell in closure:
                try:
                    closure_values.append(CALOOptimizer._compatibility_jsonable(cell.cell_contents))
                except (ValueError, TypeError):
                    closure_values.append({"unreadable_closure_cell": True})
            code = getattr(value, "__code__", None)
            if code is None and not inspect.isbuiltin(value):
                # Callable instances carry scientific parameters in instance state and executable
                # identity in their class __call__ implementation. Both are required to avoid
                # partial/callable-object compatibility collisions.
                call_impl = getattr(type(value), "__call__", None)
                try:
                    state = CALOOptimizer._compatibility_jsonable(vars(value))
                except TypeError:
                    state = {}
                if call_impl is None and not state:
                    raise ValueError(
                        f"Cannot safely canonicalize callable scientific transform {type(value)!r}"
                    )
                return {
                    "callable_kind": "callable_object",
                    "class": f"{type(value).__module__}.{type(value).__qualname__}",
                    "call_impl": CALOOptimizer._compatibility_jsonable(call_impl)
                    if call_impl is not None
                    else None,
                    "state": state,
                }
            code_identity = None
            if code is not None:
                code_payload = {
                    "co_code_sha256": hashlib.sha256(code.co_code).hexdigest(),
                    "co_consts": CALOOptimizer._compatibility_jsonable(code.co_consts),
                    "co_names": list(code.co_names),
                    "co_varnames": list(code.co_varnames),
                    "co_freevars": list(code.co_freevars),
                }
                encoded_code = json.dumps(
                    code_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                code_identity = hashlib.sha256(encoded_code).hexdigest()
            return {
                "callable_kind": "function" if code is not None else "builtin",
                "callable_module": str(getattr(value, "__module__", "")),
                "callable_qualname": str(getattr(value, "__qualname__", type(value).__qualname__)),
                "defaults": CALOOptimizer._compatibility_jsonable(defaults or ()),
                "kwdefaults": CALOOptimizer._compatibility_jsonable(kwdefaults or {}),
                "closure": closure_values,
                "code_identity_sha256": code_identity,
            }
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "__dict__"):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": CALOOptimizer._compatibility_jsonable(vars(value)),
            }
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    def _problem_compatibility_fingerprint(self) -> str:
        return scientific_problem_fingerprint(self.problem)

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
        if population_size <= 0 or int(self.config.max_evaluations) % population_size != 0:
            raise ValueError(
                "CALO strict FE fairness requires max_evaluations to be an exact multiple of "
                f"population_size; got {self.config.max_evaluations} and {population_size}. "
                "Use a divisible budget so CALO and every comparator are assigned identical requested FEs."
            )
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
                    scientific_problem_fingerprint=self._problem_compatibility_fingerprint(),
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
            raise ValueError(
                "CALO policy-assisted execution requires an explicitly imported/trained and "
                "activated policy checkpoint. No default or untrained fallback policy is permitted."
            )
        if use_ai and not bool(parameters.get("strict_policy_binding", False)):
            raise ValueError(
                "CALO policy-assisted execution requires strict immutable policy binding. "
                "Reapply CALO Intelligence after explicitly activating the intended policy."
            )
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
            allow_no_policy=not use_ai,
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

            # Native v5.9 policies observe the 24-D cognitive base plus compact HPEM,
            # dual-lane, success-memory, precision, and variable-intelligence signals (32-D total).
            # Legacy checkpoints remain isolated behind explicit compatibility/migration adapters.
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
            native_v59_policy = False
            decision = None
            if use_ai:
                _policy_started = time.perf_counter()
                decision = controller.decide(cognitive, policy_context)
                policy_inference_seconds += time.perf_counter() - _policy_started
                native_v59_policy = bool(controller.schema.get("native_v59", False))
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
            # Native v5.9 action semantics: the network's sampled global regime is the raw
            # policy action. Stagnation/recovery remain explicit controller interventions rather
            # than silently changing the PPO action distribution after sampling. Legacy/no-AI
            # behavior keeps the historical prior blend for backward-compatible trajectories.
            if severe_stagnation and not native_v59_policy:
                recovery_prior = np.asarray([0.05, 0.10, 0.10, 0.75])
                regime_probabilities = self._normalise(
                    0.50 * regime_probabilities + 0.50 * recovery_prior
                )
            if native_v59_policy and decision is not None:
                global_regime = int(decision.regime)
            else:
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
            # deterministic_policy controls only the neural policy action for native v5.9.
            # Environmental/controller stochasticity remains part of the transition kernel and
            # therefore uses the optimizer RNG in both PPO rollouts and deployed CALO.
            environment_deterministic = bool(deterministic_policy and not native_v59_policy)
            lanes = lane_controller.assign(
                population_size, learning_fraction, self.rng, environment_deterministic
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
            offspring_buffer = scratch.get("offspring", state.population.shape, np.float64)
            ai_policy_weight = float(np.clip(parameters.get("ai_policy_weight", 0.35), 0.0, 1.0))
            ai_credit_blend = float(np.clip(parameters.get("ai_credit_blend", 0.65), 0.0, 1.0))
            quality_order = self.order(state.evaluations)
            batch_variables = getattr(getattr(self.problem, "decoder", None), "variables", None)
            candidate_batch = generate_offspring(
                population=state.population,
                evaluations=state.evaluations,
                personal_best=state.personal_best,
                rng=self.rng,
                dimension=self.problem.dimension,
                variables=batch_variables,
                quality_order=quality_order,
                contexts=contexts,
                learned_lanes=lanes,
                global_regime=global_regime,
                raw_operator=int(decision.operator) if decision is not None else -1,
                native_policy=bool(native_v59_policy and decision is not None),
                ai_operator_probabilities=ai_operator_probabilities,
                adaptive=adaptive,
                memory=memory,
                hpem=hpem,
                feasible_archive=feasible_archive,
                boundary_archive=boundary_archive,
                credit=credit,
                group_intelligence=group_intelligence,
                precision=precision,
                precision_active=precision_active,
                precision_fraction=precision_fraction,
                forced_recovery=forced_recovery,
                consensus=consensus,
                environment_deterministic=environment_deterministic,
                use_mixed_variable=use_mixed_variable,
                use_diversity_recovery=use_diversity_recovery,
                use_contextual_credit=use_contextual_credit,
                use_variable_intelligence=use_variable_intelligence,
                ai_policy_weight=ai_policy_weight,
                ai_credit_blend=ai_credit_blend,
                out=offspring_buffer,
            )
            offspring = candidate_batch.offspring
            assigned_operators = candidate_batch.assigned_operators
            assigned_memory = candidate_batch.assigned_memory
            assigned_groups = candidate_batch.assigned_groups
            individual_regimes = candidate_batch.individual_regimes
            precision_mask = candidate_batch.precision_mask
            lanes = candidate_batch.learned_lanes
            candidate_generation_seconds += time.perf_counter() - _candidate_started

            _evaluation_started = time.perf_counter()
            evaluation_batch = evaluate_candidates(
                offspring,
                lambda values: (
                    cache.evaluate_requests(self, values)
                    if use_evaluation_cache
                    else self.evaluate_population(values)
                ),
            )
            evaluator_seconds += time.perf_counter() - _evaluation_started
            offspring_evaluations = evaluation_batch.evaluations
            if not evaluation_batch.complete:
                break

            _learning_started = time.perf_counter()
            transition = complete_transition(
                population=state.population,
                evaluations=state.evaluations,
                personal_best=state.personal_best,
                personal_best_evaluations=state.personal_best_evaluations,
                offspring=offspring,
                offspring_evaluations=offspring_evaluations,
                epsilon=epsilon,
                assigned_operators=assigned_operators,
                assigned_memory=assigned_memory,
                assigned_groups=assigned_groups,
                individual_regimes=individual_regimes,
                contexts=contexts,
                precision_mask=precision_mask,
                memory=memory,
                credit=credit,
                group_intelligence=group_intelligence,
                precision=precision,
                feasible_archive=feasible_archive,
                boundary_archive=boundary_archive,
                hpem=hpem,
                old_diagnostics=current_diag,
                old_diversity=current_diversity,
                diversity_weight=float(adaptive["diversity_weight"]),
                population_size=population_size,
                use_memory=use_memory,
                use_contextual_credit=use_contextual_credit,
                use_variable_intelligence=use_variable_intelligence,
                use_dual_archives=use_dual_archives,
                use_hpem=use_hpem,
            )
            precision_evaluations += transition.precision_attempts
            precision_successes += transition.precision_successes
            state.select_from_combined(
                transition.combined_population,
                transition.combined_evaluations,
                transition.selected_indices,
                transition.offspring_personal_best,
                transition.offspring_personal_best_evaluations,
            )
            new_diag = transition.new_diagnostics
            new_diversity = transition.new_diversity
            reward = transition.reward
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
            derived_parameter_action = np.clip(
                (adaptive_vector - PARAMETER_LOW)
                / np.maximum(PARAMETER_HIGH - PARAMETER_LOW, 1e-12),
                0.0,
                1.0,
            )
            if bool(parameters.get("record_policy_trajectory", True)):
                full_policy_state = build_policy_vector(
                    cognitive, policy_context, input_dim=POLICY_STATE_DIM
                )
                raw_policy_record = {
                    "regime": int(decision.regime) if decision is not None else int(global_regime),
                    "operator": int(decision.operator) if decision is not None else -1,
                    "regime_probabilities": (
                        decision.regime_probabilities.tolist()
                        if decision is not None
                        else regime_probabilities.tolist()
                    ),
                    "operator_probabilities": (
                        decision.operator_probabilities.tolist()
                        if decision is not None
                        else ai_operator_probabilities.tolist()
                    ),
                    "parameter": (
                        decision.raw_parameter_action.tolist()
                        if decision is not None
                        else derived_parameter_action.tolist()
                    ),
                    "value_estimate": float(decision.value_estimate)
                    if decision is not None
                    else None,
                }
                policy_trajectory.append(
                    {
                        "schema_version": "calo-policy-trajectory-v5.9",
                        "policy_state": full_policy_state.tolist(),
                        "cognitive_state": cognitive.vector().tolist(),
                        "raw_policy": raw_policy_record,
                        "executed_controller": {
                            "global_regime": int(global_regime),
                            "individual_regimes": individual_regimes.astype(int).tolist(),
                            "executed_operators": assigned_operators.astype(int).tolist(),
                            "memory_levels": assigned_memory.astype(int).tolist(),
                            "variable_groups": assigned_groups.astype(int).tolist(),
                            "precision_mask": precision_mask.astype(bool).tolist(),
                            "forced_recovery_indices": sorted(int(i) for i in forced_recovery),
                            "final_parameters": {
                                name: float(adaptive[name]) for name in PARAMETER_NAMES
                            },
                        },
                        # Legacy aliases remain for repository readers, but exact v5.9 pretraining
                        # uses policy_state/raw_policy above rather than these composite summaries.
                        "state": full_policy_state.tolist(),
                        "regime": int(raw_policy_record["regime"]),
                        "operator": int(raw_policy_record["operator"]),
                        "parameter": list(raw_policy_record["parameter"]),
                        "parameter_supervision": bool(decision is not None),
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
            "calo_version": "v5.9",
            "scientific_problem_fingerprint": self._problem_compatibility_fingerprint(),
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
            "exact_cache_persistent_enabled_final": bool(cache.persistent_enabled)
            if use_evaluation_cache
            else False,
            "exact_cache_adaptation_requests": int(cache.minimum_requests_before_adaptation)
            if use_evaluation_cache
            else 0,
            "exact_cache_persistent_enabled": bool(cache.persistent_enabled)
            if use_evaluation_cache
            else False,
            "runtime_profile": {
                "wall_seconds": float(max(time.perf_counter() - started, 1e-12)),
                "end_to_end_requested_fe_per_second": float(
                    self.evaluations / max(time.perf_counter() - started, 1e-12)
                ),
                "metric_definition": "full_CALO_control_plus_evaluator_wall_clock",
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
