"""End-to-end TSH-CALO optimizer consuming an immutable qualified policy ensemble."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint

from .adaptive_epsilon import AdaptiveEpsilonController
from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import (
    build_cognitive_state,
    population_diversity,
    rule_based_regime_prior,
)
from .contextual_credit import ContextualCredit, classify_contexts
from .diagnostics import population_diagnostics
from .dual_lane_controller import DualLaneController
from .environmental_selection import epsilon_sort_key
from .hierarchical_memory import HierarchicalPrefixEliteMemory
from .optimizer import CALOOptimizer
from .policy_schema import POLICY_STATE_DIM, PolicyRuntimeContext, build_policy_vector
from .precision_engine import CognitivePrecisionEngine
from .run_checkpoint import load_exact_run_checkpoint, save_exact_run_checkpoint
from .success_memory import SuccessMemory
from .tensor_state import CALOTensorState
from .transition_kernel import evaluate_candidates
from .tsh_calo_inference import TSHCALOInferenceController
from .tsh_calo_policy import GroupActionMask
from .tsh_calo_physics_repair import (
    PhysicsRepairConfig,
    PhysicsRepairOperator,
    physics_repair_context_from_counted_evaluation,
    physics_repair_context_is_usable,
)
from .tsh_calo_runtime_context import build_runtime_topology_policy_context
from .tsh_calo_schema import (
    N_OPERATORS,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSHCALOFeatureFlags,
)
from .tsh_calo_shield import (
    FallbackDisposition,
    OODCalibration,
    SafetyEnvelope,
    SlidingWindowContextualBandit,
)
from .tsh_calo_transition_kernel import (
    complete_tsh_transition,
    effective_group_parameter_values,
    effective_recovery_fraction,
    generate_tsh_offspring,
)
from .variable_intelligence import VariableGroupIntelligence


class TSHCALOPolicyRejected(RuntimeError):
    """Raised before unsafe continuation when the bound policy cannot govern TSH-CALO."""


class TSHCALOBaselineFallbackRequired(TSHCALOPolicyRejected):
    """Signals an explicit relaunch as frozen CALO; never relabels that run as TSH-CALO."""

    algorithm_identity = "CALO-v5.9"


def _normalise_first_six(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)[:6]
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full(6, 1.0 / 6.0)


def _learner_bandit_rewards(transition) -> np.ndarray:
    objective = np.asarray(transition.objective_gain, dtype=float)
    feasibility = np.asarray(transition.feasibility_gain, dtype=float)

    def scale(values: np.ndarray) -> np.ndarray:
        positive_infinity = np.isposinf(values)
        finite = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
        peak = float(np.max(finite, initial=0.0))
        scaled = finite / peak if peak > 0.0 else np.zeros_like(finite)
        scaled[positive_infinity] = 1.0
        return scaled

    rewards = (
        0.40 * scale(objective)
        + 0.35 * scale(feasibility)
        + 0.15 * np.asarray(transition.successful, dtype=float)
        + 0.10 * np.asarray(transition.feasibility_transition, dtype=float)
    )
    if not np.all(np.isfinite(rewards)):
        raise RuntimeError("TSH-CALO bandit reward construction produced non-finite values")
    return rewards


def _ood_calibration(parameters: dict) -> OODCalibration:
    payload = dict(parameters.get("ood_calibration", {}) or {})
    required = {"mean", "scale"}
    if not required.issubset(payload):
        raise ValueError("TSH-CALO requires a frozen qualified OOD calibration")
    calibration = OODCalibration(
        np.asarray(payload["mean"], dtype=float),
        np.asarray(payload["scale"], dtype=float),
        float(payload.get("attenuation_start", 2.0)),
        float(payload.get("minimum_neural_weight", 0.0)),
    )
    calibration.validate()
    return calibration


class TSHCALOOptimizer(BaseOptimizer):
    """Topology-aware, hierarchical CALO with policy-first fail-closed execution."""

    name = TSH_CALO_ALGORITHM_ID
    supports_exact_resume = True
    CHECKPOINT_SCHEMA = "tsh-calo-exact-runtime-v1"

    def _physics_repair_enabled(self, features: TSHCALOFeatureFlags) -> bool:
        """Return the production feature choice; ablation subclasses may only remove E."""

        return bool(features.physics_repair)

    def _build_inference_controller(
        self, parameters: dict, calibration: OODCalibration
    ) -> TSHCALOInferenceController:
        """Build the production controller; qualification overrides this non-public seam."""

        return TSHCALOInferenceController(
            parameters,
            ood_calibration=calibration,
            expected_ood_calibration_sha256=str(
                parameters.get("policy_ood_calibration_sha256", "")
            ),
            deterministic=bool(parameters.get("deterministic_policy", False)),
            seed=int(parameters.get("ai_inference_seed", self.seed + 7919)),
            requested_device=str(parameters.get("inference_device", "auto")),
            allow_cpu_fallback=bool(parameters.get("allow_cpu_fallback", True)),
            baseline_fallback_permitted=bool(parameters.get("baseline_fallback_permitted", False)),
        )

    def _evaluate_population_with_context(self, population):
        evaluator = getattr(self.problem, "evaluate_with_context", None)
        if not callable(evaluator):
            raise TypeError("TSH-CALO requires the counted ORPD evaluate_with_context API")
        rows = np.asarray(population, dtype=float)
        remaining = max(0, int(self.config.max_evaluations) - int(self.evaluations))
        if remaining <= 0 or self.cancelled():
            return [], []
        clipped_rows = np.asarray(
            [self._repair_to_bounds(row) for row in rows[:remaining]], dtype=float
        )
        batch_evaluator = getattr(self.problem, "evaluate_population_with_context", None)
        if callable(batch_evaluator):
            records = list(
                batch_evaluator(
                    clipped_rows,
                    retain_control_linearization=bool(
                        getattr(self, "_retain_control_linearization", False)
                    ),
                )
            )
            if len(records) != len(clipped_rows):
                raise RuntimeError(
                    "Counted ORPD batch-context evaluator returned an incomplete population"
                )
            evaluations = [
                self._register_evaluation(clipped, record[0])
                for clipped, record in zip(clipped_rows, records, strict=True)
            ]
            return evaluations, [record[1] for record in records]

        evaluations: list[object] = []
        contexts: list[object] = []
        for clipped in clipped_rows:
            if not self.can_evaluate():
                break
            evaluation, context = evaluator(
                clipped,
                retain_control_linearization=bool(
                    getattr(self, "_retain_control_linearization", False)
                ),
            )
            evaluations.append(self._register_evaluation(clipped, evaluation))
            contexts.append(context)
        return evaluations, contexts

    def _compatibility(self, parameters: dict, controller: TSHCALOInferenceController) -> dict:
        ignored = {
            "run_checkpoint_path",
            "resume_run_checkpoint",
            "checkpoint_interval_evaluations",
            "extended_evaluation_target",
            "continuation_segment_index",
        }
        stable = {str(key): value for key, value in parameters.items() if str(key) not in ignored}
        serializable = CALOOptimizer._compatibility_jsonable(stable)
        encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {
            "schema_version": self.CHECKPOINT_SCHEMA,
            "algorithm": self.name,
            "algorithm_version": TSH_CALO_ALGORITHM_VERSION,
            "seed": int(self.seed),
            "dimension": int(self.problem.dimension),
            "population_size": int(self.config.population_size),
            "case_checksum": str(self.problem.case.checksum()),
            "problem_fingerprint": scientific_problem_fingerprint(self.problem),
            "policy_sha256": str(controller.binding.get("policy_sha256", "")),
            "parameters_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        }

    def _base_checkpoint_state(self, controller: TSHCALOInferenceController) -> dict:
        return {
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
                key: list(values) for key, values in self.constraint_component_histories.items()
            },
            "rng_state": self.rng.bit_generator.state,
            "policy_generator_state": controller.generator.get_state().cpu(),
        }

    def _restore_base_state(self, base: dict, controller: TSHCALOInferenceController) -> None:
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
            str(key): list(values)
            for key, values in dict(base.get("constraint_component_histories", {})).items()
        }
        self.rng.bit_generator.state = base["rng_state"]
        controller.generator.set_state(base["policy_generator_state"].cpu())

    def _save_checkpoint(
        self,
        path: str,
        parameters: dict,
        controller: TSHCALOInferenceController,
        runtime: dict,
    ) -> str:
        return save_exact_run_checkpoint(
            path,
            {
                "compatibility": self._compatibility(parameters, controller),
                "base_state": self._base_checkpoint_state(controller),
                "runtime_state": runtime,
            },
        )

    @staticmethod
    def _fallback_guard(result) -> None:
        if result.fallback.disposition is FallbackDisposition.EXECUTE_POLICY:
            return
        if result.fallback.disposition is FallbackDisposition.EXPLICIT_BASELINE:
            raise TSHCALOBaselineFallbackRequired(
                "TSH-CALO policy was rejected; explicitly relaunch as CALO-v5.9: "
                + result.fallback.reason
            )
        raise TSHCALOPolicyRejected("TSH-CALO policy was rejected: " + result.fallback.reason)

    def run(self):
        started = time.perf_counter()
        parameters = dict(self.config.parameters or {})
        population_size = int(self.config.population_size)
        if population_size < 2 or int(self.config.max_evaluations) % population_size != 0:
            raise ValueError(
                "TSH-CALO strict FE fairness requires a population of at least two and an "
                "evaluation budget exactly divisible by population size"
            )
        features = TSHCALOFeatureFlags(**dict(parameters.get("policy_feature_flags", {}) or {}))
        features.validate()
        if features.population_schedule:
            raise ValueError(
                "Experimental Change F is not admitted by the fixed-population production path"
            )
        physics_repair_enabled = self._physics_repair_enabled(features)
        if physics_repair_enabled and not features.physics_repair:
            raise ValueError(
                "TSH-CALO ablation cannot add physics repair absent from the candidate"
            )
        self._retain_control_linearization = physics_repair_enabled

        calibration = _ood_calibration(parameters)
        controller = self._build_inference_controller(parameters, calibration)
        preflight = controller.fallback_decision()
        if preflight.disposition is FallbackDisposition.EXPLICIT_BASELINE:
            raise TSHCALOBaselineFallbackRequired(
                "TSH-CALO preflight requires explicit CALO-v5.9 relaunch: " + preflight.reason
            )
        if preflight.disposition is FallbackDisposition.BLOCK:
            raise TSHCALOPolicyRejected("TSH-CALO preflight blocked: " + preflight.reason)

        run_checkpoint_path = str(parameters.get("run_checkpoint_path", "") or "").strip()
        resume_path = str(parameters.get("resume_run_checkpoint", "") or "").strip()
        checkpoint_interval = max(
            population_size,
            int(parameters.get("checkpoint_interval_evaluations", population_size)),
        )
        next_checkpoint = checkpoint_interval

        variables = getattr(getattr(self.problem, "decoder", None), "variables", None) or []
        feasible_archive = FeasibleEliteArchive(
            int(parameters.get("feasible_archive_capacity", 32))
        )
        boundary_archive = ConstraintBoundaryArchive(
            int(parameters.get("boundary_archive_capacity", 48))
        )
        hpem = HierarchicalPrefixEliteMemory(self.problem.dimension, variables=variables)
        memory = SuccessMemory(
            int(parameters.get("memory_capacity", 256)),
            float(parameters.get("memory_decay", 0.97)),
            n_operators=8,
        )
        credit = ContextualCredit(
            4,
            N_OPERATORS,
            4,
            4,
            decay=float(parameters.get("credit_decay", 0.90)),
            floor=float(parameters.get("credit_floor", 0.02)),
        )
        groups = VariableGroupIntelligence(
            variables, decay=float(parameters.get("group_credit_decay", 0.90))
        )
        lanes = DualLaneController(
            max_learning=float(parameters.get("max_learning_lane_fraction", 0.92))
        )
        precision = CognitivePrecisionEngine(
            initial_radius=float(parameters.get("precision_start_radius", 0.04)),
            min_radius=float(parameters.get("precision_min_radius", 5e-4)),
            max_radius=float(parameters.get("precision_max_radius", 0.15)),
        )
        bandit = SlidingWindowContextualBandit(
            int(parameters.get("bandit_window_size", 32)),
            float(parameters.get("bandit_exploration", 0.35)),
        )
        epsilon_controller = None
        evaluation_contexts: list[object]
        trajectory: list[dict] = []
        scenario_solver_calls = 0
        physics_repair_operator = PhysicsRepairOperator(
            PhysicsRepairConfig(enabled=physics_repair_enabled)
        )
        physics_repair_available_generations = 0
        physics_repair_proposal_count = 0
        physics_repair_linear_algebra_seconds = 0.0
        previous_best_violation = float("inf")
        previous_best_objective = float("inf")
        constraint_stagnation = 0
        objective_stagnation = 0
        violation_improving = False

        if resume_path:
            payload = load_exact_run_checkpoint(resume_path)
            if dict(payload.get("compatibility", {})) != self._compatibility(
                parameters, controller
            ):
                raise RuntimeError(
                    "TSH-CALO run checkpoint is incompatible with the scientific configuration"
                )
            self._restore_base_state(dict(payload["base_state"]), controller)
            runtime = dict(payload["runtime_state"])
            state = runtime["state"]
            evaluation_contexts = list(runtime["evaluation_contexts"])
            feasible_archive = runtime["feasible_archive"]
            boundary_archive = runtime["boundary_archive"]
            hpem = runtime["hpem"]
            memory = runtime["memory"]
            credit = runtime["credit"]
            groups = runtime["groups"]
            lanes = runtime["lanes"]
            precision = runtime["precision"]
            bandit = SlidingWindowContextualBandit.from_state_dict(runtime["bandit"])
            epsilon_controller = runtime["epsilon_controller"]
            trajectory = list(runtime["trajectory"])
            scenario_solver_calls = int(runtime["scenario_solver_calls"])
            previous_best_violation = float(runtime["previous_best_violation"])
            previous_best_objective = float(runtime["previous_best_objective"])
            constraint_stagnation = int(runtime["constraint_stagnation"])
            objective_stagnation = int(runtime["objective_stagnation"])
            violation_improving = bool(runtime["violation_improving"])
            next_checkpoint = ((self.evaluations // checkpoint_interval) + 1) * checkpoint_interval
        else:
            population = self.random_population(population_size)
            evaluations, evaluation_contexts = self._evaluate_population_with_context(population)
            if len(evaluations) != population_size:
                return self.finalize(population[: len(evaluations)], started=started)
            scenario_solver_calls = sum(len(context.scenarios) for context in evaluation_contexts)
            state = CALOTensorState.initialize(population, evaluations)
            feasible_archive.update(state.population, state.evaluations)
            boundary_archive.update(state.population, state.evaluations)
            hpem.update(state.population, state.evaluations)
            finite_violations = [
                float(item.violation)
                for item in state.evaluations
                if np.isfinite(float(item.violation))
            ]
            initial_epsilon = (
                float(
                    np.quantile(finite_violations, float(parameters.get("epsilon_quantile", 0.75)))
                )
                if finite_violations
                else 0.0
            )
            epsilon_controller = AdaptiveEpsilonController(
                initial_epsilon,
                float(parameters.get("epsilon_control_fraction", 0.65)),
                float(parameters.get("epsilon_exponent", 2.0)),
            )

        assert epsilon_controller is not None
        stagnation_window = max(4, int(parameters.get("stagnation_window", 12)))

        def runtime_payload() -> dict:
            return {
                "state": state,
                "evaluation_contexts": evaluation_contexts,
                "feasible_archive": feasible_archive,
                "boundary_archive": boundary_archive,
                "hpem": hpem,
                "memory": memory,
                "credit": credit,
                "groups": groups,
                "lanes": lanes,
                "precision": precision,
                "bandit": bandit.state_dict(),
                "epsilon_controller": epsilon_controller,
                "trajectory": trajectory,
                "scenario_solver_calls": scenario_solver_calls,
                "previous_best_violation": previous_best_violation,
                "previous_best_objective": previous_best_objective,
                "constraint_stagnation": constraint_stagnation,
                "objective_stagnation": objective_stagnation,
                "violation_improving": violation_improving,
            }

        while self.iteration < self.config.max_iterations and self.can_evaluate(population_size):
            self.iteration += 1
            progress = float(self.evaluations / max(self.config.max_evaluations, 1))
            rough = population_diagnostics(state.evaluations, epsilon_controller.current)
            epsilon = epsilon_controller.value(
                self.evaluations,
                self.config.max_evaluations,
                rough.feasible_ratio,
                violation_improving,
                min(constraint_stagnation / stagnation_window, 1.0),
            )
            diagnostics = population_diagnostics(state.evaluations, epsilon)
            diversity = population_diversity(state.population)
            cognition = build_cognitive_state(
                state.population,
                state.evaluations,
                epsilon=epsilon,
                previous_best_violation=previous_best_violation,
                previous_best_objective=previous_best_objective,
                constraint_stagnation=min(constraint_stagnation / stagnation_window, 1.0),
                objective_stagnation=min(objective_stagnation / stagnation_window, 1.0),
                remaining_budget=max(0.0, 1.0 - progress),
                operator_credit=_normalise_first_six(credit.global_operator_probabilities()),
                feasible_archive_size=len(feasible_archive),
                feasible_archive_capacity=feasible_archive.capacity,
                boundary_archive_size=len(boundary_archive),
                boundary_archive_capacity=boundary_archive.capacity,
            )
            consensus = hpem.consensus(state.population.mean(axis=0)) if len(hpem) else 0.0
            readiness = lanes.memory_readiness(
                diagnostics.feasible_ratio,
                hpem.occupancy,
                memory.density,
                min(
                    self.iteration / max(int(parameters.get("memory_evidence_batches", 6)), 1), 1.0
                ),
                consensus,
            )
            severe_stagnation = (
                max(constraint_stagnation, objective_stagnation) >= stagnation_window
            )
            learning_fraction = lanes.learning_fraction(
                readiness, progress, diversity, severe_stagnation
            )
            learned_lanes = lanes.assign(population_size, learning_fraction, self.rng, False)
            precision_active = precision.active(
                diagnostics.feasible_ratio,
                min(objective_stagnation / stagnation_window, 1.0),
                progress,
                len(hpem),
            )
            precision_fraction = (
                float(np.clip(0.12 + 0.28 * objective_stagnation / stagnation_window, 0.12, 0.55))
                if precision_active
                else 0.0
            )
            policy_context = PolicyRuntimeContext(
                hpem_occupancy=float(hpem.occupancy),
                memory_consensus=float(consensus),
                memory_readiness=float(readiness),
                success_memory_density=float(memory.density),
                learning_lane_fraction=float(learning_fraction),
                precision_active=float(precision_active),
                precision_radius=float(
                    np.clip(precision.radius / max(precision.max_radius, 1e-12), 0.0, 1.0)
                ),
                variable_group_concentration=float(
                    np.max(groups.probabilities(int(np.argmax(rule_based_regime_prior(cognition)))))
                ),
            )
            aggregate = build_policy_vector(cognition, policy_context, input_dim=POLICY_STATE_DIM)
            quality_order = self.order(state.evaluations)
            reference_index = int(quality_order[0])
            runtime_context = build_runtime_topology_policy_context(
                aggregate,
                self.problem,
                evaluation_contexts[reference_index],
            )
            learner_contexts = classify_contexts(
                state.population, state.evaluations, violation_improving
            )
            provisional_regime = int(np.argmax(rule_based_regime_prior(cognition)))
            learner_groups = np.asarray(
                [
                    groups.choose(provisional_regime, self.rng, False)
                    for _ in range(population_size)
                ],
                dtype=np.int8,
            )
            physics_contexts = tuple(
                physics_repair_context_from_counted_evaluation(context)
                for context in evaluation_contexts
            )
            physics_repair_available = bool(
                physics_repair_enabled
                and all(
                    physics_repair_context_is_usable(
                        context,
                        maximum_condition_number=physics_repair_operator.config.maximum_condition_number,
                    )
                    for context in physics_contexts
                )
            )
            if physics_repair_available:
                physics_repair_available_generations += 1
            action_mask = GroupActionMask.from_control_groups(
                groups.variable_groups,
                mixed_variable_enabled=True,
                diversity_recovery_enabled=True,
                physics_repair_enabled=physics_repair_available,
            )
            decision = controller.decide(
                runtime_context.policy_state,
                action_mask,
                learner_groups,
                learner_contexts,
                bandit=bandit,
                safety=SafetyEnvelope(
                    self.config.max_evaluations - self.evaluations,
                    population_size,
                    mixed_variable_lattice_valid=True,
                ),
            )
            self._fallback_guard(decision)
            assert decision.regime is not None
            assert decision.learner_operators is not None
            assert decision.group_parameters is not None

            raw_group_parameters = decision.group_parameters.detach().cpu().numpy()
            effective_group_parameters = effective_group_parameter_values(raw_group_parameters)
            recovery_fraction_ceiling = float(parameters.get("recovery_fraction", 0.18))
            selected_recovery_fraction = effective_recovery_fraction(
                raw_group_parameters,
                learner_groups,
                maximum_fraction=recovery_fraction_ceiling,
            )
            forced_recovery: set[int] = set()
            recovery_triggered = bool(
                severe_stagnation
                and diversity < float(parameters.get("recovery_diversity_threshold", 0.06))
            )
            if recovery_triggered:
                recovery_fraction = selected_recovery_fraction
                count = max(
                    1,
                    min(population_size - 1, int(round(population_size * recovery_fraction))),
                )
                weakest = sorted(
                    range(population_size),
                    key=lambda index: epsilon_sort_key(state.evaluations[index], epsilon),
                    reverse=True,
                )
                forced_recovery = set(weakest[:count])

            candidate_batch = generate_tsh_offspring(
                population=state.population,
                evaluations=state.evaluations,
                personal_best=state.personal_best,
                rng=self.rng,
                dimension=self.problem.dimension,
                variables=variables,
                quality_order=quality_order,
                contexts=learner_contexts,
                learner_groups=learner_groups,
                learned_lanes=learned_lanes,
                global_regime=int(decision.regime),
                learner_operators=decision.learner_operators.detach().cpu().numpy(),
                group_parameter_actions=decision.group_parameters.detach().cpu().numpy(),
                memory=memory,
                hpem=hpem,
                feasible_archive=feasible_archive,
                boundary_archive=boundary_archive,
                credit=credit,
                group_intelligence=groups,
                precision=precision,
                precision_active=precision_active,
                precision_fraction=precision_fraction,
                forced_recovery=forced_recovery,
                consensus=consensus,
                environment_deterministic=False,
                physics_repair_operator=(
                    physics_repair_operator if physics_repair_available else None
                ),
                physics_contexts=physics_contexts,
            )
            repair_traces = [
                {
                    "status": proposal.status.value,
                    "source_evaluation_id": proposal.source_evaluation_id,
                    "condition_number": proposal.condition_number,
                    "step_norm": proposal.step_norm,
                    "linear_algebra_seconds": proposal.linear_algebra_seconds,
                    "hidden_solver_calls": proposal.hidden_solver_calls,
                    "evaluator_calls_before_trusted_batch": proposal.evaluator_calls,
                    "declares_feasibility": proposal.declares_feasibility,
                }
                for proposal in candidate_batch.physics_repair_proposals
                if proposal is not None
            ]
            physics_repair_proposal_count += len(repair_traces)
            physics_repair_linear_algebra_seconds += sum(
                float(item["linear_algebra_seconds"]) for item in repair_traces
            )
            offspring = candidate_batch.candidates.offspring
            offspring_evaluations: list[object] = []
            offspring_contexts: list[object] = []

            def counted_evaluator(values):
                evaluated, counted = self._evaluate_population_with_context(values)
                offspring_contexts.extend(counted)
                return evaluated

            evaluated_batch = evaluate_candidates(offspring, counted_evaluator)
            offspring_evaluations = evaluated_batch.evaluations
            scenario_solver_calls += sum(len(context.scenarios) for context in offspring_contexts)
            if not evaluated_batch.complete:
                break
            batch = candidate_batch.candidates
            selected_diversity = candidate_batch.group_parameter_values[learner_groups, 4]
            transition = complete_tsh_transition(
                population=state.population,
                evaluations=state.evaluations,
                personal_best=state.personal_best,
                personal_best_evaluations=state.personal_best_evaluations,
                offspring=offspring,
                offspring_evaluations=offspring_evaluations,
                epsilon=epsilon,
                assigned_operators=batch.assigned_operators,
                assigned_memory=batch.assigned_memory,
                assigned_groups=batch.assigned_groups,
                individual_regimes=batch.individual_regimes,
                contexts=learner_contexts,
                precision_mask=batch.precision_mask,
                memory=memory,
                credit=credit,
                group_intelligence=groups,
                precision=precision,
                feasible_archive=feasible_archive,
                boundary_archive=boundary_archive,
                hpem=hpem,
                old_diagnostics=diagnostics,
                old_diversity=diversity,
                diversity_weight=float(np.mean(selected_diversity)),
                population_size=population_size,
            )
            bandit_rewards = _learner_bandit_rewards(transition)
            for index, operator in enumerate(batch.assigned_operators):
                if int(operator) >= 0:
                    bandit.update(
                        int(batch.assigned_groups[index]),
                        int(learner_contexts[index]),
                        int(operator),
                        float(bandit_rewards[index]),
                    )
            combined_contexts = evaluation_contexts + offspring_contexts
            evaluation_contexts = [
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
            violation_improving = (
                new_diagnostics.best_violation < diagnostics.best_violation - 1e-12
            )
            constraint_stagnation = 0 if violation_improving else constraint_stagnation + 1
            objective_improving = (
                np.isfinite(new_diagnostics.best_feasible_objective)
                and new_diagnostics.best_feasible_objective
                < diagnostics.best_feasible_objective - 1e-12
            )
            objective_stagnation = 0 if objective_improving else objective_stagnation + 1
            previous_best_violation = new_diagnostics.best_violation
            previous_best_objective = new_diagnostics.best_feasible_objective
            trajectory.append(
                {
                    "schema_version": "tsh-calo-runtime-generation-v2-parameter-evidence",
                    "evaluations": int(self.evaluations),
                    "policy_id": str(controller.binding.get("policy_id", "")),
                    "policy_sha256": str(controller.binding.get("policy_sha256", "")),
                    "reference_scenario": runtime_context.reference_scenario,
                    "scenario_names": list(runtime_context.scenario_names),
                    "regime": int(decision.regime),
                    "group_parameter_names": list(PARAMETER_NAMES),
                    "group_parameters_raw": raw_group_parameters.tolist(),
                    "group_parameter_values": effective_group_parameters.tolist(),
                    "recovery_fraction_ceiling": recovery_fraction_ceiling,
                    "selected_recovery_fraction": selected_recovery_fraction,
                    "recovery_triggered": recovery_triggered,
                    "learner_groups": learner_groups.astype(int).tolist(),
                    "learner_contexts": learner_contexts.astype(int).tolist(),
                    "executed_operators": batch.assigned_operators.astype(int).tolist(),
                    "operator_probabilities": decision.operator_probabilities.detach()
                    .cpu()
                    .tolist(),
                    "precision_mask": batch.precision_mask.astype(bool).tolist(),
                    "forced_recovery_indices": sorted(forced_recovery),
                    "ensemble_uncertainty": decision.shield_trace.uncertainty.detach()
                    .cpu()
                    .tolist(),
                    "shield_mixture_weights": decision.shield_trace.mixture_weights.detach()
                    .cpu()
                    .tolist(),
                    "shield_action_mask": decision.shield_trace.action_mask.detach().cpu().tolist(),
                    "shield_interventions": list(decision.shield_trace.intervention_reasons),
                    "physics_repair_available": physics_repair_available,
                    "physics_repair_proposals": repair_traces,
                    "ood_score": float(decision.shield_trace.ood_score),
                    "ood_attenuation": float(decision.shield_trace.ood_attenuation),
                    "value_estimate": float(decision.value_estimate),
                    "reward_components": {
                        "objective_improvement": float(transition.reward.objective_improvement),
                        "constraint_improvement": float(transition.reward.constraint_improvement),
                        "feasible_ratio_improvement": float(
                            transition.reward.feasible_ratio_improvement
                        ),
                        "diversity_recovery": float(transition.reward.diversity_recovery),
                        "overhead_penalty": float(transition.reward.overhead_penalty),
                    },
                    "reward": float(transition.reward.total),
                    "feasible_ratio_before": float(diagnostics.feasible_ratio),
                    "feasible_ratio_after": float(new_diagnostics.feasible_ratio),
                    "best_violation_before": float(diagnostics.best_violation),
                    "best_violation_after": float(new_diagnostics.best_violation),
                    "best_feasible_objective_before": (
                        float(diagnostics.best_feasible_objective)
                        if np.isfinite(diagnostics.best_feasible_objective)
                        else None
                    ),
                    "best_feasible_objective_after": (
                        float(new_diagnostics.best_feasible_objective)
                        if np.isfinite(new_diagnostics.best_feasible_objective)
                        else None
                    ),
                    "diversity_before": float(diversity),
                    "diversity_after": float(transition.new_diversity),
                }
            )
            self.record(
                {
                    "policy_id": str(controller.binding.get("policy_id", "")),
                    "device": controller.admission.selected_device,
                    "feasible_ratio": new_diagnostics.feasible_ratio,
                    "best_constraint_violation": new_diagnostics.best_violation,
                    "reward": float(transition.reward.total),
                }
            )
            if run_checkpoint_path and self.evaluations >= next_checkpoint:
                self._save_checkpoint(
                    run_checkpoint_path, parameters, controller, runtime_payload()
                )
                next_checkpoint = (
                    (self.evaluations // checkpoint_interval) + 1
                ) * checkpoint_interval

        if run_checkpoint_path:
            self._save_checkpoint(run_checkpoint_path, parameters, controller, runtime_payload())
        metadata = {
            "algorithm_id": self.name,
            "algorithm_version": TSH_CALO_ALGORITHM_VERSION,
            "policy_identity": {
                "policy_id": str(controller.binding.get("policy_id", "")),
                "policy_sha256": str(controller.binding.get("policy_sha256", "")),
                "qualification_status": str(
                    controller.binding.get("policy_qualification_status", "")
                ),
                "active_at_binding": bool(
                    controller.binding.get("policy_active_at_binding", False)
                ),
            },
            "feature_flags": asdict(features),
            "device_admission": asdict(controller.admission),
            "computation_semantics": {
                "policy_inference": controller.admission.computation_device,
                "trusted_orpd_evaluator": "cpu",
                "memory_role": "admitted storage, not an independent compute device",
            },
            "candidate_evaluations": int(self.evaluations),
            "scenario_power_flow_calls": int(scenario_solver_calls),
            "runtime_trajectory": trajectory,
            "physics_repair_runtime": (
                {
                    "status": (
                        "enabled_counted_proposal_only"
                        if physics_repair_enabled
                        else "disabled_by_immutable_policy_feature_flags"
                    ),
                    "available_generations": physics_repair_available_generations,
                    "proposal_count": physics_repair_proposal_count,
                    "linear_algebra_seconds": physics_repair_linear_algebra_seconds,
                    "hidden_solver_calls": 0,
                    "feasibility_authority": False,
                    "trusted_evaluations_remain_in_candidate_fe_budget": True,
                    "masking": (
                        "dynamic fail-closed mask unless every learner has a finite, conditioned, "
                        "nonzero counted constraint/sensitivity context"
                        if physics_repair_enabled
                        else "immutable feature disabled"
                    ),
                }
            ),
            "population_schedule_runtime": "disabled",
            "fallback_semantics": "block or require explicit CALO-v5.9 relaunch",
            "resumed_from": resume_path,
            "checkpoint_path": run_checkpoint_path,
        }
        return self.finalize(state.population, metadata=metadata, started=started)
