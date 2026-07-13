"""Cognitive Adaptive Learning Optimizer — CALO Core v2.

CALO Core v2 is a constraint-aware, mixed-variable optimizer with:

* adaptive epsilon-feasibility;
* separate feasible and constraint-boundary archives;
* per-individual operator allocation;
* mixed-variable neighbourhood moves;
* environmental selection from parents and offspring;
* success-distribution memory;
* online operator credit blended with the learned policy;
* separate objective and constraint stagnation states.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np

from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer
from .ai_controller import AIController, PARAMETER_HIGH, PARAMETER_LOW, PARAMETER_NAMES
from .archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from .cognitive_state import (
    REGIME_NAMES,
    build_cognitive_state,
    population_diversity,
    rule_based_regime_prior,
)
from .diagnostics import CONSTRAINT_COMPONENTS, diagnostic_history_template, population_diagnostics
from .environmental_selection import environmental_select, epsilon_better
from .learning_operators import (
    OPERATOR_NAMES,
    cognitive_teacher_learning,
    constraint_boundary_differential,
    diversity_recovery,
    feasible_elite_learning,
    mixed_variable_neighbourhood,
    success_distribution_memory,
)
from .operator_credit import OperatorCredit, blend_probabilities
from .reward import calculate_reward
from .success_memory import SuccessMemory


REGIME_OPERATOR_PRIORS = np.asarray(
    [
        [0.05, 0.33, 0.12, 0.08, 0.30, 0.12],  # feasibility
        [0.18, 0.24, 0.18, 0.14, 0.18, 0.08],  # transition
        [0.34, 0.08, 0.22, 0.20, 0.12, 0.04],  # objective refinement
        [0.08, 0.15, 0.10, 0.10, 0.12, 0.45],  # recovery
    ],
    dtype=float,
)


class CALOOptimizer(BaseOptimizer):
    name = "CALO"

    def _default_checkpoint(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "trained_models" / "calo_policy_v2.pt"

    @staticmethod
    def _epsilon(initial_epsilon: float, evaluations: int, max_evaluations: int,
                 control_fraction: float, exponent: float) -> float:
        control_end = max(1.0, float(max_evaluations) * float(control_fraction))
        if evaluations >= control_end:
            return 0.0
        ratio = max(0.0, 1.0 - evaluations / control_end)
        return float(max(initial_epsilon, 0.0) * ratio ** float(exponent))

    @staticmethod
    def _rule_operator_probabilities(regime: int) -> np.ndarray:
        values = REGIME_OPERATOR_PRIORS[int(regime)].copy()
        return values / values.sum()

    def _select_distinct(self, population: np.ndarray, index: int, count: int = 2) -> list[np.ndarray]:
        candidates = [i for i in range(len(population)) if i != index]
        if len(candidates) < count:
            return [population[index].copy() for _ in range(count)]
        chosen = self.rng.choice(candidates, size=count, replace=False)
        return [population[int(i)].copy() for i in chosen]

    def _candidate(
        self,
        operator: int,
        index: int,
        population: np.ndarray,
        evaluations,
        pbest: np.ndarray,
        memory: SuccessMemory,
        feasible_archive: FeasibleEliteArchive,
        boundary_archive: ConstraintBoundaryArchive,
        parameters: dict[str, float],
        regime: int,
    ) -> np.ndarray:
        x = population[index]
        r1, r2 = self._select_distinct(population, index, 2)
        quality_order = self.order(evaluations)
        best = population[quality_order[0]]
        mean = population.mean(axis=0)
        feasible_teacher = feasible_archive.sample(self.rng, best)
        boundary_teacher = boundary_archive.sample(self.rng, best)
        variables = getattr(getattr(self.problem, "decoder", None), "variables", None)

        if operator == 0:
            teacher = feasible_teacher if len(feasible_archive) else boundary_teacher
            return feasible_elite_learning(
                x,
                teacher,
                r1,
                r2,
                self.rng,
                parameters["attraction"],
                parameters["differential"],
            )
        if operator == 1:
            return constraint_boundary_differential(
                x,
                boundary_teacher,
                r1,
                r2,
                self.rng,
                parameters["attraction"],
                parameters["differential"],
            )
        if operator == 2:
            teacher = feasible_teacher if regime >= 2 and len(feasible_archive) else boundary_teacher
            return cognitive_teacher_learning(
                x,
                teacher,
                mean,
                self.rng,
                parameters["attraction"],
                0.35 * parameters["exploration_sigma"],
            )
        if operator == 3:
            direction = memory.sample_direction(
                self.problem.dimension,
                self.rng,
                prefer_feasibility=regime <= 1,
            )
            return success_distribution_memory(
                x,
                pbest[index],
                direction,
                self.rng,
                0.55,
                parameters["memory_weight"],
            )
        if operator == 4:
            return mixed_variable_neighbourhood(
                x,
                variables,
                self.rng,
                continuous_sigma=max(parameters["exploration_sigma"] * 0.35, 0.006),
                discrete_radius=2 if regime == 3 else 1,
            )
        reference = boundary_teacher if regime <= 1 else feasible_teacher
        return diversity_recovery(
            reference,
            population,
            self.rng,
            sigma=max(parameters["exploration_sigma"], 0.06),
        )

    def run(self):
        started = time.perf_counter()
        parameters = dict(self.config.parameters)
        population_size = self.config.population_size

        historical_repository = None
        historical_prior_applied: dict[str, float] = {}
        historical_repository_path = str(parameters.get("historical_repository", "") or "").strip()
        if historical_repository_path and (
            bool(parameters.get("use_historical_parameter_priors", False))
            or bool(parameters.get("use_cross_algorithm_warm_start", False))
        ):
            from calo_rpd_studio.learning.experience_repository import load_experience_repository

            historical_repository = load_experience_repository(historical_repository_path)
            case_checksum = self.problem.case.checksum()
            case_name = self.problem.case.name
            if bool(parameters.get("use_historical_parameter_priors", False)):
                prior = historical_repository.calo_parameter_prior(
                    case_checksum=case_checksum,
                    case_name=case_name,
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
                    "local_intensification_start",
                }
                for name, prior_value in prior.items():
                    if name not in tunable or not isinstance(prior_value, (int, float)):
                        continue
                    current = parameters.get(name, prior_value)
                    if isinstance(current, (int, float)):
                        blended = (1.0 - blend) * float(current) + blend * float(prior_value)
                        parameters[name] = int(round(blended)) if name == "stagnation_window" else blended
                        historical_prior_applied[name] = parameters[name]

        population = self.random_population()
        historical_warm_start_count = 0
        if historical_repository is not None and bool(parameters.get("use_cross_algorithm_warm_start", False)):
            solutions = historical_repository.compatible_solutions(
                case_checksum=self.problem.case.checksum(),
                case_name=self.problem.case.name,
                dimension=self.problem.dimension,
            )
            fraction = float(np.clip(parameters.get("historical_warm_start_fraction", 0.15), 0.0, 0.50))
            count = min(int(round(population_size * fraction)), len(solutions), population_size)
            for index, item in enumerate(solutions[:count]):
                vector = np.asarray(item.get("best_vector") or [], dtype=float)
                if vector.shape == (self.problem.dimension,):
                    population[index] = np.clip(vector, 0.0, 1.0)
                    historical_warm_start_count += 1

        evaluations = self.evaluate_population(population)
        if len(evaluations) < len(population):
            return self.finalize(population[: len(evaluations)], started=started)

        use_ai = bool(parameters.get("use_ai", True))
        use_memory = bool(parameters.get("use_memory", True))
        use_dual_archives = bool(parameters.get("use_dual_archives", True))
        use_epsilon = bool(parameters.get("use_epsilon", True))
        use_mixed_variable = bool(parameters.get("use_mixed_variable", True))
        use_diversity_recovery = bool(parameters.get("use_diversity_recovery", True))
        use_local_intensification = bool(parameters.get("use_local_intensification", True))
        deterministic_policy = bool(parameters.get("deterministic_policy", False))

        feasible_archive = FeasibleEliteArchive(int(parameters.get("feasible_archive_capacity", 32)))
        boundary_archive = ConstraintBoundaryArchive(int(parameters.get("boundary_archive_capacity", 48)))
        feasible_archive.update(population, evaluations)
        boundary_archive.update(population, evaluations)
        memory = SuccessMemory(
            int(parameters.get("memory_capacity", 256)),
            float(parameters.get("memory_decay", 0.97)),
        )
        credit = OperatorCredit(6, decay=float(parameters.get("credit_decay", 0.90)))

        pbest = population.copy()
        pbest_evaluations = list(evaluations)
        initial_violations = [e.violation for e in evaluations if np.isfinite(e.violation)]
        initial_epsilon = (
            float(np.quantile(initial_violations, float(parameters.get("epsilon_quantile", 0.75))))
            if initial_violations
            else 0.0
        )
        if not use_epsilon:
            initial_epsilon = 0.0

        checkpoint = parameters.get("policy_checkpoint", str(self._default_checkpoint()))
        controller = AIController(
            checkpoint if use_ai else None,
            seed=int(parameters.get("ai_inference_seed", self.seed + 7919)),
            deterministic=deterministic_policy,
            device=str(parameters.get("inference_device", "auto")),
        )

        diagnostics_history = diagnostic_history_template()
        operator_usage_history: list[dict[str, int]] = []
        operator_success_history: list[dict[str, float]] = []
        regime_history: list[str] = []
        reward_history: list[float] = []
        previous_best_violation = float("inf")
        previous_best_objective = float("inf")
        constraint_stagnation = 0
        objective_stagnation = 0
        stagnation_window = max(4, int(parameters.get("stagnation_window", 12)))
        recovery_cooldown = 0
        local_intensification_evaluations = 0
        policy_trajectory: list[dict] = []

        while self.iteration < self.config.max_iterations and self.can_evaluate(population_size):
            self.iteration += 1
            epsilon = self._epsilon(
                initial_epsilon,
                self.evaluations,
                self.config.max_evaluations,
                float(parameters.get("epsilon_control_fraction", 0.65)),
                float(parameters.get("epsilon_exponent", 2.0)),
            )
            current_diag = population_diagnostics(evaluations, epsilon)
            current_diversity = population_diversity(population)
            remaining_budget = max(0.0, 1.0 - self.evaluations / max(self.config.max_evaluations, 1))
            state = build_cognitive_state(
                population,
                evaluations,
                epsilon=epsilon,
                previous_best_violation=previous_best_violation,
                previous_best_objective=previous_best_objective,
                constraint_stagnation=min(constraint_stagnation / stagnation_window, 1.0),
                objective_stagnation=min(objective_stagnation / stagnation_window, 1.0),
                remaining_budget=remaining_budget,
                operator_credit=credit.probabilities(),
                feasible_archive_size=len(feasible_archive),
                feasible_archive_capacity=feasible_archive.capacity,
                boundary_archive_size=len(boundary_archive),
                boundary_archive_capacity=boundary_archive.capacity,
            )

            if use_ai:
                decision = controller.decide(state)
                regime_probabilities = decision.regime_probabilities
                ai_operator_probabilities = decision.operator_probabilities
                adaptive = dict(decision.parameters)
            else:
                prior = rule_based_regime_prior(state)
                regime_probabilities = prior
                adaptive = {
                    "attraction": 0.65,
                    "differential": 0.35,
                    "exploration_sigma": 0.08,
                    "memory_weight": 0.35,
                    "diversity_weight": 0.18,
                    "recovery_fraction": 0.18,
                }
                ai_operator_probabilities = np.full(6, 1 / 6)

            # Recovery is temporary and never becomes a permanent hard override.
            if recovery_cooldown > 0:
                recovery_cooldown -= 1
            severe_stagnation = max(constraint_stagnation, objective_stagnation) >= stagnation_window
            if severe_stagnation and recovery_cooldown == 0:
                regime_probabilities = 0.45 * regime_probabilities + 0.55 * np.asarray([0.05, 0.10, 0.10, 0.75])
                regime_probabilities /= regime_probabilities.sum()
                recovery_cooldown = 3
                constraint_stagnation = max(0, constraint_stagnation - stagnation_window // 2)
                objective_stagnation = max(0, objective_stagnation - stagnation_window // 2)

            regime = int(np.argmax(regime_probabilities)) if deterministic_policy else int(
                self.rng.choice(4, p=regime_probabilities)
            )
            regime_prior = self._rule_operator_probabilities(regime)
            ai_policy_weight = float(parameters.get("ai_policy_weight", 0.35))
            ai_policy_weight = float(np.clip(ai_policy_weight, 0.0, 1.0))
            learned = ai_policy_weight * ai_operator_probabilities + (1.0 - ai_policy_weight) * regime_prior
            learned /= learned.sum()
            operator_probabilities = blend_probabilities(
                learned,
                credit.probabilities(),
                alpha=float(parameters.get("ai_credit_blend", 0.65)),
            )
            if not use_mixed_variable:
                operator_probabilities[4] = 0.0
            if not use_diversity_recovery:
                operator_probabilities[5] = 0.0
            if operator_probabilities.sum() <= 0:
                operator_probabilities[:] = 1.0
            operator_probabilities /= operator_probabilities.sum()

            # Final-budget intensification: increase physically meaningful neighbourhood search
            # after a feasible archive exists, without taking away the complete adaptive portfolio.
            if len(feasible_archive) and remaining_budget < 0.18 and use_mixed_variable:
                operator_probabilities = 0.80 * operator_probabilities
                operator_probabilities[4] += 0.20
                operator_probabilities /= operator_probabilities.sum()

            offspring = []
            assigned_operators = []
            for index in range(population_size):
                operator = int(np.argmax(operator_probabilities)) if deterministic_policy else int(
                    self.rng.choice(6, p=operator_probabilities)
                )
                assigned_operators.append(operator)
                offspring.append(
                    self._candidate(
                        operator,
                        index,
                        population,
                        evaluations,
                        pbest,
                        memory,
                        feasible_archive,
                        boundary_archive,
                        adaptive,
                        regime,
                    )
                )
            offspring = np.asarray(offspring)
            offspring_evaluations = self.evaluate_population(offspring)
            if len(offspring_evaluations) != len(offspring):
                break

            # Credit and memory are based on genuine child-versus-parent progress under the same
            # epsilon rule used by environmental selection.
            for index, (child, child_ev, operator) in enumerate(
                zip(offspring, offspring_evaluations, assigned_operators)
            ):
                parent_ev = evaluations[index]
                successful = epsilon_better(child_ev, parent_ev, epsilon)
                objective_gain = 0.0
                if parent_ev.feasible and child_ev.feasible and np.isfinite(parent_ev.value):
                    objective_gain = max(
                        (parent_ev.value - child_ev.value) / max(abs(parent_ev.value), 1.0), 0.0
                    )
                feasibility_gain = max(float(parent_ev.violation - child_ev.violation), 0.0)
                local_reward = objective_gain + feasibility_gain
                credit.update(operator, local_reward, successful)
                if successful and use_memory:
                    memory.add(child - population[index], operator, objective_gain, feasibility_gain)
                if epsilon_better(child_ev, pbest_evaluations[index], epsilon):
                    pbest[index] = child.copy()
                    pbest_evaluations[index] = child_ev

            combined_population = np.vstack([population, offspring])
            combined_evaluations = list(evaluations) + list(offspring_evaluations)
            population, evaluations = environmental_select(
                combined_population,
                combined_evaluations,
                population_size,
                epsilon,
                diversity_weight=adaptive["diversity_weight"],
            )
            # Environmental selection changes learner identity/order. Re-align personal references
            # with the selected population rather than attaching history to the wrong learner.
            pbest = population.copy()
            pbest_evaluations = list(evaluations)
            if use_dual_archives:
                feasible_archive.update(combined_population, combined_evaluations)
                boundary_archive.update(combined_population, combined_evaluations)
            else:
                # The ablation retains only current-population references and discards persistent
                # archive history, isolating the contribution of the dual-archive mechanism.
                feasible_archive.entries = []
                boundary_archive.entries = []
                feasible_archive.update(population, evaluations)
                boundary_archive.update(population, evaluations)

            # Controlled local intensification is activated only after an exact-feasible archive
            # exists and the remaining budget is small. It uses the same mixed-variable decoder
            # semantics as the main search and remains fully counted in the evaluation budget.
            local_interval = max(1, int(parameters.get("local_intensification_interval", 5)))
            if (
                use_local_intensification
                and len(feasible_archive)
                and remaining_budget < float(parameters.get("local_intensification_start", 0.15))
                and self.iteration % local_interval == 0
            ):
                variables = getattr(getattr(self.problem, "decoder", None), "variables", None)
                local_count = min(
                    max(1, int(parameters.get("local_intensification_candidates", max(1, population_size // 10)))),
                    max(self.config.max_evaluations - self.evaluations, 0),
                )
                if local_count > 0 and self.can_evaluate(local_count):
                    elite_vector = feasible_archive.best.vector.copy()
                    local_candidates = np.asarray([
                        mixed_variable_neighbourhood(
                            elite_vector,
                            variables,
                            self.rng,
                            continuous_sigma=max(adaptive["exploration_sigma"] * 0.12, 0.0025),
                            discrete_radius=1,
                        )
                        for _ in range(local_count)
                    ])
                    local_evaluations = self.evaluate_population(local_candidates)
                    local_intensification_evaluations += len(local_evaluations)
                    if local_evaluations:
                        local_population = np.vstack([population, local_candidates[:len(local_evaluations)]])
                        local_all_evaluations = list(evaluations) + list(local_evaluations)
                        population, evaluations = environmental_select(
                            local_population,
                            local_all_evaluations,
                            population_size,
                            epsilon,
                            diversity_weight=adaptive["diversity_weight"],
                        )
                        if use_dual_archives:
                            feasible_archive.update(local_population, local_all_evaluations)
                            boundary_archive.update(local_population, local_all_evaluations)
                        else:
                            feasible_archive.entries = []
                            boundary_archive.entries = []
                            feasible_archive.update(population, evaluations)
                            boundary_archive.update(population, evaluations)
                        pbest = population.copy()
                        pbest_evaluations = list(evaluations)

            new_diag = population_diagnostics(evaluations, epsilon)
            new_diversity = population_diversity(population)
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
            reward_history.append(reward.total)

            if new_diag.best_violation < current_diag.best_violation - 1e-12:
                constraint_stagnation = 0
            else:
                constraint_stagnation += 1
            if (
                np.isfinite(new_diag.best_feasible_objective)
                and new_diag.best_feasible_objective < current_diag.best_feasible_objective - 1e-12
            ):
                objective_stagnation = 0
            elif np.isfinite(new_diag.best_feasible_objective):
                objective_stagnation += 1

            previous_best_violation = new_diag.best_violation
            previous_best_objective = new_diag.best_feasible_objective
            usage = Counter(assigned_operators)
            operator_usage_history.append({OPERATOR_NAMES[k]: int(usage.get(k, 0)) for k in range(6)})
            rates = credit.success_rates()
            operator_success_history.append({OPERATOR_NAMES[k]: float(rates[k]) for k in range(6)})
            regime_history.append(REGIME_NAMES[regime])

            diagnostics_history["best_total_violation"].append(new_diag.best_violation)
            diagnostics_history["mean_total_violation"].append(new_diag.mean_violation)
            diagnostics_history["feasible_ratio"].append(new_diag.feasible_ratio)
            diagnostics_history["epsilon_feasible_ratio"].append(new_diag.epsilon_feasible_ratio)
            diagnostics_history["population_diversity"].append(new_diversity)
            diagnostics_history["elite_diversity"].append(state.elite_spread)
            diagnostics_history["epsilon"].append(epsilon)
            for key in CONSTRAINT_COMPONENTS:
                diagnostics_history[f"best_{key}"].append(new_diag.component_best.get(key, 0.0))
                diagnostics_history[f"mean_{key}"].append(new_diag.component_mean.get(key, 0.0))

            dominant_operator = int(np.argmax(np.bincount(assigned_operators, minlength=6)))
            adaptive_vector = np.asarray([adaptive[name] for name in PARAMETER_NAMES], dtype=float)
            raw_parameter_action = np.clip(
                (adaptive_vector - PARAMETER_LOW) / np.maximum(PARAMETER_HIGH - PARAMETER_LOW, 1e-12),
                0.0,
                1.0,
            )
            if bool(parameters.get("record_policy_trajectory", True)):
                policy_trajectory.append(
                    {
                        "state": state.vector().tolist(),
                        "regime": int(regime),
                        "operator": int(dominant_operator),
                        "parameter": raw_parameter_action.tolist(),
                        "reward": float(reward.total),
                        "evaluations": int(self.evaluations),
                        "source_policy": "ai" if use_ai else "rule_based",
                    }
                )
            self.record(
                {
                    "calo_operator": OPERATOR_NAMES[dominant_operator],
                    "calo_regime": REGIME_NAMES[regime],
                    "operator_probabilities": {
                        OPERATOR_NAMES[k]: float(operator_probabilities[k]) for k in range(6)
                    },
                    "operator_success_rates": {
                        OPERATOR_NAMES[k]: float(rates[k]) for k in range(6)
                    },
                    "diversity": new_diversity,
                    "elite_diversity": state.elite_spread,
                    "feasible_ratio": new_diag.feasible_ratio,
                    "epsilon_feasible_ratio": new_diag.epsilon_feasible_ratio,
                    "epsilon": epsilon,
                    "constraint_components": dict(new_diag.component_best),
                    "reward": reward.total,
                    "feasible_archive_size": len(feasible_archive),
                    "boundary_archive_size": len(boundary_archive),
                }
            )

        metadata = {
            "calo_version": "Core v2",
            "operator_names": list(OPERATOR_NAMES),
            "operator_attempts": credit.attempts.tolist(),
            "operator_successes": credit.successes.tolist(),
            "operator_credit": credit.probabilities().tolist(),
            "mean_reward": float(np.mean(reward_history)) if reward_history else 0.0,
            "reward_history": reward_history,
            "success_memory_size": len(memory),
            "feasible_archive_size": len(feasible_archive),
            "boundary_archive_size": len(boundary_archive),
            "local_intensification_evaluations": local_intensification_evaluations,
            "diagnostics_history": diagnostics_history,
            "operator_usage_history": operator_usage_history,
            "operator_success_history": operator_success_history,
            "regime_history": regime_history,
            "policy_checkpoint": controller.checkpoint_path,
            "policy_checksum": controller.checksum,
            "policy_metadata": controller.metadata,
            "policy_inference_device": str(controller.device),
            "policy_trajectory": policy_trajectory,
            "historical_learning": {
                "repository": historical_repository_path,
                "repository_sha256": (
                    historical_repository.payload.get("repository_sha256", "")
                    if historical_repository is not None
                    else ""
                ),
                "parameter_priors_enabled": bool(parameters.get("use_historical_parameter_priors", False)),
                "parameter_priors_applied": historical_prior_applied,
                "cross_algorithm_warm_start_enabled": bool(parameters.get("use_cross_algorithm_warm_start", False)),
                "warm_start_count": historical_warm_start_count,
            },
            "ablation": {
                "use_ai": use_ai,
                "use_memory": use_memory,
                "use_dual_archives": use_dual_archives,
                "use_epsilon": use_epsilon,
                "use_mixed_variable": use_mixed_variable,
                "use_diversity_recovery": use_diversity_recovery,
                "use_local_intensification": use_local_intensification,
            },
        }
        return self.finalize(population, metadata=metadata, started=started)
