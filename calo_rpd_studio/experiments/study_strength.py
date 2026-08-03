"""Scientifically honest, apply-once experiment-strength protocols."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import NormalDist

from calo_rpd_studio.algorithms.registry import SPECS
from calo_rpd_studio.portfolio.catalog import OUTPUT_REQUIREMENTS
from calo_rpd_studio.portfolio.models import (
    EvidenceProfile,
    PortfolioKind,
    StorageProfile,
)
from calo_rpd_studio.robustness.robust_objectives import RobustAggregation

from .evaluation_budget import BudgetPolicy


class StudyStrength(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    GOOD = "good"
    STRONG = "strong"


@dataclass(frozen=True, slots=True)
class StudyStrengthPlan:
    strength: StudyStrength
    label: str
    runs: int
    population_size: int
    max_evaluations: int
    algorithms: tuple[str, ...]
    scenario_mode: str
    scenario_count: int
    aggregation: RobustAggregation
    outputs: tuple[str, ...]
    storage: StorageProfile
    recommended_case_count: int
    default_standardized_effect: float | None
    default_power: float | None
    description: str
    limitations: str

    def output_labels(self) -> tuple[str, ...]:
        return tuple(OUTPUT_REQUIREMENTS[key].label for key in self.outputs)

    def guidance(
        self,
        selected_case: str,
        *,
        standardized_effect: float | None = None,
        target_power: float | None = None,
    ) -> str:
        recommendation = recommend_paired_runs(
            self.strength,
            standardized_effect=standardized_effect,
            target_power=target_power,
        )
        repeated = f"{recommendation.runs} paired independent runs per algorithm and case"
        cases = (
            f"at least {self.recommended_case_count} independently declared case/formulation blocks"
            if self.recommended_case_count > 1
            else f"the selected {selected_case} formulation"
        )
        robust = (
            "deterministic operating conditions"
            if self.scenario_mode == "deterministic"
            else f"{self.scenario_count} declared scenarios with {self.aggregation.value} aggregation"
        )
        return (
            f"{self.label}: {repeated}; {len(self.algorithms)} comparison algorithms; "
            f"{self.max_evaluations} equal evaluations per run; {robust}; {cases}.\n\n"
            f"Recommended outputs: {', '.join(self.output_labels())}.\n\n"
            f"Run-planning basis: {recommendation.explanation}\n\n"
            f"Interpretation boundary: {self.limitations}"
        )


@dataclass(frozen=True, slots=True)
class PairedRunRecommendation:
    runs: int
    minimum_runs: int
    standardized_effect: float | None
    target_power: float | None
    family_alpha: float
    comparisons: int
    failure_allowance: float
    explanation: str


_BASE_OUTPUTS = (
    "objective_convergence",
    "constraint_convergence",
    "constraint_decomposition",
    "voltage_profile",
    "branch_loading",
    "generator_reactive_power",
    "control_changes",
    "median_convergence",
    "objective_boxplot",
    "feasible_run_probability",
    "descriptive_statistics",
)

_PLANS = {
    StudyStrength.LOW: StudyStrengthPlan(
        StudyStrength.LOW,
        "Low-cost screening",
        5,
        40,
        2_000,
        ("CALO", "TLBO", "PSO"),
        "deterministic",
        1,
        RobustAggregation.EXPECTED,
        _BASE_OUTPUTS,
        StorageProfile.FULL_SINGLE_RUN,
        1,
        None,
        None,
        "Fast screening for software checks, feasibility discovery, and early hypothesis formation.",
        "Useful for screening only; five runs do not support broad superiority or generalization claims.",
    ),
    StudyStrength.MODERATE: StudyStrengthPlan(
        StudyStrength.MODERATE,
        "Moderate comparative study",
        15,
        50,
        5_000,
        ("CALO", "TLBO", "PSO", "QODE", "GWO"),
        "deterministic",
        1,
        RobustAggregation.EXPECTED,
        _BASE_OUTPUTS
        + (
            "convergence_uncertainty_band",
            "objective_violin",
            "evaluations_to_feasibility",
            "wilcoxon_holm",
            "effect_sizes",
            "calo_regime_timeline",
            "calo_operator_usage",
        ),
        StorageProfile.REPEATED_STATISTICS,
        2,
        0.80,
        0.80,
        "A balanced paired comparison with uncertainty, significance, and effect-size evidence.",
        "Evidence is still sensitive to case choice; repeat on another declared case before generalizing.",
    ),
    StudyStrength.GOOD: StudyStrengthPlan(
        StudyStrength.GOOD,
        "Rigorous robust study",
        30,
        50,
        7_500,
        ("CALO", "TLBO", "PSO", "QODE", "CLPSO", "MTLA-DE", "GWO", "MVO"),
        "monte_carlo",
        30,
        RobustAggregation.MEAN_RISK,
        _BASE_OUTPUTS
        + (
            "convergence_uncertainty_band",
            "objective_violin",
            "evaluations_to_feasibility",
            "wilcoxon_holm",
            "effect_sizes",
            "scenario_loss_heatmap",
            "scenario_feasibility_heatmap",
            "best_validated_voltage_profile",
            "best_validated_branch_heatmap",
            "calo_regime_timeline",
            "calo_operator_usage",
            "calo_operator_success",
        ),
        StorageProfile.ROBUST_FULL,
        3,
        0.60,
        0.90,
        "A robust, validated comparison with paired statistics and multiple competitive baselines.",
        "Strong evidence requires all declared cases, seeds, exclusions, and failed runs to be reported.",
    ),
    StudyStrength.STRONG: StudyStrengthPlan(
        StudyStrength.STRONG,
        "Comprehensive confirmatory study",
        50,
        60,
        12_000,
        ("CALO", "TLBO", "PSO", "QODE", "CLPSO", "MTLA-DE", "GWO", "MVO"),
        "monte_carlo",
        50,
        RobustAggregation.CVAR,
        _BASE_OUTPUTS
        + (
            "convergence_uncertainty_band",
            "objective_violin",
            "evaluations_to_feasibility",
            "wilcoxon_holm",
            "effect_sizes",
            "friedman_ranking",
            "critical_difference",
            "scenario_loss_heatmap",
            "scenario_feasibility_heatmap",
            "cvar_curve",
            "best_validated_voltage_profile",
            "best_validated_branch_heatmap",
            "calo_regime_timeline",
            "calo_operator_usage",
            "calo_operator_success",
            "parity_scatter",
        ),
        StorageProfile.ROBUST_FULL,
        4,
        0.50,
        0.95,
        "A preregistration-ready confirmatory protocol with robust tails, validation, and cross-block ranks.",
        "No preset guarantees acceptance or universal superiority; claims remain limited to executed cases, budgets, and uncertainty models.",
    ),
}


def study_strength_plan(strength: StudyStrength | str) -> StudyStrengthPlan:
    return _PLANS[StudyStrength(strength)]


def recommend_paired_runs(
    strength: StudyStrength | str,
    *,
    standardized_effect: float | None = None,
    target_power: float | None = None,
    family_alpha: float = 0.05,
    failure_allowance: float = 0.10,
    planned_comparisons: int | None = None,
) -> PairedRunRecommendation:
    """Return a conservative paired-run planning approximation.

    The calculation uses the normal approximation for a paired standardized mean difference and
    the smallest Holm threshold (family alpha divided by CALO-versus-baseline comparisons), then
    inflates for failed/non-converged runs. It is a planning aid: confirmatory work should replace
    the assumed effect with a preregistered pilot estimate and, where material, simulation-based
    power for the final nonparametric analysis.
    """

    plan = study_strength_plan(strength)
    comparisons = (
        max(1, len(plan.algorithms) - 1)
        if planned_comparisons is None
        else int(planned_comparisons)
    )
    if comparisons < 1:
        raise ValueError("planned_comparisons must be at least 1")
    if plan.strength is StudyStrength.LOW:
        return PairedRunRecommendation(
            runs=plan.runs,
            minimum_runs=plan.runs,
            standardized_effect=None,
            target_power=None,
            family_alpha=float(family_alpha),
            comparisons=comparisons,
            failure_allowance=float(failure_allowance),
            explanation=(
                f"{plan.runs}-run screening floor; no inferential power claim is made. "
                "Use pilot paired differences before escalating this protocol."
            ),
        )

    effect_source = (
        plan.default_standardized_effect if standardized_effect is None else standardized_effect
    )
    power_source = plan.default_power if target_power is None else target_power
    if effect_source is None or power_source is None:
        raise RuntimeError("Inferential study plans require default effect and power assumptions")
    effect = float(effect_source)
    power = float(power_source)
    alpha = float(family_alpha)
    allowance = float(failure_allowance)
    if not math.isfinite(effect) or not 0.0 < effect <= 3.0:
        raise ValueError("standardized_effect must be finite and lie in (0, 3]")
    if not math.isfinite(power) or not 0.50 <= power < 1.0:
        raise ValueError("target_power must be finite and lie in [0.50, 1)")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1)")
    if not math.isfinite(allowance) or not 0.0 <= allowance < 0.50:
        raise ValueError("failure_allowance must lie in [0, 0.50)")

    per_comparison_alpha = alpha / comparisons
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - per_comparison_alpha / 2.0)
    z_power = normal.inv_cdf(power)
    analytical = ((z_alpha + z_power) / effect) ** 2
    inflated = math.ceil(analytical / (1.0 - allowance))
    runs = max(int(plan.runs), int(inflated))
    return PairedRunRecommendation(
        runs=runs,
        minimum_runs=int(plan.runs),
        standardized_effect=effect,
        target_power=power,
        family_alpha=alpha,
        comparisons=comparisons,
        failure_allowance=allowance,
        explanation=(
            f"planning approximation for standardized paired effect {effect:.2f}, "
            f"power {power:.0%}, two-sided family α={alpha:.2f} across {comparisons} "
            f"CALO-versus-baseline comparisons, with {allowance:.0%} failed-run allowance. "
            "Replace the assumed effect with preregistered pilot paired differences before the "
            "confirmatory campaign."
        ),
    )


def summarize_study_protocol_change(before, after) -> tuple[str, ...]:
    """Describe the scientist-relevant fields changed by a guided protocol."""

    before_values = {
        "Evidence strength": str(before.study_strength),
        "Reference case": str(before.case_name),
        "Paired runs per algorithm/case": str(before.runs),
        "Primary algorithms": ", ".join(before.algorithms),
        "Population size": str(before.population_size),
        "Evaluation budget": str(before.budget.max_evaluations),
        "Scenario plan": f"{before.scenarios.mode} ({before.scenarios.count})",
        "Required outputs": str(len(before.portfolio.requested_outputs)),
        "Result directory": str(before.output_directory),
    }
    after_values = {
        "Evidence strength": str(after.study_strength),
        "Reference case": str(after.case_name),
        "Paired runs per algorithm/case": str(after.runs),
        "Primary algorithms": ", ".join(after.algorithms),
        "Population size": str(after.population_size),
        "Evaluation budget": str(after.budget.max_evaluations),
        "Scenario plan": f"{after.scenarios.mode} ({after.scenarios.count})",
        "Required outputs": str(len(after.portfolio.requested_outputs)),
        "Result directory": str(after.output_directory),
    }
    return tuple(
        f"{label}: {before_values[label]} → {after_values[label]}"
        for label in before_values
        if before_values[label] != after_values[label]
    )


def apply_study_strength(
    config,
    strength: StudyStrength | str,
    *,
    case_name: str,
    standardized_effect: float | None = None,
    target_power: float | None = None,
) -> StudyStrengthPlan:
    """Apply one coherent protocol without touching CALO policy-training settings."""
    plan = study_strength_plan(strength)
    algorithms = [name for name in plan.algorithms if name in SPECS]
    if "CALO" not in algorithms:
        raise RuntimeError("The active algorithm registry does not expose CALO.")
    run_plan = recommend_paired_runs(
        plan.strength,
        standardized_effect=standardized_effect,
        target_power=target_power,
    )
    config.study_strength = plan.strength.value
    config.study_case_plan = [str(case_name)]
    config.case_name = str(case_name)
    config.algorithms = algorithms
    config.runs = int(run_plan.runs)
    config.study_standardized_effect = run_plan.standardized_effect
    config.study_target_power = run_plan.target_power
    config.study_family_alpha = float(run_plan.family_alpha)
    config.study_failure_allowance = float(run_plan.failure_allowance)
    config.study_run_planning_method = (
        "screening_floor"
        if plan.strength is StudyStrength.LOW
        else "paired_normal_holm_approximation"
    )
    config.population_size = int(plan.population_size)
    config.max_iterations = int(plan.max_evaluations // plan.population_size)
    config.budget.policy = BudgetPolicy.EQUAL_EVALUATIONS
    config.budget.max_evaluations = int(plan.max_evaluations)
    config.budget.wall_clock_seconds = None
    config.scenarios.mode = plan.scenario_mode
    config.scenarios.count = int(plan.scenario_count)
    config.scenarios.active_load_std = 0.0 if plan.scenario_mode == "deterministic" else 0.05
    config.scenarios.reactive_load_std = 0.0 if plan.scenario_mode == "deterministic" else 0.05
    config.scenarios.branch_outages = []
    config.scenarios.generator_outages = []
    config.robust_objective.aggregation = plan.aggregation
    config.portfolio.kind = PortfolioKind.OVERALL_EXPERIMENT
    # The powered paired-run recommendation is authoritative. A legacy fixed-count profile would
    # silently reduce it when Portfolio Manager is reapplied, so every guided study uses the
    # neutral custom-count contract with the calculated run count persisted below.
    config.portfolio.evidence_profile = EvidenceProfile.CUSTOM
    config.portfolio.custom_runs = int(run_plan.runs)
    config.portfolio.requested_outputs = list(plan.outputs)
    config.portfolio.storage_profile = plan.storage
    config.portfolio.require_independent_validation = True
    config.portfolio.reuse_compatible_results = True
    config.portfolio.enable_resume = True
    config.portfolio.auto_validate = False
    config.portfolio.auto_export = False
    config.portfolio.name = plan.label
    config.checkpoint_interval_evaluations = min(500, int(plan.max_evaluations))
    config.resume_enabled = True
    config.safe_pause = True
    config.reuse_compatible_results = True
    config.output_directory = f"results_data/{plan.strength.value}_{case_name}"
    return plan
