"""Evidence-output catalogue and minimum-data dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputRequirement:
    key: str
    label: str
    category: str
    minimum_runs: int = 1
    minimum_algorithms: int = 1
    minimum_blocks: int = 1
    requires_validation: bool = False
    requires_calo: bool = False
    robust_only: bool = False
    accelerator_records: bool = False
    required_fields: tuple[str, ...] = ()


OUTPUT_REQUIREMENTS = {
    item.key: item
    for item in (
        OutputRequirement(
            "objective_convergence",
            "Objective convergence",
            "Convergence",
            required_fields=("convergence",),
        ),
        OutputRequirement(
            "constraint_convergence",
            "Constraint-violation convergence",
            "Convergence",
            required_fields=("convergence", "constraint_history"),
        ),
        OutputRequirement(
            "constraint_decomposition",
            "Constraint decomposition",
            "Feasibility",
            required_fields=("constraint_components",),
        ),
        OutputRequirement(
            "voltage_profile",
            "Initial/optimized voltage profile",
            "Power system",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "voltage_heatmap",
            "Bus-voltage heatmap",
            "Power system",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "branch_loading",
            "Branch-loading chart",
            "Power system",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "branch_loading_heatmap",
            "Branch-loading heatmap",
            "Power system",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "generator_reactive_power",
            "Generator reactive-power profile",
            "Power system",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "control_changes",
            "Initial/final control changes",
            "Controls",
            required_fields=("decoded_controls",),
        ),
        OutputRequirement(
            "objective_violation_scatter",
            "Objective–violation scatter",
            "Relationships",
            required_fields=("population_samples",),
        ),
        OutputRequirement(
            "calo_regime_timeline",
            "CALO cognitive-regime timeline",
            "CALO diagnostics",
            requires_calo=True,
            required_fields=("calo_diagnostics",),
        ),
        OutputRequirement(
            "calo_operator_usage",
            "CALO operator utilization",
            "CALO diagnostics",
            requires_calo=True,
            required_fields=("calo_diagnostics",),
        ),
        OutputRequirement(
            "calo_operator_success",
            "CALO operator success",
            "CALO diagnostics",
            requires_calo=True,
            required_fields=("calo_diagnostics",),
        ),
        OutputRequirement(
            "median_convergence",
            "Median convergence",
            "Repeated-run statistics",
            minimum_runs=5,
            required_fields=("convergence",),
        ),
        OutputRequirement(
            "convergence_uncertainty_band",
            "Convergence uncertainty band",
            "Repeated-run statistics",
            minimum_runs=5,
            required_fields=("convergence",),
        ),
        OutputRequirement(
            "objective_boxplot", "Objective boxplot", "Repeated-run statistics", minimum_runs=5
        ),
        OutputRequirement(
            "objective_violin", "Objective violin plot", "Repeated-run statistics", minimum_runs=5
        ),
        OutputRequirement(
            "feasible_run_probability",
            "Feasible-run probability",
            "Repeated-run statistics",
            minimum_runs=5,
        ),
        OutputRequirement(
            "evaluations_to_feasibility",
            "Evaluations-to-feasibility distribution",
            "Repeated-run statistics",
            minimum_runs=5,
            required_fields=("first_feasible_evaluation",),
        ),
        OutputRequirement(
            "descriptive_statistics", "Descriptive statistics", "Statistics", minimum_runs=5
        ),
        OutputRequirement(
            "wilcoxon_holm",
            "Wilcoxon tests with Holm correction",
            "Statistics",
            minimum_runs=10,
            minimum_algorithms=2,
        ),
        OutputRequirement(
            "effect_sizes", "Effect-size plot", "Statistics", minimum_runs=10, minimum_algorithms=2
        ),
        OutputRequirement(
            "friedman_ranking",
            "Friedman ranking",
            "Statistics",
            minimum_runs=10,
            minimum_algorithms=3,
            minimum_blocks=2,
        ),
        OutputRequirement(
            "critical_difference",
            "Critical-difference diagram",
            "Statistics",
            minimum_runs=10,
            minimum_algorithms=3,
            minimum_blocks=2,
        ),
        OutputRequirement(
            "best_validated_voltage_profile",
            "Best validated voltage profile",
            "Publication",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "best_validated_branch_heatmap",
            "Best validated branch heatmap",
            "Publication",
            requires_validation=True,
            required_fields=("solution_state",),
        ),
        OutputRequirement(
            "scenario_loss_heatmap",
            "Scenario-loss heatmap",
            "Robustness",
            minimum_runs=5,
            robust_only=True,
            required_fields=("scenario_metrics",),
        ),
        OutputRequirement(
            "scenario_feasibility_heatmap",
            "Scenario-feasibility heatmap",
            "Robustness",
            minimum_runs=5,
            robust_only=True,
            required_fields=("scenario_metrics",),
        ),
        OutputRequirement(
            "cvar_curve",
            "CVaR curve",
            "Robustness",
            minimum_runs=5,
            robust_only=True,
            required_fields=("scenario_metrics",),
        ),
        OutputRequirement(
            "contingency_matrix",
            "Contingency violation matrix",
            "Robustness",
            robust_only=True,
            required_fields=("scenario_metrics",),
        ),
        OutputRequirement(
            "throughput_batch_scaling",
            "Throughput versus batch size",
            "Accelerator",
            accelerator_records=True,
        ),
        OutputRequirement(
            "device_speedup", "CPU/CUDA/XPU speedup", "Accelerator", accelerator_records=True
        ),
        OutputRequirement(
            "parity_scatter",
            "CPU/accelerator parity scatter",
            "Accelerator",
            accelerator_records=True,
        ),
    )
}


def categories() -> dict[str, list[OutputRequirement]]:
    result: dict[str, list[OutputRequirement]] = {}
    for item in OUTPUT_REQUIREMENTS.values():
        result.setdefault(item.category, []).append(item)
    return result
