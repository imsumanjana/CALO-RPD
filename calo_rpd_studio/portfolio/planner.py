"""Dependency-aware minimal task planning for run, experiment, and article portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import OUTPUT_REQUIREMENTS
from .models import ArticlePreset, EvidenceProfile, PortfolioConfig, PortfolioKind, StorageProfile


@dataclass(slots=True)
class PortfolioPlan:
    required_runs: int
    algorithms: list[str]
    total_jobs: int
    requested_outputs: list[str]
    required_fields: list[str]
    warnings: list[str] = field(default_factory=list)
    disabled_outputs: dict[str, str] = field(default_factory=dict)
    storage_profile: str = StorageProfile.REPEATED_STATISTICS.value
    require_validation: bool = False
    benchmark_blocks_required: int = 1

    def summary(self) -> str:
        text = (
            f"{len(self.algorithms)} algorithm(s) × {self.required_runs} paired run(s) = "
            f"{self.total_jobs} optimizer job(s)."
        )
        if self.disabled_outputs:
            text += f" {len(self.disabled_outputs)} selected output(s) are currently unavailable."
        if self.warnings:
            text += " " + " ".join(self.warnings)
        return text


class PortfolioPlanner:
    """Translate selected evidence into only the evaluations and stored fields it requires."""

    @staticmethod
    def apply_article_preset(config, portfolio: PortfolioConfig) -> None:
        preset = portfolio.article_preset
        if preset is ArticlePreset.NONE:
            return
        portfolio.kind = PortfolioKind.OVERALL_EXPERIMENT
        portfolio.evidence_profile = EvidenceProfile.TRANSACTIONS
        portfolio.require_independent_validation = True
        if preset is ArticlePreset.TLBO_MTLBO:
            from calo_rpd_studio.algorithms.registry import SPECS

            # Legacy MTLBO is not one of the 20 primary baselines in every release. Select it only
            # when the registry explicitly exposes it; otherwise retain TLBO and let the dedicated
            # CALO-ablation/legacy workflow supply MTLBO evidence.
            config.algorithms = [name for name in ("TLBO", "MTLBO") if name in SPECS]
            portfolio.requested_outputs = [
                "median_convergence",
                "convergence_uncertainty_band",
                "objective_boxplot",
                "feasible_run_probability",
                "wilcoxon_holm",
                "effect_sizes",
                "best_validated_voltage_profile",
                "control_changes",
            ]
        elif preset is ArticlePreset.CALO_DETERMINISTIC:
            from calo_rpd_studio.algorithms.registry import SPECS

            preferred = ["CALO", "TLBO", "QODE", "CLPSO", "MTLA-DE", "GWO", "MVO", "PSO"]
            config.algorithms = [name for name in preferred if name in SPECS]
            portfolio.requested_outputs = [
                "median_convergence",
                "convergence_uncertainty_band",
                "objective_boxplot",
                "objective_violin",
                "feasible_run_probability",
                "evaluations_to_feasibility",
                "friedman_ranking",
                "critical_difference",
                "constraint_decomposition",
                "best_validated_voltage_profile",
                "best_validated_branch_heatmap",
                "calo_regime_timeline",
                "calo_operator_usage",
                "calo_operator_success",
            ]
        elif preset is ArticlePreset.CALO_ROBUST:
            from calo_rpd_studio.algorithms.registry import SPECS

            preferred = ["CALO", "TLBO", "QODE", "CLPSO", "MTLA-DE", "GWO", "MVO", "PSO"]
            config.algorithms = [name for name in preferred if name in SPECS]
            portfolio.requested_outputs = [
                "scenario_loss_heatmap",
                "scenario_feasibility_heatmap",
                "cvar_curve",
                "contingency_matrix",
                "objective_boxplot",
                "feasible_run_probability",
                "friedman_ranking",
                "critical_difference",
            ]
            portfolio.storage_profile = StorageProfile.ROBUST_FULL
        elif preset is ArticlePreset.CALO_TRANSFER_ACCELERATOR:
            from calo_rpd_studio.algorithms.registry import SPECS

            preferred = ["CALO", "TLBO", "QODE", "CLPSO", "MTLA-DE", "GWO", "MVO", "PSO"]
            config.algorithms = [name for name in preferred if name in SPECS]
            portfolio.requested_outputs = [
                "median_convergence",
                "feasible_run_probability",
                "throughput_batch_scaling",
                "device_speedup",
                "parity_scatter",
                "calo_regime_timeline",
            ]

    @staticmethod
    def plan(config, portfolio: PortfolioConfig, benchmark_blocks: int = 1) -> PortfolioPlan:
        portfolio.validate()
        algorithms = list(config.algorithms)
        runs = portfolio.required_runs()
        disabled: dict[str, str] = {}
        warnings: list[str] = []
        required_fields: set[str] = {"final_metrics", "decoded_controls", "seed_provenance"}
        require_validation = bool(portfolio.require_independent_validation)
        minimum_required_runs = runs
        minimum_blocks = 1

        robust = str(getattr(config.scenarios, "mode", "deterministic")) != "deterministic"
        for output in portfolio.requested_outputs:
            req = OUTPUT_REQUIREMENTS.get(output)
            if req is None:
                disabled[output] = "Unknown output definition"
                continue
            if req.requires_calo and "CALO" not in algorithms:
                disabled[output] = "CALO is not selected"
                continue
            if req.robust_only and not robust:
                disabled[output] = "Requires a robust/scenario study"
                continue
            if len(algorithms) < req.minimum_algorithms:
                disabled[output] = f"Requires at least {req.minimum_algorithms} algorithms"
                continue
            if benchmark_blocks < req.minimum_blocks:
                disabled[output] = f"Requires at least {req.minimum_blocks} benchmark blocks"
                minimum_blocks = max(minimum_blocks, req.minimum_blocks)
                continue
            if portfolio.kind is PortfolioKind.SINGLE_RUN and req.minimum_runs > 1:
                disabled[output] = "Requires repeated independent runs"
                continue
            minimum_required_runs = max(minimum_required_runs, req.minimum_runs)
            require_validation = require_validation or req.requires_validation
            required_fields.update(req.required_fields)
            if req.accelerator_records:
                required_fields.add("accelerator_telemetry")

        if portfolio.kind is PortfolioKind.SINGLE_RUN:
            minimum_required_runs = 1
            if any(
                OUTPUT_REQUIREMENTS.get(key, None) and OUTPUT_REQUIREMENTS[key].minimum_runs > 1
                for key in portfolio.requested_outputs
            ):
                warnings.append(
                    "Repeated-run statistics were excluded from the single-run diagnostic portfolio."
                )
        elif minimum_required_runs > runs:
            warnings.append(
                f"Selected evidence requires at least {minimum_required_runs} runs; the plan was increased from {runs}."
            )
            runs = minimum_required_runs

        if minimum_blocks > benchmark_blocks:
            warnings.append(
                "Ranking/critical-difference evidence needs multiple benchmark blocks. Add cases or formulations, or remove those outputs."
            )
        if portfolio.evidence_profile is EvidenceProfile.EXPLORATORY:
            warnings.append("Exploratory evidence is not intended for final publication claims.")
        if portfolio.evidence_profile is EvidenceProfile.JOURNAL and runs < 30:
            warnings.append("Journal profile normally requires at least 30 repeated runs.")
        if portfolio.evidence_profile is EvidenceProfile.TRANSACTIONS and runs < 50:
            warnings.append("Transactions profile normally requires 50 repeated runs.")

        return PortfolioPlan(
            required_runs=runs,
            algorithms=algorithms,
            total_jobs=runs * len(algorithms),
            requested_outputs=list(portfolio.requested_outputs),
            required_fields=sorted(required_fields),
            warnings=warnings,
            disabled_outputs=disabled,
            storage_profile=portfolio.storage_profile.value,
            require_validation=require_validation,
            benchmark_blocks_required=minimum_blocks,
        )
