"""Serializable portfolio configuration for evidence-aware experiment planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class PortfolioKind(str, Enum):
    SINGLE_RUN = "single_run"
    OVERALL_EXPERIMENT = "overall_experiment"


class EvidenceProfile(str, Enum):
    DIAGNOSTIC = "diagnostic"
    EXPLORATORY = "exploratory"
    JOURNAL = "journal"
    TRANSACTIONS = "transactions"
    CUSTOM = "custom"


class StorageProfile(str, Enum):
    MINIMAL = "minimal"
    FULL_SINGLE_RUN = "full_single_run"
    REPEATED_STATISTICS = "repeated_statistics"
    ROBUST_FULL = "robust_full"


class ArticlePreset(str, Enum):
    NONE = "none"
    TLBO_MTLBO = "tlbo_mtlbo"
    CALO_DETERMINISTIC = "calo_deterministic"
    CALO_ROBUST = "calo_robust"
    CALO_TRANSFER_ACCELERATOR = "calo_transfer_accelerator"


DEFAULT_SINGLE_RUN_OUTPUTS = [
    "objective_convergence",
    "constraint_convergence",
    "constraint_decomposition",
    "voltage_profile",
    "voltage_heatmap",
    "branch_loading",
    "branch_loading_heatmap",
    "generator_reactive_power",
    "control_changes",
    "objective_violation_scatter",
    "calo_regime_timeline",
    "calo_operator_usage",
]

DEFAULT_EXPERIMENT_OUTPUTS = [
    "median_convergence",
    "convergence_uncertainty_band",
    "objective_boxplot",
    "objective_violin",
    "feasible_run_probability",
    "evaluations_to_feasibility",
    "descriptive_statistics",
    "wilcoxon_holm",
    "effect_sizes",
    "friedman_ranking",
    "critical_difference",
    "best_validated_voltage_profile",
    "best_validated_branch_heatmap",
]


@dataclass(slots=True)
class PortfolioConfig:
    kind: PortfolioKind = PortfolioKind.OVERALL_EXPERIMENT
    evidence_profile: EvidenceProfile = EvidenceProfile.JOURNAL
    article_preset: ArticlePreset = ArticlePreset.NONE
    requested_outputs: list[str] = field(default_factory=lambda: list(DEFAULT_EXPERIMENT_OUTPUTS))
    custom_runs: int = 30
    require_independent_validation: bool = True
    reuse_compatible_results: bool = True
    enable_resume: bool = True
    checkpoint_interval_evaluations: int = 500
    storage_profile: StorageProfile = StorageProfile.REPEATED_STATISTICS
    auto_validate: bool = False
    auto_export: bool = False
    name: str = "Overall experiment portfolio"

    def required_runs(self) -> int:
        if self.kind is PortfolioKind.SINGLE_RUN:
            return 1
        if self.evidence_profile is EvidenceProfile.DIAGNOSTIC:
            base = 1
        elif self.evidence_profile is EvidenceProfile.EXPLORATORY:
            base = 10
        elif self.evidence_profile is EvidenceProfile.JOURNAL:
            base = 30
        elif self.evidence_profile is EvidenceProfile.TRANSACTIONS:
            base = 50
        else:
            base = max(1, int(self.custom_runs))
        # Persist the output-driven minimum in the serializable portfolio itself so a reloaded or
        # resumed configuration cannot silently revert to fewer runs than the selected evidence
        # requires.
        from .catalog import OUTPUT_REQUIREMENTS
        evidence_minimum = max(
            (OUTPUT_REQUIREMENTS[key].minimum_runs for key in self.requested_outputs if key in OUTPUT_REQUIREMENTS),
            default=1,
        )
        return max(base, int(evidence_minimum))

    def validate(self) -> None:
        if self.custom_runs <= 0:
            raise ValueError("custom_runs must be positive")
        if self.checkpoint_interval_evaluations <= 0:
            raise ValueError("checkpoint_interval_evaluations must be positive")
        if not self.requested_outputs:
            raise ValueError("Select at least one portfolio output")
        if self.kind is PortfolioKind.SINGLE_RUN and self.evidence_profile not in {
            EvidenceProfile.DIAGNOSTIC,
            EvidenceProfile.CUSTOM,
        }:
            # Single-run mode is diagnostic by definition. Normalize instead of silently planning
            # publication statistics from one stochastic run.
            self.evidence_profile = EvidenceProfile.DIAGNOSTIC

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, Enum):
                payload[key] = value.value
        return payload

    @classmethod
    def from_dict(cls, data: dict | None) -> "PortfolioConfig":
        data = dict(data or {})
        return cls(
            kind=PortfolioKind(data.get("kind", PortfolioKind.OVERALL_EXPERIMENT.value)),
            evidence_profile=EvidenceProfile(
                data.get("evidence_profile", EvidenceProfile.JOURNAL.value)
            ),
            article_preset=ArticlePreset(data.get("article_preset", ArticlePreset.NONE.value)),
            requested_outputs=list(data.get("requested_outputs", DEFAULT_EXPERIMENT_OUTPUTS)),
            custom_runs=int(data.get("custom_runs", 30)),
            require_independent_validation=bool(data.get("require_independent_validation", True)),
            reuse_compatible_results=bool(data.get("reuse_compatible_results", True)),
            enable_resume=bool(data.get("enable_resume", True)),
            checkpoint_interval_evaluations=int(data.get("checkpoint_interval_evaluations", 500)),
            storage_profile=StorageProfile(
                data.get("storage_profile", StorageProfile.REPEATED_STATISTICS.value)
            ),
            auto_validate=bool(data.get("auto_validate", False)),
            auto_export=bool(data.get("auto_export", False)),
            name=str(data.get("name", "Overall experiment portfolio")),
        )
