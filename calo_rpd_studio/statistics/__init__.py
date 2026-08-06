"""Scientific statistics with explicit exact-pair and method provenance contracts."""

from .paired import (
    PAIRED_ANALYSIS_SCHEMA_VERSION,
    RELATIVE_IMPROVEMENT_VERSION,
    PairIntegrityError,
    exact_keyed_pairs,
    exploratory_pair_status,
    matched_pairs_rank_biserial,
    relative_objective_improvement,
    wilcoxon_signed_rank_evidence,
)

__all__ = [
    "PAIRED_ANALYSIS_SCHEMA_VERSION",
    "RELATIVE_IMPROVEMENT_VERSION",
    "PairIntegrityError",
    "exact_keyed_pairs",
    "exploratory_pair_status",
    "matched_pairs_rank_biserial",
    "relative_objective_improvement",
    "wilcoxon_signed_rank_evidence",
]
