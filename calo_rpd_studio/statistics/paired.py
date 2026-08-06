"""Versioned exact-pair statistics for v12 qualification evidence.

Positive relative improvement and positive rank-biserial effect always mean that the candidate is
better for a minimization objective. Formal callers must provide exact keyed pairs; this module
never truncates arrays, invents positional pairing, or silently substitutes a different test.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.stats import rankdata, wilcoxon as _scipy_wilcoxon


PAIRED_ANALYSIS_SCHEMA_VERSION = "calo-paired-analysis-v2-exact-keyed-signed-rank"
RELATIVE_IMPROVEMENT_VERSION = "calo-relative-improvement-v2-symmetric-max-scale"
WILCOXON_METHOD = "scipy.stats.wilcoxon"
DEFAULT_ZERO_TOLERANCE = 1e-15
DEFAULT_OBJECTIVE_SCALE_FLOOR = 1e-12


class PairIntegrityError(ValueError):
    """Raised before analysis when the declared paired experiment is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class ExactPair:
    """One explicitly keyed candidate/comparator record pair."""

    key: tuple[object, ...]
    candidate: Mapping[str, object]
    comparator: Mapping[str, object]


def _record_key(record: Mapping[str, object], key_fields: Sequence[str]) -> tuple[object, ...]:
    missing = [field for field in key_fields if field not in record]
    if missing:
        raise PairIntegrityError("Paired record is missing key field(s): " + ", ".join(missing))
    key = tuple(record[field] for field in key_fields)
    try:
        hash(key)
    except TypeError as exc:
        raise PairIntegrityError(f"Paired record key is not hashable: {key!r}") from exc
    return key


def _index_records(
    records: Iterable[Mapping[str, object]],
    *,
    label: str,
    key_fields: Sequence[str],
) -> dict[tuple[object, ...], Mapping[str, object]]:
    indexed: dict[tuple[object, ...], Mapping[str, object]] = {}
    for record in records:
        key = _record_key(record, key_fields)
        if key in indexed:
            raise PairIntegrityError(f"Duplicate {label} paired-record key: {key!r}")
        indexed[key] = record
    return indexed


def exact_keyed_pairs(
    candidate_records: Iterable[Mapping[str, object]],
    comparator_records: Iterable[Mapping[str, object]],
    *,
    key_fields: Sequence[str] = ("case", "run_index"),
    expected_keys: Iterable[tuple[object, ...]] | None = None,
) -> tuple[ExactPair, ...]:
    """Return deterministic exact pairs or fail before any statistical calculation.

    Input order is not scientific identity. Explicit keys are authoritative, so differently ordered
    inputs align safely. Duplicate, missing, extra, unhashable, or undeclared keys fail closed.
    """

    fields = tuple(str(field) for field in key_fields)
    if not fields:
        raise PairIntegrityError("Exact pairing requires at least one key field")
    candidate = _index_records(candidate_records, label="candidate", key_fields=fields)
    comparator = _index_records(comparator_records, label="comparator", key_fields=fields)
    candidate_keys = set(candidate)
    comparator_keys = set(comparator)
    if candidate_keys != comparator_keys:
        missing_candidate = sorted(comparator_keys - candidate_keys, key=repr)
        missing_comparator = sorted(candidate_keys - comparator_keys, key=repr)
        raise PairIntegrityError(
            "Paired record sets differ; "
            f"missing candidate keys={missing_candidate!r}; "
            f"missing comparator keys={missing_comparator!r}"
        )
    if expected_keys is not None:
        expected = set(expected_keys)
        if candidate_keys != expected:
            missing = sorted(expected - candidate_keys, key=repr)
            extra = sorted(candidate_keys - expected, key=repr)
            raise PairIntegrityError(
                f"Paired record set does not match the preregistered keys; missing={missing!r}; "
                f"extra={extra!r}"
            )
    return tuple(
        ExactPair(key=key, candidate=candidate[key], comparator=comparator[key])
        for key in sorted(candidate_keys, key=repr)
    )


def exploratory_pair_status(
    candidate_records: Iterable[Mapping[str, object]],
    comparator_records: Iterable[Mapping[str, object]],
    *,
    key_fields: Sequence[str] = ("case", "run_index"),
    expected_keys: Iterable[tuple[object, ...]] | None = None,
) -> dict[str, object]:
    """Describe exploratory pair completeness without granting qualification authority."""

    try:
        pairs = exact_keyed_pairs(
            candidate_records,
            comparator_records,
            key_fields=key_fields,
            expected_keys=expected_keys,
        )
    except PairIntegrityError as exc:
        return {
            "status": "incomplete_pairs",
            "qualifying": False,
            "pair_count": 0,
            "pair_integrity_error": str(exc),
        }
    return {
        "status": "complete_pairs",
        "qualifying": False,
        "pair_count": len(pairs),
        "pair_integrity_error": "",
    }


def relative_objective_improvement(
    candidate: float,
    comparator: float,
    *,
    scale_floor: float = DEFAULT_OBJECTIVE_SCALE_FLOOR,
) -> float:
    """Return a symmetric-scale minimization improvement where positive favors the candidate."""

    candidate_value = float(candidate)
    comparator_value = float(comparator)
    floor = float(scale_floor)
    if not math.isfinite(candidate_value) or not math.isfinite(comparator_value):
        raise ValueError("Relative objective improvement requires finite paired objectives")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("Relative objective improvement scale_floor must be finite and positive")
    scale = max(abs(candidate_value), abs(comparator_value), floor)
    return float((comparator_value - candidate_value) / scale)


def matched_pairs_rank_biserial(
    improvements: Iterable[float],
    *,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> float:
    """Return the Wilcoxon matched-pairs rank-biserial effect.

    Absolute nonzero improvements are ranked with average ranks for ties. Positive rank mass means
    candidate improvement; zero differences follow the Wilcoxon ``wilcox`` convention and are
    excluded.
    """

    values = np.asarray(tuple(improvements), dtype=float).ravel()
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Rank-biserial effect requires finite paired improvements")
    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    nonzero = values[np.abs(values) > tolerance]
    if nonzero.size == 0:
        return 0.0
    ranks = np.asarray(rankdata(np.abs(nonzero), method="average"), dtype=float)
    positive = float(np.sum(ranks[nonzero > 0.0]))
    negative = float(np.sum(ranks[nonzero < 0.0]))
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def wilcoxon_signed_rank_evidence(
    improvements: Iterable[float],
    *,
    alternative: str,
    zero_method: str = "wilcox",
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> dict[str, object]:
    """Execute the declared Wilcoxon test without changing test family on failure."""

    values = np.asarray(tuple(improvements), dtype=float).ravel()
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Wilcoxon evidence requires finite paired improvements")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("Wilcoxon alternative must be two-sided, greater, or less")
    if zero_method != "wilcox":
        raise ValueError("v12 formal evidence currently requires zero_method='wilcox'")
    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    nonzero = values[np.abs(values) > tolerance]
    base: dict[str, object] = {
        "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
        "statistical_test": WILCOXON_METHOD,
        "library": "scipy",
        "library_version": str(scipy.__version__),
        "alternative": alternative,
        "zero_method": zero_method,
        "zero_tolerance": tolerance,
        "n_input_pairs": int(values.size),
        "n_nonzero_pairs": int(nonzero.size),
        "fallback_used": False,
        "fallback_reason": "",
    }
    if nonzero.size < 2:
        return {
            **base,
            "status": "insufficient_nonzero_pairs",
            "statistic": None,
            "p_value": None,
        }
    try:
        result = _scipy_wilcoxon(
            nonzero,
            alternative=alternative,
            zero_method=zero_method,
        )
    except ValueError as exc:
        return {
            **base,
            "status": "declared_test_invalid",
            "statistic": None,
            "p_value": None,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        "status": "ok",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def pair_manifest(pairs: Iterable[ExactPair], key_fields: Sequence[str]) -> list[dict[str, object]]:
    """Return a JSON-safe ordered manifest of exact pair identities."""

    fields = tuple(str(field) for field in key_fields)
    return [dict(zip(fields, pair.key, strict=True)) for pair in pairs]


__all__ = [
    "DEFAULT_OBJECTIVE_SCALE_FLOOR",
    "DEFAULT_ZERO_TOLERANCE",
    "ExactPair",
    "PAIRED_ANALYSIS_SCHEMA_VERSION",
    "PairIntegrityError",
    "RELATIVE_IMPROVEMENT_VERSION",
    "WILCOXON_METHOD",
    "exact_keyed_pairs",
    "exploratory_pair_status",
    "matched_pairs_rank_biserial",
    "pair_manifest",
    "relative_objective_improvement",
    "wilcoxon_signed_rank_evidence",
]
