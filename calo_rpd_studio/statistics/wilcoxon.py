"""Guarded exact-pair Wilcoxon signed-rank testing."""

import numpy as np

from .paired import PairIntegrityError, wilcoxon_signed_rank_evidence


def wilcoxon_signed_rank(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size != b.size:
        raise PairIntegrityError(
            f"Paired Wilcoxon inputs must have equal lengths; received {a.size} and {b.size}"
        )
    if a.size and (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
        raise PairIntegrityError("Paired Wilcoxon inputs must contain only finite paired values")
    evidence = wilcoxon_signed_rank_evidence(a - b, alternative="two-sided")
    statistic = evidence["statistic"]
    p_value = evidence["p_value"]
    return {
        "statistic": float(statistic) if statistic is not None else float("nan"),
        "p_value": float(p_value) if p_value is not None else float("nan"),
        "n_pairs": int(a.size),
        "method": evidence,
    }
