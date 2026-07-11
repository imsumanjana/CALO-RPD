"""Paired Wilcoxon signed-rank testing."""
from scipy.stats import wilcoxon as _wilcoxon

def wilcoxon_signed_rank(a, b):
    result = _wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
