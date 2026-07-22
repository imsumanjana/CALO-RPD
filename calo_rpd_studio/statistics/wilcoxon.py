"""Guarded paired Wilcoxon signed-rank testing."""
import numpy as np
from scipy.stats import wilcoxon as _wilcoxon


def wilcoxon_signed_rank(a, b):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel(); n=min(a.size,b.size)
    a,b=a[:n],b[:n]; mask=np.isfinite(a)&np.isfinite(b); a,b=a[mask],b[mask]
    if a.size < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": int(a.size)}
    if np.allclose(a,b,rtol=0,atol=0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": int(a.size)}
    result=_wilcoxon(a,b,zero_method="wilcox",alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_pairs": int(a.size)}
