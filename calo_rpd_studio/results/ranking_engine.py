"""Median-performance ranking."""
def rank_summary(summary):
    out = summary.copy()
    out["rank"] = out["median"].rank(method="average")
    return out.sort_values("rank")
