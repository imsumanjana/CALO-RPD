"""Median-performance ranking."""

import pandas as pd  # noqa: F401


def rank_summary(summary):
    out = summary.copy()
    out["rank"] = out["median"].rank(method="average")
    return out.sort_values("rank")
