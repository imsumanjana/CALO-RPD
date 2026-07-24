"""Friedman repeated-measures rank testing with degenerate-data guards."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import friedmanchisquare


def friedman_test(*groups):
    """Return finite, explicit evidence even when every paired block is tied.

    SciPy can return NaN/NaN for degenerate all-tied inputs.  Such data contain no rank evidence,
    so CALO-RPD records a neutral non-significant result instead of silently propagating NaNs into
    qualification tables or publication exports.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = friedmanchisquare(*groups)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "status": "degenerate_or_all_tied",
        }
    return {"statistic": statistic, "p_value": p_value, "status": "ok"}
