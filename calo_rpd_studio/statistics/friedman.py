"""Friedman repeated-measures rank testing."""
from scipy.stats import friedmanchisquare

def friedman_test(*groups):
    result = friedmanchisquare(*groups)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
