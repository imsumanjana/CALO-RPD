"""Cliff's delta nonparametric effect size."""
import numpy as np

def cliffs_delta(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    greater = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    return float((greater - less) / (len(a) * len(b)))
