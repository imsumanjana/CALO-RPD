"""Weighted empirical conditional value at risk."""
from __future__ import annotations
import numpy as np
def weighted_cvar(values,weights,alpha=0.95):
    v=np.asarray(values,float);w=np.asarray(weights,float);w=w/w.sum();order=np.argsort(v);v=v[order];w=w[order];cdf=np.cumsum(w);var=v[np.searchsorted(cdf,alpha,side='left')];mask=v>=var;return float(np.sum(v[mask]*w[mask])/max(np.sum(w[mask]),1e-15))
