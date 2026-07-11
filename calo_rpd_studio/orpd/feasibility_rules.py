"""Deb-style feasibility-first comparisons."""
from __future__ import annotations
def better(a,b,tol=1e-12):
    if b is None:return True
    if a.feasible and not b.feasible:return True
    if b.feasible and not a.feasible:return False
    if a.feasible:return a.value<b.value-tol
    if a.violation<b.violation-tol:return True
    if abs(a.violation-b.violation)<=tol:return a.value<b.value
    return False
def sort_key(e):return (0 if e.feasible else 1,e.value if e.feasible else e.violation,e.value)
