"""Batch evaluation adapter."""

from __future__ import annotations


def evaluate_population(problem, population):
    return [problem.evaluate(x) for x in population]
