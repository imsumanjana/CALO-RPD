"""Structured repeated-run statistical report."""

from .descriptive import descriptive_statistics


def build_statistical_report(groups):
    return {name: descriptive_statistics(values) for name, values in groups.items()}
