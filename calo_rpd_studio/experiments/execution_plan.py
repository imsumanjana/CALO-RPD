"""Canonical experiment-plan construction used by GUI and execution backends."""
from __future__ import annotations

from dataclasses import dataclass

from calo_rpd_studio.experiments.calo_ablation import ABLATION_SPECS, AblationSpec


@dataclass(frozen=True, slots=True)
class PlannedItem:
    """One optimizer/variant item inside one repeated run."""

    job_index: int
    run_index: int
    label: str
    ablation_spec: AblationSpec | None = None


COMPARISON_MODE = "comparison"
ABLATION_MODE = "ablation"


def labels_for_mode(config, mode: str) -> tuple[str, ...]:
    if mode == COMPARISON_MODE:
        return tuple(config.algorithms)
    if mode == ABLATION_MODE:
        return tuple(spec.label for spec in ABLATION_SPECS)
    raise ValueError(f"Unsupported experiment mode: {mode}")


def planned_item_count(config, mode: str) -> int:
    return int(config.runs) * len(labels_for_mode(config, mode))


def build_execution_plan(config, mode: str) -> list[PlannedItem]:
    """Return a stable run-major execution plan.

    Primary comparisons use exactly the algorithms selected in ``config.algorithms``.
    CALO ablation studies intentionally use the the fixed CALO v4 scientific ablation variants and do not
    consume the primary-algorithm selection.
    """

    if mode == COMPARISON_MODE:
        base = [(name, None) for name in config.algorithms]
    elif mode == ABLATION_MODE:
        base = [(spec.label, spec) for spec in ABLATION_SPECS]
    else:
        raise ValueError(f"Unsupported experiment mode: {mode}")

    plan: list[PlannedItem] = []
    job_index = 0
    for run_index in range(int(config.runs)):
        for label, spec in base:
            plan.append(PlannedItem(job_index, run_index, label, spec))
            job_index += 1
    return plan
