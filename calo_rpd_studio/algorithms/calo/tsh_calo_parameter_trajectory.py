"""Read-only summaries of policy parameter behavior retained in TSH-CALO trajectories."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np

from .ai_controller import PARAMETER_NAMES


TSH_CALO_PARAMETER_TRAJECTORY_SUMMARY_SCHEMA = "tsh-calo-parameter-trajectory-summary-v1"


def summarize_parameter_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Parameter trajectory is empty")
    by_parameter: dict[str, list[float]] = defaultdict(list)
    reward_associations: dict[str, list[tuple[float, float]]] = defaultdict(list)
    trigger_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        names = list(row.get("group_parameter_names", PARAMETER_NAMES))
        values = np.asarray(row.get("group_parameter_values", []), dtype=float)
        if values.shape != (3, len(names)) or not np.all(np.isfinite(values)):
            raise ValueError("Parameter trajectory contains invalid scaled group values")
        reward = float(row.get("reward", float("nan")))
        if not math.isfinite(reward):
            raise ValueError("Parameter trajectory contains a non-finite reward")
        triggered = bool(row.get("recovery_triggered", False))
        for index, name in enumerate(names):
            group_values = values[:, index]
            mean_value = float(np.mean(group_values))
            by_parameter[name].append(mean_value)
            reward_associations[name].append((mean_value, reward))
            if triggered:
                trigger_values[name].append(mean_value)
    parameters = []
    for name in PARAMETER_NAMES:
        values = np.asarray(by_parameter.get(name, []), dtype=float)
        pairs = reward_associations.get(name, [])
        correlation = None
        if len(pairs) >= 3:
            x = np.asarray([pair[0] for pair in pairs], dtype=float)
            y = np.asarray([pair[1] for pair in pairs], dtype=float)
            if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
                correlation = float(np.corrcoef(x, y)[0, 1])
        parameters.append(
            {
                "parameter": name,
                "observations": int(values.size),
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "reward_correlation": correlation,
                "mean_when_recovery_triggered": (
                    float(np.mean(trigger_values[name])) if trigger_values[name] else None
                ),
            }
        )
    return {
        "schema_version": TSH_CALO_PARAMETER_TRAJECTORY_SUMMARY_SCHEMA,
        "generation_count": len(rows),
        "parameters": parameters,
        "interpretation": "descriptive_within_run_association_only",
        "automatic_parameter_change": False,
    }
