"""Portfolio-aware compressed storage for numeric optimizer arrays."""
from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np


class ResultStore:
    def __init__(self, directory, *, storage_profile: str = "repeated_statistics", required_fields=()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.storage_profile = str(storage_profile)
        self.required_fields = set(required_fields or ())

    def _keep_population(self) -> bool:
        return self.storage_profile in {"full_single_run", "robust_full"} or "population_samples" in self.required_fields

    def save_arrays(self, result):
        path = self.directory / f"{uuid.uuid4()}.npz"
        metadata = dict(getattr(result, "metadata", {}) or {})
        arrays: dict[str, np.ndarray] = {
            "best_vector": np.asarray(result.best_vector, dtype=float),
            "convergence_history": np.asarray(result.convergence_history, dtype=float),
            "final_population": (
                np.asarray(result.final_population, dtype=float)
                if self._keep_population() and result.final_population is not None
                else np.asarray([], dtype=float)
            ),
        }
        # Frequently plotted histories are duplicated into the compressed trace so publication
        # generation does not need to parse large JSON values. JSON metadata remains the canonical
        # human-readable record for backward compatibility.
        for key in (
            "convergence_evaluations",
            "best_feasible_objective_history",
            "best_constraint_violation_history",
            "incumbent_objective_history",
        ):
            values = metadata.get(key)
            if values is not None:
                arrays[key] = np.asarray(values, dtype=float)
        components = metadata.get("constraint_component_histories", {}) or {}
        if "constraint_components" in self.required_fields or self.storage_profile in {"full_single_run", "robust_full"}:
            for key, values in components.items():
                arrays[f"constraint_component__{key}"] = np.asarray(values, dtype=float)
        np.savez_compressed(path, **arrays)
        return path

    @staticmethod
    def load(path):
        return dict(np.load(path, allow_pickle=False))
