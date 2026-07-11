"""Compressed storage for numeric optimizer arrays."""
from __future__ import annotations
from pathlib import Path
import uuid
import numpy as np

class ResultStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_arrays(self, result):
        path = self.directory / f"{uuid.uuid4()}.npz"
        np.savez_compressed(
            path,
            best_vector=result.best_vector,
            convergence_history=np.asarray(result.convergence_history, float),
            final_population=np.asarray([]) if result.final_population is None else result.final_population,
        )
        return path

    @staticmethod
    def load(path):
        return dict(np.load(path, allow_pickle=False))
