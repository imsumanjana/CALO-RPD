"""Small reusable NumPy scratch-buffer pool used by CALO v4 host orchestration.

The scientific evaluator may keep its own CUDA/XPU workspaces; this pool avoids
repeated host allocation of candidate/direction buffers in the CALO control loop.
"""
from __future__ import annotations

import numpy as np


class ScratchPool:
    def __init__(self) -> None:
        self._buffers: dict[tuple[str, tuple[int, ...], str], np.ndarray] = {}

    def get(self, name: str, shape, dtype=np.float64, *, clear: bool = False) -> np.ndarray:
        shape = tuple(int(v) for v in shape)
        dtype = np.dtype(dtype)
        key = (str(name), shape, dtype.str)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = np.empty(shape, dtype=dtype)
            self._buffers[key] = buffer
        if clear:
            buffer.fill(0)
        return buffer

    @property
    def allocated_bytes(self) -> int:
        return int(sum(array.nbytes for array in self._buffers.values()))

    def clear(self) -> None:
        self._buffers.clear()
