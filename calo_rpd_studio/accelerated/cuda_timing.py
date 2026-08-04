"""Fail-closed CUDA event timing for accelerator-eligible numerical windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, TypeVar


CUDA_NUMERICAL_TIME_SHARE_TARGET = 0.95
POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY = 100
POLICY_EPOCHS_PER_BOUNDARY = 10

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class CudaWindowTiming:
    """Timing evidence for one explicitly bounded CUDA numerical operation."""

    label: str
    device: str
    wall_seconds: float
    cuda_event_seconds: float
    raw_cuda_time_share: float
    cuda_time_share: float
    target_cuda_time_share: float
    measurement_consistent: bool
    target_met: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_cuda_window_timing(
    *,
    label: str,
    device: str,
    wall_seconds: float,
    cuda_event_seconds: float,
    target_cuda_time_share: float = CUDA_NUMERICAL_TIME_SHARE_TARGET,
) -> CudaWindowTiming:
    """Build a bounded timing record without hiding invalid or contradictory samples."""

    wall = float(wall_seconds)
    cuda = float(cuda_event_seconds)
    target = float(target_cuda_time_share)
    if not 0.0 < target <= 1.0:
        raise ValueError("CUDA numerical-time share target must be in (0, 1]")
    if wall <= 0.0 or cuda <= 0.0:
        raw_share = 0.0
        consistent = False
    else:
        raw_share = cuda / wall
        # Event and host clocks are independent. Permit at most 2% or 2 ms of clock noise, while
        # retaining the raw ratio so inconsistent samples can never qualify silently.
        tolerance = max(0.002, wall * 0.02)
        consistent = cuda <= wall + tolerance
    share = min(max(raw_share, 0.0), 1.0)
    return CudaWindowTiming(
        label=str(label),
        device=str(device),
        wall_seconds=wall,
        cuda_event_seconds=cuda,
        raw_cuda_time_share=raw_share,
        cuda_time_share=share,
        target_cuda_time_share=target,
        measurement_consistent=consistent,
        target_met=bool(consistent and share >= target),
    )


def measure_cuda_window(
    operation: Callable[[], _T],
    *,
    device: str,
    label: str,
    target_cuda_time_share: float = CUDA_NUMERICAL_TIME_SHARE_TARGET,
    torch_module=None,
    clock: Callable[[], float] = perf_counter,
) -> tuple[_T, CudaWindowTiming]:
    """Execute and time one CUDA window, synchronizing only at its outer boundaries."""

    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    target_device = torch.device(device)
    if target_device.type != "cuda" or not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA window timing requires an available physical CUDA device")
    device_index = (
        int(target_device.index)
        if target_device.index is not None
        else int(torch.cuda.current_device())
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device_index)
    wall_started = float(clock())
    start_event.record()
    result = operation()
    end_event.record()
    end_event.synchronize()
    wall_seconds = float(clock()) - wall_started
    cuda_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    timing = summarize_cuda_window_timing(
        label=label,
        device=str(target_device),
        wall_seconds=wall_seconds,
        cuda_event_seconds=cuda_seconds,
        target_cuda_time_share=target_cuda_time_share,
    )
    return result, timing
