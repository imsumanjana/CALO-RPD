"""PyTorch device selection and numerical-safety helpers.

The accelerator backend is deliberately optional at import time.  The first-launch bootstrap
installs a hardware-appropriate PyTorch build, while CPU-only scientific/reference workflows can
still import the rest of CALO-RPD Studio without importing torch eagerly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceContext:
    requested: str
    resolved: str
    backend: str
    name: str
    dtype_name: str = "float64"
    accelerator_available: bool = False


def _torch():
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "The tensor scientific backend requires PyTorch. Run bootstrap.py to install or repair prerequisites."
        ) from exc


def resolve_device(requested: str = "auto", *, require_accelerator: bool = False) -> DeviceContext:
    """Resolve ``auto``/CUDA/XPU/CPU to a verified PyTorch device.

    The returned identifier follows PyTorch numbering and does not necessarily match Windows Task
    Manager adapter numbers.  Double precision is mandatory for the ORPD scientific evaluator.
    """

    torch = _torch()
    requested = str(requested or "auto").lower().strip()
    if requested in {"gpu", "nvidia"}:
        requested = "cuda:0"
    if requested in {"intel", "intel_gpu"}:
        requested = "xpu:0"

    candidates: list[str]
    if requested == "auto":
        candidates = ["cuda:0", "xpu:0", "cpu"]
    else:
        candidates = [requested]

    for candidate in candidates:
        if candidate.startswith("cuda"):
            available = bool(torch.cuda.is_available())
            if not available:
                continue
            index = int(candidate.split(":", 1)[1]) if ":" in candidate else 0
            if index >= int(torch.cuda.device_count()):
                continue
            return DeviceContext(
                requested,
                f"cuda:{index}",
                "cuda",
                str(torch.cuda.get_device_name(index)),
                accelerator_available=True,
            )
        if candidate.startswith("xpu"):
            xpu = getattr(torch, "xpu", None)
            available = bool(xpu is not None and xpu.is_available())
            if not available:
                continue
            index = int(candidate.split(":", 1)[1]) if ":" in candidate else 0
            count = int(xpu.device_count()) if hasattr(xpu, "device_count") else 1
            if index >= count:
                continue
            try:
                name = str(xpu.get_device_name(index))
            except Exception:
                name = f"Intel XPU {index}"
            return DeviceContext(
                requested,
                f"xpu:{index}",
                "xpu",
                name,
                accelerator_available=True,
            )
        if candidate == "cpu":
            if require_accelerator:
                continue
            return DeviceContext(requested, "cpu", "cpu", "CPU", accelerator_available=False)

    if require_accelerator:
        raise RuntimeError(f"Requested accelerator {requested!r} is not available to this PyTorch runtime")
    return DeviceContext(requested, "cpu", "cpu", "CPU", accelerator_available=False)


def torch_dtype(dtype_name: str = "float64") -> Any:
    torch = _torch()
    normalized = str(dtype_name).lower().strip()
    if normalized not in {"float64", "double", "fp64"}:
        raise ValueError("The v3 scientific backend requires float64/double precision")
    return torch.float64


def reflect_unit_interval(values):
    """Reflection boundary handling for tensors in normalized decision space.

    Reflection avoids the population collapse that hard clipping can create at 0 and 1 while still
    preserving the common bounded search space used by every optimizer.
    """

    torch = _torch()
    x = torch.remainder(values, 2.0)
    return torch.where(x <= 1.0, x, 2.0 - x)
