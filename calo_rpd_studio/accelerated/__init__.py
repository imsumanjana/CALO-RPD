"""Accelerator-native ORPD evaluation and canonical optimizer support."""
from .device import DeviceContext, resolve_device
from .torch_orpd import AcceleratedORPDProblem, ParityReport, parity_check

__all__ = [
    "DeviceContext",
    "resolve_device",
    "AcceleratedORPDProblem",
    "ParityReport",
    "parity_check",
]

from .throughput_engine import CrossRunBatchBroker, ThroughputProfile, calibrate_evaluator
