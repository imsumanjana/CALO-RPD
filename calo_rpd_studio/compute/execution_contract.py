"""Versioned runtime resolution for every CALO-RPD experiment entry point.

Scientist-facing execution modes remain ``cuda_preferred`` and ``cpu_only``.  A formal run is a
strict purpose layered over ``cuda_preferred``: it requires a concretely identified NVIDIA CUDA
device and forbids CPU fallback.  Exploratory CUDA-preferred runs may perform one explicitly
recorded full-request CPU restart when the configuration permits it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import os
import platform
import re
from typing import Any

from .resource_scheduler import DeviceSnapshot, ResourceMonitor, ResourceSnapshot


EXECUTION_CONTRACT_SCHEMA = "calo-runtime-execution-contract-v2"
CURRENT_EXECUTION_MODES = frozenset({"cuda_preferred", "cpu_only"})


class ExecutionPurpose(str, Enum):
    EXPLORATORY = "exploratory"
    FORMAL = "formal"


class FallbackPolicy(str, Enum):
    FORBIDDEN = "forbidden"
    FULL_REQUEST_CPU_RESTART = "full_request_cpu_restart"
    NOT_APPLICABLE = "not_applicable"


class ExecutionResolutionError(RuntimeError):
    """The requested execution contract cannot be resolved truthfully."""


class FormalCudaRequired(ExecutionResolutionError):
    """A formal CUDA-only run has no admissible concrete CUDA device."""


class PhysicalDeviceIdentityUnavailable(ExecutionResolutionError):
    """A CUDA device lacks the UUID/PCI identity required for a physical lease."""


def _normalise_pci_bus_id(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.removeprefix("00000000:").removeprefix("0000:")
    text = re.sub(r"[^0-9a-f:.]", "", text)
    return text


def host_scope() -> str:
    """Return the operator-controlled physical-host scope used by device leases."""

    configured = os.environ.get("CALO_DEVICE_HOST_SCOPE", "").strip()
    return configured or platform.node().strip() or "local-host"


def container_scope() -> str:
    """Return a descriptive container/process scope without splitting a physical GPU lease."""

    configured = os.environ.get("CALO_DEVICE_CONTAINER_SCOPE", "").strip()
    return configured or "host-process"


def physical_device_identity(device: DeviceSnapshot) -> tuple[str, str]:
    """Return the stable lease identity and its authority for one CUDA snapshot."""

    uuid = str(device.hardware_uuid or "").strip().lower()
    if uuid:
        return f"gpu-uuid:{uuid}", "hardware_uuid"
    pci = _normalise_pci_bus_id(device.pci_bus_id)
    if pci:
        return f"pci:{pci}", "normalized_pci_bus_id"
    raise PhysicalDeviceIdentityUnavailable(
        f"{device.device_id} has neither a stable hardware UUID nor a PCI bus identity"
    )


@dataclass(frozen=True, slots=True)
class RuntimeExecutionResolution:
    """Immutable requested-versus-resolved device and fallback record."""

    requested_mode: str
    execution_purpose: str
    requested_device: str
    assigned_physical_device: str
    physical_identity_authority: str
    assigned_logical_device: str
    runtime_compute_device: str
    fallback_policy: str
    fallback_permitted: bool
    fallback_reason: str
    actual_computation_device: str
    device_name: str
    lease_host_scope: str
    lease_container_scope: str
    safe_memory_fraction: float
    cuda_only_claims_eligible: bool
    fallback_stratification_required: bool
    schema_version: str = EXECUTION_CONTRACT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _device_for_request(snapshot: ResourceSnapshot, requested: str) -> DeviceSnapshot | None:
    devices = tuple(
        device for device in snapshot.devices if device.backend == "cuda" and device.available
    )
    if requested in {"", "auto", "cuda"}:
        return devices[0] if devices else None
    return next((device for device in devices if device.device_id == requested), None)


def resolve_execution(
    config,
    *,
    snapshot: ResourceSnapshot | None = None,
    monitor: ResourceMonitor | None = None,
    assigned_device: str | None = None,
) -> RuntimeExecutionResolution:
    """Resolve one configuration to a concrete CPU or ``cuda:N`` contract.

    ``assigned_device`` is used by the GUI/parallel scheduler after it has selected a lane.  CLI
    entry points omit it and receive the same CUDA-first resolution from the sampled topology.
    """

    mode = str(getattr(config, "execution_backend", "cuda_preferred") or "").strip().lower()
    purpose = str(getattr(config, "execution_purpose", "exploratory") or "").strip().lower()
    requested = str(getattr(config, "requested_compute_device", "auto") or "auto").strip().lower()
    candidate = str(assigned_device or "").strip().lower()
    if any("xpu" in value for value in (mode, purpose, requested, candidate)):
        raise ExecutionResolutionError("Intel XPU execution is historical/view-only and cannot run")
    if mode not in CURRENT_EXECUTION_MODES:
        raise ExecutionResolutionError(f"Unsupported current execution mode: {mode!r}")
    if purpose not in {item.value for item in ExecutionPurpose}:
        raise ExecutionResolutionError(f"Unsupported execution purpose: {purpose!r}")
    if requested not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", requested):
        raise ExecutionResolutionError(
            "requested_compute_device must be auto, cpu, cuda, or a concrete cuda:N"
        )
    if mode == "cpu_only" and requested not in {"auto", "cpu"}:
        raise ExecutionResolutionError("CPU-only mode cannot request a CUDA runtime")
    if candidate and re.fullmatch(r"cuda:\d+", requested) and candidate != requested:
        raise ExecutionResolutionError(
            f"Scheduler assignment {candidate!r} does not match requested device {requested!r}"
        )
    if purpose == ExecutionPurpose.FORMAL.value and mode != "cuda_preferred":
        raise FormalCudaRequired("Formal execution requires cuda_preferred mode")

    sampled = snapshot
    selected: DeviceSnapshot | None = None
    if mode == "cuda_preferred":
        sampled = sampled or (monitor or ResourceMonitor()).sample()
        effective_request = candidate or requested
        if effective_request == "cpu":
            selected = None
        else:
            selected = _device_for_request(sampled, effective_request)

    fallback_enabled = bool(getattr(config, "cuda_cpu_fallback_enabled", True))
    if mode == "cpu_only":
        logical = runtime = actual = "cpu"
        physical = f"cpu:{host_scope()}"
        authority = "host_scope"
        fallback_policy = FallbackPolicy.NOT_APPLICABLE
        fallback_reason = "explicit_cpu_only_mode"
        name = "CPU"
    elif selected is not None:
        physical, authority = physical_device_identity(selected)
        logical = runtime = actual = str(selected.device_id)
        fallback_policy = (
            FallbackPolicy.FORBIDDEN
            if purpose == ExecutionPurpose.FORMAL.value or not fallback_enabled
            else FallbackPolicy.FULL_REQUEST_CPU_RESTART
        )
        fallback_reason = ""
        name = str(selected.name)
    else:
        if purpose == ExecutionPurpose.FORMAL.value:
            raise FormalCudaRequired(
                "Formal CUDA-only execution requires an available, concretely identified CUDA device"
            )
        if not fallback_enabled:
            raise ExecutionResolutionError(
                "CUDA-preferred execution could not resolve CUDA and CPU fallback is forbidden"
            )
        logical = runtime = actual = "cpu"
        physical = f"cpu:{host_scope()}"
        authority = "host_scope"
        fallback_policy = FallbackPolicy.FULL_REQUEST_CPU_RESTART
        fallback_reason = (
            "scheduler_assigned_cpu_after_cuda_admission"
            if candidate == "cpu"
            else (
                "explicit_cpu_device_request"
                if requested == "cpu"
                else "cuda_unavailable_at_resolution"
            )
        )
        name = "CPU"

    fallback_permitted = fallback_policy is FallbackPolicy.FULL_REQUEST_CPU_RESTART
    formal_cuda = bool(
        purpose == ExecutionPurpose.FORMAL.value
        and runtime.startswith("cuda:")
        and not fallback_permitted
    )
    return RuntimeExecutionResolution(
        requested_mode=mode,
        execution_purpose=purpose,
        requested_device=requested,
        assigned_physical_device=physical,
        physical_identity_authority=authority,
        assigned_logical_device=logical,
        runtime_compute_device=runtime,
        fallback_policy=fallback_policy.value,
        fallback_permitted=fallback_permitted,
        fallback_reason=fallback_reason,
        actual_computation_device=actual,
        device_name=name,
        lease_host_scope=host_scope(),
        lease_container_scope=container_scope(),
        safe_memory_fraction=0.80,
        cuda_only_claims_eligible=formal_cuda,
        fallback_stratification_required=bool(fallback_permitted or fallback_reason),
    )


def execution_claim_eligibility(resolution: RuntimeExecutionResolution) -> dict[str, bool]:
    """Return explicit claim gates so fallback results cannot enter CUDA-only strata."""

    eligible = bool(resolution.cuda_only_claims_eligible)
    return {
        "cuda_only_timing": eligible,
        "cuda_only_energy": eligible,
        "cuda_only_parity": eligible,
        "cuda_only_utilization": eligible,
        "cuda_only_equivalence": eligible,
    }


__all__ = [
    "CURRENT_EXECUTION_MODES",
    "EXECUTION_CONTRACT_SCHEMA",
    "ExecutionPurpose",
    "FallbackPolicy",
    "ExecutionResolutionError",
    "FormalCudaRequired",
    "PhysicalDeviceIdentityUnavailable",
    "RuntimeExecutionResolution",
    "container_scope",
    "execution_claim_eligibility",
    "host_scope",
    "physical_device_identity",
    "resolve_execution",
]
