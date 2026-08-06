"""Canonical CUDA/CPU runtime-device binding and execution attestation.

All experiment execution paths must bind the scientific evaluator, optimizer kernels, and CALO
policy inference settings to the same runtime identifier. This module is the single authority for
that binding so a job labelled ``cuda:0`` cannot silently retain a CPU evaluator.
"""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

from calo_rpd_studio.continuation.runtime_binding import bind_exact_run_checkpoint
from calo_rpd_studio.compute.execution_contract import (
    execution_claim_eligibility,
    resolve_execution,
)


def _apply_resolution(local, resolution) -> None:
    payload = resolution.to_dict()
    payload["claim_eligibility"] = execution_claim_eligibility(resolution)
    local.runtime_assigned_physical_device = resolution.assigned_physical_device
    local.runtime_assigned_logical_device = resolution.assigned_logical_device
    local.runtime_compute_device = resolution.runtime_compute_device
    local.runtime_fallback_policy = resolution.fallback_policy
    local.runtime_fallback_reason = resolution.fallback_reason
    local.runtime_device_resolution = payload
    local.runtime_resolution_process_id = os.getpid()
    local.cuda_cpu_fallback_enabled = bool(resolution.fallback_permitted)


def bind_config_to_device(config, compute_device: str, item=None):
    """Return a deep-copied config fully bound to ``compute_device``.

    Runtime IDs are PyTorch IDs (``cuda:N`` or ``cpu``), not Windows Task Manager GPU
    numbers.  The binding is intentionally applied to every configured algorithm plus the canonical
    CALO/TLBO entries used by continuation/ablation paths.
    """

    local = deepcopy(config)
    resolution = resolve_execution(local, assigned_device=str(compute_device or ""))
    _apply_resolution(local, resolution)
    device = resolution.runtime_compute_device
    parameters = dict(local.algorithm_parameters)
    names = set(getattr(local, "algorithms", ()) or ())
    names.update(parameters)
    names.update(("CALO", "TLBO"))
    for name in sorted(names):
        values = dict(parameters.get(name, {}))
        values["execution_device"] = device
        values["runtime_fallback_policy"] = resolution.fallback_policy
        values["runtime_physical_device"] = resolution.assigned_physical_device
        if str(getattr(local, "scientific_backend", "cpu_reference")) == "torch_fp64":
            values["optimizer_backend"] = "torch"
        if name == "CALO":
            values["inference_device"] = device
            values["policy_control_plane"] = "bound_to_assigned_runtime_v67"
        parameters[name] = values
    local.algorithm_parameters = parameters
    return bind_exact_run_checkpoint(local, item)


def resolve_config_for_entrypoint(config, *, monitor=None):
    """Return a copied configuration resolved before a CLI or direct runner starts."""

    local = deepcopy(config)
    if int(getattr(local, "runtime_resolution_process_id", 0)) == os.getpid() and dict(
        getattr(local, "runtime_device_resolution", {}) or {}
    ):
        return local
    resolution = resolve_execution(local, monitor=monitor)
    _apply_resolution(local, resolution)
    parameters = dict(local.algorithm_parameters)
    for name in sorted(set(local.algorithms) | set(parameters) | {"CALO", "TLBO"}):
        values = dict(parameters.get(name, {}))
        values["execution_device"] = resolution.runtime_compute_device
        values["runtime_fallback_policy"] = resolution.fallback_policy
        values["runtime_physical_device"] = resolution.assigned_physical_device
        if str(getattr(local, "scientific_backend", "cpu_reference")) == "torch_fp64":
            values["optimizer_backend"] = "torch"
        if name == "CALO":
            values["inference_device"] = resolution.runtime_compute_device
            values["policy_control_plane"] = "bound_to_assigned_runtime_v12"
        parameters[name] = values
    local.algorithm_parameters = parameters
    return local


def runtime_device_attestation(requested_device: str) -> dict[str, Any]:
    """Probe the exact runtime device with a tiny tensor and return auditable identity metadata.

    This is not a performance benchmark.  It proves that the current interpreter can allocate and
    execute on the requested runtime device and records the runtime-resolved hardware name.
    """

    requested = str(requested_device or "cpu")
    attestation: dict[str, Any] = {
        "requested_device": requested,
        "resolved_device": "",
        "device_name": "",
        "runtime": "torch" if requested != "cpu" else "host",
        "available": False,
        "tensor_probe_passed": False,
        "error": "",
    }
    if requested == "cpu":
        attestation.update(
            {
                "resolved_device": "cpu",
                "device_name": "CPU",
                "available": True,
                "tensor_probe_passed": True,
            }
        )
        return attestation
    try:
        import torch

        if requested.startswith("cuda:"):
            index = int(requested.split(":", 1)[1])
            if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
                raise RuntimeError(f"Requested CUDA runtime {requested} is not available")
            name = str(torch.cuda.get_device_name(index))
        else:
            raise ValueError(f"Unsupported runtime device identifier: {requested}")
        device = torch.device(requested)
        probe = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
        value = float((probe * probe).sum().detach().cpu().item())
        if abs(value - 14.0) > 1e-12:
            raise RuntimeError(f"Device tensor probe returned unexpected value {value!r}")
        if requested.startswith("cuda:"):
            torch.cuda.synchronize(device)
        attestation.update(
            {
                "resolved_device": str(device),
                "device_name": name,
                "available": True,
                "tensor_probe_passed": True,
            }
        )
    except (ImportError, RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
        attestation["error"] = f"{type(exc).__name__}: {exc}"
    return attestation


def result_device_attestation(config, problem, result) -> dict[str, Any]:
    """Build truthful planned-vs-actual device metadata for one completed optimizer run."""

    requested = str(getattr(config, "runtime_compute_device", "cpu"))
    runtime = runtime_device_attestation(requested)
    evaluator_device = str(getattr(problem, "device", "cpu"))
    execution_provenance = (
        dict(problem.execution_provenance())
        if callable(getattr(problem, "execution_provenance", None))
        else {}
    )
    actual_evaluator_device = str(
        execution_provenance.get("actual_computation_device", evaluator_device)
    )
    evaluator_context = getattr(problem, "device_context", None)
    evaluator_name = str(
        getattr(evaluator_context, "name", "CPU" if evaluator_device == "cpu" else "")
    )
    metadata = dict(getattr(result, "metadata", {}) or {})
    optimizer_device = str(metadata.get("optimizer_device", ""))
    if not optimizer_device:
        # CALO's cognitive candidate-generation/control plane remains NumPy/CPU while its evaluator
        # and configured policy can be accelerator-resident. Do not falsely label that CPU work as
        # an accelerator optimizer kernel. Legacy/reference optimizers are likewise reported as CPU.
        if str(getattr(result, "algorithm", "")) == "CALO":
            optimizer_device = "cpu_control_plane"
        elif str(getattr(config, "scientific_backend", "cpu_reference")) != "torch_fp64":
            optimizer_device = "cpu_legacy_optimizer"
        else:
            optimizer_device = requested
    policy_device = str(metadata.get("policy_inference_device", ""))
    resolution = dict(getattr(config, "runtime_device_resolution", {}) or {})
    return {
        "execution_contract_schema": resolution.get("schema_version", "unresolved"),
        "requested_mode": str(getattr(config, "execution_backend", "")),
        "execution_purpose": str(getattr(config, "execution_purpose", "exploratory")),
        "requested_device": str(getattr(config, "requested_compute_device", "auto")),
        "assigned_physical_device": str(getattr(config, "runtime_assigned_physical_device", "")),
        "assigned_logical_device": str(
            getattr(config, "runtime_assigned_logical_device", requested)
        ),
        "planned_device": requested,
        "runtime_probe": runtime,
        "actual_evaluator_device": actual_evaluator_device,
        "actual_evaluator_device_name": evaluator_name,
        "actual_optimizer_device": optimizer_device,
        "actual_policy_device": policy_device,
        "fallback_policy": str(getattr(config, "runtime_fallback_policy", "unresolved")),
        "fallback_reason": str(getattr(config, "runtime_fallback_reason", "")),
        "runtime_fallback": execution_provenance,
        "claim_eligibility": dict(resolution.get("claim_eligibility", {}) or {}),
        "binding_consistent": bool(
            runtime.get("tensor_probe_passed")
            and (
                actual_evaluator_device == requested
                or bool(execution_provenance.get("fallback_used", False))
            )
            and (optimizer_device in {requested, "cpu_control_plane", "cpu_legacy_optimizer"})
            and (not policy_device or policy_device == requested)
        ),
        "attestation_schema": 2,
    }


__all__ = [
    "bind_config_to_device",
    "resolve_config_for_entrypoint",
    "runtime_device_attestation",
    "result_device_attestation",
]
