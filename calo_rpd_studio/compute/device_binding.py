"""Canonical runtime-device binding and execution attestation.

All experiment execution paths (primary process, persistent CUDA worker, persistent XPU sidecar,
and one-shot XPU worker) must bind the same scientific evaluator, optimizer kernels, and CALO
policy inference settings to the same runtime identifier.  This module is the single authority for
that binding so a job labelled ``xpu:0`` or ``cuda:0`` cannot silently retain a CPU evaluator.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from calo_rpd_studio.continuation.runtime_binding import bind_exact_run_checkpoint


def bind_config_to_device(config, compute_device: str, item=None):
    """Return a deep-copied config fully bound to ``compute_device``.

    Runtime IDs are PyTorch IDs (``cuda:N``, ``xpu:N``, ``cpu``), not Windows Task Manager GPU
    numbers.  The binding is intentionally applied to every configured algorithm plus the canonical
    CALO/TLBO entries used by continuation/ablation paths.
    """

    device = str(compute_device or "cpu")
    local = deepcopy(config)
    local.runtime_compute_device = device
    parameters = dict(local.algorithm_parameters)
    names = set(getattr(local, "algorithms", ()) or ())
    names.update(parameters)
    names.update(("CALO", "TLBO"))
    for name in sorted(names):
        values = dict(parameters.get(name, {}))
        values["execution_device"] = device
        if str(getattr(local, "scientific_backend", "cpu_reference")) == "torch_fp64":
            values["optimizer_backend"] = "torch"
        if name == "CALO":
            values["inference_device"] = device
            values["policy_control_plane"] = "bound_to_assigned_runtime_v67"
        parameters[name] = values
    local.algorithm_parameters = parameters
    return bind_exact_run_checkpoint(local, item)


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
        elif requested.startswith("xpu:"):
            index = int(requested.split(":", 1)[1])
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError(f"Requested XPU runtime {requested} is not available")
            if index >= int(torch.xpu.device_count()):
                raise RuntimeError(f"Requested XPU runtime {requested} is outside the device count")
            props = torch.xpu.get_device_properties(index)
            name = str(getattr(props, "name", requested))
        else:
            raise ValueError(f"Unsupported runtime device identifier: {requested}")
        device = torch.device(requested)
        probe = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
        value = float((probe * probe).sum().detach().cpu().item())
        if abs(value - 14.0) > 1e-12:
            raise RuntimeError(f"Device tensor probe returned unexpected value {value!r}")
        if requested.startswith("cuda:"):
            torch.cuda.synchronize(device)
        elif requested.startswith("xpu:"):
            torch.xpu.synchronize(device)
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
    evaluator_context = getattr(problem, "device_context", None)
    evaluator_name = str(getattr(evaluator_context, "name", "CPU" if evaluator_device == "cpu" else ""))
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
    return {
        "planned_device": requested,
        "runtime_probe": runtime,
        "actual_evaluator_device": evaluator_device,
        "actual_evaluator_device_name": evaluator_name,
        "actual_optimizer_device": optimizer_device,
        "actual_policy_device": policy_device,
        "binding_consistent": bool(
            runtime.get("tensor_probe_passed")
            and evaluator_device == requested
            and (optimizer_device in {requested, "cpu_control_plane", "cpu_legacy_optimizer"})
            and (not policy_device or policy_device == requested)
        ),
        "attestation_schema": 1,
    }


__all__ = ["bind_config_to_device", "runtime_device_attestation", "result_device_attestation"]
