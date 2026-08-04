"""Run a reproducible CPU/accelerator parity audit from the command line."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

from calo_rpd_studio.accelerated.device import resolve_device
from calo_rpd_studio.accelerated.parity_audit import run_configuration_parity_audit
from calo_rpd_studio.ai.model_io import durable_write_bytes
from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_source_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("Git did not return a full source commit")
    return commit.lower(), bool(tracked)


def _runtime_snapshot(device: str) -> dict:
    import psutil
    import torch

    memory = psutil.virtual_memory()
    cpu_admission = calculate_available_memory_admission(
        total_bytes=int(memory.total),
        available_bytes=int(memory.available),
        requested_fraction=0.80,
    )
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda or ""),
        "cpu_memory_admission": cpu_admission.to_dict(),
    }
    selected = str(device)
    if selected.startswith("cuda"):
        target = torch.device(selected)
        free_bytes, total_bytes = torch.cuda.mem_get_info(target)
        allocated = int(torch.cuda.memory_allocated(target))
        reserved = int(torch.cuda.memory_reserved(target))
        cuda_admission = calculate_available_memory_admission(
            total_bytes=int(total_bytes),
            available_bytes=int(free_bytes),
            requested_fraction=0.80,
            baseline_reserved_bytes=reserved,
        )
        properties = torch.cuda.get_device_properties(target)
        payload["cuda"] = {
            "device": str(target),
            "name": str(torch.cuda.get_device_name(target)),
            "device_count": int(torch.cuda.device_count()),
            "total_bytes": int(total_bytes),
            "free_bytes_at_sample": int(free_bytes),
            "process_allocated_bytes": allocated,
            "process_reserved_bytes": reserved,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "hardware_uuid": str(getattr(properties, "uuid", "") or ""),
            "pci_bus_id": str(getattr(properties, "pci_bus_id", "") or ""),
            "admission": cuda_admission.to_dict(),
        }
    return payload


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_new_evidence(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite accelerator evidence: {destination}")
    encoded = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    durable_write_bytes(destination, encoded)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="case30", choices=["case30", "case57", "case118", "case300"]
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", help="new durable JSON evidence path; existing paths are refused"
    )
    parser.add_argument("--run-id", default="", help="stable qualification-run identity")
    parser.add_argument(
        "--require-physical-cuda",
        action="store_true",
        help="fail unless the resolved computation device is physical NVIDIA CUDA",
    )
    args = parser.parse_args()

    config = ExperimentConfig(case_name=args.case, master_seed=args.seed)
    config.tensor_batch_size = args.batch_size
    started_at = _utc_now()
    source_commit = ""
    if args.output:
        if not args.run_id.strip():
            raise ValueError("--run-id is required when --output is used")
        source_commit, tracked_dirty = _git_source_identity()
        if tracked_dirty:
            raise RuntimeError("Durable accelerator evidence requires a clean tracked source tree")
    else:
        tracked_dirty = False
    import torch

    device_context = resolve_device(args.device, require_accelerator=args.require_physical_cuda)
    resolved_before = str(device_context.resolved)
    runtime_before = _runtime_snapshot(resolved_before)
    if resolved_before.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(resolved_before))
    report = run_configuration_parity_audit(config, device=args.device, candidates=args.candidates)
    resolved = str(report.get("device", ""))
    if resolved != resolved_before:
        raise RuntimeError(
            f"Device changed across the parity boundary: {resolved_before!r} -> {resolved!r}"
        )
    physical_cuda = resolved.startswith("cuda") and bool(torch.cuda.is_available())
    qualification_passed = bool(report.get("passed")) and (
        physical_cuda or not args.require_physical_cuda
    )
    report["physical_cuda_exercised"] = physical_cuda
    report["physical_cuda_required"] = bool(args.require_physical_cuda)
    report["qualification_passed"] = qualification_passed
    if args.output:
        runtime_after = _runtime_snapshot(resolved)
        runtime = {"before": runtime_before, "after": runtime_after}
        if physical_cuda:
            target = torch.device(resolved)
            peak_allocated = int(torch.cuda.max_memory_allocated(target))
            peak_reserved = int(torch.cuda.max_memory_reserved(target))
            before_cuda = runtime_before["cuda"]
            allowance = int(before_cuda["admission"]["additional_allowance_bytes"])
            additional_allocated = max(
                0, peak_allocated - int(before_cuda["process_allocated_bytes"])
            )
            additional_reserved = max(0, peak_reserved - int(before_cuda["process_reserved_bytes"]))
            runtime["cuda_peak"] = {
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "additional_allocated_bytes": additional_allocated,
                "additional_reserved_bytes": additional_reserved,
                "admission_allowance_bytes": allowance,
                "additional_allocated_within_admission": additional_allocated <= allowance,
                "additional_reserved_within_admission": additional_reserved <= allowance,
            }
        evidence = {
            "schema_version": "calo-rpd-physical-accelerator-parity-v1",
            "run_id": args.run_id.strip(),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "source_commit": source_commit,
            "tracked_source_clean": not tracked_dirty,
            "parameters": {
                "case": args.case,
                "requested_device": args.device,
                "random_candidates": int(args.candidates),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "require_physical_cuda": bool(args.require_physical_cuda),
            },
            "runtime": runtime,
            "parity": report,
            "qualification_passed": qualification_passed,
            "protected_cases_opened": False,
            "claim_scope": (
                "physical FP64 CPU/CUDA parity for this development case, candidate battery, "
                "source commit, device, and software stack only"
                if qualification_passed and physical_cuda
                else "no physical CPU/CUDA parity claim"
            ),
        }
        destination = _write_new_evidence(args.output, evidence)
        print(f"evidence_path={destination}")
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualification_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
