"""Retain bounded physical CUDA pressure, staging, fallback, lease, and recovery evidence."""

from __future__ import annotations

import argparse
import gc
import multiprocessing
from pathlib import Path
import queue
import tempfile
import time

import numpy as np

from calo_rpd_studio.accelerated.device import resolve_device
from calo_rpd_studio.accelerated.vram_residency import (
    CudaCapacityExhausted,
    VramResidencyGovernor,
    VramResidencyPolicy,
)
from calo_rpd_studio.compute.device_lease import DeviceLeaseUnavailable, ExclusiveDeviceLease
from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import build_problem
from calo_rpd_studio.scripts.validate_accelerator import (
    _git_source_identity,
    _runtime_snapshot,
    _utc_now,
    _write_new_evidence,
)


_MIB = 1024**2


def _bounded_pressure_bytes(free_bytes: int, fraction: float, maximum_mib: int) -> int:
    """Return a deliberately conservative physical-allocation probe size."""
    free = int(free_bytes)
    share = float(fraction)
    maximum = int(maximum_mib)
    if free <= 0:
        raise ValueError("CUDA free bytes must be positive")
    if not 0.0 < share <= 0.25:
        raise ValueError("pressure fraction must be in (0, 0.25]")
    if not 16 <= maximum <= 512:
        raise ValueError("maximum pressure allocation must be between 16 and 512 MiB")
    requested = min(int(free * share), maximum * _MIB)
    if requested < 16 * _MIB:
        raise RuntimeError("Insufficient free CUDA memory for the bounded 16 MiB pressure probe")
    return requested


def _recovery_within_tolerance(
    free_bytes_before: int, free_bytes_after: int, tolerance_mib: int
) -> bool:
    tolerance = int(tolerance_mib)
    if tolerance < 0:
        raise ValueError("recovery tolerance cannot be negative")
    return int(free_bytes_after) >= int(free_bytes_before) - tolerance * _MIB


def _pressure_probe(
    device: str, *, fraction: float, maximum_mib: int, recovery_tolerance_mib: int
) -> dict:
    import torch

    target = torch.device(device)
    free_before, total_bytes = torch.cuda.mem_get_info(target)
    requested = _bounded_pressure_bytes(free_before, fraction, maximum_mib)
    admission_before = calculate_available_memory_admission(
        total_bytes=int(total_bytes),
        available_bytes=int(free_before),
        requested_fraction=0.80,
    )
    pressure = None
    try:
        pressure = torch.empty(requested, dtype=torch.uint8, device=target)
        pressure.fill_(1)
        torch.cuda.synchronize(target)
        free_during, total_during = torch.cuda.mem_get_info(target)
        admission_during = calculate_available_memory_admission(
            total_bytes=int(total_during),
            available_bytes=int(free_during),
            requested_fraction=0.80,
        )
    finally:
        del pressure
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(target)
    time.sleep(0.1)
    free_after, total_after = torch.cuda.mem_get_info(target)
    observed = max(0, int(free_before) - int(free_during))
    pressure_observed = observed >= int(requested * 0.90)
    recovery_ok = _recovery_within_tolerance(free_before, free_after, int(recovery_tolerance_mib))
    admission_contract_ok = (
        admission_before.requested_available_fraction == 0.80
        and admission_during.requested_available_fraction == 0.80
        and admission_during.additional_allowance_bytes
        <= admission_before.additional_allowance_bytes
        and int(total_bytes) == int(total_during) == int(total_after)
    )
    return {
        "requested_allocation_bytes": int(requested),
        "observed_free_memory_reduction_bytes": observed,
        "free_bytes_before": int(free_before),
        "free_bytes_during": int(free_during),
        "free_bytes_after": int(free_after),
        "recovery_tolerance_bytes": int(recovery_tolerance_mib) * _MIB,
        "pressure_observed": bool(pressure_observed),
        "recovery_within_tolerance": bool(recovery_ok),
        "admission_before": admission_before.to_dict(),
        "admission_during": admission_during.to_dict(),
        "admission_contract_passed": bool(admission_contract_ok),
        "passed": bool(pressure_observed and recovery_ok and admission_contract_ok),
    }


def _build_cuda_problem(case_name: str, device: str, seed: int, batch_size: int):
    config = ExperimentConfig(case_name=case_name, master_seed=int(seed))
    config.scientific_backend = "torch_fp64"
    config.runtime_compute_device = str(device)
    config.tensor_batch_size = int(batch_size)
    config.cuda_cpu_fallback_enabled = True
    config.validate()
    return build_problem(config, int(seed) + 99173)


def _close_problem(problem) -> None:
    evaluator = getattr(problem, "_device_resident_evaluator", None)
    governor = getattr(evaluator, "vram_governor", None)
    if governor is not None:
        governor.close()


def _staged_host_probe(
    device: str, *, case_name: str, seed: int, candidates: int, batch_size: int
) -> dict:
    problem = _build_cuda_problem(case_name, device, seed, batch_size)
    try:
        if getattr(problem, "_device_resident_evaluator", None) is None:
            raise RuntimeError("Development case did not select the device-resident CUDA evaluator")
        population = np.random.default_rng(int(seed)).random((int(candidates), problem.dimension))
        batch = problem.evaluate_population_tensor(population)
        residency = dict(batch.metadata.get("vram_residency", {}))
        passed = (
            str(batch.objective.device).startswith("cuda")
            and residency.get("execution_state") == "cuda_staged_host"
            and residency.get("input_staged_from_host") is True
            and residency.get("cpu_inner_loop_participation") is False
            and int(residency.get("request_oom_retries", -1)) == 0
            and int(batch.count) == int(candidates)
        )
        return {
            "case": str(case_name),
            "candidate_count": int(batch.count),
            "output_device": str(batch.objective.device),
            "residency": residency,
            "passed": bool(passed),
        }
    finally:
        _close_problem(problem)


def _controlled_oom_backoff_probe(device: str) -> dict:
    """Exercise the real CUDA governor with a clearly labelled injected OOM boundary."""
    import torch

    governor = VramResidencyGovernor(
        device,
        VramResidencyPolicy(
            budget_fraction=0.80,
            oom_retry_count=4,
            minimum_microbatch=1,
        ),
    )
    calls: list[int] = []
    population = torch.arange(15, dtype=torch.float64).reshape(5, 3)

    def evaluate_once(chunk):
        calls.append(int(chunk.shape[0]))
        if int(chunk.shape[0]) > 2:
            raise torch.OutOfMemoryError("controlled G2 validation fault")
        return chunk.to(device=device).sum(dim=1)

    try:
        values, metadata = governor.run_microbatched(
            population,
            evaluate_once,
            lambda chunks, meta: (torch.cat(chunks, dim=0), meta),
            preferred_microbatch=5,
        )
        expected = population.sum(dim=1)
        numerically_complete = bool(
            torch.equal(values.detach().to("cpu"), expected.detach().to("cpu"))
        )
        passed = (
            calls[:2] == [5, 2]
            and int(metadata.get("request_oom_retries", 0)) == 1
            and int(metadata.get("cpu_fallbacks", -1)) == 0
            and metadata.get("cpu_inner_loop_participation") is False
            and numerically_complete
        )
        return {
            "fault_injection": "torch.OutOfMemoryError raised before allocations for chunks > 2",
            "natural_hardware_oom_claimed": False,
            "attempted_microbatches": calls,
            "numerically_complete": numerically_complete,
            "residency": metadata,
            "passed": bool(passed),
        }
    finally:
        governor.close()


def _controlled_cpu_restart_and_cuda_recovery_probe(
    device: str, *, case_name: str, seed: int, batch_size: int
) -> dict:
    problem = _build_cuda_problem(case_name, device, seed, batch_size)
    evaluator = getattr(problem, "_device_resident_evaluator", None)
    if evaluator is None:
        _close_problem(problem)
        raise RuntimeError("Development case did not select the device-resident CUDA evaluator")
    population = np.random.default_rng(int(seed) + 1).random((2, problem.dimension))
    original = evaluator.evaluate_tensor

    def controlled_exhaustion(_population):
        raise CudaCapacityExhausted(
            "controlled G2 capacity exhaustion",
            {
                "execution_state": "cuda_capacity_exhausted",
                "last_fallback_reason": "controlled_validation_fault",
                "injected_for_validation": True,
            },
        )

    try:
        evaluator.evaluate_tensor = controlled_exhaustion
        fallback_results = problem.evaluate_population(population)
        fallback_metadata = [dict(item.metadata or {}) for item in fallback_results]
        fallback_passed = all(
            row.get("scientific_backend") == "cpu_reference_after_cuda_capacity_exhaustion"
            and row.get("compute_fallback") == "cpu_reference_full_request_restart"
            and row.get("runtime_timing_comparable_to_cuda_only") is False
            and row.get("cuda_capacity_exhaustion", {}).get("injected_for_validation") is True
            for row in fallback_metadata
        )
        evaluator.evaluate_tensor = original
        recovered = problem.evaluate_population_tensor(population)
        recovery_residency = dict(recovered.metadata.get("vram_residency", {}))
        recovery_passed = (
            str(recovered.objective.device).startswith("cuda")
            and int(recovered.count) == len(population)
            and recovery_residency.get("cpu_inner_loop_participation") is False
        )
        return {
            "fault_injection": "typed CudaCapacityExhausted before candidate evaluation",
            "natural_hardware_oom_claimed": False,
            "fallback_candidate_count": len(fallback_results),
            "fallback_metadata": fallback_metadata,
            "fallback_passed": bool(fallback_passed),
            "cuda_recovery_residency": recovery_residency,
            "cuda_recovery_passed": bool(recovery_passed),
            "passed": bool(fallback_passed and recovery_passed),
        }
    finally:
        evaluator.evaluate_tensor = original
        _close_problem(problem)


def _lease_attempt(device: str, root: str, result_queue) -> None:
    try:
        lease = ExclusiveDeviceLease(device, root=root)
    except DeviceLeaseUnavailable:
        result_queue.put("busy")
    except BaseException as exc:
        result_queue.put(f"error:{type(exc).__name__}:{exc}")
    else:
        result_queue.put("acquired")
        lease.close()


def _child_lease_result(context, device: str, root: str) -> dict:
    result_queue = context.Queue()
    process = context.Process(target=_lease_attempt, args=(device, root, result_queue))
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return {"status": "timeout", "exitcode": process.exitcode}
    try:
        status = result_queue.get(timeout=3)
    except queue.Empty:
        status = "missing_result"
    return {"status": str(status), "exitcode": process.exitcode}


def _cross_process_lease_probe(device: str) -> dict:
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="calo-rpd-g2-lease-") as root:
        owner = ExclusiveDeviceLease(device, root=root)
        try:
            while_owned = _child_lease_result(context, device, root)
        finally:
            owner.close()
        after_release = _child_lease_result(context, device, root)
    passed = while_owned == {"status": "busy", "exitcode": 0} and after_release == {
        "status": "acquired",
        "exitcode": 0,
    }
    return {
        "device": str(device),
        "contender_while_owned": while_owned,
        "contender_after_release": after_release,
        "passed": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", choices=["case30", "case57"], default="case30")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--pressure-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-pressure-mib", type=int, default=256)
    parser.add_argument("--recovery-tolerance-mib", type=int, default=64)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-physical-cuda", action="store_true")
    args = parser.parse_args()

    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite accelerator evidence: {destination}")
    if int(args.candidates) < 1 or int(args.batch_size) < 1:
        raise ValueError("candidates and batch size must be positive")
    source_commit, tracked_dirty = _git_source_identity()
    if tracked_dirty:
        raise RuntimeError("Durable resource evidence requires a clean tracked source tree")
    selected = str(
        resolve_device(args.device, require_accelerator=args.require_physical_cuda).resolved
    )
    import torch

    physical_cuda = selected.startswith("cuda") and bool(torch.cuda.is_available())
    if args.require_physical_cuda and not physical_cuda:
        raise RuntimeError("Physical NVIDIA CUDA was required but not selected")

    started_at = _utc_now()
    runtime_before = _runtime_snapshot(selected)
    evidence = {
        "schema_version": "calo-rpd-physical-resource-recovery-v1",
        "run_id": str(args.run_id).strip(),
        "started_at": started_at,
        "source_commit": source_commit,
        "tracked_source_clean": not tracked_dirty,
        "parameters": {
            "device": str(args.device),
            "case": str(args.case),
            "candidates": int(args.candidates),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "pressure_fraction": float(args.pressure_fraction),
            "maximum_pressure_mib": int(args.maximum_pressure_mib),
            "recovery_tolerance_mib": int(args.recovery_tolerance_mib),
            "require_physical_cuda": bool(args.require_physical_cuda),
        },
        "selected_device": selected,
        "physical_cuda_exercised": physical_cuda,
        "runtime_before": runtime_before,
        "protected_cases_opened": False,
        "controlled_faults_are_not_natural_oom_evidence": True,
    }
    exit_code = 1
    try:
        evidence["pressure"] = _pressure_probe(
            selected,
            fraction=float(args.pressure_fraction),
            maximum_mib=int(args.maximum_pressure_mib),
            recovery_tolerance_mib=int(args.recovery_tolerance_mib),
        )
        evidence["staged_host"] = _staged_host_probe(
            selected,
            case_name=str(args.case),
            seed=int(args.seed),
            candidates=int(args.candidates),
            batch_size=int(args.batch_size),
        )
        evidence["oom_backoff"] = _controlled_oom_backoff_probe(selected)
        evidence["cpu_restart_and_cuda_recovery"] = _controlled_cpu_restart_and_cuda_recovery_probe(
            selected,
            case_name=str(args.case),
            seed=int(args.seed),
            batch_size=int(args.batch_size),
        )
        evidence["cross_process_lease"] = _cross_process_lease_probe(selected)
        qualification_passed = physical_cuda and all(
            bool(evidence[name].get("passed"))
            for name in (
                "pressure",
                "staged_host",
                "oom_backoff",
                "cpu_restart_and_cuda_recovery",
                "cross_process_lease",
            )
        )
        evidence["qualification_passed"] = bool(qualification_passed)
        evidence["claim_scope"] = (
            "bounded physical VRAM pressure/recovery, host-staged CUDA execution, and cross-process "
            "device-lease evidence on this source/device; OOM and CPU-restart boundaries used "
            "explicit controlled fault injection"
            if qualification_passed
            else "no physical resource-recovery qualification claim"
        )
        exit_code = 0 if qualification_passed else 1
    except Exception as exc:
        evidence["qualification_passed"] = False
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        evidence["claim_scope"] = "no physical resource-recovery qualification claim"
    evidence["runtime_after"] = _runtime_snapshot(selected)
    evidence["completed_at"] = _utc_now()
    written = _write_new_evidence(destination, evidence)
    print(f"evidence_path={written}")
    print(f"qualification_passed={str(bool(evidence['qualification_passed'])).lower()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
