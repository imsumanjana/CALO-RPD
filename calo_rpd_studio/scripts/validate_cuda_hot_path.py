"""Measure a CUDA-resident ORPD numerical window without opening protected cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from calo_rpd_studio.accelerated.cuda_timing import (
    CUDA_NUMERICAL_TIME_SHARE_TARGET,
    POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY,
    measure_cuda_window,
)
from calo_rpd_studio.ai.model_io import durable_write_bytes
from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import build_problem


DEVELOPMENT_CASES = ("case30", "case57")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new_json(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite CUDA hot-path evidence: {destination}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    durable_write_bytes(destination, encoded)
    return destination


def _build_cuda_problem(case_name: str, seed: int, batch_size: int):
    config = ExperimentConfig(case_name=case_name, master_seed=seed)
    config.scientific_backend = "torch_fp64"
    config.runtime_compute_device = "cuda"
    config.device_resident_execution = True
    config.cuda_resident_hot_loop = True
    config.cuda_cpu_fallback_enabled = False
    config.tensor_batch_size = int(batch_size)
    config.validate()
    return config, build_problem(config, int(seed) + 99173)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=DEVELOPMENT_CASES, default="case30")
    parser.add_argument("--batch-size", type=int, default=POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size < POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY:
        raise ValueError("CUDA qualification requires at least 100 evaluations per host boundary")
    if args.batches < 1 or args.warmup_batches < 0:
        raise ValueError("Batch count must be positive and warmup count must be non-negative")
    if not args.run_id.strip():
        raise ValueError("CUDA hot-path evidence requires a non-empty run ID")

    source = resolve_source_identity(require_durable=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Physical CUDA is required for CUDA hot-path qualification")
    started_at = _utc_now()
    target = torch.device("cuda:0")
    target_index = 0
    config, problem = _build_cuda_problem(args.case, args.seed, args.batch_size)
    if str(problem.device) != str(target):
        raise RuntimeError(f"CUDA problem resolved to unexpected device {problem.device!r}")
    # Problem construction creates the CUDA context and resident scientific tensors. Some PyTorch
    # builds reject allocator-stat operations before that first context initialization.
    torch.cuda.reset_peak_memory_stats(target_index)
    generator = torch.Generator(device=target).manual_seed(int(args.seed))
    population = torch.rand(
        (int(args.batch_size), int(problem.dimension)),
        dtype=torch.float64,
        device=target,
        generator=generator,
    )

    last_batch = None
    with torch.no_grad():
        for _ in range(int(args.warmup_batches)):
            last_batch = problem.evaluate_population_tensor(population)
        torch.cuda.synchronize(target_index)

        def _run_window():
            result = None
            for _ in range(int(args.batches)):
                result = problem.evaluate_population_tensor(population)
            return result

        last_batch, timing = measure_cuda_window(
            _run_window,
            device=str(target),
            label="device_resident_orpd",
        )
    if last_batch is None:
        raise RuntimeError("CUDA hot-path window returned no device-resident result")
    residency = dict(last_batch.metadata.get("vram_residency", {}))
    tensors_on_cuda = all(
        tensor.device.type == "cuda"
        for tensor in (
            last_batch.objective,
            last_batch.violation,
            last_batch.feasible,
            last_batch.normalized_values,
            last_batch.decoded_values,
            last_batch.scenario_values,
        )
    )
    full_request_admitted = bool(residency.get("full_request_residency_admitted", False))
    no_inner_transfers = int(residency.get("cpu_cuda_inner_loop_transfers", -1)) == 0
    no_fallback = int(residency.get("cpu_fallbacks", -1)) == 0
    target_boundary = (
        int(residency.get("target_evaluations_per_host_boundary", 0))
        == POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY
    )
    qualification_passed = bool(
        timing.target_met
        and tensors_on_cuda
        and full_request_admitted
        and no_inner_transfers
        and no_fallback
        and target_boundary
    )
    evidence = {
        "schema_version": "calo-rpd-cuda-hot-path-v1",
        "run_id": args.run_id.strip(),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source_commit": source.source_commit,
        "source_identity_kind": source.source_identity_kind,
        "tracked_source_clean": source.tracked_source_clean,
        "parameters": {
            "case": args.case,
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "measured_batches": int(args.batches),
            "warmup_batches": int(args.warmup_batches),
            "measured_candidate_evaluations": int(args.batch_size) * int(args.batches),
        },
        "runtime": {
            "torch": str(torch.__version__),
            "torch_cuda_runtime": str(torch.version.cuda or ""),
            "device": str(target),
            "device_name": str(torch.cuda.get_device_name(target_index)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_index)),
        },
        "timing": timing.to_dict(),
        "residency_checks": {
            "result_tensors_on_cuda": tensors_on_cuda,
            "full_request_residency_admitted": full_request_admitted,
            "cpu_cuda_inner_loop_transfers": int(
                residency.get("cpu_cuda_inner_loop_transfers", -1)
            ),
            "cpu_fallbacks": int(residency.get("cpu_fallbacks", -1)),
            "target_evaluations_per_host_boundary": int(
                residency.get("target_evaluations_per_host_boundary", 0)
            ),
        },
        "qualification_passed": qualification_passed,
        "protected_cases_opened": False,
        "claim_scope": (
            "more than or equal to 95% CUDA event-time share for this steady-state, "
            "accelerator-eligible ORPD numerical window only"
            if qualification_passed
            else "no CUDA numerical-time-share qualification claim"
        ),
        "excluded_from_metric": [
            "startup",
            "UI",
            "database",
            "filesystem I/O",
            "orchestration",
            "final serialization",
        ],
        "target_cuda_time_share": CUDA_NUMERICAL_TIME_SHARE_TARGET,
        "scientific_configuration": config.to_dict(),
    }
    destination = _write_new_json(args.output, evidence)
    print(f"evidence_path={destination}")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualification_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
