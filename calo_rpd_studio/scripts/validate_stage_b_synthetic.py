"""Validate v6.4 Stage-B synthetic curriculum parity and microbatch throughput.

This is a target-machine qualification utility. It does not mark a CUDA/XPU device scientifically
qualified merely because it is visible; every generated task is checked against the NumPy reference
before throughput is reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from calo_rpd_studio.algorithms.calo.training import CurriculumProblem
from calo_rpd_studio.algorithms.calo.device_resident_synthetic import (
    DeviceResidentCurriculumProblem,
    SyntheticCrossEpisodeBatchBroker,
)


def _resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", default="stage_b_synthetic_validation.json")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    episodes = max(1, int(args.episodes))
    population_size = max(4, int(args.population))
    repetitions = max(1, int(args.repetitions))

    problems = []
    populations = []
    for episode in range(episodes):
        rng = np.random.default_rng(2026 + 10007 * episode)
        reference = CurriculumProblem(rng, episode % 4)
        populations.append(rng.random((population_size, reference.dimension)))
        problems.append(reference)

    reference_start = time.perf_counter()
    for _ in range(repetitions):
        for problem, population in zip(problems, populations):
            [problem.evaluate(candidate) for candidate in population]
    reference_elapsed = max(time.perf_counter() - reference_start, 1e-12)

    with SyntheticCrossEpisodeBatchBroker(
        device=device,
        batch_window_ms=2.0,
        max_candidates=max(4096, episodes * population_size),
    ) as broker:
        wrapped = [
            DeviceResidentCurriculumProblem(
                problem,
                device=device,
                broker=broker,
                require_startup_parity=True,
            )
            for problem in problems
        ]
        # First pass is the fail-closed parity qualification pass.
        with ThreadPoolExecutor(max_workers=episodes) as executor:
            first = [
                executor.submit(wrapper.evaluate_population, population)
                for wrapper, population in zip(wrapped, populations)
            ]
            [future.result() for future in first]
        if not all(wrapper.parity_verified for wrapper in wrapped):
            raise RuntimeError("Not every Stage-B synthetic task passed startup parity")

        start = time.perf_counter()
        for _ in range(repetitions):
            with ThreadPoolExecutor(max_workers=episodes) as executor:
                futures = [
                    executor.submit(wrapper.evaluate_population, population)
                    for wrapper, population in zip(wrapped, populations)
                ]
                [future.result() for future in futures]
        elapsed = max(time.perf_counter() - start, 1e-12)
        candidates = repetitions * episodes * population_size
        metrics = broker.metrics()

    payload = {
        "schema": "calo-stage-b-synthetic-validation-v1",
        "device": device,
        "device_type": torch.device(device).type,
        "physical_accelerator": torch.device(device).type in {"cuda", "xpu"},
        "episodes": episodes,
        "population_size": population_size,
        "repetitions": repetitions,
        "candidates": candidates,
        "elapsed_seconds": elapsed,
        "candidates_per_second": candidates / elapsed,
        "numpy_reference_elapsed_seconds": reference_elapsed,
        "numpy_reference_candidates_per_second": candidates / reference_elapsed,
        "throughput_ratio_vs_numpy_reference": reference_elapsed / elapsed,
        "parity_verified": True,
        "maximum_startup_parity_error": max(wrapper.parity_max_error for wrapper in wrapped),
        "broker": metrics,
        "qualification_note": (
            "Physical accelerator Stage-B qualification requires this command to run on the intended "
            "CUDA/XPU device under the target software stack. CPU execution validates logic only."
        ),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
