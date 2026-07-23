"""Configuration-level CPU/accelerator parity audit with adversarial candidate coverage."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import numpy as np

from calo_rpd_studio.experiments.experiment_runner import build_scenarios
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
from calo_rpd_studio.power_system.case_loader import CaseLoader

from .torch_orpd import AcceleratedORPDProblem, parity_check


def _parity_candidate_battery(reference: ORPDProblem, *, seed: int, random_candidates: int) -> np.ndarray:
    """Return deterministic bounds/center/corner probes plus seeded random samples.

    This is intentionally stronger than a handful of ordinary random points.  Mixed-variable
    decode corners and near-bound probes exercise clipping, discrete controls, voltage/tap/shunt
    extremes and commonly trigger different PF/Q-limit regimes.
    """
    n = int(reference.dimension)
    probes: list[np.ndarray] = [
        np.full(n, 0.5),
        np.zeros(n),
        np.ones(n),
        np.full(n, 1e-9),
        np.full(n, 1.0 - 1e-9),
    ]
    if n:
        alternating_a = np.asarray([(i % 2) for i in range(n)], dtype=float)
        alternating_b = 1.0 - alternating_a
        probes.extend([alternating_a, alternating_b])
        for index in range(min(n, 4)):
            low = np.full(n, 0.5); low[index] = 0.0
            high = np.full(n, 0.5); high[index] = 1.0
            probes.extend([low, high])
    rng = np.random.default_rng(int(seed))
    for _ in range(max(1, int(random_candidates))):
        probes.append(rng.random(n))
    # Preserve order while removing exact duplicates.
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for row in probes:
        key = np.ascontiguousarray(row, dtype=np.float64).tobytes()
        if key not in seen:
            seen.add(key); unique.append(row)
    return np.asarray(unique, dtype=float)


def _scientific_fingerprint(reference: ORPDProblem) -> str:
    # Reuse the same canonical scientific identity used by continuation/history compatibility so
    # callable scenario transforms, objective/control semantics, PF options and tolerance schema are
    # all part of the parity certificate rather than only scenario names.
    return scientific_problem_fingerprint(reference)


def run_configuration_parity_audit(config, *, device: str = "auto", candidates: int = 12):
    """Compare the exact configured scientific problem against the accelerator backend."""
    config.validate()
    case = CaseLoader.load(config.case_name)
    problem_config = ORPDProblemConfig(
        objective=config.objective,
        variables=config.variables,
        robust=config.robust_objective,
        power_flow=config.power_flow,
        constraint_tolerances=config.constraint_tolerances,
    )
    scenarios = build_scenarios(config, int(config.master_seed) + 99173, case)
    reference = ORPDProblem(case, problem_config, scenarios)
    accelerated = AcceleratedORPDProblem(
        case,
        problem_config,
        scenarios,
        device=device,
        dtype_name="float64",
        batch_size=int(getattr(config, "tensor_batch_size", 64)),
    )
    sample = _parity_candidate_battery(
        reference, seed=int(config.master_seed) + 4711, random_candidates=max(1, int(candidates))
    )
    report = parity_check(
        reference,
        accelerated,
        sample,
        objective_tolerance=float(getattr(config, "parity_objective_tolerance", 1e-5)),
        violation_tolerance=float(getattr(config, "parity_violation_tolerance", 1e-6)),
        voltage_tolerance=float(getattr(config, "parity_voltage_tolerance", 1e-5)),
        angle_tolerance_deg=float(getattr(config, "parity_angle_tolerance_deg", 1e-4)),
    )
    payload = asdict(report)
    payload.update(
        {
            "case": config.case_name,
            "device": accelerated.device,
            "device_name": accelerated.device_context.name,
            "dtype": "float64",
            "scenario_count": len(scenarios),
            "candidate_battery_count": int(len(sample)),
            "scientific_configuration_sha256": _scientific_fingerprint(reference),
            "scientific_problem_fingerprint": scientific_problem_fingerprint(reference),
            "power_flow_options": asdict(config.power_flow),
            "constraint_tolerances": asdict(config.constraint_tolerances),
            "tolerances": {
                "objective": float(getattr(config, "parity_objective_tolerance", 1e-5)),
                "violation": float(getattr(config, "parity_violation_tolerance", 1e-6)),
                "voltage_pu": float(getattr(config, "parity_voltage_tolerance", 1e-5)),
                "angle_deg": float(getattr(config, "parity_angle_tolerance_deg", 1e-4)),
            },
        }
    )
    return payload
