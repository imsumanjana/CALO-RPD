"""Configuration-level CPU/accelerator parity audit."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from calo_rpd_studio.experiments.experiment_runner import build_scenarios
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader

from .torch_orpd import AcceleratedORPDProblem, parity_check


def run_configuration_parity_audit(config, *, device: str = "auto", candidates: int = 5):
    """Compare deterministic candidate evaluations against the trusted CPU reference.

    The candidate set is generated from a fixed seed derived from the experiment master seed, so
    repeated audits are reproducible.  The same objective, mixed-variable decoder, scenarios,
    robust aggregation, and power-flow settings are used on both backends.
    """

    case = CaseLoader.load(config.case_name)
    problem_config = ORPDProblemConfig(config.objective, config.variables, config.robust_objective)
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
    rng = np.random.default_rng(int(config.master_seed) + 4711)
    sample = rng.random((max(1, int(candidates)), reference.dimension))
    report = parity_check(
        reference,
        accelerated,
        sample,
        objective_tolerance=float(getattr(config, "parity_objective_tolerance", 1e-5)),
        violation_tolerance=float(getattr(config, "parity_violation_tolerance", 1e-6)),
        voltage_tolerance=float(getattr(config, "parity_voltage_tolerance", 1e-5)),
    )
    payload = asdict(report)
    payload.update(
        {
            "case": config.case_name,
            "device": accelerated.device,
            "device_name": accelerated.device_context.name,
            "dtype": "float64",
            "scenario_count": len(scenarios),
            "tolerances": {
                "objective": float(getattr(config, "parity_objective_tolerance", 1e-5)),
                "violation": float(getattr(config, "parity_violation_tolerance", 1e-6)),
                "voltage_pu": float(getattr(config, "parity_voltage_tolerance", 1e-5)),
            },
        }
    )
    return payload
