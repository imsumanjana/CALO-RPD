"""Independent best-solution reconstruction and constraint audit."""

from __future__ import annotations
import json
import numpy as np
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import build_problem
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.orpd.objectives import calculate_objective
from calo_rpd_studio.orpd.constraints import evaluate_constraints


def validate_stored_run(database, run_id, rtol=1e-8, atol=1e-8):
    row = database.get_run(run_id)
    if row is None:
        raise KeyError(run_id)
    exp = database.get_experiment(row["experiment_id"])
    config = ExperimentConfig.from_dict(json.loads(exp["config_json"]))
    stored = json.loads(row["result_json"])
    seeds = json.loads(row["seed_json"])
    problem = build_problem(config, int(seeds["scenario_seed"]))
    z = np.asarray(stored["best_vector"], float)
    reported = float(stored["best_objective"])
    recomputed = problem.evaluate(z)
    absolute = abs(recomputed.value - reported)
    relative = absolute / max(abs(reported), 1e-15)
    controlled, _ = problem.decoder.decode(z)
    audits = []
    for scenario in problem.scenarios:
        pf = run_ac_power_flow(scenario.apply(controlled), problem.config.power_flow)
        obj = calculate_objective(pf, problem.config.objective)
        con = evaluate_constraints(pf)
        audits.append(
            {
                "scenario": scenario.name,
                "converged": pf.converged,
                "objective": obj.value,
                "constraint_violation": con.total,
                "voltage_limits": con.components.get("bus_voltage", float("inf")) <= 1e-12,
                "generator_q_limits": con.components.get("generator_q", float("inf")) <= 1e-12,
                "generator_p_limits": con.components.get("generator_p", float("inf")) <= 1e-12,
                "branch_limits": con.components.get("branch_thermal", float("inf")) <= 1e-12,
            }
        )
    passed = bool(
        np.isfinite(recomputed.value)
        and np.isclose(recomputed.value, reported, rtol=rtol, atol=atol)
        and recomputed.feasible == bool(stored["feasible"])
        and problem.decoder.control_validity(z)
    )
    result = {
        "passed": passed,
        "reported_objective": reported,
        "recomputed_objective": recomputed.value,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "maximum_constraint_violation": recomputed.violation,
        "power_flow_convergence": all(x["converged"] for x in audits),
        "voltage_limit_status": all(x["voltage_limits"] for x in audits),
        "generator_q_limit_status": all(x["generator_q_limits"] for x in audits),
        "generator_p_limit_status": all(x["generator_p_limits"] for x in audits),
        "branch_limit_status": all(x["branch_limits"] for x in audits),
        "decision_vector_valid": problem.decoder.control_validity(z),
        "scenario_audit": audits,
        "final_integrity_result": "VERIFIED" if passed else "FAILED",
    }
    database.add_validation(run_id, result)
    return result
