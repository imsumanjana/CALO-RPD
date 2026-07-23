"""Shared ORPD evaluator used without algorithm-specific physics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from calo_rpd_studio.power_system.ac_power_flow import PowerFlowOptions, run_ac_power_flow
from calo_rpd_studio.power_system.case_model import *
from calo_rpd_studio.power_system.voltage_stability import kessel_glavitsch_l_index
from calo_rpd_studio.robustness.robust_objectives import (
    RobustObjectiveConfig, aggregate_robust, aggregate_constraint_violation, normalize_scenario_weights,
)
from calo_rpd_studio.robustness.scenario import Scenario
from .constraints import ConstraintToleranceConfig, evaluate_constraints
from .objectives import ObjectiveConfig, calculate_objective
from .variable_decoder import ORPDVariableConfig, ORPDVariableDecoder


@dataclass(slots=True)
class ORPDProblemConfig:
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    variables: ORPDVariableConfig = field(default_factory=ORPDVariableConfig)
    robust: RobustObjectiveConfig = field(default_factory=RobustObjectiveConfig)
    power_flow: PowerFlowOptions = field(default_factory=PowerFlowOptions)
    constraint_tolerances: ConstraintToleranceConfig = field(default_factory=ConstraintToleranceConfig)

    def __post_init__(self) -> None:
        self.objective.validate()
        self.variables.validate()
        self.robust.validate()
        self.power_flow.validate()
        self.constraint_tolerances.validate()


@dataclass(slots=True)
class Evaluation:
    value: float
    feasible: bool
    violation: float
    components: dict[str, float] = field(default_factory=dict)
    physical_controls: dict[str, float] = field(default_factory=dict)
    scenario_values: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ORPDProblem:
    def __init__(self, case, config=None, scenarios=None):
        self.case = case.clone()
        self.config = config or ORPDProblemConfig()
        self.decoder = ORPDVariableDecoder(self.case, self.config.variables)
        self.scenarios = [Scenario("base")] if scenarios is None else list(scenarios)
        if not self.scenarios:
            raise ValueError(
                "At least one robust scenario is required; an empty scenario set is invalid."
            )

    @property
    def dimension(self):
        return self.decoder.dimension

    def evaluate(self, normalized):
        z = np.clip(np.asarray(normalized, float), 0, 1)
        controlled, physical = self.decoder.decode(z)
        values = []
        violations = []
        weights = []
        scenario_values = []
        comp_acc = {}
        constraint_acc = {}
        scenario_constraint_components = []
        for scenario in self.scenarios:
            formulation_case = scenario.apply(controlled)
            pf = run_ac_power_flow(formulation_case, self.config.power_flow)
            obj = calculate_objective(pf, self.config.objective, formulation_case=formulation_case)
            con = evaluate_constraints(pf, self.config.constraint_tolerances)
            value = float(obj.value)
            values.append(value)
            violations.append(float(con.total))
            weights.append(float(scenario.weight))
            scenario_values.append(value)
            scenario_constraint_components.append(dict(con.components))
            for k, v in obj.components.items():
                comp_acc.setdefault(k, []).append(float(v))
            for k, v in con.components.items():
                constraint_acc.setdefault(k, []).append(float(v))
        w = normalize_scenario_weights(weights)
        finite = np.asarray(values, float)
        robust = aggregate_robust(values, w, self.config.robust)
        violation = aggregate_constraint_violation(violations, w, self.config.robust)
        feasible = violation <= float(self.config.constraint_tolerances.feasibility_total) and np.isfinite(robust)
        components = {k: float(np.sum(w * np.asarray(v))) for k, v in comp_acc.items()}
        components["scenario_objective_mean"] = (
            float(np.sum(w * finite)) if np.all(np.isfinite(finite)) else float("inf")
        )
        components["scenario_objective_std"] = (
            float(np.sqrt(np.sum(w * (finite - components["scenario_objective_mean"]) ** 2)))
            if np.all(np.isfinite(finite))
            else float("inf")
        )
        constraint_components = {
            k: aggregate_constraint_violation(v, w, self.config.robust)
            for k, v in constraint_acc.items()
        }
        metadata = {
            "scenario_count": len(self.scenarios),
            "constraint_components": constraint_components,
            "scenario_constraint_components": scenario_constraint_components,
        }
        return Evaluation(
            robust, feasible, violation, components, physical, scenario_values, metadata
        )

    def solution_state(self, normalized):
        z = np.clip(np.asarray(normalized, float), 0, 1)
        controlled, physical = self.decoder.decode(z)
        records = []
        for sc in self.scenarios:
            formulation_case = sc.apply(controlled)
            pf = run_ac_power_flow(formulation_case, self.config.power_flow)
            obj = calculate_objective(pf, self.config.objective, formulation_case=formulation_case)
            con = evaluate_constraints(pf, self.config.constraint_tolerances)
            online = np.where(pf.case.gen[:, GEN_STATUS] > 0)[0]
            rec = {
                "scenario": sc.name,
                "weight": float(sc.weight),
                "converged": bool(pf.converged),
                "iterations": int(pf.iterations),
                "max_mismatch": float(pf.max_mismatch),
                "bus_numbers": pf.case.bus[:, BUS_I].astype(int).tolist(),
                "bus_types": pf.case.bus[:, BUS_TYPE].astype(int).tolist(),
                "vm_pu": pf.vm_pu.tolist(),
                "va_deg": pf.va_deg.tolist(),
                "generator_bus": pf.case.gen[online, GEN_BUS].astype(int).tolist(),
                "pg_mw": pf.case.gen[online, PG].tolist(),
                "qg_mvar": pf.case.gen[online, QG].tolist(),
                "objective": float(obj.value),
                "objective_components": dict(obj.components),
                "constraint_components": dict(con.components),
                "total_constraint_violation": float(con.total),
                "total_loss_mw": float(pf.total_loss_mw),
                "l_index_max": float(kessel_glavitsch_l_index(pf.case, pf.voltage, partition_case=formulation_case).maximum)
                if pf.converged
                else float("inf"),
            }
            if pf.branch is not None:
                rec.update(
                    {
                        "branch_from_bus": pf.case.branch[:, F_BUS].astype(int).tolist(),
                        "branch_to_bus": pf.case.branch[:, T_BUS].astype(int).tolist(),
                        "p_from_mw": np.real(pf.branch.s_from_mva).tolist(),
                        "q_from_mvar": np.imag(pf.branch.s_from_mva).tolist(),
                        "p_to_mw": np.real(pf.branch.s_to_mva).tolist(),
                        "q_to_mvar": np.imag(pf.branch.s_to_mva).tolist(),
                        "loading_percent": pf.branch.loading_percent.tolist(),
                    }
                )
            records.append(rec)
        return {
            "normalized_decision_vector": z.tolist(),
            "decoded_controls": physical,
            "case_checksum": self.case.checksum(),
            "scenarios": records,
        }
