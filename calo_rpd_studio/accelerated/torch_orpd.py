"""Accelerator-native ORPD problem with CPU-reference parity support."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from calo_rpd_studio.orpd.objectives import ObjectiveKind
from calo_rpd_studio.orpd.problem import Evaluation, ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
from calo_rpd_studio.power_system.case_model import (
    BUS_I,
    BUS_TYPE,
    GEN_BUS,
    GEN_STATUS,
    PMAX,
    PMIN,
    PQ,
    QMAX,
    QMIN,
    RATE_A,
    BR_STATUS,
    VMAX,
    VMIN,
)
from calo_rpd_studio.robustness.robust_objectives import (
    RobustAggregation, aggregate_constraint_violation, normalize_scenario_weights,
)
from calo_rpd_studio.robustness.cvar import weighted_cvar_torch
from calo_rpd_studio.robustness.scenario import Scenario
from calo_rpd_studio.orpd.constraints import branch_angle_limit_violation, generator_limit_violation

from .device import resolve_device, torch_dtype
from .torch_power_flow import (
    TorchPowerFlowOptions,
    run_torch_ac_power_flow,
    run_torch_ac_power_flow_batch,
    torch_l_index,
)
from .torch_decoder import TorchVariableDecoder
from .throughput_engine import GLOBAL_LEDGER, timed_stage
from .runtime_context import get_cross_run_broker


@dataclass(frozen=True, slots=True)
class ParityReport:
    candidate_count: int
    passed: bool
    max_objective_error: float
    max_violation_error: float
    feasibility_mismatches: int
    max_voltage_error: float
    details: tuple[dict[str, Any], ...] = ()
    max_constraint_component_error: float = 0.0
    max_objective_component_error: float = 0.0
    max_angle_error_deg: float = 0.0
    convergence_mismatches: int = 0
    bus_type_mismatches: int = 0
    scenario_count_mismatches: int = 0


class AcceleratedORPDProblem:
    """The same ORPD formulation evaluated through double-precision PyTorch kernels.

    Decision decoding remains governed by the single common ``ORPDVariableDecoder``.  Power flow,
    branch flows, objective components, normalized constraints, robust aggregation, and L-index are
    evaluated on the requested device.  Final publication state reconstruction deliberately uses
    the trusted CPU evaluator as an independent audit.
    """

    def __init__(
        self,
        case,
        config: ORPDProblemConfig | None = None,
        scenarios: list[Scenario] | None = None,
        *,
        device: str = "auto",
        dtype_name: str = "float64",
        batch_size: int = 64,
        device_resident: bool = True,
    ):
        self.case = case.clone()
        self.config = config or ORPDProblemConfig()
        self.scenarios = [Scenario("base")] if scenarios is None else list(scenarios)
        if not self.scenarios:
            raise ValueError(
                "At least one robust scenario is required; an empty scenario set is invalid."
            )
        self.reference = ORPDProblem(self.case, self.config, self.scenarios)
        self.decoder = self.reference.decoder
        self.device_context = resolve_device(device)
        self.device = self.device_context.resolved
        self.dtype = torch_dtype(dtype_name)
        self.batch_size = max(1, int(batch_size))
        self.device_resident_enabled = bool(device_resident)
        self.tensor_decoder = TorchVariableDecoder(self.decoder, self.device, self.dtype)
        pf = self.config.power_flow
        self._broker = get_cross_run_broker()
        self._batch_signature_cache = None
        self.power_flow_options = TorchPowerFlowOptions(
            tolerance=float(pf.tolerance),
            max_iterations=int(pf.max_iterations),
            enforce_q_limits=bool(pf.enforce_q_limits),
            max_q_limit_rounds=int(pf.max_q_limit_rounds),
            q_limit_tolerance_mvar=float(pf.q_limit_tolerance_mvar),
        )
        self._device_resident_evaluator = None
        if self.device_resident_enabled:
            # The device-resident evaluator currently stores generator capability by bus.
            # When multiple online units share a bus, use the batched torch path below so
            # per-generator PMIN/PMAX/QMIN/QMAX enforcement remains exact.
            online = self.case.gen[self.case.gen[:, GEN_STATUS] > 0]
            buses = online[:, GEN_BUS].astype(int) if len(online) else np.asarray([], dtype=int)
            has_colocated_units = len(set(buses.tolist())) != len(buses)
            if not has_colocated_units:
                from .device_resident_orpd import DeviceResidentORPDEvaluator

                self._device_resident_evaluator = DeviceResidentORPDEvaluator(self)

    @property
    def dimension(self) -> int:
        return self.decoder.dimension

    def _objective(self, pf, formulation_case=None):
        import torch

        if not pf.converged or pf.branch is None:
            inf = float("inf")
            return inf, {
                "active_power_loss_mw": inf,
                "voltage_deviation_pu": inf,
                "l_index_max": inf,
            }
        reference = formulation_case if formulation_case is not None else pf.case
        pq = np.where(reference.bus[:, BUS_TYPE].astype(int) == PQ)[0]
        pq_t = torch.as_tensor(pq, dtype=torch.long, device=self.device)
        loss = float(pf.total_loss_mw)
        voltage_deviation = (
            float(torch.sum(torch.abs(pf.vm_pu[pq_t] - 1.0)).detach().cpu()) if pq.size else 0.0
        )
        partitioned_case = pf.case.clone()
        partitioned_case.bus[:, BUS_TYPE] = reference.bus[:, BUS_TYPE]
        _, l_index = torch_l_index(partitioned_case, pf.voltage, pf.ybus)
        components = {
            "active_power_loss_mw": loss,
            "voltage_deviation_pu": voltage_deviation,
            "l_index_max": float(l_index),
        }
        config = self.config.objective
        config.validate()
        if config.kind is ObjectiveKind.ACTIVE_POWER_LOSS:
            value = loss
        elif config.kind is ObjectiveKind.VOLTAGE_DEVIATION:
            value = voltage_deviation
        elif config.kind is ObjectiveKind.L_INDEX:
            value = l_index
        else:
            value = (
                config.weight_loss * loss / config.loss_scale
                + config.weight_voltage_deviation
                * voltage_deviation
                / config.voltage_deviation_scale
                + config.weight_l_index * l_index / config.l_index_scale
            )
        return float(value), components

    def _constraints(self, pf):
        import torch

        if not pf.converged or pf.actual_pg_mw is None or pf.actual_qg_mvar is None:
            return float("inf"), {"power_flow": float("inf")}
        case = pf.case
        dtype = pf.vm_pu.dtype
        device = pf.vm_pu.device
        tolerances = self.config.constraint_tolerances
        lower = torch.as_tensor(case.bus[:, VMIN], dtype=dtype, device=device)
        upper = torch.as_tensor(case.bus[:, VMAX], dtype=dtype, device=device)
        span = torch.clamp(upper - lower, min=1e-12)
        below = lower - pf.vm_pu
        above = pf.vm_pu - upper
        below = torch.where(below > float(tolerances.voltage_pu), below, torch.zeros_like(below))
        above = torch.where(above > float(tolerances.voltage_pu), above, torch.zeros_like(above))
        bus_voltage = torch.sum(torch.relu(below) / span + torch.relu(above) / span)

        qv, pv = generator_limit_violation(case, tolerances)
        generator_q = torch.as_tensor(qv, dtype=dtype, device=device)
        generator_p = torch.as_tensor(pv, dtype=dtype, device=device)

        branch_thermal = torch.zeros((), dtype=dtype, device=device)
        if pf.branch is not None:
            rated = torch.as_tensor(
                (case.branch[:, BR_STATUS] > 0) & (case.branch[:, RATE_A] > 0),
                dtype=torch.bool,
                device=device,
            )
            if bool(torch.any(rated)):
                overload = pf.branch.loading_percent[rated] - 100.0
                overload = torch.where(
                    overload > float(tolerances.branch_loading_percent),
                    overload,
                    torch.zeros_like(overload),
                )
                branch_thermal = torch.sum(torch.relu(overload) / 100.0)
        branch_angle = branch_angle_limit_violation(
            case, np.asarray(pf.va_deg.detach().cpu(), dtype=float), tolerances
        )
        components = {
            "bus_voltage": float(bus_voltage.detach().cpu()),
            "generator_q": float(generator_q.detach().cpu()),
            "generator_p": float(generator_p.detach().cpu()),
            "branch_thermal": float(branch_thermal.detach().cpu()),
            "branch_angle": float(branch_angle),
            "power_flow": 0.0,
        }
        return float(sum(components.values())), components

    def _aggregate_robust(self, values, weights):
        import torch

        v = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        w = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        w = w / torch.sum(w)
        mean = torch.sum(v * w)
        config = self.config.robust
        if config.aggregation is RobustAggregation.EXPECTED:
            result = mean
        elif config.aggregation is RobustAggregation.MEAN_RISK:
            result = mean + config.risk_lambda * torch.sqrt(torch.sum(w * (v - mean).square()))
        elif config.aggregation is RobustAggregation.WORST_CASE:
            result = torch.max(v)
        else:
            result = weighted_cvar_torch(v, w, config.cvar_alpha)
        return float(result.detach().cpu())

    def batch_signature(self) -> str:
        """Strict scientific compatibility key for cross-run evaluation batching.

        Cross-run microbatching is permitted only for problems with the exact same canonical
        scientific formulation. v5.9 therefore reuses the same fingerprint as continuation,
        policy-development evidence and accelerator parity, then binds device/dtype as execution
        identity. No broad fallback to scenario names is allowed.
        """
        if self._batch_signature_cache is None:
            payload = {
                "scientific_problem_fingerprint": scientific_problem_fingerprint(self.reference),
                "device": str(self.device),
                "dtype": str(self.dtype),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self._batch_signature_cache = hashlib.sha256(encoded).hexdigest()
        return str(self._batch_signature_cache)

    def attach_broker(self, broker) -> None:
        self._broker = broker

    def evaluate(self, normalized):
        return self.evaluate_population([normalized])[0]

    def evaluate_population_tensor(self, population):
        """Return one device-resident population result without host materialisation."""
        if self._device_resident_evaluator is None:
            raise RuntimeError("Device-resident evaluation is disabled for this problem")
        return self._device_resident_evaluator.evaluate_tensor(population)

    def _evaluate_population_tensor_direct(self, population):
        return self.evaluate_population_tensor(population)

    def evaluate_population(self, population: Iterable):
        if self._broker is not None:
            return self._broker.submit(self, population)
        return self._evaluate_population_direct(population)

    def _evaluate_population_direct(self, population: Iterable):
        if self._device_resident_evaluator is not None:
            return self._device_resident_evaluator.evaluate_tensor(population).to_evaluations()
        try:
            import torch

            if isinstance(population, torch.Tensor):
                candidates = population.detach().to("cpu", dtype=torch.float64).numpy()
            else:
                candidates = np.asarray(population, dtype=float)
        except Exception:
            candidates = np.asarray(population, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        candidates = np.clip(candidates, 0.0, 1.0)
        with timed_stage("mixed_variable_decode", len(candidates), GLOBAL_LEDGER):
            controlled_cases, physical_controls = self.tensor_decoder.decode_batch(candidates)
        count = len(candidates)
        values = [[] for _ in range(count)]
        violations = [[] for _ in range(count)]
        weights = [[] for _ in range(count)]
        scenario_values = [[] for _ in range(count)]
        objective_components = [dict() for _ in range(count)]
        constraint_components = [dict() for _ in range(count)]
        scenario_constraint_components = [[] for _ in range(count)]
        converged_all = [True] * count

        for scenario in self.scenarios:
            with timed_stage("scenario_prepare", count, GLOBAL_LEDGER):
                scenario_cases = [scenario.apply(case) for case in controlled_cases]
            scenario_results = []
            with timed_stage("batched_ac_power_flow", count, GLOBAL_LEDGER):
                for offset in range(0, count, self.batch_size):
                    scenario_results.extend(
                        run_torch_ac_power_flow_batch(
                            scenario_cases[offset : offset + self.batch_size],
                            device=self.device,
                            dtype=self.dtype,
                            options=self.power_flow_options,
                        )
                    )
            with timed_stage("objective_constraint_aggregation", count, GLOBAL_LEDGER):
                for index, pf in enumerate(scenario_results):
                    value, obj_components = self._objective(pf, scenario_cases[index])
                    violation, con_components = self._constraints(pf)
                    converged_all[index] = converged_all[index] and bool(pf.converged)
                    values[index].append(float(value))
                    violations[index].append(float(violation))
                    weights[index].append(float(scenario.weight))
                    scenario_values[index].append(float(value))
                    scenario_constraint_components[index].append(dict(con_components))
                    for key, component in obj_components.items():
                        objective_components[index].setdefault(key, []).append(float(component))
                    for key, component in con_components.items():
                        constraint_components[index].setdefault(key, []).append(float(component))

        results: list[Evaluation] = []
        with timed_stage("robust_result_finalize", count, GLOBAL_LEDGER):
            for index in range(count):
                weights_np = normalize_scenario_weights(weights[index])
                finite = np.asarray(values[index], dtype=float)
                robust_value = (
                    float("inf")
                    if not np.all(np.isfinite(finite))
                    else self._aggregate_robust(values[index], weights_np)
                )
                violation = aggregate_constraint_violation(
                    violations[index], weights_np, self.config.robust
                )
                feasible = bool(
                    converged_all[index] and np.isfinite(robust_value) and violation <= float(self.config.constraint_tolerances.feasibility_total)
                )
                components = {
                    key: float(np.sum(weights_np * np.asarray(series, dtype=float)))
                    for key, series in objective_components[index].items()
                }
                if np.all(np.isfinite(finite)):
                    mean = float(np.sum(weights_np * finite))
                    components["scenario_objective_mean"] = mean
                    components["scenario_objective_std"] = float(
                        np.sqrt(np.sum(weights_np * (finite - mean) ** 2))
                    )
                else:
                    components["scenario_objective_mean"] = float("inf")
                    components["scenario_objective_std"] = float("inf")
                weighted_constraints = {
                    key: aggregate_constraint_violation(series, weights_np, self.config.robust)
                    for key, series in constraint_components[index].items()
                }
                metadata = {
                    "scenario_count": len(self.scenarios),
                    "constraint_components": weighted_constraints,
                    "scenario_constraint_components": scenario_constraint_components[index],
                    "scientific_backend": "torch_batched_dense_newton_raphson",
                    "throughput_engine_version": "3.1",
                    "compute_device": self.device,
                    "device_name": self.device_context.name,
                    "dtype": "float64",
                    "batch_size": self.batch_size,
                    "cross_run_batching": self._broker is not None,
                    "q_limit_switching": bool(self.power_flow_options.enforce_q_limits),
                    "candidate_specific_q_limit_fallback": True,
                }
                results.append(
                    Evaluation(
                        robust_value,
                        feasible,
                        violation,
                        components,
                        physical_controls[index],
                        scenario_values[index],
                        metadata,
                    )
                )
        return results

    def accelerator_solution_state(self, normalized):
        """Return state reconstructed directly by the accelerator solver for parity auditing."""
        z = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
        controlled, physical = self.decoder.decode(z)
        records = []
        for scenario in self.scenarios:
            pf = run_torch_ac_power_flow(
                scenario.apply(controlled),
                device=self.device,
                dtype=self.dtype,
                options=self.power_flow_options,
            )
            record = {
                "scenario": scenario.name,
                "weight": float(scenario.weight),
                "converged": bool(pf.converged),
                "iterations": int(pf.iterations),
                "max_mismatch": float(pf.max_mismatch),
                "bus_types": pf.case.bus[:, BUS_TYPE].astype(int).tolist(),
                "vm_pu": np.asarray(pf.vm_pu.detach().cpu(), dtype=float).tolist(),
                "va_deg": np.asarray(pf.va_deg.detach().cpu(), dtype=float).tolist(),
                "total_loss_mw": float(pf.total_loss_mw),
            }
            if pf.branch is not None:
                record["loading_percent"] = np.asarray(
                    pf.branch.loading_percent.detach().cpu(), dtype=float
                ).tolist()
            records.append(record)
        return {
            "normalized_decision_vector": z.tolist(),
            "decoded_controls": physical,
            "case_checksum": self.case.checksum(),
            "scenarios": records,
            "backend": "torch_batched_dense_newton_raphson",
            "device": self.device,
        }

    def solution_state(self, normalized):
        state = self.reference.solution_state(normalized)
        state["optimization_backend"] = {
            "name": "torch_dense_newton_raphson",
            "device": self.device,
            "device_name": self.device_context.name,
            "dtype": "float64",
        }
        state["publication_state_reconstructed_with"] = "independent_cpu_reference"
        return state


def parity_check(
    reference: ORPDProblem,
    accelerated: AcceleratedORPDProblem,
    candidates,
    *,
    objective_tolerance=1e-5,
    violation_tolerance=1e-6,
    voltage_tolerance=1e-6,
    angle_tolerance_deg=1e-4,
):
    """Fail-closed scientific parity across objectives, constraints and solved states."""
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    details = []
    max_objective_error = 0.0
    max_violation_error = 0.0
    max_voltage_error = 0.0
    max_angle_error = 0.0
    max_constraint_component_error = 0.0
    max_objective_component_error = 0.0
    feasibility_mismatches = 0
    convergence_mismatches = 0
    bus_type_mismatches = 0
    scenario_count_mismatches = 0

    def scalar_error(a, b):
        a, b = float(a), float(b)
        if np.isfinite(a) and np.isfinite(b):
            return abs(a - b)
        return 0.0 if a == b else float("inf")

    for index, candidate in enumerate(candidates):
        cpu = reference.evaluate(candidate)
        gpu = accelerated.evaluate(candidate)
        objective_error = scalar_error(cpu.value, gpu.value)
        violation_error = scalar_error(cpu.violation, gpu.violation)
        feasibility_mismatch = bool(cpu.feasible) != bool(gpu.feasible)

        cpu_constraints = dict((cpu.metadata or {}).get("constraint_components", {}) or {})
        gpu_constraints = dict((gpu.metadata or {}).get("constraint_components", {}) or {})
        constraint_component_error = 0.0
        for key in sorted(set(cpu_constraints) | set(gpu_constraints)):
            constraint_component_error = max(
                constraint_component_error,
                scalar_error(cpu_constraints.get(key, 0.0), gpu_constraints.get(key, 0.0)),
            )
        objective_component_error = 0.0
        for key in sorted(set(cpu.components) | set(gpu.components)):
            objective_component_error = max(
                objective_component_error,
                scalar_error(cpu.components.get(key, 0.0), gpu.components.get(key, 0.0)),
            )

        cpu_state = reference.solution_state(candidate)
        gpu_state = accelerated.accelerator_solution_state(candidate)
        voltage_error = 0.0
        angle_error = 0.0
        convergence_mismatch = 0
        bus_type_mismatch = 0
        scenario_count_mismatch = int(len(cpu_state.get("scenarios", [])) != len(gpu_state.get("scenarios", [])))
        if not scenario_count_mismatch:
            try:
                for cpu_scenario, gpu_scenario in zip(
                    cpu_state["scenarios"], gpu_state["scenarios"], strict=True
                ):
                    convergence_mismatch += int(
                        bool(cpu_scenario.get("converged")) != bool(gpu_scenario.get("converged"))
                    )
                    cpu_types = np.asarray(cpu_scenario.get("bus_types", []), dtype=int)
                    gpu_types = np.asarray(gpu_scenario.get("bus_types", []), dtype=int)
                    if cpu_types.shape != gpu_types.shape or not np.array_equal(cpu_types, gpu_types):
                        bus_type_mismatch += 1
                    cpu_vm = np.asarray(cpu_scenario["vm_pu"], dtype=float)
                    gpu_vm = np.asarray(gpu_scenario["vm_pu"], dtype=float)
                    cpu_va = np.asarray(cpu_scenario["va_deg"], dtype=float)
                    gpu_va = np.asarray(gpu_scenario["va_deg"], dtype=float)
                    if cpu_vm.shape != gpu_vm.shape or cpu_va.shape != gpu_va.shape:
                        voltage_error = angle_error = float("inf")
                        break
                    if cpu_vm.size:
                        voltage_error = max(voltage_error, float(np.max(np.abs(cpu_vm - gpu_vm))))
                    if cpu_va.size:
                        # Voltage angles have a fixed reference bus in both solvers; direct comparison is valid.
                        angle_error = max(angle_error, float(np.max(np.abs(cpu_va - gpu_va))))
            except Exception:
                voltage_error = angle_error = float("inf")
                convergence_mismatch += 1

        max_objective_error = max(max_objective_error, objective_error)
        max_violation_error = max(max_violation_error, violation_error)
        max_voltage_error = max(max_voltage_error, voltage_error)
        max_angle_error = max(max_angle_error, angle_error)
        max_constraint_component_error = max(max_constraint_component_error, constraint_component_error)
        max_objective_component_error = max(max_objective_component_error, objective_component_error)
        feasibility_mismatches += int(feasibility_mismatch)
        convergence_mismatches += int(convergence_mismatch)
        bus_type_mismatches += int(bus_type_mismatch)
        scenario_count_mismatches += int(scenario_count_mismatch)
        details.append(
            {
                "candidate": index,
                "objective_error": objective_error,
                "violation_error": violation_error,
                "constraint_component_error": constraint_component_error,
                "objective_component_error": objective_component_error,
                "voltage_error": voltage_error,
                "angle_error_deg": angle_error,
                "feasibility_mismatch": feasibility_mismatch,
                "convergence_mismatches": int(convergence_mismatch),
                "bus_type_mismatches": int(bus_type_mismatch),
                "scenario_count_mismatch": bool(scenario_count_mismatch),
            }
        )
    passed = bool(
        feasibility_mismatches == 0
        and convergence_mismatches == 0
        and bus_type_mismatches == 0
        and scenario_count_mismatches == 0
        and max_objective_error <= objective_tolerance
        and max_objective_component_error <= objective_tolerance
        and max_violation_error <= violation_tolerance
        and max_constraint_component_error <= violation_tolerance
        and max_voltage_error <= voltage_tolerance
        and max_angle_error <= angle_tolerance_deg
    )
    return ParityReport(
        candidate_count=int(candidates.shape[0]),
        passed=passed,
        max_objective_error=max_objective_error,
        max_violation_error=max_violation_error,
        feasibility_mismatches=feasibility_mismatches,
        max_voltage_error=max_voltage_error,
        details=tuple(details),
        max_constraint_component_error=max_constraint_component_error,
        max_objective_component_error=max_objective_component_error,
        max_angle_error_deg=max_angle_error,
        convergence_mismatches=convergence_mismatches,
        bus_type_mismatches=bus_type_mismatches,
        scenario_count_mismatches=scenario_count_mismatches,
    )

