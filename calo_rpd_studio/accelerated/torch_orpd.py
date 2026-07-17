"""Accelerator-native ORPD problem with CPU-reference parity support."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from calo_rpd_studio.orpd.objectives import ObjectiveKind
from calo_rpd_studio.orpd.problem import Evaluation, ORPDProblem, ORPDProblemConfig
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
    VMAX,
    VMIN,
)
from calo_rpd_studio.robustness.robust_objectives import RobustAggregation
from calo_rpd_studio.robustness.scenario import Scenario

from .device import resolve_device, torch_dtype
from .torch_power_flow import TorchPowerFlowOptions, run_torch_ac_power_flow, run_torch_ac_power_flow_batch, torch_l_index
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
        self.scenarios = scenarios or [Scenario("base")]
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
            from .device_resident_orpd import DeviceResidentORPDEvaluator

            self._device_resident_evaluator = DeviceResidentORPDEvaluator(self)

    @property
    def dimension(self) -> int:
        return self.decoder.dimension

    def _objective(self, pf):
        import torch

        if not pf.converged or pf.branch is None:
            inf = float("inf")
            return inf, {
                "active_power_loss_mw": inf,
                "voltage_deviation_pu": inf,
                "l_index_max": inf,
            }
        pq = np.where(pf.case.bus[:, BUS_TYPE].astype(int) == PQ)[0]
        pq_t = torch.as_tensor(pq, dtype=torch.long, device=self.device)
        loss = float(pf.total_loss_mw)
        voltage_deviation = float(torch.sum(torch.abs(pf.vm_pu[pq_t] - 1.0)).detach().cpu()) if pq.size else 0.0
        _, l_index = torch_l_index(pf.case, pf.voltage, pf.ybus)
        components = {
            "active_power_loss_mw": loss,
            "voltage_deviation_pu": voltage_deviation,
            "l_index_max": float(l_index),
        }
        config = self.config.objective
        if config.kind is ObjectiveKind.ACTIVE_POWER_LOSS:
            value = loss
        elif config.kind is ObjectiveKind.VOLTAGE_DEVIATION:
            value = voltage_deviation
        elif config.kind is ObjectiveKind.L_INDEX:
            value = l_index
        else:
            value = (
                config.weight_loss * loss / max(config.loss_scale, 1e-15)
                + config.weight_voltage_deviation
                * voltage_deviation
                / max(config.voltage_deviation_scale, 1e-15)
                + config.weight_l_index * l_index / max(config.l_index_scale, 1e-15)
            )
        return float(value), components

    def _constraints(self, pf):
        import torch

        if not pf.converged or pf.actual_pg_mw is None or pf.actual_qg_mvar is None:
            return float("inf"), {"power_flow": float("inf")}
        case = pf.case
        dtype = pf.vm_pu.dtype
        device = pf.vm_pu.device
        lower = torch.as_tensor(case.bus[:, VMIN], dtype=dtype, device=device)
        upper = torch.as_tensor(case.bus[:, VMAX], dtype=dtype, device=device)
        span = torch.clamp(upper - lower, min=1e-12)
        bus_voltage = torch.sum(torch.relu(lower - pf.vm_pu) / span + torch.relu(pf.vm_pu - upper) / span)

        index = case.bus_index_map()
        generator_q = torch.zeros((), dtype=dtype, device=device)
        generator_p = torch.zeros((), dtype=dtype, device=device)
        for bus_number in case.bus[:, BUS_I].astype(int):
            generators = np.where(
                (case.gen[:, GEN_STATUS] > 0) & (case.gen[:, GEN_BUS].astype(int) == bus_number)
            )[0]
            if not generators.size:
                continue
            bus_index = index[bus_number]
            qmin = float(case.gen[generators, QMIN].sum())
            qmax = float(case.gen[generators, QMAX].sum())
            qspan = max(qmax - qmin, 1.0)
            actual_q = pf.actual_qg_mvar[bus_index]
            generator_q = generator_q + torch.relu(
                torch.as_tensor(qmin, dtype=dtype, device=device) - actual_q
            ) / qspan + torch.relu(actual_q - torch.as_tensor(qmax, dtype=dtype, device=device)) / qspan
            pmin = float(case.gen[generators, PMIN].sum())
            pmax = float(case.gen[generators, PMAX].sum())
            pspan = max(pmax - pmin, 1.0)
            actual_p = pf.actual_pg_mw[bus_index]
            generator_p = generator_p + torch.relu(
                torch.as_tensor(pmin, dtype=dtype, device=device) - actual_p
            ) / pspan + torch.relu(actual_p - torch.as_tensor(pmax, dtype=dtype, device=device)) / pspan

        branch_thermal = torch.zeros((), dtype=dtype, device=device)
        if pf.branch is not None:
            rated = torch.as_tensor(case.branch[:, RATE_A] > 0, dtype=torch.bool, device=device)
            if bool(torch.any(rated)):
                branch_thermal = torch.sum(torch.relu(pf.branch.loading_percent[rated] - 100.0) / 100.0)
        components = {
            "bus_voltage": float(bus_voltage.detach().cpu()),
            "generator_q": float(generator_q.detach().cpu()),
            "generator_p": float(generator_p.detach().cpu()),
            "branch_thermal": float(branch_thermal.detach().cpu()),
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
            order = torch.argsort(v)
            sorted_v = v[order]
            sorted_w = w[order]
            cdf = torch.cumsum(sorted_w, dim=0)
            threshold_index = int(torch.searchsorted(cdf, torch.as_tensor(config.cvar_alpha, dtype=self.dtype, device=self.device)).item())
            threshold_index = min(max(threshold_index, 0), sorted_v.numel() - 1)
            var = sorted_v[threshold_index]
            mask = sorted_v >= var
            result = torch.sum(sorted_v[mask] * sorted_w[mask]) / torch.clamp(torch.sum(sorted_w[mask]), min=1e-15)
        return float(result.detach().cpu())

    def batch_signature(self) -> str:
        """Stable compatibility key for cross-run evaluation batching."""
        if self._batch_signature_cache is None:
            scenario_records = []
            for scenario in self.scenarios:
                try:
                    checksum = scenario.apply(self.case).checksum()
                except Exception:
                    checksum = scenario.name
                scenario_records.append((scenario.name, float(scenario.weight), checksum))
            objective = self.config.objective
            robust = self.config.robust
            payload = {
                "case": self.case.checksum(),
                "dimension": self.dimension,
                "scenarios": scenario_records,
                "objective": {
                    "kind": objective.kind.value,
                    "weights": [objective.weight_loss, objective.weight_voltage_deviation, objective.weight_l_index],
                    "scales": [objective.loss_scale, objective.voltage_deviation_scale, objective.l_index_scale],
                },
                "robust": {
                    "aggregation": robust.aggregation.value,
                    "risk_lambda": robust.risk_lambda,
                    "cvar_alpha": robust.cvar_alpha,
                },
                "device": self.device,
                "dtype": str(self.dtype),
                "power_flow": {
                    "tolerance": self.power_flow_options.tolerance,
                    "max_iterations": self.power_flow_options.max_iterations,
                    "enforce_q_limits": self.power_flow_options.enforce_q_limits,
                },
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
                    value, obj_components = self._objective(pf)
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
                weights_np = np.asarray(weights[index], dtype=float)
                weights_np = weights_np / weights_np.sum()
                finite = np.asarray(values[index], dtype=float)
                robust_value = (
                    float("inf")
                    if not np.all(np.isfinite(finite))
                    else self._aggregate_robust(values[index], weights_np)
                )
                violation = (
                    float(np.sum(weights_np * np.asarray(violations[index], dtype=float)))
                    if np.all(np.isfinite(violations[index]))
                    else float("inf")
                )
                feasible = bool(
                    converged_all[index]
                    and np.isfinite(robust_value)
                    and violation <= 1e-12
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
                    key: float(np.sum(weights_np * np.asarray(series, dtype=float)))
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


def parity_check(reference: ORPDProblem, accelerated: AcceleratedORPDProblem, candidates, *, objective_tolerance=1e-5, violation_tolerance=1e-6, voltage_tolerance=1e-6):
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    details = []
    max_objective_error = 0.0
    max_violation_error = 0.0
    max_voltage_error = 0.0
    feasibility_mismatches = 0
    for index, candidate in enumerate(candidates):
        cpu = reference.evaluate(candidate)
        gpu = accelerated.evaluate(candidate)
        objective_error = abs(float(cpu.value) - float(gpu.value)) if np.isfinite(cpu.value) and np.isfinite(gpu.value) else (0.0 if cpu.value == gpu.value else float("inf"))
        violation_error = abs(float(cpu.violation) - float(gpu.violation)) if np.isfinite(cpu.violation) and np.isfinite(gpu.violation) else (0.0 if cpu.violation == gpu.violation else float("inf"))
        feasibility_mismatch = bool(cpu.feasible) != bool(gpu.feasible)
        cpu_state = reference.solution_state(candidate)
        gpu_state = accelerated.accelerator_solution_state(candidate)
        voltage_error = 0.0
        try:
            for cpu_scenario, gpu_scenario in zip(cpu_state["scenarios"], gpu_state["scenarios"]):
                voltage_error = max(
                    voltage_error,
                    float(np.max(np.abs(np.asarray(cpu_scenario["vm_pu"]) - np.asarray(gpu_scenario["vm_pu"])))),
                )
        except Exception:
            voltage_error = float("inf")
        max_objective_error = max(max_objective_error, objective_error)
        max_violation_error = max(max_violation_error, violation_error)
        max_voltage_error = max(max_voltage_error, voltage_error)
        feasibility_mismatches += int(feasibility_mismatch)
        details.append(
            {
                "candidate": index,
                "objective_error": objective_error,
                "violation_error": violation_error,
                "voltage_error": voltage_error,
                "feasibility_mismatch": feasibility_mismatch,
            }
        )
    passed = bool(
        feasibility_mismatches == 0
        and max_objective_error <= objective_tolerance
        and max_violation_error <= violation_tolerance
        and max_voltage_error <= voltage_tolerance
    )
    return ParityReport(
        int(candidates.shape[0]),
        passed,
        max_objective_error,
        max_violation_error,
        feasibility_mismatches,
        max_voltage_error,
        tuple(details),
    )
