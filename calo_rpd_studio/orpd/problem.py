"""Shared ORPD evaluator used without algorithm-specific physics."""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from typing import Any
import numpy as np
from calo_rpd_studio.power_system.ac_power_flow import PowerFlowOptions, run_ac_power_flow
from calo_rpd_studio.power_system.case_model import *
from calo_rpd_studio.power_system.voltage_stability import kessel_glavitsch_l_index
from calo_rpd_studio.power_system.ybus import build_ybus
from calo_rpd_studio.robustness.robust_objectives import (
    RobustObjectiveConfig,
    aggregate_robust,
    aggregate_constraint_violation,
    normalize_scenario_weights,
)
from calo_rpd_studio.robustness.scenario import Scenario
from .constraints import ConstraintToleranceConfig, evaluate_constraints
from .objectives import ObjectiveConfig, calculate_objective
from .variable_decoder import ORPDVariableConfig, ORPDVariableDecoder


COUNTED_CONTROL_LINEARIZATION_SCHEMA = "calo-rpd-counted-control-linearization-v1"
MAX_RETAINED_CONDITION_DIMENSION = 1200


@dataclass(slots=True)
class ORPDProblemConfig:
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    variables: ORPDVariableConfig = field(default_factory=ORPDVariableConfig)
    robust: RobustObjectiveConfig = field(default_factory=RobustObjectiveConfig)
    power_flow: PowerFlowOptions = field(default_factory=PowerFlowOptions)
    constraint_tolerances: ConstraintToleranceConfig = field(
        default_factory=ConstraintToleranceConfig
    )

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
    feasibility_tolerance: float = 1e-12


@dataclass(frozen=True, slots=True)
class ScenarioEvaluationContext:
    """One already-counted scenario solve retained outside serializable result metadata."""

    name: str
    weight: float
    power_flow: Any
    control_linearization: ORPDControlLinearization | None = None


@dataclass(frozen=True, slots=True)
class ORPDControlLinearization:
    """Proposal-only control sensitivity derived from one counted converged solve."""

    schema_version: str
    source_evaluation_id: str
    jacobian: object
    control_sensitivity: np.ndarray
    constraint_residual: np.ndarray
    condition_number: float | None
    sensitivity_semantics: str


@dataclass(frozen=True, slots=True)
class ORPDEvaluationContext:
    """Ephemeral solver context for topology/sensitivity consumers; never an extra evaluation."""

    normalized_controls: np.ndarray
    scenarios: tuple[ScenarioEvaluationContext, ...]

    def primary_converged_scenario(self) -> ScenarioEvaluationContext:
        converged = [
            item
            for item in self.scenarios
            if bool(getattr(item.power_flow, "converged", False))
            and getattr(item.power_flow, "branch", None) is not None
        ]
        if not converged:
            raise ValueError("No already-counted converged power-flow context is available")
        base = next((item for item in converged if item.name.lower() == "base"), None)
        return base or max(converged, key=lambda item: float(item.weight))

    def primary_converged_power_flow(self):
        return self.primary_converged_scenario().power_flow

    def primary_control_linearization(self) -> ORPDControlLinearization:
        selected = self.primary_converged_scenario()
        if selected.control_linearization is None:
            raise ValueError("No counted control linearization is available")
        return selected.control_linearization


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
        evaluation, _context = self._evaluate(
            normalized,
            retain_context=False,
            retain_control_linearization=False,
        )
        return evaluation

    def evaluate_with_context(
        self,
        normalized,
        *,
        retain_control_linearization: bool = False,
    ) -> tuple[Evaluation, ORPDEvaluationContext]:
        """Evaluate once and retain the exact scenario solves without rerunning power flow."""

        evaluation, context = self._evaluate(
            normalized,
            retain_context=True,
            retain_control_linearization=bool(retain_control_linearization),
        )
        assert context is not None
        return evaluation, context

    @staticmethod
    def _matrix_vector(matrix, vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ np.asarray(vector, dtype=float), dtype=float).reshape(-1)

    @staticmethod
    def _selected_complex_derivative(derivative, rows: np.ndarray, column: int) -> np.ndarray:
        selected = derivative[rows, int(column)]
        if hasattr(selected, "toarray"):
            selected = selected.toarray()
        return np.asarray(selected).reshape(-1)

    def _constraint_state_correction(self, pf, linearization) -> np.ndarray:
        pvpq = np.asarray(linearization.pvpq, dtype=int)
        pq = np.asarray(linearization.pq, dtype=int)
        state_dimension = len(pvpq) + len(pq)
        rows: list[np.ndarray] = []
        residuals: list[float] = []
        angle_position = {int(bus): position for position, bus in enumerate(pvpq)}
        magnitude_offset = len(pvpq)
        magnitude_position = {int(bus): position for position, bus in enumerate(pq)}
        tolerances = self.config.constraint_tolerances

        dS_dVa = linearization.dS_dVa
        dS_dVm = linearization.dS_dVm
        if hasattr(dS_dVa, "toarray"):
            dS_dVa = dS_dVa.toarray()
        if hasattr(dS_dVm, "toarray"):
            dS_dVm = dS_dVm.toarray()
        state_power_derivative = np.hstack(
            [
                np.asarray(dS_dVa)[:, pvpq],
                np.asarray(dS_dVm)[:, pq],
            ]
        ) * float(pf.case.base_mva)

        def append_constraint(row, residual: float) -> None:
            row = np.asarray(row, dtype=float).reshape(-1)
            if (
                row.shape == (state_dimension,)
                and np.all(np.isfinite(row))
                and np.isfinite(float(residual))
                and float(np.linalg.norm(row)) > 1e-15
                and abs(float(residual)) > 0.0
            ):
                rows.append(row)
                residuals.append(float(residual))

        for bus in pq:
            bus = int(bus)
            voltage = float(pf.vm_pu[bus])
            lower = float(pf.case.bus[bus, VMIN])
            upper = float(pf.case.bus[bus, VMAX])
            span = max(upper - lower, 1.0)
            if lower - voltage > float(tolerances.voltage_pu):
                row = np.zeros(state_dimension, dtype=float)
                row[magnitude_offset + magnitude_position[bus]] = 1.0 / span
                append_constraint(row, (voltage - lower) / span)
            elif voltage - upper > float(tolerances.voltage_pu):
                row = np.zeros(state_dimension, dtype=float)
                row[magnitude_offset + magnitude_position[bus]] = 1.0 / span
                append_constraint(row, (voltage - upper) / span)

        branch = np.asarray(pf.case.branch, dtype=float)
        if branch.shape[1] > ANGMAX:
            bus_index = pf.case.bus_index_map()
            for row in branch:
                if row[BR_STATUS] <= 0.0:
                    continue
                lower = float(row[ANGMIN])
                upper = float(row[ANGMAX])
                if lower == 0.0 and upper == 0.0:
                    continue
                from_bus = bus_index[int(row[F_BUS])]
                to_bus = bus_index[int(row[T_BUS])]
                difference = float(pf.va_deg[from_bus] - pf.va_deg[to_bus])
                target = difference
                if lower > -360.0 and lower - difference > float(tolerances.branch_angle_deg):
                    target = lower
                elif upper < 360.0 and difference - upper > float(tolerances.branch_angle_deg):
                    target = upper
                if target == difference:
                    continue
                span = max(upper - lower, 1.0) if lower > -360.0 and upper < 360.0 else 360.0
                row = np.zeros(state_dimension, dtype=float)
                from_position = angle_position.get(from_bus)
                to_position = angle_position.get(to_bus)
                degrees_per_radian = 180.0 / np.pi
                if from_position is not None:
                    row[from_position] += degrees_per_radian / span
                if to_position is not None:
                    row[to_position] -= degrees_per_radian / span
                append_constraint(row, (difference - target) / span)

        online = np.where(pf.case.gen[:, GEN_STATUS] > 0)[0]
        bus_index = pf.case.bus_index_map()
        for bus_number in sorted(set(pf.case.gen[online, GEN_BUS].astype(int))):
            generators = online[pf.case.gen[online, GEN_BUS].astype(int) == bus_number]
            bus = bus_index[int(bus_number)]
            qspan = np.maximum(pf.case.gen[generators, QMAX] - pf.case.gen[generators, QMIN], 1.0)
            raw_span = np.maximum(
                pf.case.gen[generators, QMAX] - pf.case.gen[generators, QMIN], 0.0
            )
            total_span = float(np.sum(raw_span))
            q_total = float(np.sum(pf.case.gen[generators, QG]))
            q_minimum = float(np.sum(pf.case.gen[generators, QMIN]))
            q_maximum = float(np.sum(pf.case.gen[generators, QMAX]))
            if int(pf.case.bus[bus, BUS_TYPE]) == REF and (
                q_total < q_minimum or q_total > q_maximum
            ):
                q_coefficients = np.zeros(len(generators), dtype=float)
                q_coefficients[0] = 1.0
            elif total_span > 0.0:
                q_coefficients = raw_span / total_span
            else:
                q_coefficients = np.full(len(generators), 1.0 / len(generators))
            q_state_derivative = np.asarray(state_power_derivative[bus].imag, dtype=float)
            for local, generator in enumerate(generators):
                value = float(pf.case.gen[generator, QG])
                lower = float(pf.case.gen[generator, QMIN])
                upper = float(pf.case.gen[generator, QMAX])
                if lower - value > float(tolerances.generator_q_mvar):
                    append_constraint(
                        q_coefficients[local] * q_state_derivative / qspan[local],
                        (value - lower) / qspan[local],
                    )
                elif value - upper > float(tolerances.generator_q_mvar):
                    append_constraint(
                        q_coefficients[local] * q_state_derivative / qspan[local],
                        (value - upper) / qspan[local],
                    )

            if int(pf.case.bus[bus, BUS_TYPE]) == REF and len(generators):
                generator = int(generators[0])
                value = float(pf.case.gen[generator, PG])
                lower = float(pf.case.gen[generator, PMIN])
                upper = float(pf.case.gen[generator, PMAX])
                span = max(upper - lower, 1.0)
                p_state_derivative = np.asarray(state_power_derivative[bus].real, dtype=float)
                if lower - value > float(tolerances.generator_p_mw):
                    append_constraint(p_state_derivative / span, (value - lower) / span)
                elif value - upper > float(tolerances.generator_p_mw):
                    append_constraint(p_state_derivative / span, (value - upper) / span)

        if pf.branch is not None and state_dimension:
            admittance = build_ybus(pf.case)
            voltage = np.asarray(pf.voltage, dtype=complex)
            voltage_state_derivative = np.zeros((pf.case.n_bus, state_dimension), dtype=complex)
            for position, bus in enumerate(pvpq):
                voltage_state_derivative[int(bus), position] = 1j * voltage[int(bus)]
            for position, bus in enumerate(pq):
                bus = int(bus)
                voltage_state_derivative[bus, magnitude_offset + position] = voltage[bus] / max(
                    abs(voltage[bus]), 1e-15
                )
            current_from = np.asarray(admittance.y_from @ voltage).reshape(-1)
            current_to = np.asarray(admittance.y_to @ voltage).reshape(-1)
            current_from_derivative = np.asarray(admittance.y_from @ voltage_state_derivative)
            current_to_derivative = np.asarray(admittance.y_to @ voltage_state_derivative)
            from_rows = np.asarray([bus_index[int(value)] for value in branch[:, F_BUS]], dtype=int)
            to_rows = np.asarray([bus_index[int(value)] for value in branch[:, T_BUS]], dtype=int)
            base_mva = float(pf.case.base_mva)
            s_from_derivative = base_mva * (
                voltage_state_derivative[from_rows] * np.conj(current_from)[:, None]
                + voltage[from_rows, None] * np.conj(current_from_derivative)
            )
            s_to_derivative = base_mva * (
                voltage_state_derivative[to_rows] * np.conj(current_to)[:, None]
                + voltage[to_rows, None] * np.conj(current_to_derivative)
            )
            for index, row in enumerate(branch):
                rate = float(row[RATE_A])
                if row[BR_STATUS] <= 0.0 or rate <= 0.0:
                    continue
                loading = float(pf.branch.loading_percent[index])
                if loading - 100.0 <= float(tolerances.branch_loading_percent):
                    continue
                s_from = complex(pf.branch.s_from_mva[index])
                s_to = complex(pf.branch.s_to_mva[index])
                if abs(s_from) >= abs(s_to):
                    selected_flow = s_from
                    selected_derivative = s_from_derivative[index]
                else:
                    selected_flow = s_to
                    selected_derivative = s_to_derivative[index]
                if abs(selected_flow) <= 1e-15:
                    continue
                magnitude_derivative = np.real(
                    np.conj(selected_flow) * selected_derivative / abs(selected_flow)
                )
                append_constraint(magnitude_derivative / rate, loading / 100.0 - 1.0)

        if not rows:
            return np.zeros(state_dimension, dtype=float)
        matrix = np.vstack(rows)
        correction = np.linalg.lstsq(
            matrix,
            -np.asarray(residuals, dtype=float),
            rcond=None,
        )[0]
        return np.asarray(correction, dtype=float)

    def _control_forcing(self, pf, linearization) -> np.ndarray:
        pvpq = np.asarray(linearization.pvpq, dtype=int)
        pq = np.asarray(linearization.pq, dtype=int)
        forcing = np.zeros((len(pvpq) + len(pq), self.dimension), dtype=float)
        bus_index = pf.case.bus_index_map()
        pq_position = {int(bus): position for position, bus in enumerate(pq)}
        voltage = np.asarray(pf.voltage, dtype=complex)

        for column, control in enumerate(self.decoder.relaxed_control_derivatives()):
            scale = float(control.normalized_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            if control.kind == "vg":
                bus = bus_index[int(control.target)]
                if int(pf.case.bus[bus, BUS_TYPE]) not in {REF, PV}:
                    # Aggregate Q-limit switching converted this controlled bus to PQ. Its
                    # declared VG is no longer an active equation control in this local regime.
                    continue
                derivative = linearization.dS_dVm
                forcing[: len(pvpq), column] = (
                    self._selected_complex_derivative(derivative, pvpq, bus).real * scale
                )
                forcing[len(pvpq) :, column] = (
                    self._selected_complex_derivative(derivative, pq, bus).imag * scale
                )
            elif control.kind in {"shunt", "shunt_delta"}:
                bus = bus_index[int(control.target)]
                position = pq_position.get(bus)
                if position is not None:
                    forcing[len(pvpq) + position, column] = (
                        -(abs(voltage[bus]) ** 2) * scale / float(pf.case.base_mva)
                    )
            elif control.kind == "tap":
                row = np.asarray(pf.case.branch[int(control.target)], dtype=float)
                tap = float(row[TAP]) if float(row[TAP]) != 0.0 else 1.0
                series = 1.0 / complex(float(row[BR_R]), float(row[BR_X]))
                charging = 1j * float(row[BR_B]) / 2.0
                shift = np.deg2rad(float(row[SHIFT]))
                ratio = tap * np.exp(1j * shift)
                yff = (series + charging) / (ratio * np.conj(ratio))
                yft = -series / np.conj(ratio)
                ytf = -series / ratio
                from_bus = bus_index[int(row[F_BUS])]
                to_bus = bus_index[int(row[T_BUS])]
                current_derivative = np.zeros(pf.case.n_bus, dtype=complex)
                current_derivative[from_bus] = (-2.0 * yff / tap) * voltage[from_bus] + (
                    -yft / tap
                ) * voltage[to_bus]
                current_derivative[to_bus] = (-ytf / tap) * voltage[from_bus]
                power_derivative = voltage * np.conj(current_derivative)
                forcing[: len(pvpq), column] = power_derivative[pvpq].real * scale
                forcing[len(pvpq) :, column] = power_derivative[pq].imag * scale
        return forcing

    @staticmethod
    def _solve_control_sensitivity(jacobian, forcing: np.ndarray) -> np.ndarray:
        try:
            from scipy.sparse import issparse
            from scipy.sparse.linalg import spsolve

            if issparse(jacobian):
                solved = spsolve(jacobian, -forcing)
                return np.asarray(solved, dtype=float)
        except (ImportError, RuntimeError, ValueError, TypeError):
            pass
        dense = jacobian.toarray() if hasattr(jacobian, "toarray") else np.asarray(jacobian)
        return np.asarray(np.linalg.solve(dense, -forcing), dtype=float)

    def _build_control_linearization(
        self, normalized, scenario, pf
    ) -> ORPDControlLinearization | None:
        linearization = getattr(pf, "linearization", None)
        if not bool(getattr(pf, "converged", False)) or linearization is None:
            return None
        try:
            jacobian = linearization.jacobian
            dimension = int(jacobian.shape[0])
            if dimension < 1 or dimension > MAX_RETAINED_CONDITION_DIMENSION:
                return None
            forcing = self._control_forcing(pf, linearization)
            sensitivity = self._solve_control_sensitivity(jacobian, forcing)
            if sensitivity.shape != forcing.shape or not np.all(np.isfinite(sensitivity)):
                return None
            dense = jacobian.toarray() if hasattr(jacobian, "toarray") else np.asarray(jacobian)
            condition = float(np.linalg.cond(np.asarray(dense, dtype=float)))
            if not np.isfinite(condition):
                return None
            correction = self._constraint_state_correction(pf, linearization)
            constraint_residual = -self._matrix_vector(jacobian, correction)
            digest = hashlib.sha256()
            digest.update(COUNTED_CONTROL_LINEARIZATION_SCHEMA.encode("ascii"))
            digest.update(str(scenario.name).encode("utf-8"))
            digest.update(np.ascontiguousarray(normalized, dtype=np.float64).tobytes())
            digest.update(np.ascontiguousarray(pf.voltage, dtype=np.complex128).tobytes())
            return ORPDControlLinearization(
                COUNTED_CONTROL_LINEARIZATION_SCHEMA,
                digest.hexdigest(),
                jacobian,
                sensitivity,
                constraint_residual,
                condition,
                "exact analytic relaxed-control and active-constraint derivatives at the counted "
                "converged AC state; "
                "discrete candidates remain lattice-snapped and require a trusted FE",
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError, TypeError, AttributeError):
            return None

    def _evaluate(
        self,
        normalized,
        *,
        retain_context: bool,
        retain_control_linearization: bool,
    ):
        z = np.clip(np.asarray(normalized, float), 0, 1)
        controlled, physical = self.decoder.decode_reusable(z)
        values = []
        violations = []
        weights = []
        scenario_values = []
        comp_acc = {}
        constraint_acc = {}
        scenario_constraint_components = []
        retained_scenarios: list[ScenarioEvaluationContext] = []
        for scenario in self.scenarios:
            formulation_case = scenario.apply(controlled, copy_base=False)
            pf = run_ac_power_flow(
                formulation_case,
                self.config.power_flow,
                retain_linearization=retain_context and retain_control_linearization,
            )
            if retain_context:
                control_linearization = (
                    self._build_control_linearization(z, scenario, pf)
                    if retain_control_linearization
                    else None
                )
                retained_scenarios.append(
                    ScenarioEvaluationContext(
                        str(scenario.name),
                        float(scenario.weight),
                        pf,
                        control_linearization,
                    )
                )
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
        feasible = violation <= float(
            self.config.constraint_tolerances.feasibility_total
        ) and np.isfinite(robust)
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
            "normalized_decision_vector": z.astype(float).tolist(),
        }
        evaluation = Evaluation(
            robust,
            feasible,
            violation,
            components,
            physical,
            scenario_values,
            metadata,
            float(self.config.constraint_tolerances.feasibility_total),
        )
        context = (
            ORPDEvaluationContext(z.copy(), tuple(retained_scenarios)) if retain_context else None
        )
        return evaluation, context

    def solution_state(self, normalized):
        z = np.clip(np.asarray(normalized, float), 0, 1)
        controlled, physical = self.decoder.decode_reusable(z)
        records = []
        for sc in self.scenarios:
            formulation_case = sc.apply(controlled, copy_base=False)
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
                "l_index_max": float(
                    kessel_glavitsch_l_index(
                        pf.case, pf.voltage, partition_case=formulation_case
                    ).maximum
                )
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
