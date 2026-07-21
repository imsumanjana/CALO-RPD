"""End-to-end tensor-resident ORPD evaluation for CALO-RPD Studio v3.4.

The module keeps candidate populations, mixed-variable decoding, scenario expansion, AC Newton-
Raphson power flow, objective/constraint evaluation, and robust aggregation on the selected
PyTorch device.  Host materialisation occurs once per completed population request to create the
stable public ``Evaluation`` objects used by the database, GUI, and publication pipeline.

The implementation intentionally retains FP64 arithmetic and the same formulations used by the
trusted CPU reference.  Candidate-specific PV-to-PQ switching is handled in grouped tensor batches
by bus-type pattern instead of falling back to one Python power-flow solve per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from calo_rpd_studio.orpd.objectives import ObjectiveKind
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.power_system.case_model import (
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    BS,
    BUS_TYPE,
    F_BUS,
    GEN_BUS,
    GEN_STATUS,
    GS,
    PD,
    PG,
    PMAX,
    PMIN,
    PQ,
    PV,
    QD,
    QG,
    QMAX,
    QMIN,
    RATE_A,
    REF,
    SHIFT,
    TAP,
    T_BUS,
    VA,
    VM,
    VMAX,
    VMIN,
)
from calo_rpd_studio.robustness.robust_objectives import RobustAggregation
from calo_rpd_studio.robustness.cvar import weighted_cvar_torch

from .torch_power_flow import solve_newton_raphson_batch_torch


OBJECTIVE_COMPONENT_NAMES = (
    "active_power_loss_mw",
    "voltage_deviation_pu",
    "l_index_max",
    "scenario_objective_mean",
    "scenario_objective_std",
)
CONSTRAINT_COMPONENT_NAMES = (
    "bus_voltage",
    "generator_q",
    "generator_p",
    "branch_thermal",
    "power_flow",
)


def _torch():
    import torch

    return torch


@dataclass(slots=True)
class DeviceResidentBatch:
    """Tensor result for one population request.

    All fields remain on the execution device until ``to_evaluations`` is called.  This lets
    optimizer kernels inspect objective/violation/feasibility tensors without an intermediate
    CUDA->CPU->CUDA round trip.
    """

    objective: Any
    violation: Any
    feasible: Any
    normalized_values: Any
    decoded_values: Any
    scenario_values: Any
    objective_components: dict[str, Any]
    constraint_components: dict[str, Any]
    scenario_constraint_components: Any
    variable_names: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def count(self) -> int:
        return int(self.objective.shape[0])

    def to_evaluations(self) -> list[Evaluation]:
        """Materialise the complete population with one packed host transfer."""
        torch = _torch()
        objective_columns = [self.objective_components[name] for name in OBJECTIVE_COMPONENT_NAMES]
        constraint_columns = [
            self.constraint_components[name] for name in CONSTRAINT_COMPONENT_NAMES
        ]
        packed = torch.cat(
            (
                self.objective[:, None],
                self.violation[:, None],
                self.feasible.to(self.objective.dtype)[:, None],
                self.normalized_values,
                self.decoded_values,
                self.scenario_values,
                torch.stack(objective_columns, dim=1),
                torch.stack(constraint_columns, dim=1),
                self.scenario_constraint_components.reshape(self.count, -1),
            ),
            dim=1,
        )
        host = np.asarray(packed.detach().to("cpu"), dtype=float)
        n_normalized = int(self.normalized_values.shape[1])
        n_controls = int(self.decoded_values.shape[1])
        n_scenarios = int(self.scenario_values.shape[1])
        cursor = 3
        normalized = host[:, cursor : cursor + n_normalized]
        cursor += n_normalized
        controls = host[:, cursor : cursor + n_controls]
        cursor += n_controls
        scenario_values = host[:, cursor : cursor + n_scenarios]
        cursor += n_scenarios
        objective_matrix = host[:, cursor : cursor + len(OBJECTIVE_COMPONENT_NAMES)]
        cursor += len(OBJECTIVE_COMPONENT_NAMES)
        constraint_matrix = host[:, cursor : cursor + len(CONSTRAINT_COMPONENT_NAMES)]
        cursor += len(CONSTRAINT_COMPONENT_NAMES)
        scenario_constraints = host[:, cursor:].reshape(
            self.count, n_scenarios, len(CONSTRAINT_COMPONENT_NAMES)
        )

        out: list[Evaluation] = []
        for row in range(self.count):
            physical = {
                name: float(controls[row, index]) for index, name in enumerate(self.variable_names)
            }
            components = {
                name: float(objective_matrix[row, index])
                for index, name in enumerate(OBJECTIVE_COMPONENT_NAMES)
            }
            constraints = {
                name: float(constraint_matrix[row, index])
                for index, name in enumerate(CONSTRAINT_COMPONENT_NAMES)
            }
            scenario_constraint_records = [
                {
                    name: float(scenario_constraints[row, scenario, index])
                    for index, name in enumerate(CONSTRAINT_COMPONENT_NAMES)
                }
                for scenario in range(n_scenarios)
            ]
            metadata = dict(self.metadata)
            metadata.update(
                {
                    "constraint_components": constraints,
                    "scenario_constraint_components": scenario_constraint_records,
                    "host_materializations_per_population": 1,
                    "normalized_decision_vector": normalized[row].astype(float).tolist(),
                }
            )
            out.append(
                Evaluation(
                    float(host[row, 0]),
                    bool(host[row, 2] > 0.5),
                    float(host[row, 1]),
                    components,
                    physical,
                    scenario_values[row].astype(float).tolist(),
                    metadata,
                )
            )
        return out


class DeviceResidentORPDEvaluator:
    """Prepared device cache and tensor-resident ORPD evaluation pipeline."""

    def __init__(self, problem) -> None:
        torch = _torch()
        self.problem = problem
        self.case = problem.case
        self.config = problem.config
        self.scenarios = tuple(problem.scenarios)
        self.decoder = problem.decoder
        self.device = torch.device(problem.device)
        self.dtype = problem.dtype
        self.cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        self.options = problem.power_flow_options
        self.variable_names = tuple(variable.name for variable in self.decoder.variables)
        self._prepare_host_arrays()
        self._prepare_device_tensors()

    def _prepare_host_arrays(self) -> None:
        cases = [scenario.apply(self.case) for scenario in self.scenarios]
        if not cases:
            raise ValueError("At least one ORPD scenario is required")
        first = cases[0]
        self.scenario_cases = cases
        self.scenario_count = len(cases)
        self.n_bus = first.n_bus
        self.n_branch = first.n_branch
        self.base_mva_np = np.asarray([case.base_mva for case in cases], dtype=float)
        index = first.bus_index_map()
        self.fidx_np = np.asarray([index[int(v)] for v in first.branch[:, F_BUS]], dtype=np.int64)
        self.tidx_np = np.asarray([index[int(v)] for v in first.branch[:, T_BUS]], dtype=np.int64)
        for case in cases[1:]:
            if case.n_bus != self.n_bus or case.n_branch != self.n_branch:
                raise ValueError("All scenarios must preserve network dimensions")
            if not np.array_equal(
                case.branch[:, F_BUS].astype(int), first.branch[:, F_BUS].astype(int)
            ):
                raise ValueError("All scenarios must preserve branch from-bus topology")
            if not np.array_equal(
                case.branch[:, T_BUS].astype(int), first.branch[:, T_BUS].astype(int)
            ):
                raise ValueError("All scenarios must preserve branch to-bus topology")

        self.bus_np = np.stack([case.bus for case in cases], axis=0)
        self.gen_np = np.stack([case.gen for case in cases], axis=0)
        self.branch_np = np.stack([case.branch for case in cases], axis=0)

        shape = (self.scenario_count, self.n_bus)
        pg = np.zeros(shape, dtype=float)
        qg = np.zeros(shape, dtype=float)
        pmin = np.zeros(shape, dtype=float)
        pmax = np.zeros(shape, dtype=float)
        qmin = np.zeros(shape, dtype=float)
        qmax = np.zeros(shape, dtype=float)
        gen_mask = np.zeros(shape, dtype=bool)
        for scenario_index, case in enumerate(cases):
            case_index = case.bus_index_map()
            for generator in case.gen[case.gen[:, GEN_STATUS] > 0]:
                bus_index = case_index[int(generator[GEN_BUS])]
                gen_mask[scenario_index, bus_index] = True
                pg[scenario_index, bus_index] += float(generator[PG])
                qg[scenario_index, bus_index] += float(generator[QG])
                pmin[scenario_index, bus_index] += float(generator[PMIN])
                pmax[scenario_index, bus_index] += float(generator[PMAX])
                qmin[scenario_index, bus_index] += float(generator[QMIN])
                qmax[scenario_index, bus_index] += float(generator[QMAX])
        self.pg_np = pg
        self.qg_np = qg
        self.pmin_np = pmin
        self.pmax_np = pmax
        self.qmin_np = qmin
        self.qmax_np = qmax
        self.gen_mask_np = gen_mask

        # Map the reference decoder targets into dense tensor indices once.
        reference_index = self.case.bus_index_map()
        action_records = []
        for action in self.decoder._actions:
            kind, target, lower, upper, lattice = action
            if kind in {"vg", "shunt", "shunt_delta"}:
                target_index = int(reference_index[int(target)])
            else:
                target_index = int(target)
            action_records.append((str(kind), target_index, float(lower), float(upper), lattice))
        self.actions = tuple(action_records)

    def _prepare_device_tensors(self) -> None:
        torch = _torch()
        device = self.device
        dtype = self.dtype
        branch = self.branch_np
        bus = self.bus_np
        self.base_mva = torch.as_tensor(self.base_mva_np, dtype=dtype, device=device)
        self.fidx = torch.as_tensor(self.fidx_np, dtype=torch.long, device=device)
        self.tidx = torch.as_tensor(self.tidx_np, dtype=torch.long, device=device)
        self.branch_r = torch.as_tensor(branch[:, :, BR_R], dtype=dtype, device=device)
        self.branch_x = torch.as_tensor(branch[:, :, BR_X], dtype=dtype, device=device)
        self.branch_b = torch.as_tensor(branch[:, :, BR_B], dtype=dtype, device=device)
        self.branch_status = torch.as_tensor(
            branch[:, :, BR_STATUS] > 0, dtype=torch.bool, device=device
        )
        tap = branch[:, :, TAP].copy()
        tap[tap == 0] = 1.0
        self.base_tap = torch.as_tensor(tap, dtype=dtype, device=device)
        self.branch_shift = torch.deg2rad(
            torch.as_tensor(branch[:, :, SHIFT], dtype=dtype, device=device)
        )
        self.rate_a = torch.as_tensor(branch[:, :, RATE_A], dtype=dtype, device=device)

        self.base_vm = torch.as_tensor(bus[:, :, VM], dtype=dtype, device=device)
        self.base_va = torch.deg2rad(torch.as_tensor(bus[:, :, VA], dtype=dtype, device=device))
        self.base_bs = torch.as_tensor(bus[:, :, BS], dtype=dtype, device=device)
        self.base_gs = torch.as_tensor(bus[:, :, GS], dtype=dtype, device=device)
        self.pd = torch.as_tensor(bus[:, :, PD], dtype=dtype, device=device)
        self.qd = torch.as_tensor(bus[:, :, QD], dtype=dtype, device=device)
        self.vmin = torch.as_tensor(bus[:, :, VMIN], dtype=dtype, device=device)
        self.vmax = torch.as_tensor(bus[:, :, VMAX], dtype=dtype, device=device)
        self.base_types = torch.as_tensor(
            bus[:, :, BUS_TYPE].astype(np.int64), dtype=torch.long, device=device
        )
        self.original_pq = self.base_types == int(PQ)
        self.original_gen = (self.base_types == int(PV)) | (self.base_types == int(REF))

        self.pg_spec = torch.as_tensor(self.pg_np, dtype=dtype, device=device)
        self.qg_spec = torch.as_tensor(self.qg_np, dtype=dtype, device=device)
        self.pmin = torch.as_tensor(self.pmin_np, dtype=dtype, device=device)
        self.pmax = torch.as_tensor(self.pmax_np, dtype=dtype, device=device)
        self.qmin = torch.as_tensor(self.qmin_np, dtype=dtype, device=device)
        self.qmax = torch.as_tensor(self.qmax_np, dtype=dtype, device=device)
        self.gen_mask = torch.as_tensor(self.gen_mask_np, dtype=torch.bool, device=device)
        self.weights = torch.as_tensor(
            [float(scenario.weight) for scenario in self.scenarios], dtype=dtype, device=device
        )
        self.weights = self.weights / torch.clamp(self.weights.sum(), min=torch.finfo(dtype).eps)

        # Discrete lattices are cached once on the execution device.
        lattices: list[Any | None] = []
        for _kind, _target, _lower, _upper, lattice in self.actions:
            lattices.append(
                None
                if lattice is None
                else torch.as_tensor(np.asarray(lattice, dtype=float), dtype=dtype, device=device)
            )
        self.lattices = tuple(lattices)

    def decode(self, normalized):
        torch = _torch()
        if isinstance(normalized, torch.Tensor):
            z = normalized.to(device=self.device, dtype=self.dtype)
        else:
            z = torch.as_tensor(
                np.asarray(normalized, dtype=float), dtype=self.dtype, device=self.device
            )
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.shape[1] != len(self.actions):
            raise ValueError(f"Expected {len(self.actions)} decision columns, got {tuple(z.shape)}")
        z = torch.clamp(z, 0.0, 1.0)
        decoded_columns = []
        for column, (_kind, _target, lower, upper, _lattice) in enumerate(self.actions):
            lattice = self.lattices[column]
            if lattice is None:
                decoded = lower + z[:, column] * (upper - lower)
            else:
                index = torch.floor(z[:, column] * lattice.numel()).long()
                index = torch.clamp(index, 0, lattice.numel() - 1)
                decoded = lattice[index]
            decoded_columns.append(decoded)
        decoded = (
            torch.stack(decoded_columns, dim=1)
            if decoded_columns
            else torch.empty((z.shape[0], 0), dtype=self.dtype, device=self.device)
        )
        batch = z.shape[0]
        vm = self.base_vm.unsqueeze(0).expand(batch, -1, -1).clone()
        tap = self.base_tap.unsqueeze(0).expand(batch, -1, -1).clone()
        bs = self.base_bs.unsqueeze(0).expand(batch, -1, -1).clone()
        for column, (kind, target, _lower, _upper, _lattice) in enumerate(self.actions):
            value = decoded[:, column][:, None]
            if kind == "vg":
                vm[:, :, target] = value
            elif kind == "tap":
                tap[:, :, target] = value
            elif kind == "shunt":
                bs[:, :, target] = value
            elif kind == "shunt_delta":
                bs[:, :, target] = self.base_bs[:, target].unsqueeze(0) + value
        return z, decoded, vm, tap, bs

    def _admittance(self, tap, bs):
        torch = _torch()
        batch = tap.shape[0]
        scenarios = self.scenario_count
        rows = batch * scenarios
        r = self.branch_r.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_branch)
        x = self.branch_x.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_branch)
        line_b = self.branch_b.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_branch)
        status = self.branch_status.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_branch)
        tap_flat = tap.reshape(rows, self.n_branch)
        shift = self.branch_shift.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_branch)
        z = torch.complex(r, x)
        eps = torch.finfo(self.dtype).eps
        y = torch.where(torch.abs(z) > eps, 1.0 / z, torch.zeros_like(z))
        y = torch.where(status, y, torch.zeros_like(y))
        charging = torch.where(
            status,
            torch.complex(torch.zeros_like(line_b), line_b / 2.0),
            torch.zeros_like(y),
        )
        a = torch.polar(tap_flat, shift)
        yff = (y + charging) / (a * torch.conj(a))
        yft = -y / torch.conj(a)
        ytf = -y / a
        ytt = y + charging

        ybus = torch.zeros((rows, self.n_bus, self.n_bus), dtype=self.cdtype, device=self.device)
        row_index = torch.arange(rows, device=self.device)[:, None].expand(rows, self.n_branch)
        f = self.fidx[None, :].expand(rows, self.n_branch)
        t = self.tidx[None, :].expand(rows, self.n_branch)
        ybus.index_put_((row_index, f, f), yff, accumulate=True)
        ybus.index_put_((row_index, f, t), yft, accumulate=True)
        ybus.index_put_((row_index, t, f), ytf, accumulate=True)
        ybus.index_put_((row_index, t, t), ytt, accumulate=True)
        gs = self.base_gs.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_bus)
        bs_flat = bs.reshape(rows, self.n_bus)
        base = self.base_mva.unsqueeze(0).expand(batch, -1).reshape(rows, 1)
        shunt = torch.complex(gs, bs_flat) / base
        diagonal = torch.arange(self.n_bus, device=self.device)
        ybus[:, diagonal, diagonal] += shunt
        return ybus, yff, yft, ytf, ytt

    def _specified_power(self, batch: int):
        torch = _torch()
        base = self.base_mva.unsqueeze(0).expand(batch, -1).reshape(batch * self.scenario_count, 1)
        pg = (
            self.pg_spec.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_bus)
        )
        qg = (
            self.qg_spec.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_bus)
        )
        pd = (
            self.pd.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_bus)
        )
        qd = (
            self.qd.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_bus)
        )
        return torch.complex((pg - pd) / base, (qg - qd) / base)

    def _initial_voltage(self, vm):
        va = self.base_va.unsqueeze(0).expand(vm.shape[0], -1, -1)
        return _torch().polar(vm, va).reshape(vm.shape[0] * self.scenario_count, self.n_bus)

    def _required_generation(self, voltage, ybus, batch: int):
        torch = _torch()
        base = self.base_mva.unsqueeze(0).expand(batch, -1).reshape(batch * self.scenario_count, 1)
        pd = (
            self.pd.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_bus)
        )
        qd = (
            self.qd.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_bus)
        )
        injection = voltage * torch.conj(torch.bmm(ybus, voltage.unsqueeze(-1)).squeeze(-1)) * base
        return injection.real + pd, injection.imag + qd

    def _solve_power_flow(self, ybus, sbus, v0, batch: int):
        """Grouped tensor PV-to-PQ switching without per-candidate CPU fallback."""
        torch = _torch()
        rows = batch * self.scenario_count
        types = self.base_types.unsqueeze(0).expand(batch, -1, -1).reshape(rows, self.n_bus).clone()
        voltage = v0.clone()
        final_converged = torch.zeros(rows, dtype=torch.bool, device=self.device)
        failed = torch.zeros(rows, dtype=torch.bool, device=self.device)
        iterations = torch.zeros(rows, dtype=torch.long, device=self.device)
        mismatch = torch.full((rows,), float("inf"), dtype=self.dtype, device=self.device)
        q_rounds = torch.zeros(rows, dtype=torch.long, device=self.device)
        current_sbus = sbus.clone()

        for q_round in range(int(self.options.max_q_limit_rounds) + 1):
            active_rows = torch.where((~final_converged) & (~failed))[0]
            if active_rows.numel() == 0:
                break
            active_patterns = types[active_rows]
            unique_patterns, inverse = torch.unique(active_patterns, dim=0, return_inverse=True)
            round_converged = torch.zeros(rows, dtype=torch.bool, device=self.device)
            for pattern_index in range(int(unique_patterns.shape[0])):
                local = torch.where(inverse == pattern_index)[0]
                group_rows = active_rows[local]
                pattern = unique_patterns[pattern_index]
                ref = torch.where(pattern == int(REF))[0]
                pv = torch.where(pattern == int(PV))[0]
                pq = torch.where(pattern == int(PQ))[0]
                if ref.numel() != 1:
                    failed[group_rows] = True
                    continue
                conv, bad, solved, iters, max_mm, _history = solve_newton_raphson_batch_torch(
                    ybus[group_rows],
                    current_sbus[group_rows],
                    voltage[group_rows],
                    ref,
                    pv,
                    pq,
                    tolerance=float(self.options.tolerance),
                    max_iterations=int(self.options.max_iterations),
                    collect_history=False,
                )
                voltage[group_rows] = solved
                iterations[group_rows] += iters
                mismatch[group_rows] = max_mm
                group_failed = bad | (~conv)
                failed[group_rows[group_failed]] = True
                round_converged[group_rows[conv & (~bad)]] = True

            solved_rows = torch.where(round_converged & (~failed))[0]
            if solved_rows.numel() == 0:
                continue
            # Use direct per-row load/base tensors for arbitrary grouped row subsets.
            scenario_index = solved_rows % self.scenario_count
            base_rows = self.base_mva[scenario_index][:, None]
            qd_rows = self.qd[scenario_index]
            injection = (
                voltage[solved_rows]
                * torch.conj(
                    torch.bmm(ybus[solved_rows], voltage[solved_rows].unsqueeze(-1)).squeeze(-1)
                )
                * base_rows
            )
            qg = injection.imag + qd_rows
            current_types = types[solved_rows]
            qmin_rows = self.qmin[scenario_index]
            qmax_rows = self.qmax[scenario_index]
            pv_mask = current_types == int(PV)
            high = pv_mask & (qg > qmax_rows + float(self.options.q_limit_tolerance_mvar))
            low = pv_mask & (qg < qmin_rows - float(self.options.q_limit_tolerance_mvar))
            violated = high | low
            has_violation = torch.any(violated, dim=1)
            clean_rows = solved_rows[~has_violation]
            final_converged[clean_rows] = True
            if not bool(self.options.enforce_q_limits):
                final_converged[solved_rows] = True
                continue
            switching_rows = solved_rows[has_violation]
            if switching_rows.numel() == 0:
                continue
            if q_round >= int(self.options.max_q_limit_rounds):
                failed[switching_rows] = True
                continue
            local_violated = violated[has_violation]
            local_high = high[has_violation]
            switch_scenarios = switching_rows % self.scenario_count
            limit = torch.where(
                local_high, self.qmax[switch_scenarios], self.qmin[switch_scenarios]
            )
            types[switching_rows] = torch.where(
                local_violated,
                torch.full_like(types[switching_rows], int(PQ)),
                types[switching_rows],
            )
            imag = current_sbus[switching_rows].imag.clone()
            specified_q = (limit - self.qd[switch_scenarios]) / self.base_mva[switch_scenarios][
                :, None
            ]
            imag = torch.where(local_violated, specified_q, imag)
            current_sbus[switching_rows] = torch.complex(current_sbus[switching_rows].real, imag)
            q_rounds[switching_rows] = q_round + 1

        return final_converged, failed, voltage, iterations, mismatch, q_rounds, types

    def _branch_outputs(self, voltage, yff, yft, ytf, ytt, batch: int):
        torch = _torch()
        vf = voltage[:, self.fidx]
        vt = voltage[:, self.tidx]
        current_from = yff * vf + yft * vt
        current_to = ytf * vf + ytt * vt
        base = self.base_mva.unsqueeze(0).expand(batch, -1).reshape(batch * self.scenario_count, 1)
        s_from = vf * torch.conj(current_from) * base
        s_to = vt * torch.conj(current_to) * base
        rate = (
            self.rate_a.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * self.scenario_count, self.n_branch)
        )
        magnitude = torch.maximum(torch.abs(s_from), torch.abs(s_to))
        loading = torch.where(rate > 0, 100.0 * magnitude / rate, torch.zeros_like(rate))
        loss = torch.sum((s_from + s_to).real, dim=1)
        return s_from, s_to, loading, loss

    def _l_index(self, voltage, ybus, batch: int):
        torch = _torch()
        values = torch.zeros(batch * self.scenario_count, dtype=self.dtype, device=self.device)
        for scenario_index in range(self.scenario_count):
            rows = torch.arange(batch, device=self.device) * self.scenario_count + scenario_index
            types = self.base_types[scenario_index]
            load = torch.where(types == int(PQ))[0]
            gen = torch.where((types == int(PV)) | (types == int(REF)))[0]
            if load.numel() == 0 or gen.numel() == 0:
                continue
            matrices = ybus[rows]
            yll = matrices[:, load][:, :, load]
            ylg = matrices[:, load][:, :, gen]
            rhs = ylg
            solution, info = torch.linalg.solve_ex(yll, rhs, check_errors=False)
            f = -solution
            vg = voltage[rows][:, gen]
            vl = voltage[rows][:, load]
            numerator = torch.bmm(f, vg.unsqueeze(-1)).squeeze(-1)
            local = torch.abs(1.0 - numerator / vl)
            maxima = torch.max(local, dim=1).values
            maxima = torch.where(info == 0, maxima, torch.full_like(maxima, float("inf")))
            values[rows] = maxima
        return values

    def _robust(self, scenario_values):
        torch = _torch()
        weights = self.weights[None, :]
        mean = torch.sum(scenario_values * weights, dim=1)
        aggregation = self.config.robust.aggregation
        if aggregation is RobustAggregation.EXPECTED:
            return mean
        if aggregation is RobustAggregation.MEAN_RISK:
            std = torch.sqrt(torch.sum(weights * (scenario_values - mean[:, None]).square(), dim=1))
            return mean + float(self.config.robust.risk_lambda) * std
        if aggregation is RobustAggregation.WORST_CASE:
            return torch.max(scenario_values, dim=1).values
        return weighted_cvar_torch(
            scenario_values,
            self.weights,
            float(self.config.robust.cvar_alpha),
        )

    def evaluate_tensor(self, normalized) -> DeviceResidentBatch:
        torch = _torch()
        with torch.inference_mode():
            z, decoded, vm, tap, bs = self.decode(normalized)
            batch = int(z.shape[0])
            ybus, yff, yft, ytf, ytt = self._admittance(tap, bs)
            sbus = self._specified_power(batch)
            v0 = self._initial_voltage(vm)
            converged, failed, voltage, iterations, mismatch, q_rounds, _final_types = (
                self._solve_power_flow(ybus, sbus, v0, batch)
            )
            pg, qg = self._required_generation(voltage, ybus, batch)
            _s_from, _s_to, loading, loss = self._branch_outputs(voltage, yff, yft, ytf, ytt, batch)
            l_index = self._l_index(voltage, ybus, batch)

            vm_final = torch.abs(voltage)
            original_pq = (
                self.original_pq.unsqueeze(0)
                .expand(batch, -1, -1)
                .reshape(batch * self.scenario_count, self.n_bus)
            )
            voltage_deviation = torch.sum(
                torch.where(original_pq, torch.abs(vm_final - 1.0), torch.zeros_like(vm_final)),
                dim=1,
            )

            lower = (
                self.vmin.unsqueeze(0)
                .expand(batch, -1, -1)
                .reshape(batch * self.scenario_count, self.n_bus)
            )
            upper = (
                self.vmax.unsqueeze(0)
                .expand(batch, -1, -1)
                .reshape(batch * self.scenario_count, self.n_bus)
            )
            span = torch.clamp(upper - lower, min=1e-12)
            bus_voltage = torch.sum(
                torch.relu(lower - vm_final) / span + torch.relu(vm_final - upper) / span,
                dim=1,
            )
            scenario_indices = (
                torch.arange(batch * self.scenario_count, device=self.device) % self.scenario_count
            )
            gen_mask = self.gen_mask[scenario_indices]
            pmin = self.pmin[scenario_indices]
            pmax = self.pmax[scenario_indices]
            qmin = self.qmin[scenario_indices]
            qmax = self.qmax[scenario_indices]
            pspan = torch.clamp(pmax - pmin, min=1.0)
            qspan = torch.clamp(qmax - qmin, min=1.0)
            generator_p = torch.sum(
                torch.where(
                    gen_mask,
                    torch.relu(pmin - pg) / pspan + torch.relu(pg - pmax) / pspan,
                    torch.zeros_like(pg),
                ),
                dim=1,
            )
            generator_q = torch.sum(
                torch.where(
                    gen_mask,
                    torch.relu(qmin - qg) / qspan + torch.relu(qg - qmax) / qspan,
                    torch.zeros_like(qg),
                ),
                dim=1,
            )
            rated = self.rate_a[scenario_indices] > 0
            branch_thermal = torch.sum(
                torch.where(rated, torch.relu(loading - 100.0) / 100.0, torch.zeros_like(loading)),
                dim=1,
            )
            power_flow = torch.where(
                converged & (~failed),
                torch.zeros_like(loss),
                torch.full_like(loss, float("inf")),
            )
            finite_mask = converged & (~failed)
            loss = torch.where(finite_mask, loss, torch.full_like(loss, float("inf")))
            voltage_deviation = torch.where(
                finite_mask, voltage_deviation, torch.full_like(voltage_deviation, float("inf"))
            )
            l_index = torch.where(finite_mask, l_index, torch.full_like(l_index, float("inf")))
            bus_voltage = torch.where(
                finite_mask, bus_voltage, torch.full_like(bus_voltage, float("inf"))
            )
            generator_p = torch.where(
                finite_mask, generator_p, torch.full_like(generator_p, float("inf"))
            )
            generator_q = torch.where(
                finite_mask, generator_q, torch.full_like(generator_q, float("inf"))
            )
            branch_thermal = torch.where(
                finite_mask, branch_thermal, torch.full_like(branch_thermal, float("inf"))
            )

            objective_kind = self.config.objective.kind
            if objective_kind is ObjectiveKind.ACTIVE_POWER_LOSS:
                scenario_objective = loss
            elif objective_kind is ObjectiveKind.VOLTAGE_DEVIATION:
                scenario_objective = voltage_deviation
            elif objective_kind is ObjectiveKind.L_INDEX:
                scenario_objective = l_index
            else:
                objective_config = self.config.objective
                scenario_objective = (
                    float(objective_config.weight_loss)
                    * loss
                    / max(float(objective_config.loss_scale), 1e-15)
                    + float(objective_config.weight_voltage_deviation)
                    * voltage_deviation
                    / max(float(objective_config.voltage_deviation_scale), 1e-15)
                    + float(objective_config.weight_l_index)
                    * l_index
                    / max(float(objective_config.l_index_scale), 1e-15)
                )

            scenario_objective = scenario_objective.reshape(batch, self.scenario_count)
            scenario_constraints = torch.stack(
                (bus_voltage, generator_q, generator_p, branch_thermal, power_flow), dim=1
            ).reshape(batch, self.scenario_count, len(CONSTRAINT_COMPONENT_NAMES))
            scenario_violation = torch.sum(scenario_constraints, dim=2)
            robust_objective = self._robust(scenario_objective)
            weights = self.weights[None, :]
            violation = torch.sum(weights * scenario_violation, dim=1)
            feasible = (
                torch.all(finite_mask.reshape(batch, self.scenario_count), dim=1)
                & torch.isfinite(robust_objective)
                & (violation <= 1e-12)
            )
            objective_mean = torch.sum(weights * scenario_objective, dim=1)
            objective_std = torch.sqrt(
                torch.sum(weights * (scenario_objective - objective_mean[:, None]).square(), dim=1)
            )
            objective_components = {
                "active_power_loss_mw": torch.sum(
                    weights * loss.reshape(batch, self.scenario_count), dim=1
                ),
                "voltage_deviation_pu": torch.sum(
                    weights * voltage_deviation.reshape(batch, self.scenario_count), dim=1
                ),
                "l_index_max": torch.sum(
                    weights * l_index.reshape(batch, self.scenario_count), dim=1
                ),
                "scenario_objective_mean": objective_mean,
                "scenario_objective_std": objective_std,
            }
            constraint_components = {
                name: torch.sum(weights * scenario_constraints[:, :, index], dim=1)
                for index, name in enumerate(CONSTRAINT_COMPONENT_NAMES)
            }
            metadata = {
                "scenario_count": self.scenario_count,
                "scientific_backend": "torch_batched_dense_newton_raphson",
                "scientific_backend_engine": "torch_fp64_device_resident_newton_raphson",
                "throughput_engine_version": "3.3",
                "compute_device": str(self.device),
                "dtype": "float64" if self.dtype == torch.float64 else str(self.dtype),
                "device_resident_execution": True,
                "candidate_specific_q_limit_fallback": False,
                "grouped_tensor_q_limit_switching": True,
                "device_state_retained_until_population_complete": True,
                "solver_diagnostics_retained_on_device": True,
            }
            return DeviceResidentBatch(
                robust_objective,
                violation,
                feasible,
                z,
                decoded,
                scenario_objective,
                objective_components,
                constraint_components,
                scenario_constraints,
                self.variable_names,
                metadata,
            )
