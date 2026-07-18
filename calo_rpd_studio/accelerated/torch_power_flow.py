"""Double-precision PyTorch AC Newton-Raphson power flow.

This implementation mirrors the trusted CPU formulation while executing dense Jacobian assembly,
linear solves, branch-flow calculation, and L-index calculation on the requested PyTorch device.
It supports generator aggregate reactive-power limits through iterative PV-to-PQ switching.

The solver is intentionally explicit rather than opaque: every candidate has an independent
convergence mask/status, failed solves are reported as infeasible, and no lower-precision autocast
is permitted for publication runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from calo_rpd_studio.power_system.case_model import (
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    BS,
    BUS_I,
    BUS_TYPE,
    F_BUS,
    GEN_BUS,
    GEN_STATUS,
    GS,
    PG,
    PQ,
    PV,
    QD,
    QG,
    RATE_A,
    REF,
    SHIFT,
    TAP,
    T_BUS,
    VA,
    VG,
    VM,
)
from calo_rpd_studio.power_system.pv_pq_switching import (
    aggregate_q_limits,
    distribute_reactive_power,
    online_generators_at_bus,
)


@dataclass(slots=True)
class TorchPowerFlowOptions:
    tolerance: float = 1e-8
    max_iterations: int = 30
    enforce_q_limits: bool = True
    max_q_limit_rounds: int = 10
    q_limit_tolerance_mvar: float = 1e-6


@dataclass(slots=True)
class TorchBranchResult:
    s_from_mva: Any
    s_to_mva: Any
    loading_percent: Any
    total_loss_mw: float


@dataclass(slots=True)
class TorchPowerFlowResult:
    converged: bool
    case: Any
    voltage: Any
    vm_pu: Any
    va_deg: Any
    iterations: int
    q_limit_rounds: int
    max_mismatch: float
    mismatch_history: list[float]
    branch: TorchBranchResult | None
    ybus: Any
    actual_pg_mw: Any | None = None
    actual_qg_mvar: Any | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_loss_mw(self) -> float:
        return float("inf") if self.branch is None else float(self.branch.total_loss_mw)


def _torch():
    import torch

    return torch


def _types(case):
    types = case.bus[:, BUS_TYPE].astype(int)
    ref = np.where(types == REF)[0]
    pv = np.where(types == PV)[0]
    pq = np.where(types == PQ)[0]
    if ref.size != 1:
        raise ValueError(f"Exactly one reference bus is required; found {ref.size}")
    return ref.astype(int), pv.astype(int), pq.astype(int)


def build_dense_admittance(case, device: str, dtype=None):
    torch = _torch()
    dtype = dtype or torch.float64
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    n = case.n_bus
    nl = case.n_branch
    idx = case.bus_index_map()
    ybus = torch.zeros((n, n), dtype=cdtype, device=device)
    yf = torch.zeros((nl, n), dtype=cdtype, device=device)
    yt = torch.zeros((nl, n), dtype=cdtype, device=device)
    for k, br in enumerate(case.branch):
        if br[BR_STATUS] <= 0:
            continue
        f = idx[int(br[F_BUS])]
        t = idx[int(br[T_BUS])]
        z = complex(float(br[BR_R]), float(br[BR_X]))
        y = 0j if abs(z) == 0 else 1 / z
        b = 1j * float(br[BR_B]) / 2
        tap = float(br[TAP]) if float(br[TAP]) != 0 else 1.0
        shift = np.deg2rad(float(br[SHIFT]))
        a = tap * np.exp(1j * shift)
        yff = (y + b) / (a * np.conj(a))
        yft = -y / np.conj(a)
        ytf = -y / a
        ytt = y + b
        vals = torch.tensor([yff, yft, ytf, ytt], dtype=cdtype, device=device)
        ybus[f, f] += vals[0]
        ybus[f, t] += vals[1]
        ybus[t, f] += vals[2]
        ybus[t, t] += vals[3]
        yf[k, f] = vals[0]
        yf[k, t] = vals[1]
        yt[k, f] = vals[2]
        yt[k, t] = vals[3]
    sh = torch.as_tensor((case.bus[:, GS] + 1j * case.bus[:, BS]) / case.base_mva, dtype=cdtype, device=device)
    ybus = ybus + torch.diag(sh)
    return ybus, yf, yt


def _sbus(case, device: str, dtype):
    torch = _torch()
    pg = torch.zeros(case.n_bus, dtype=dtype, device=device)
    qg = torch.zeros(case.n_bus, dtype=dtype, device=device)
    idx = case.bus_index_map()
    for gi in np.where(case.gen[:, GEN_STATUS] > 0)[0]:
        bi = idx[int(case.gen[gi, GEN_BUS])]
        pg[bi] += float(case.gen[gi, PG])
        qg[bi] += float(case.gen[gi, QG])
    pd = torch.as_tensor(case.bus[:, 2], dtype=dtype, device=device)
    qd = torch.as_tensor(case.bus[:, QD], dtype=dtype, device=device)
    return torch.complex((pg - pd) / case.base_mva, (qg - qd) / case.base_mva)


def _initial_voltage(case, device: str, dtype):
    torch = _torch()
    vm = torch.as_tensor(case.bus[:, VM].copy(), dtype=dtype, device=device)
    va = torch.as_tensor(np.deg2rad(case.bus[:, VA].copy()), dtype=dtype, device=device)
    idx = case.bus_index_map()
    for gen in case.gen[case.gen[:, GEN_STATUS] > 0]:
        vm[idx[int(gen[GEN_BUS])]] = float(gen[VG])
    return torch.polar(vm, va)


def solve_newton_raphson_torch(ybus, sbus, v0, ref, pv, pq, *, tolerance=1e-8, max_iterations=30):
    torch = _torch()
    device = v0.device
    dtype = v0.real.dtype
    pv_t = torch.as_tensor(pv, dtype=torch.long, device=device)
    pq_t = torch.as_tensor(pq, dtype=torch.long, device=device)
    pvpq = torch.cat((pv_t, pq_t))
    v = v0.clone()
    history: list[float] = []

    for iteration in range(max_iterations + 1):
        current = ybus @ v
        calc = v * torch.conj(current)
        mismatch = sbus - calc
        f = torch.cat((mismatch[pvpq].real, mismatch[pq_t].imag))
        norm = float(torch.max(torch.abs(f)).detach().cpu()) if f.numel() else 0.0
        history.append(norm)
        if norm < tolerance:
            return True, v, iteration, norm, history
        if iteration == max_iterations:
            break

        vm = torch.abs(v)
        va = torch.angle(v)
        g = ybus.real
        b = ybus.imag
        p = calc.real
        q = calc.imag
        theta = va[:, None] - va[None, :]
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        vm_i = vm[:, None]
        vm_j = vm[None, :]

        h = vm_i * vm_j * (g * sin_t - b * cos_t)
        n = vm_i * (g * cos_t + b * sin_t)
        m = -vm_i * vm_j * (g * cos_t + b * sin_t)
        ell = vm_i * (g * sin_t - b * cos_t)

        diag = torch.arange(v.numel(), device=device)
        safe_vm = torch.clamp(vm, min=torch.finfo(dtype).eps)
        h[diag, diag] = -q - torch.diagonal(b) * vm.square()
        n[diag, diag] = p / safe_vm + torch.diagonal(g) * vm
        m[diag, diag] = p - torch.diagonal(g) * vm.square()
        ell[diag, diag] = q / safe_vm - torch.diagonal(b) * vm

        top = torch.cat((h[pvpq][:, pvpq], n[pvpq][:, pq_t]), dim=1)
        bottom = torch.cat((m[pq_t][:, pvpq], ell[pq_t][:, pq_t]), dim=1)
        jacobian = torch.cat((top, bottom), dim=0)
        try:
            dx = torch.linalg.solve(jacobian, f)
        except RuntimeError:
            return False, v, iteration, norm, history
        if not bool(torch.all(torch.isfinite(dx))):
            return False, v, iteration, norm, history
        va = va.clone()
        vm = vm.clone()
        va[pvpq] += dx[: pvpq.numel()]
        vm[pq_t] += dx[pvpq.numel() :]
        if bool(torch.any(vm <= 0)):
            return False, v, iteration, norm, history
        v = torch.polar(vm, va)

    return False, v, max_iterations, history[-1], history


def _required_generation(case, voltage, ybus):
    torch = _torch()
    dtype = voltage.real.dtype
    device = voltage.device
    injection = voltage * torch.conj(ybus @ voltage) * float(case.base_mva)
    pd = torch.as_tensor(case.bus[:, 2], dtype=dtype, device=device)
    qd = torch.as_tensor(case.bus[:, QD], dtype=dtype, device=device)
    return injection.real + pd, injection.imag + qd


def _update_outputs(case, pg, qg):
    pg_np = np.asarray(pg.detach().cpu(), dtype=float)
    qg_np = np.asarray(qg.detach().cpu(), dtype=float)
    for i, bus_number in enumerate(case.bus[:, BUS_I].astype(int)):
        generators = online_generators_at_bus(case, bus_number)
        if not generators.size:
            continue
        distribute_reactive_power(case, bus_number, float(qg_np[i]))
        if int(case.bus[i, BUS_TYPE]) == REF:
            fixed = float(np.sum(case.gen[generators[1:], PG])) if generators.size > 1 else 0.0
            case.gen[generators[0], PG] = float(pg_np[i] - fixed)


def _branch_flows(case, voltage, yf, yt):
    torch = _torch()
    device = voltage.device
    dtype = voltage.real.dtype
    current_from = yf @ voltage
    current_to = yt @ voltage
    idx = case.bus_index_map()
    fidx = torch.as_tensor([idx[int(v)] for v in case.branch[:, F_BUS]], dtype=torch.long, device=device)
    tidx = torch.as_tensor([idx[int(v)] for v in case.branch[:, T_BUS]], dtype=torch.long, device=device)
    s_from = voltage[fidx] * torch.conj(current_from) * float(case.base_mva)
    s_to = voltage[tidx] * torch.conj(current_to) * float(case.base_mva)
    rate = torch.as_tensor(case.branch[:, RATE_A], dtype=dtype, device=device)
    magnitude = torch.maximum(torch.abs(s_from), torch.abs(s_to))
    loading = torch.where(rate > 0, 100.0 * magnitude / rate, torch.zeros_like(rate))
    loss = float(torch.sum((s_from + s_to).real).detach().cpu())
    return TorchBranchResult(s_from, s_to, loading, loss)


def run_torch_ac_power_flow(input_case, *, device="cpu", dtype=None, options: TorchPowerFlowOptions | None = None):
    torch = _torch()
    dtype = dtype or torch.float64
    options = options or TorchPowerFlowOptions()
    case = input_case.clone()
    warnings: list[str] = []
    ybus, yf, yt = build_dense_admittance(case, device, dtype)
    voltage = _initial_voltage(case, device, dtype)
    total_iterations = 0
    history: list[float] = []

    for q_round in range(options.max_q_limit_rounds + 1):
        ref, pv, pq = _types(case)
        converged, voltage, iterations, max_mismatch, local_history = solve_newton_raphson_torch(
            ybus,
            _sbus(case, device, dtype),
            voltage,
            ref,
            pv,
            pq,
            tolerance=options.tolerance,
            max_iterations=options.max_iterations,
        )
        total_iterations += int(iterations)
        history.extend(local_history)
        if not converged:
            return TorchPowerFlowResult(
                False,
                case,
                voltage,
                torch.abs(voltage),
                torch.rad2deg(torch.angle(voltage)),
                total_iterations,
                q_round,
                max_mismatch,
                history,
                None,
                ybus,
                warnings=warnings,
            )

        pg, qg = _required_generation(case, voltage, ybus)
        if not options.enforce_q_limits:
            _update_outputs(case, pg, qg)
            return TorchPowerFlowResult(
                True,
                case,
                voltage,
                torch.abs(voltage),
                torch.rad2deg(torch.angle(voltage)),
                total_iterations,
                q_round,
                max_mismatch,
                history,
                _branch_flows(case, voltage, yf, yt),
                ybus,
                pg,
                qg,
                warnings,
            )

        violations: list[tuple[int, float]] = []
        qg_cpu = np.asarray(qg.detach().cpu(), dtype=float)
        for bus_index in pv:
            bus_number = int(case.bus[bus_index, BUS_I])
            qmin, qmax = aggregate_q_limits(case, bus_number)
            required = float(qg_cpu[bus_index])
            if required > qmax + options.q_limit_tolerance_mvar:
                violations.append((int(bus_index), float(qmax)))
            elif required < qmin - options.q_limit_tolerance_mvar:
                violations.append((int(bus_index), float(qmin)))

        if not violations:
            _update_outputs(case, pg, qg)
            return TorchPowerFlowResult(
                True,
                case,
                voltage,
                torch.abs(voltage),
                torch.rad2deg(torch.angle(voltage)),
                total_iterations,
                q_round,
                max_mismatch,
                history,
                _branch_flows(case, voltage, yf, yt),
                ybus,
                pg,
                qg,
                warnings,
            )

        if q_round >= options.max_q_limit_rounds:
            warnings.append("Reactive-power limit switching reached the configured round limit.")
            return TorchPowerFlowResult(
                False,
                case,
                voltage,
                torch.abs(voltage),
                torch.rad2deg(torch.angle(voltage)),
                total_iterations,
                q_round,
                max_mismatch,
                history,
                _branch_flows(case, voltage, yf, yt),
                ybus,
                pg,
                qg,
                warnings,
            )

        for bus_index, limit in violations:
            bus_number = int(case.bus[bus_index, BUS_I])
            distribute_reactive_power(case, bus_number, limit)
            case.bus[bus_index, BUS_TYPE] = PQ
            warnings.append(f"Bus {bus_number} converted from PV to PQ at aggregate Q limit {limit:g} MVAr.")
        # Bus types and specified Q changed, while network admittance did not.  Keep ybus/yf/yt.

    raise RuntimeError("Unreachable torch power-flow state")


def torch_l_index(case, voltage, ybus):
    torch = _torch()
    device = voltage.device
    types = case.bus[:, BUS_TYPE].astype(int)
    load = np.where(types == PQ)[0]
    gen = np.where((types == PV) | (types == REF))[0]
    if not load.size or not gen.size:
        return torch.zeros(load.size, dtype=voltage.real.dtype, device=device), 0.0
    load_t = torch.as_tensor(load, dtype=torch.long, device=device)
    gen_t = torch.as_tensor(gen, dtype=torch.long, device=device)
    yll = ybus[load_t][:, load_t]
    ylg = ybus[load_t][:, gen_t]
    try:
        f = -torch.linalg.solve(yll, ylg)
    except RuntimeError:
        return torch.full((load.size,), float("inf"), dtype=voltage.real.dtype, device=device), float("inf")
    values = torch.abs(1.0 - (f @ voltage[gen_t]) / voltage[load_t])
    return values, float(torch.max(values).detach().cpu())


def solve_newton_raphson_batch_torch(ybus, sbus, v0, ref, pv, pq, *, tolerance=1e-8, max_iterations=30, collect_history=True):
    """Batched dense Newton-Raphson solve for candidates sharing bus-type sets.

    Singular/non-finite candidates are isolated with ``torch.linalg.solve_ex`` and do not abort the
    remaining batch.  Returned histories are candidate-specific.
    """
    torch = _torch()
    device = v0.device
    dtype = v0.real.dtype
    batch, nbus = v0.shape
    pv_t = torch.as_tensor(pv, dtype=torch.long, device=device)
    pq_t = torch.as_tensor(pq, dtype=torch.long, device=device)
    pvpq = torch.cat((pv_t, pq_t))
    v = v0.clone()
    converged = torch.zeros(batch, dtype=torch.bool, device=device)
    failed = torch.zeros(batch, dtype=torch.bool, device=device)
    iterations = torch.zeros(batch, dtype=torch.long, device=device)
    max_mismatch = torch.full((batch,), float("inf"), dtype=dtype, device=device)
    histories: list[list[float]] = [[] for _ in range(batch)]

    for iteration in range(max_iterations + 1):
        current = torch.bmm(ybus, v.unsqueeze(-1)).squeeze(-1)
        calc = v * torch.conj(current)
        mismatch = sbus - calc
        f = torch.cat((mismatch[:, pvpq].real, mismatch[:, pq_t].imag), dim=1)
        norms = torch.max(torch.abs(f), dim=1).values if f.shape[1] else torch.zeros(batch, dtype=dtype, device=device)
        if collect_history:
            norms_cpu = norms.detach().cpu().numpy()
            active_history = ((~failed) & (~converged)).detach().cpu().numpy()
            for i, value in enumerate(norms_cpu):
                if bool(active_history[i]):
                    histories[i].append(float(value))
        newly_converged = (~failed) & (~converged) & (norms < tolerance)
        iterations[newly_converged] = iteration
        converged = converged | newly_converged
        max_mismatch = torch.where((~failed) & (~converged), norms, max_mismatch)
        active = (~failed) & (~converged)
        if not bool(torch.any(active)) or iteration == max_iterations:
            break

        vm = torch.abs(v)
        va = torch.angle(v)
        g = ybus.real
        b = ybus.imag
        p = calc.real
        q = calc.imag
        theta = va[:, :, None] - va[:, None, :]
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        vm_i = vm[:, :, None]
        vm_j = vm[:, None, :]
        h = vm_i * vm_j * (g * sin_t - b * cos_t)
        n = vm_i * (g * cos_t + b * sin_t)
        m = -vm_i * vm_j * (g * cos_t + b * sin_t)
        ell = vm_i * (g * sin_t - b * cos_t)
        diag = torch.arange(nbus, device=device)
        safe_vm = torch.clamp(vm, min=torch.finfo(dtype).eps)
        h[:, diag, diag] = -q - torch.diagonal(b, dim1=1, dim2=2) * vm.square()
        n[:, diag, diag] = p / safe_vm + torch.diagonal(g, dim1=1, dim2=2) * vm
        m[:, diag, diag] = p - torch.diagonal(g, dim1=1, dim2=2) * vm.square()
        ell[:, diag, diag] = q / safe_vm - torch.diagonal(b, dim1=1, dim2=2) * vm
        top = torch.cat((h[:, pvpq][:, :, pvpq], n[:, pvpq][:, :, pq_t]), dim=2)
        bottom = torch.cat((m[:, pq_t][:, :, pvpq], ell[:, pq_t][:, :, pq_t]), dim=2)
        jacobian = torch.cat((top, bottom), dim=1)

        inactive = ~active
        if bool(torch.any(inactive)):
            size = jacobian.shape[-1]
            identity = torch.eye(size, dtype=dtype, device=device)
            jacobian = jacobian.clone()
            f = f.clone()
            jacobian[inactive] = identity
            f[inactive] = 0.0
        try:
            dx, info = torch.linalg.solve_ex(jacobian, f.unsqueeze(-1), check_errors=False)
            dx = dx.squeeze(-1)
        except Exception:
            # Conservative fallback for runtimes without batched solve_ex support.
            dx = torch.zeros_like(f)
            info = torch.zeros(batch, dtype=torch.int32, device=device)
            for i in range(batch):
                if not bool(active[i]):
                    continue
                try:
                    dx[i] = torch.linalg.solve(jacobian[i], f[i])
                except RuntimeError:
                    info[i] = 1
        bad = active & ((info != 0) | (~torch.all(torch.isfinite(dx), dim=1)))
        failed = failed | bad
        active = active & (~bad)
        va_new = va.clone()
        vm_new = vm.clone()
        # Advanced indexing by active batch rows keeps the bus-index vectors unchanged.
        rows = torch.where(active)[0]
        if rows.numel():
            va_new[rows[:, None], pvpq[None, :]] += dx[rows, : pvpq.numel()]
            vm_new[rows[:, None], pq_t[None, :]] += dx[rows, pvpq.numel() :]
        invalid_vm = active & torch.any(vm_new <= 0, dim=1)
        failed = failed | invalid_vm
        valid_rows = torch.where(active & (~invalid_vm))[0]
        if valid_rows.numel():
            v[valid_rows] = torch.polar(vm_new[valid_rows], va_new[valid_rows])

    max_mismatch = torch.where(converged, torch.zeros_like(max_mismatch), max_mismatch)
    if collect_history:
        converged_cpu = converged.detach().cpu().numpy()
        iterations_cpu = iterations.detach().cpu().numpy()
        for i in range(batch):
            if not bool(converged_cpu[i]) and histories[i]:
                max_mismatch[i] = histories[i][-1]
            if not bool(converged_cpu[i]) and int(iterations_cpu[i]) == 0:
                iterations[i] = min(max_iterations, len(histories[i]) - 1 if histories[i] else 0)
    else:
        # Preserve a compact device-side mismatch summary without per-iteration host transfers.
        unconverged = ~converged
        iterations = torch.where(
            unconverged & (iterations == 0),
            torch.full_like(iterations, int(max_iterations)),
            iterations,
        )
    return converged, failed, v, iterations, max_mismatch, histories




@dataclass(slots=True)
class TorchBatchedAdmittance:
    ybus: Any
    yff: Any
    yft: Any
    ytf: Any
    ytt: Any
    fidx: Any
    tidx: Any


def build_batched_admittance(cases, device: str, dtype=None):
    """Build candidate-specific Y-bus and branch coefficients in one tensor operation.

    Topology is required to be identical, while tap ratios, phase shifts, branch status and bus
    shunts may vary per candidate/scenario.  The construction performs one host-to-device transfer
    per matrix family instead of creating thousands of tiny tensors in Python branch loops.
    """
    torch = _torch()
    dtype = dtype or torch.float64
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    if not cases:
        raise ValueError("At least one case is required")
    first = cases[0]
    batch = len(cases)
    n = first.n_bus
    nl = first.n_branch
    index = first.bus_index_map()
    fidx_np = np.asarray([index[int(v)] for v in first.branch[:, F_BUS]], dtype=np.int64)
    tidx_np = np.asarray([index[int(v)] for v in first.branch[:, T_BUS]], dtype=np.int64)
    for case in cases[1:]:
        if case.n_bus != n or case.n_branch != nl:
            raise ValueError("Batched cases must share network dimensions")
        if not np.array_equal(case.branch[:, F_BUS].astype(int), first.branch[:, F_BUS].astype(int)):
            raise ValueError("Batched cases must share branch from-bus topology")
        if not np.array_equal(case.branch[:, T_BUS].astype(int), first.branch[:, T_BUS].astype(int)):
            raise ValueError("Batched cases must share branch to-bus topology")

    branch = np.stack([case.branch for case in cases], axis=0)
    r = torch.as_tensor(branch[:, :, BR_R], dtype=dtype, device=device)
    x = torch.as_tensor(branch[:, :, BR_X], dtype=dtype, device=device)
    line_b = torch.as_tensor(branch[:, :, BR_B], dtype=dtype, device=device)
    status = torch.as_tensor(branch[:, :, BR_STATUS] > 0, dtype=torch.bool, device=device)
    tap_np = branch[:, :, TAP].copy()
    tap_np[tap_np == 0] = 1.0
    tap = torch.as_tensor(tap_np, dtype=dtype, device=device)
    shift = torch.deg2rad(torch.as_tensor(branch[:, :, SHIFT], dtype=dtype, device=device))
    z = torch.complex(r, x)
    eps = torch.finfo(dtype).eps
    y = torch.where(torch.abs(z) > eps, 1.0 / z, torch.zeros_like(z))
    y = torch.where(status, y, torch.zeros_like(y))
    charging = torch.where(status, torch.complex(torch.zeros_like(line_b), line_b / 2.0), torch.zeros_like(y))
    a = torch.polar(tap, shift)
    yff = (y + charging) / (a * torch.conj(a))
    yft = -y / torch.conj(a)
    ytf = -y / a
    ytt = y + charging

    fidx = torch.as_tensor(fidx_np, dtype=torch.long, device=device)
    tidx = torch.as_tensor(tidx_np, dtype=torch.long, device=device)
    ybus = torch.zeros((batch, n, n), dtype=cdtype, device=device)
    rows = torch.arange(batch, device=device)[:, None].expand(batch, nl)
    f = fidx[None, :].expand(batch, nl)
    t = tidx[None, :].expand(batch, nl)
    ybus.index_put_((rows, f, f), yff, accumulate=True)
    ybus.index_put_((rows, f, t), yft, accumulate=True)
    ybus.index_put_((rows, t, f), ytf, accumulate=True)
    ybus.index_put_((rows, t, t), ytt, accumulate=True)
    shunt_np = np.stack(
        [(case.bus[:, GS] + 1j * case.bus[:, BS]) / float(case.base_mva) for case in cases],
        axis=0,
    )
    shunt = torch.as_tensor(shunt_np, dtype=cdtype, device=device)
    diagonal = torch.arange(n, device=device)
    ybus[:, diagonal, diagonal] += shunt
    return TorchBatchedAdmittance(ybus, yff, yft, ytf, ytt, fidx, tidx)


def _sbus_batch(cases, device: str, dtype):
    torch = _torch()
    batch = len(cases)
    n = cases[0].n_bus
    pg = np.zeros((batch, n), dtype=float)
    qg = np.zeros((batch, n), dtype=float)
    pd = np.stack([case.bus[:, 2] for case in cases], axis=0)
    qd = np.stack([case.bus[:, QD] for case in cases], axis=0)
    for row, case in enumerate(cases):
        index = case.bus_index_map()
        online = np.where(case.gen[:, GEN_STATUS] > 0)[0]
        for gi in online:
            bi = index[int(case.gen[gi, GEN_BUS])]
            pg[row, bi] += float(case.gen[gi, PG])
            qg[row, bi] += float(case.gen[gi, QG])
    base = np.asarray([case.base_mva for case in cases], dtype=float)[:, None]
    real = torch.as_tensor((pg - pd) / base, dtype=dtype, device=device)
    imag = torch.as_tensor((qg - qd) / base, dtype=dtype, device=device)
    return torch.complex(real, imag)


def _initial_voltage_batch(cases, device: str, dtype):
    torch = _torch()
    vm = np.stack([case.bus[:, VM].copy() for case in cases], axis=0)
    va = np.stack([np.deg2rad(case.bus[:, VA].copy()) for case in cases], axis=0)
    for row, case in enumerate(cases):
        index = case.bus_index_map()
        for gen in case.gen[case.gen[:, GEN_STATUS] > 0]:
            vm[row, index[int(gen[GEN_BUS])]] = float(gen[VG])
    return torch.polar(
        torch.as_tensor(vm, dtype=dtype, device=device),
        torch.as_tensor(va, dtype=dtype, device=device),
    )


def _required_generation_batch(cases, voltage, ybus):
    torch = _torch()
    dtype = voltage.real.dtype
    device = voltage.device
    base = torch.as_tensor([case.base_mva for case in cases], dtype=dtype, device=device)[:, None]
    injection = voltage * torch.conj(torch.matmul(ybus, voltage.unsqueeze(-1)).squeeze(-1)) * base
    pd = torch.as_tensor(np.stack([case.bus[:, 2] for case in cases]), dtype=dtype, device=device)
    qd = torch.as_tensor(np.stack([case.bus[:, QD] for case in cases]), dtype=dtype, device=device)
    return injection.real + pd, injection.imag + qd


def _branch_flows_batch(cases, voltage, admittance: TorchBatchedAdmittance):
    torch = _torch()
    vf = voltage[:, admittance.fidx]
    vt = voltage[:, admittance.tidx]
    current_from = admittance.yff * vf + admittance.yft * vt
    current_to = admittance.ytf * vf + admittance.ytt * vt
    base = torch.as_tensor([case.base_mva for case in cases], dtype=voltage.real.dtype, device=voltage.device)[:, None]
    s_from = vf * torch.conj(current_from) * base
    s_to = vt * torch.conj(current_to) * base
    rate = torch.as_tensor(np.stack([case.branch[:, RATE_A] for case in cases]), dtype=voltage.real.dtype, device=voltage.device)
    magnitude = torch.maximum(torch.abs(s_from), torch.abs(s_to))
    loading = torch.where(rate > 0, 100.0 * magnitude / rate, torch.zeros_like(rate))
    losses = torch.sum((s_from + s_to).real, dim=1)
    return s_from, s_to, loading, losses

def run_torch_ac_power_flow_batch(input_cases, *, device="cpu", dtype=None, options: TorchPowerFlowOptions | None = None):
    """Evaluate one candidate batch with genuinely batched FP64 network construction and NR solves.

    Candidates sharing the initial bus-type signature use batched Jacobian assembly and batched
    ``torch.linalg.solve_ex``.  Only candidates requiring candidate-specific PV-to-PQ switching are
    sent to the exact single-candidate fallback.  All non-switching branch-flow calculations are
    also completed in one tensor batch.
    """
    torch = _torch()
    cases = [case.clone() for case in input_cases]
    if not cases:
        return []
    dtype = dtype or torch.float64
    options = options or TorchPowerFlowOptions()
    first_types = _types(cases[0])

    def _same_types(other):
        return all(np.array_equal(a, b) for a, b in zip(first_types, _types(other)))

    if any(not _same_types(case) for case in cases[1:]):
        return [run_torch_ac_power_flow(case, device=device, dtype=dtype, options=options) for case in cases]

    ref, pv, pq = first_types
    admittance = build_batched_admittance(cases, device, dtype)
    sbus = _sbus_batch(cases, device, dtype)
    v0 = _initial_voltage_batch(cases, device, dtype)
    converged, failed, voltage, iterations, mismatch, histories = solve_newton_raphson_batch_torch(
        admittance.ybus,
        sbus,
        v0,
        ref,
        pv,
        pq,
        tolerance=options.tolerance,
        max_iterations=options.max_iterations,
    )
    pg_batch, qg_batch = _required_generation_batch(cases, voltage, admittance.ybus)
    qg_cpu = np.asarray(qg_batch.detach().cpu(), dtype=float)
    s_from, s_to, loading, losses = _branch_flows_batch(cases, voltage, admittance)
    loss_cpu = np.asarray(losses.detach().cpu(), dtype=float)
    iteration_cpu = np.asarray(iterations.detach().cpu(), dtype=int)
    mismatch_cpu = np.asarray(mismatch.detach().cpu(), dtype=float)

    results = []
    for i, case in enumerate(cases):
        if not bool(converged[i]) or bool(failed[i]):
            results.append(
                TorchPowerFlowResult(
                    False,
                    case,
                    voltage[i],
                    torch.abs(voltage[i]),
                    torch.rad2deg(torch.angle(voltage[i])),
                    int(iteration_cpu[i]),
                    0,
                    float(mismatch_cpu[i]),
                    histories[i],
                    None,
                    admittance.ybus[i],
                )
            )
            continue

        violations = []
        if options.enforce_q_limits:
            for bus_index in pv:
                bus_number = int(case.bus[bus_index, BUS_I])
                qmin, qmax = aggregate_q_limits(case, bus_number)
                required = float(qg_cpu[i, bus_index])
                if required > qmax + options.q_limit_tolerance_mvar or required < qmin - options.q_limit_tolerance_mvar:
                    violations.append(bus_index)
        if violations:
            # Exact candidate-specific bus-type switching remains the reference-equivalent fallback.
            results.append(run_torch_ac_power_flow(case, device=device, dtype=dtype, options=options))
            continue

        _update_outputs(case, pg_batch[i], qg_batch[i])
        branch = TorchBranchResult(s_from[i], s_to[i], loading[i], float(loss_cpu[i]))
        results.append(
            TorchPowerFlowResult(
                True,
                case,
                voltage[i],
                torch.abs(voltage[i]),
                torch.rad2deg(torch.angle(voltage[i])),
                int(iteration_cpu[i]),
                0,
                float(mismatch_cpu[i]),
                histories[i],
                branch,
                admittance.ybus[i],
                pg_batch[i],
                qg_batch[i],
                [],
            )
        )
    return results

