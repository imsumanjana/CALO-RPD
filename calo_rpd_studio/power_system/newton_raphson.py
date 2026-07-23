"""Sparse Newton-Raphson AC power-flow reference kernel.

v5.9 constructs the polar Jacobian directly from sparse complex-voltage derivatives when SciPy is
available. It no longer densifies Ybus or allocates full NxN angle/trigonometric matrices before
converting back to sparse form. A deterministic vectorized dense fallback is retained only for
minimal environments without SciPy sparse support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class NewtonResult:
    converged: bool
    voltage: np.ndarray
    iterations: int
    max_mismatch: float
    mismatch_history: list[float]


def _mismatch(ybus, sbus, voltage, pvpq, pq):
    mis = voltage * np.conj(ybus @ voltage) - sbus
    return np.r_[mis[pvpq].real, mis[pq].imag]


def _dense_jacobian(ybus, voltage, pvpq, pq):
    """Deterministic compatibility fallback used only when SciPy sparse is unavailable."""
    y = ybus.toarray() if hasattr(ybus, "toarray") else np.asarray(ybus)
    g = y.real
    b = y.imag
    vm = np.abs(voltage)
    va = np.angle(voltage)
    theta = va[:, None] - va[None, :]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    s = voltage * np.conj(y @ voltage)
    p = s.real
    q = s.imag
    vprod = vm[:, None] * vm[None, :]
    h = vprod * (g * sin_t - b * cos_t)
    n = vm[:, None] * (g * cos_t + b * sin_t)
    m = -vprod * (g * cos_t + b * sin_t)
    ell = vm[:, None] * (g * sin_t - b * cos_t)
    diag = np.arange(len(vm))
    h[diag, diag] = -q - np.diag(b) * vm**2
    n[diag, diag] = p / np.maximum(vm, 1e-15) + np.diag(g) * vm
    m[diag, diag] = p - np.diag(g) * vm**2
    ell[diag, diag] = q / np.maximum(vm, 1e-15) - np.diag(b) * vm
    return np.block(
        [
            [h[np.ix_(pvpq, pvpq)], n[np.ix_(pvpq, pq)]],
            [m[np.ix_(pq, pvpq)], ell[np.ix_(pq, pq)]],
        ]
    )


def _jacobian(ybus, voltage, pvpq, pq):
    """Build MATPOWER-style dS/dVa,dS/dVm blocks without dense Ybus/Jacobian construction."""
    try:
        from scipy.sparse import csr_matrix, diags, hstack, vstack

        y = csr_matrix(ybus)
        voltage = np.asarray(voltage, dtype=complex)
        nbus = voltage.size
        ibus = y @ voltage
        vm = np.abs(voltage)
        vnorm = voltage / np.maximum(vm, 1e-15)
        diag_v = diags(voltage, 0, shape=(nbus, nbus), format="csr")
        diag_i = diags(ibus, 0, shape=(nbus, nbus), format="csr")
        diag_vnorm = diags(vnorm, 0, shape=(nbus, nbus), format="csr")

        dS_dVm = diag_v @ (y @ diag_vnorm).conjugate() + diag_i.conjugate() @ diag_vnorm
        dS_dVa = 1j * diag_v @ (diag_i - y @ diag_v).conjugate()

        pvpq = np.asarray(pvpq, dtype=int)
        pq = np.asarray(pq, dtype=int)
        j11 = dS_dVa[pvpq, :][:, pvpq].real.tocsr()
        j12 = dS_dVm[pvpq, :][:, pq].real.tocsr()
        j21 = dS_dVa[pq, :][:, pvpq].imag.tocsr()
        j22 = dS_dVm[pq, :][:, pq].imag.tocsr()
        return vstack([hstack([j11, j12], format="csr"), hstack([j21, j22], format="csr")], format="csr")
    except ImportError:
        return _dense_jacobian(ybus, voltage, pvpq, pq)


def _solve_linear(jacobian, rhs):
    try:
        from scipy.sparse import csr_matrix, issparse
        from scipy.sparse.linalg import spsolve

        matrix = jacobian if issparse(jacobian) else csr_matrix(jacobian)
        with np.errstate(all="ignore"):
            dx = np.asarray(spsolve(matrix, rhs), dtype=float)
        if np.all(np.isfinite(dx)):
            return dx
    except (ImportError, RuntimeError, ValueError, TypeError):
        _LOG.debug("Sparse linear solve unavailable/failed; using deterministic dense fallback", exc_info=True)
    dense = jacobian.toarray() if hasattr(jacobian, "toarray") else np.asarray(jacobian)
    return np.linalg.solve(dense, rhs)


def solve_newton_raphson(
    ybus,
    sbus,
    v0,
    ref,
    pv,
    pq,
    tolerance=1e-8,
    max_iterations=30,
    *,
    minimum_damping=1.0 / 32.0,
):
    pvpq = np.r_[pv, pq].astype(int)
    pq = np.asarray(pq, dtype=int)
    voltage = np.asarray(v0, dtype=complex).copy()
    history: list[float] = []

    for iteration in range(int(max_iterations) + 1):
        f = _mismatch(ybus, sbus, voltage, pvpq, pq)
        norm = float(np.max(np.abs(f))) if f.size else 0.0
        history.append(norm)
        if norm <= float(tolerance):
            return NewtonResult(True, voltage, iteration, norm, history)
        if iteration >= int(max_iterations):
            break
        try:
            jacobian = _jacobian(ybus, voltage, pvpq, pq)
            dx = _solve_linear(jacobian, -f)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            break
        if not np.all(np.isfinite(dx)):
            break

        vm = np.abs(voltage)
        va = np.angle(voltage)
        best_voltage = None
        best_norm = float("inf")
        damping = 1.0
        while damping >= float(minimum_damping) - 1e-15:
            va_trial = va.copy()
            vm_trial = vm.copy()
            va_trial[pvpq] += damping * dx[: len(pvpq)]
            vm_trial[pq] += damping * dx[len(pvpq) :]
            if np.any(vm_trial[pq] <= 0.0) or not np.all(np.isfinite(vm_trial)):
                damping *= 0.5
                continue
            trial = vm_trial * np.exp(1j * va_trial)
            trial_f = _mismatch(ybus, sbus, trial, pvpq, pq)
            trial_norm = float(np.max(np.abs(trial_f))) if trial_f.size else 0.0
            if trial_norm < best_norm:
                best_norm = trial_norm
                best_voltage = trial
            if trial_norm < norm:
                best_voltage = trial
                break
            damping *= 0.5
        if best_voltage is None or not np.all(np.isfinite(best_voltage)):
            break
        voltage = best_voltage

    final = _mismatch(ybus, sbus, voltage, pvpq, pq)
    max_mismatch = float(np.max(np.abs(final))) if final.size else 0.0
    return NewtonResult(False, voltage, min(int(max_iterations), len(history) - 1), max_mismatch, history)
