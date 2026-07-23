"""Sparse bus and branch admittance matrix construction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from .case_model import *


@dataclass(slots=True)
class AdmittanceMatrices:
    ybus: csr_matrix
    y_from: csr_matrix
    y_to: csr_matrix


def build_ybus(case: PowerSystemCase) -> AdmittanceMatrices:
    """Build Ybus/Yf/Yt directly from sparse non-zero triplets.

    Each in-service branch contributes four Ybus entries and two entries to each branch-current
    incidence matrix.  No dense ``n_branch x n_bus`` temporary is created, which keeps memory
    proportional to network sparsity for large/robust scenario campaigns.
    """
    n = case.n_bus
    nl = case.n_branch
    idx = case.bus_index_map()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    yf_rows: list[int] = []
    yf_cols: list[int] = []
    yf_vals: list[complex] = []
    yt_rows: list[int] = []
    yt_cols: list[int] = []
    yt_vals: list[complex] = []
    for k, br in enumerate(case.branch):
        if br[BR_STATUS] <= 0:
            continue
        f = idx[int(br[F_BUS])]
        t = idx[int(br[T_BUS])]
        z = complex(br[BR_R], br[BR_X])
        if abs(z) <= 1e-12:
            raise ValueError(
                f"In-service branch {k} has zero impedance; validate or regularize the source case explicitly."
            )
        y = 1 / z
        b = 1j * br[BR_B] / 2
        tap = br[TAP] if br[TAP] != 0 else 1.0
        shift = np.deg2rad(br[SHIFT])
        a = tap * np.exp(1j * shift)
        yff = (y + b) / (a * np.conj(a))
        yft = -y / np.conj(a)
        ytf = -y / a
        ytt = y + b
        yf_rows.extend((k, k)); yf_cols.extend((f, t)); yf_vals.extend((yff, yft))
        yt_rows.extend((k, k)); yt_cols.extend((f, t)); yt_vals.extend((ytf, ytt))
        for r, c, v in ((f, f, yff), (f, t, yft), (t, f, ytf), (t, t, ytt)):
            rows.append(r); cols.append(c); vals.append(v)
    ybus = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)
    sh = (case.bus[:, GS] + 1j * case.bus[:, BS]) / case.base_mva
    ybus = ybus + csr_matrix((sh, (np.arange(n), np.arange(n))), shape=(n, n))
    y_from = csr_matrix((yf_vals, (yf_rows, yf_cols)), shape=(nl, n), dtype=complex)
    y_to = csr_matrix((yt_vals, (yt_rows, yt_cols)), shape=(nl, n), dtype=complex)
    return AdmittanceMatrices(ybus, y_from, y_to)
