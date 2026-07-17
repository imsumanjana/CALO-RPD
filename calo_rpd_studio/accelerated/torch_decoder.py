"""Tensorized mixed-variable decoding for the common normalized ORPD search space."""
from __future__ import annotations

import numpy as np

from calo_rpd_studio.power_system.case_model import BS, GEN_BUS, GEN_STATUS, TAP, VG, VM


class TorchVariableDecoder:
    """Decode continuous and discrete controls on the selected PyTorch device.

    Network-case mutation remains lightweight Python orchestration, but all continuous scaling,
    discrete lattice indexing, and bounds enforcement are performed as tensor operations.  The
    action definitions are inherited from the single reference ``ORPDVariableDecoder`` so CPU,
    CUDA, and XPU cannot silently use different device grids.
    """

    def __init__(self, reference_decoder, device, dtype):
        import torch

        self.reference = reference_decoder
        self.case = reference_decoder.case
        self.device = device
        self.dtype = dtype
        self._torch = torch

    @property
    def dimension(self):
        return self.reference.dimension

    def decode_values(self, normalized):
        torch = self._torch
        z = torch.as_tensor(normalized, dtype=self.dtype, device=self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.shape[1] != self.dimension:
            raise ValueError(f"Expected decision matrix with {self.dimension} columns, got {tuple(z.shape)}")
        z = torch.clamp(z, 0.0, 1.0)
        columns = []
        for column, action in enumerate(self.reference._actions):
            _kind, _target, lower, upper, values = action
            if values is None:
                decoded = float(lower) + z[:, column] * (float(upper) - float(lower))
            else:
                lattice = torch.as_tensor(values, dtype=self.dtype, device=self.device)
                index = torch.floor(z[:, column] * lattice.numel()).long()
                index = torch.clamp(index, 0, lattice.numel() - 1)
                decoded = lattice[index]
            columns.append(decoded)
        return torch.stack(columns, dim=1) if columns else torch.empty((z.shape[0], 0), dtype=self.dtype, device=self.device)

    def decode_batch(self, normalized):
        decoded = self.decode_values(normalized)
        values = np.asarray(decoded.detach().cpu(), dtype=float)
        cases = []
        physical = []
        for row in values:
            case = self.case.clone()
            controls = {}
            index = case.bus_index_map()
            for value, action, variable in zip(row, self.reference._actions, self.reference.variables):
                kind, target, _lower, _upper, _lattice = action
                scalar = float(value)
                controls[variable.name] = scalar
                if kind == "vg":
                    generators = np.where(
                        (case.gen[:, GEN_STATUS] > 0)
                        & (case.gen[:, GEN_BUS].astype(int) == int(target))
                    )[0]
                    case.gen[generators, VG] = scalar
                    case.bus[index[int(target)], VM] = scalar
                elif kind == "tap":
                    case.branch[int(target), TAP] = scalar
                elif kind == "shunt":
                    case.bus[index[int(target)], BS] = scalar
            cases.append(case)
            physical.append(controls)
        return cases, physical
