"""Security-constrained N-1 branch/generator scenario construction with intact base state."""

from __future__ import annotations
import numpy as np
from calo_rpd_studio.power_system.case_model import *
from .scenario import Scenario


def n_minus_one_branch_scenarios(indices, *, include_base: bool = True):
    """Return intact P0 plus selected single-branch P1 contingencies by default."""
    out = [Scenario("base")] if include_base else []
    for k in indices:
        def transform(case, k=int(k)):
            if k < 0 or k >= case.n_branch:
                raise IndexError(k)
            case.branch[k, BR_STATUS] = 0
            return case
        out.append(Scenario(f"branch_out_{k}", 1.0, transform))
    return out or [Scenario("base")]


def n_minus_one_generator_scenarios(indices, *, include_base: bool = True):
    """Return intact P0 plus selected single-generator P1 contingencies by default."""
    out = [Scenario("base")] if include_base else []
    for k in indices:
        def transform(case, k=int(k)):
            if k < 0 or k >= case.n_gen:
                raise IndexError(k)
            old_bus = int(case.gen[k, GEN_BUS])
            case.gen[k, GEN_STATUS] = 0
            idx = case.bus_index_map()
            bi = idx[old_bus]
            remaining = np.where(
                (case.gen[:, GEN_STATUS] > 0) & (case.gen[:, GEN_BUS].astype(int) == old_bus)
            )[0]
            if not remaining.size and int(case.bus[bi, BUS_TYPE]) in (PV, REF):
                case.bus[bi, BUS_TYPE] = PQ
            refs = np.where(case.bus[:, BUS_TYPE].astype(int) == REF)[0]
            if not len(refs):
                online = np.where(case.gen[:, GEN_STATUS] > 0)[0]
                if not len(online):
                    return case
                new_bus = int(case.gen[online[0], GEN_BUS])
                case.bus[idx[new_bus], BUS_TYPE] = REF
            return case
        out.append(Scenario(f"generator_out_{k}", 1.0, transform))
    return out or [Scenario("base")]
