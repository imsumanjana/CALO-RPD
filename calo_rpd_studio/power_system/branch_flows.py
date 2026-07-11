"""Branch complex power flows, loading, and losses."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .case_model import RATE_A
from .ybus import AdmittanceMatrices

@dataclass(slots=True)
class BranchFlowResult:
    s_from_mva:np.ndarray; s_to_mva:np.ndarray; loading_percent:np.ndarray; total_loss_mw:float

def calculate_branch_flows(case,voltage,admittance:AdmittanceMatrices):
    i_from=admittance.y_from@voltage; i_to=admittance.y_to@voltage
    s_from=np.zeros(case.n_branch,dtype=complex); s_to=np.zeros(case.n_branch,dtype=complex)
    idx=case.bus_index_map()
    for k,br in enumerate(case.branch):
        f=idx[int(br[0])];t=idx[int(br[1])]
        s_from[k]=voltage[f]*np.conj(i_from[k])*case.base_mva; s_to[k]=voltage[t]*np.conj(i_to[k])*case.base_mva
    rate=case.branch[:,RATE_A]; mag=np.maximum(np.abs(s_from),np.abs(s_to)); loading=np.where(rate>0,100*mag/rate,0.0)
    return BranchFlowResult(s_from,s_to,loading,float(np.sum((s_from+s_to).real)))
