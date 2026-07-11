"""Network summary metrics."""
from __future__ import annotations
import numpy as np
from .case_model import GEN_STATUS,TAP,BR_STATUS
def summarize_case(case):
    taps=case.branch[:,TAP];return {'buses':case.n_bus,'generators':int(np.sum(case.gen[:,GEN_STATUS]>0)),'branches':int(np.sum(case.branch[:,BR_STATUS]>0)),'transformers':int(np.sum(taps!=0)),'shunt_buses':int(np.sum(np.abs(case.bus[:,5])>0)),'base_mva':float(case.base_mva),'checksum':case.checksum()}
