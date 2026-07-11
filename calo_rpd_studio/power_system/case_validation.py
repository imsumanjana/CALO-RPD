"""Structural and engineering validation for case data."""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from .case_model import *
@dataclass(slots=True)
class CaseValidationReport:
    valid:bool; errors:list[str]=field(default_factory=list); warnings:list[str]=field(default_factory=list)
def validate_case(case):
    e=[];w=[];numbers=case.bus[:,BUS_I].astype(int)
    if len(set(numbers))!=len(numbers):e.append('Bus numbers must be unique.')
    if np.sum(case.bus[:,BUS_TYPE].astype(int)==REF)!=1:e.append('Exactly one reference bus is required.')
    known=set(numbers) 
    for b in case.gen[:,GEN_BUS].astype(int):
        if b not in known:e.append(f'Generator references unknown bus {b}.')
    for f,t in case.branch[:,[F_BUS,T_BUS]].astype(int):
        if f not in known or t not in known:e.append(f'Branch references unknown buses {f}-{t}.')
    if np.any(case.bus[:,VMAX]<=case.bus[:,VMIN]):e.append('Every bus voltage maximum must exceed its minimum.')
    if np.any(case.gen[:,QMAX]<case.gen[:,QMIN]):e.append('Generator reactive-power limits are inconsistent.')
    if np.any(case.gen[:,PMAX]<case.gen[:,PMIN]):e.append('Generator active-power limits are inconsistent.')
    if np.any(case.branch[:,BR_X]==0):w.append('At least one branch has zero reactance; verify source data.')
    return CaseValidationReport(not e,e,w)
