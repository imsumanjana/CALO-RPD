"""Kessel-Glavitsch voltage-stability L-index."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .case_model import BUS_TYPE,PQ,PV,REF
from .ybus import build_ybus
@dataclass(slots=True)
class LIndexResult: values:np.ndarray; load_bus_indices:np.ndarray; maximum:float
def kessel_glavitsch_l_index(case,voltage):
    types=case.bus[:,BUS_TYPE].astype(int);load=np.where(types==PQ)[0];gen=np.where((types==PV)|(types==REF))[0]
    if not load.size or not gen.size:return LIndexResult(np.zeros(load.size),load,0.0)
    y=build_ybus(case).ybus.toarray();yll=y[np.ix_(load,load)];ylg=y[np.ix_(load,gen)]
    try:f=-np.linalg.solve(yll,ylg)
    except np.linalg.LinAlgError:return LIndexResult(np.full(load.size,np.inf),load,float('inf'))
    values=np.abs(1-(f@voltage[gen])/voltage[load]);return LIndexResult(values,load,float(np.max(values)))
