"""Seeded renewable-injection uncertainty scenarios."""
from __future__ import annotations
import numpy as np
from .scenario import Scenario
def renewable_scenarios(count,bus_number,rated_mw,mean_capacity_factor,std_capacity_factor,rng):
    out=[]
    for i in range(count):
        cf=float(np.clip(rng.normal(mean_capacity_factor,std_capacity_factor),0,1));mw=rated_mw*cf
        def transform(case,mw=mw):
            idx=case.bus_index_map()[int(bus_number)];case.bus[idx,2]=max(0.0,case.bus[idx,2]-mw);return case
        out.append(Scenario(f'renewable_{i+1:03d}',1/count,transform))
    return out
