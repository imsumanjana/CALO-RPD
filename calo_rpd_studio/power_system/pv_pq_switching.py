"""Generator reactive-limit aggregation and allocation."""
from __future__ import annotations
import numpy as np
from .case_model import *

def online_generators_at_bus(case,bus_number):
    return np.where((case.gen[:,GEN_STATUS]>0)&(case.gen[:,GEN_BUS].astype(int)==int(bus_number)))[0]
def aggregate_q_limits(case,bus_number):
    g=online_generators_at_bus(case,bus_number)
    return (float(np.sum(case.gen[g,QMIN])),float(np.sum(case.gen[g,QMAX]))) if g.size else (0.0,0.0)
def distribute_reactive_power(case,bus_number,required_q):
    g=online_generators_at_bus(case,bus_number)
    if not g.size:return
    qmin=case.gen[g,QMIN]; qmax=case.gen[g,QMAX]; span=np.maximum(qmax-qmin,0); total=float(span.sum())
    if total>0: q=qmin+(np.clip(required_q,float(qmin.sum()),float(qmax.sum()))-float(qmin.sum()))*span/total
    else: q=np.full(g.size,required_q/g.size)
    case.gen[g,QG]=q
