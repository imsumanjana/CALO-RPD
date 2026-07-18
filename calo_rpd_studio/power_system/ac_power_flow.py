"""High-level AC Newton-Raphson power flow with aggregate Q-limit switching."""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from .case_model import *
from .ybus import build_ybus
from .newton_raphson import solve_newton_raphson
from .pv_pq_switching import aggregate_q_limits,distribute_reactive_power,online_generators_at_bus
from .branch_flows import calculate_branch_flows,BranchFlowResult

@dataclass(slots=True)
class PowerFlowOptions:
    tolerance:float=1e-8; max_iterations:int=30; enforce_q_limits:bool=True; max_q_limit_rounds:int=10; q_limit_tolerance_mvar:float=1e-6
@dataclass(slots=True)
class PowerFlowResult:
    converged:bool; case:object; voltage:np.ndarray; vm_pu:np.ndarray; va_deg:np.ndarray; iterations:int; q_limit_rounds:int; max_mismatch:float; mismatch_history:list[float]; branch:BranchFlowResult|None; warnings:list[str]=field(default_factory=list)
    @property
    def total_loss_mw(self): return float('inf') if self.branch is None else self.branch.total_loss_mw

def _types(case):
    t=case.bus[:,BUS_TYPE].astype(int); ref=np.where(t==REF)[0]; pv=np.where(t==PV)[0]; pq=np.where(t==PQ)[0]
    if ref.size!=1: raise ValueError(f'Exactly one reference bus is required; found {ref.size}')
    return ref,pv,pq
def _sbus(case):
    idx=case.bus_index_map(); pg=np.zeros(case.n_bus);qg=np.zeros(case.n_bus)
    for g in np.where(case.gen[:,GEN_STATUS]>0)[0]:
        b=idx[int(case.gen[g,GEN_BUS])];pg[b]+=case.gen[g,PG];qg[b]+=case.gen[g,QG]
    return ((pg-case.bus[:,PD])+1j*(qg-case.bus[:,QD]))/case.base_mva
def _v0(case):
    vm=case.bus[:,VM].copy(); va=np.deg2rad(case.bus[:,VA].copy());idx=case.bus_index_map()
    for gen in case.gen[case.gen[:,GEN_STATUS]>0]:vm[idx[int(gen[GEN_BUS])]]=gen[VG]
    return vm*np.exp(1j*va)
def _required(case,v,y):
    inj=v*np.conj(y@v)*case.base_mva;return inj.real+case.bus[:,PD],inj.imag+case.bus[:,QD]
def _update_outputs(case,pg,qg):
    for i,bus_number in enumerate(case.bus[:,BUS_I].astype(int)):
        gens=online_generators_at_bus(case,bus_number)
        if not gens.size:continue
        is_ref=int(case.bus[i,BUS_TYPE])==REF
        distribute_reactive_power(case,bus_number,float(qg[i]),clip_to_limits=not is_ref)
        if is_ref:
            fixed=float(np.sum(case.gen[gens[1:],PG])) if gens.size>1 else 0.0;case.gen[gens[0],PG]=float(pg[i]-fixed)

def run_ac_power_flow(input_case,options:PowerFlowOptions|None=None):
    options=options or PowerFlowOptions();case=input_case.clone();warnings=[];adm=build_ybus(case);v=_v0(case);total_it=0;history=[]
    for qround in range(options.max_q_limit_rounds+1):
        ref,pv,pq=_types(case); nr=solve_newton_raphson(adm.ybus,_sbus(case),v,ref,pv,pq,options.tolerance,options.max_iterations)
        total_it+=nr.iterations;history.extend(nr.mismatch_history);v=nr.voltage
        if not nr.converged:
            return PowerFlowResult(False,case,v,np.abs(v),np.rad2deg(np.angle(v)),total_it,qround,nr.max_mismatch,history,None,warnings)
        pg,qg=_required(case,v,adm.ybus)
        if not options.enforce_q_limits:
            _update_outputs(case,pg,qg);br=calculate_branch_flows(case,v,adm);return PowerFlowResult(True,case,v,np.abs(v),np.rad2deg(np.angle(v)),total_it,qround,nr.max_mismatch,history,br,warnings)
        violations=[]
        for bi in pv:
            busnum=int(case.bus[bi,BUS_I]);qmin,qmax=aggregate_q_limits(case,busnum);req=float(qg[bi])
            if req>qmax+options.q_limit_tolerance_mvar: violations.append((bi,qmax))
            elif req<qmin-options.q_limit_tolerance_mvar: violations.append((bi,qmin))
        if not violations:
            _update_outputs(case,pg,qg);br=calculate_branch_flows(case,v,adm);return PowerFlowResult(True,case,v,np.abs(v),np.rad2deg(np.angle(v)),total_it,qround,nr.max_mismatch,history,br,warnings)
        if qround>=options.max_q_limit_rounds:
            warnings.append('Reactive-power limit switching reached the configured round limit.');br=calculate_branch_flows(case,v,adm);return PowerFlowResult(False,case,v,np.abs(v),np.rad2deg(np.angle(v)),total_it,qround,nr.max_mismatch,history,br,warnings)
        for bi,limit in violations:
            busnum=int(case.bus[bi,BUS_I]);distribute_reactive_power(case,busnum,limit);case.bus[bi,BUS_TYPE]=PQ;warnings.append(f'Bus {busnum} converted from PV to PQ at aggregate Q limit {limit:g} MVAr.')
        adm=build_ybus(case)
    raise RuntimeError('Unreachable power-flow state')
