"""Shared ORPD evaluator used without algorithm-specific physics."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any
import numpy as np
from calo_rpd_studio.power_system.ac_power_flow import PowerFlowOptions,run_ac_power_flow
from calo_rpd_studio.power_system.case_model import *
from calo_rpd_studio.power_system.voltage_stability import kessel_glavitsch_l_index
from calo_rpd_studio.robustness.robust_objectives import RobustObjectiveConfig,aggregate_robust
from calo_rpd_studio.robustness.scenario import Scenario
from .constraints import evaluate_constraints
from .objectives import ObjectiveConfig,calculate_objective
from .variable_decoder import ORPDVariableConfig,ORPDVariableDecoder
@dataclass(slots=True)
class ORPDProblemConfig:
    objective:ObjectiveConfig=field(default_factory=ObjectiveConfig);variables:ORPDVariableConfig=field(default_factory=ORPDVariableConfig);robust:RobustObjectiveConfig=field(default_factory=RobustObjectiveConfig);power_flow:PowerFlowOptions=field(default_factory=PowerFlowOptions)
@dataclass(slots=True)
class Evaluation:
    value:float;feasible:bool;violation:float;components:dict[str,float]=field(default_factory=dict);physical_controls:dict[str,float]=field(default_factory=dict);scenario_values:list[float]=field(default_factory=list);metadata:dict[str,Any]=field(default_factory=dict)
class ORPDProblem:
    def __init__(self,case,config=None,scenarios=None):
        self.case=case.clone();self.config=config or ORPDProblemConfig();self.decoder=ORPDVariableDecoder(self.case,self.config.variables);self.scenarios=scenarios or [Scenario('base')]
    @property
    def dimension(self):return self.decoder.dimension
    def evaluate(self,normalized):
        z=np.clip(np.asarray(normalized,float),0,1);controlled,physical=self.decoder.decode(z);values=[];violations=[];weights=[];scenario_values=[];comp_acc={}
        for scenario in self.scenarios:
            pf=run_ac_power_flow(scenario.apply(controlled),self.config.power_flow);obj=calculate_objective(pf,self.config.objective);con=evaluate_constraints(pf)
            value=float(obj.value);values.append(value);violations.append(float(con.total));weights.append(float(scenario.weight));scenario_values.append(value)
            for k,v in obj.components.items():comp_acc.setdefault(k,[]).append(float(v))
        w=np.asarray(weights,float);w=w/w.sum();finite=np.asarray(values,float)
        robust=float('inf') if not np.all(np.isfinite(finite)) else aggregate_robust(values,w,self.config.robust)
        violation=float(np.sum(w*np.asarray(violations))) if np.all(np.isfinite(violations)) else float('inf');feasible=violation<=1e-12 and np.isfinite(robust)
        components={k:float(np.sum(w*np.asarray(v))) for k,v in comp_acc.items()};components['scenario_objective_mean']=float(np.sum(w*finite)) if np.all(np.isfinite(finite)) else float('inf');components['scenario_objective_std']=float(np.sqrt(np.sum(w*(finite-components['scenario_objective_mean'])**2))) if np.all(np.isfinite(finite)) else float('inf')
        return Evaluation(robust,feasible,violation,components,physical,scenario_values,{'scenario_count':len(self.scenarios)})
    def solution_state(self,normalized):
        z=np.clip(np.asarray(normalized,float),0,1);controlled,physical=self.decoder.decode(z);records=[]
        for sc in self.scenarios:
            pf=run_ac_power_flow(sc.apply(controlled),self.config.power_flow);obj=calculate_objective(pf,self.config.objective);con=evaluate_constraints(pf);online=np.where(pf.case.gen[:,GEN_STATUS]>0)[0]
            rec={'scenario':sc.name,'weight':float(sc.weight),'converged':bool(pf.converged),'iterations':int(pf.iterations),'max_mismatch':float(pf.max_mismatch),'bus_numbers':pf.case.bus[:,BUS_I].astype(int).tolist(),'vm_pu':pf.vm_pu.tolist(),'va_deg':pf.va_deg.tolist(),'generator_bus':pf.case.gen[online,GEN_BUS].astype(int).tolist(),'pg_mw':pf.case.gen[online,PG].tolist(),'qg_mvar':pf.case.gen[online,QG].tolist(),'objective':float(obj.value),'objective_components':dict(obj.components),'constraint_components':dict(con.components),'total_constraint_violation':float(con.total),'total_loss_mw':float(pf.total_loss_mw),'l_index_max':float(kessel_glavitsch_l_index(pf.case,pf.voltage).maximum) if pf.converged else float('inf')}
            if pf.branch is not None:rec.update({'branch_from_bus':pf.case.branch[:,F_BUS].astype(int).tolist(),'branch_to_bus':pf.case.branch[:,T_BUS].astype(int).tolist(),'p_from_mw':np.real(pf.branch.s_from_mva).tolist(),'q_from_mvar':np.imag(pf.branch.s_from_mva).tolist(),'p_to_mw':np.real(pf.branch.s_to_mva).tolist(),'q_to_mvar':np.imag(pf.branch.s_to_mva).tolist(),'loading_percent':pf.branch.loading_percent.tolist()})
            records.append(rec)
        return {'normalized_decision_vector':z.tolist(),'decoded_controls':physical,'case_checksum':self.case.checksum(),'scenarios':records}
