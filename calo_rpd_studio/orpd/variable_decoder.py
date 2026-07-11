"""Common normalized ORPD variable encoder/decoder used by every optimizer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from calo_rpd_studio.power_system.case_model import *
from .decision_variables import DecisionVariable,VariableKind
from .mixed_variable_handler import decode_continuous,decode_discrete,stepped_values
@dataclass(slots=True)
class ShuntControlDefinition:
    bus_number:int; minimum_mvar:float=0.0; maximum_mvar:float=5.0; step_mvar:float=1.0
@dataclass(slots=True)
class ORPDVariableConfig:
    generator_voltages:bool=True; transformer_taps:bool=True; shunt_compensation:bool=True
    discrete_transformer_taps:bool=True; discrete_shunts:bool=True
    transformer_minimum:float=0.90; transformer_maximum:float=1.10; transformer_step:float=0.0125
    shunt_controls:tuple[ShuntControlDefinition,...]=()

def default_shunt_controls(case):
    known={
      'case30':(10,12,15,17,20,21,23,24),
      'case57':(18,25,53),
      'case118':(5,34,37,44,45,46,48,74,79,82,83,105,107,110),
    }
    buses=set(case.bus[:,BUS_I].astype(int));return tuple(ShuntControlDefinition(b,0.0,5.0,1.0) for b in known.get(case.name,()) if b in buses)

class ORPDVariableDecoder:
    def __init__(self,case,config:ORPDVariableConfig):
        self.case=case;self.config=config;self.variables=[];self._actions=[];idx=case.bus_index_map()
        if config.generator_voltages:
            online=np.where(case.gen[:,GEN_STATUS]>0)[0]
            # one voltage control per unique online generator bus
            seen=set()
            for gi in online:
                bus=int(case.gen[gi,GEN_BUS])
                if bus in seen:continue
                seen.add(bus);bi=idx[bus];lo=float(case.bus[bi,VMIN]);hi=float(case.bus[bi,VMAX])
                self.variables.append(DecisionVariable(f'Vg@{bus}',lo,hi));self._actions.append(('vg',bus,lo,hi,None))
        if config.transformer_taps:
            taps=np.where((case.branch[:,BR_STATUS]>0)&(case.branch[:,TAP]!=0))[0]
            vals=stepped_values(config.transformer_minimum,config.transformer_maximum,config.transformer_step)
            for bi in taps:
                f=int(case.branch[bi,F_BUS]);t=int(case.branch[bi,T_BUS]);kind=VariableKind.DISCRETE if config.discrete_transformer_taps else VariableKind.CONTINUOUS
                self.variables.append(DecisionVariable(f'Tap {f}-{t}',config.transformer_minimum,config.transformer_maximum,kind,vals if kind is VariableKind.DISCRETE else ()))
                self._actions.append(('tap',int(bi),config.transformer_minimum,config.transformer_maximum,vals if kind is VariableKind.DISCRETE else None))
        controls=config.shunt_controls or default_shunt_controls(case)
        if config.shunt_compensation:
            for c in controls:
                vals=stepped_values(c.minimum_mvar,c.maximum_mvar,c.step_mvar);kind=VariableKind.DISCRETE if config.discrete_shunts else VariableKind.CONTINUOUS
                self.variables.append(DecisionVariable(f'Qsh@{c.bus_number}',c.minimum_mvar,c.maximum_mvar,kind,vals if kind is VariableKind.DISCRETE else ()))
                self._actions.append(('shunt',c.bus_number,c.minimum_mvar,c.maximum_mvar,vals if kind is VariableKind.DISCRETE else None))
    @property
    def dimension(self):return len(self.variables)
    def decode(self,normalized):
        z=np.asarray(normalized,dtype=float)
        if z.shape!=(self.dimension,):raise ValueError(f'Expected decision vector shape ({self.dimension},), got {z.shape}')
        out=self.case.clone();physical={};idx=out.bus_index_map()
        for value,action,var in zip(z,self._actions,self.variables):
            typ,target,lo,hi,values=action;decoded=decode_discrete(value,values) if values is not None else decode_continuous(value,lo,hi);physical[var.name]=decoded
            if typ=='vg':
                gens=np.where((out.gen[:,GEN_STATUS]>0)&(out.gen[:,GEN_BUS].astype(int)==target))[0];out.gen[gens,VG]=decoded;out.bus[idx[target],VM]=decoded
            elif typ=='tap':out.branch[target,TAP]=decoded
            elif typ=='shunt':out.bus[idx[target],BS]=decoded/out.base_mva*out.base_mva # MVAr at 1 pu equals BS MVAr
        return out,physical
    def control_validity(self,normalized):
        z=np.asarray(normalized,dtype=float);return bool(z.shape==(self.dimension,) and np.all(np.isfinite(z)) and np.all((z>=0)&(z<=1)))
