"""Serializable complete experiment configuration."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from enum import Enum
import json
from pathlib import Path
import yaml
from calo_rpd_studio.orpd.objectives import ObjectiveConfig,ObjectiveKind
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig,ShuntControlDefinition
from calo_rpd_studio.robustness.robust_objectives import RobustObjectiveConfig,RobustAggregation
from .evaluation_budget import EvaluationBudget,BudgetPolicy
@dataclass(slots=True)
class RobustScenarioSettings:
    mode:str='deterministic';count:int=20;active_load_std:float=.05;reactive_load_std:float=.05;branch_outages:list[int]=field(default_factory=list);generator_outages:list[int]=field(default_factory=list);renewable_bus:int=0;renewable_rated_mw:float=0.0;renewable_mean_capacity_factor:float=.5;renewable_std_capacity_factor:float=.15
@dataclass(slots=True)
class ExperimentConfig:
    name:str='CALO-RPD comparative experiment';case_name:str='case30';algorithms:list[str]=field(default_factory=lambda:['CALO','TLBO','PSO']);runs:int=5;master_seed:int=2026;population_size:int=50;max_iterations:int=1000;budget:EvaluationBudget=field(default_factory=EvaluationBudget);objective:ObjectiveConfig=field(default_factory=ObjectiveConfig);variables:ORPDVariableConfig=field(default_factory=ORPDVariableConfig);robust_objective:RobustObjectiveConfig=field(default_factory=RobustObjectiveConfig);scenarios:RobustScenarioSettings=field(default_factory=RobustScenarioSettings);algorithm_parameters:dict[str,dict]=field(default_factory=dict);output_directory:str='results_data';parallel_workers:int=1
    def validate(self):
        from calo_rpd_studio.algorithms.registry import SPECS
        if self.runs<=0:raise ValueError('runs must be positive')
        if self.population_size<=0:raise ValueError('population_size must be positive')
        if not self.algorithms:raise ValueError('At least one algorithm must be selected')
        unknown=[x for x in self.algorithms if x not in SPECS]
        if unknown:raise ValueError(f'Unknown primary algorithms: {unknown}')
        if self.parallel_workers<=0:raise ValueError('parallel_workers must be positive')
        self.budget.validate()
    def to_dict(self):
        def cv(v):
            if isinstance(v,Enum):return v.value
            if isinstance(v,dict):return {str(k):cv(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [cv(x) for x in v]
            return v
        return cv(asdict(self))
    def save(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);data=self.to_dict();p.write_text(yaml.safe_dump(data,sort_keys=False) if p.suffix.lower() in {'.yaml','.yml'} else json.dumps(data,indent=2),encoding='utf-8');return p
    @classmethod
    def from_dict(cls,d):
        od=d.get('objective',{});obj=ObjectiveConfig(ObjectiveKind(od.get('kind',ObjectiveKind.ACTIVE_POWER_LOSS.value)),float(od.get('weight_loss',1)),float(od.get('weight_voltage_deviation',0)),float(od.get('weight_l_index',0)),float(od.get('loss_scale',1)),float(od.get('voltage_deviation_scale',1)),float(od.get('l_index_scale',1)))
        vd=d.get('variables',{});sh=tuple(ShuntControlDefinition(**x) for x in vd.get('shunt_controls',[]));var=ORPDVariableConfig(bool(vd.get('generator_voltages',True)),bool(vd.get('transformer_taps',True)),bool(vd.get('shunt_compensation',True)),bool(vd.get('discrete_transformer_taps',True)),bool(vd.get('discrete_shunts',True)),float(vd.get('transformer_minimum',.9)),float(vd.get('transformer_maximum',1.1)),float(vd.get('transformer_step',.0125)),sh)
        rd=d.get('robust_objective',{});rob=RobustObjectiveConfig(RobustAggregation(rd.get('aggregation',RobustAggregation.EXPECTED.value)),float(rd.get('risk_lambda',1)),float(rd.get('cvar_alpha',.95)))
        bd=d.get('budget',{});bud=EvaluationBudget(BudgetPolicy(bd.get('policy',BudgetPolicy.EQUAL_EVALUATIONS.value)),int(bd.get('max_evaluations',5000)),bd.get('wall_clock_seconds'))
        return cls(d.get('name','CALO-RPD comparative experiment'),d.get('case_name','case30'),list(d.get('algorithms',['CALO','TLBO','PSO'])),int(d.get('runs',5)),int(d.get('master_seed',2026)),int(d.get('population_size',50)),int(d.get('max_iterations',1000)),bud,obj,var,rob,RobustScenarioSettings(**d.get('scenarios',{})),dict(d.get('algorithm_parameters',{})),d.get('output_directory','results_data'),int(d.get('parallel_workers',1)))
    @classmethod
    def load(cls,path):
        p=Path(path);text=p.read_text(encoding='utf-8');data=yaml.safe_load(text) if p.suffix.lower() in {'.yaml','.yml'} else json.loads(text);return cls.from_dict(data)
