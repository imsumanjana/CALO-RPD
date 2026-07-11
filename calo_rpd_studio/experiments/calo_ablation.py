"""Predefined CALO ablation suite."""
from dataclasses import dataclass
from copy import deepcopy
from calo_rpd_studio.algorithms.legacy_mtlbo import LegacyMTLBOOptimizer
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from .experiment_runner import run_single,CompletedRun,build_problem
@dataclass(frozen=True,slots=True)
class AblationSpec:label:str;algorithm:str='CALO';parameters:dict|None=None
ABLATION_SPECS=(AblationSpec('Classical TLBO','TLBO',{}),AblationSpec('Legacy Gaussian MTLBO','LEGACY',{}),AblationSpec('CALO without AI','CALO',{'use_ai':False}),AblationSpec('CALO without success memory','CALO',{'use_memory':False}),AblationSpec('CALO without stagnation recovery','CALO',{'use_recovery':False}),AblationSpec('CALO without diversity feedback','CALO',{'use_diversity':False}),AblationSpec('Complete CALO','CALO',{}))
def run_ablation(config,spec,run_index,seeds,progress_callback=None,cancel_callback=None):
    if spec.algorithm!='LEGACY':
        local=deepcopy(config);local.algorithm_parameters=dict(local.algorithm_parameters);local.algorithm_parameters[spec.algorithm]={**local.algorithm_parameters.get(spec.algorithm,{}),**(spec.parameters or {})};completed=run_single(local,spec.algorithm,run_index,seeds,progress_callback,cancel_callback);completed.result.algorithm=spec.label;completed.algorithm=spec.label;return completed
    problem=build_problem(config,seeds.scenario_seed);opt=LegacyMTLBOOptimizer(problem,OptimizerConfig(config.population_size,config.budget.max_evaluations,max(config.max_iterations,config.budget.max_evaluations),{}),seeds.algorithm_seed,progress_callback,cancel_callback);res=opt.run();res.algorithm=spec.label;return CompletedRun(spec.label,run_index,seeds,res)
