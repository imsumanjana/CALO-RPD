"""Run construction, robust scenarios, budget enforcement, and failure isolation."""
from __future__ import annotations
from dataclasses import dataclass,field
import time,traceback
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.registry import SPECS,create_optimizer
from calo_rpd_studio.algorithms.result import OptimizerResult
from calo_rpd_studio.orpd.problem import ORPDProblem,ORPDProblemConfig
from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.robustness.scenario import Scenario
from calo_rpd_studio.robustness.scenario_generator import ScenarioGeneratorConfig,generate_load_scenarios
from calo_rpd_studio.robustness.renewable_uncertainty import renewable_scenarios
from calo_rpd_studio.robustness.contingencies import n_minus_one_branch_scenarios,n_minus_one_generator_scenarios
from .evaluation_budget import BudgetPolicy
from .seed_manager import SeedManager,RunSeeds
@dataclass(slots=True)
class CompletedRun:algorithm:str;run_index:int;seeds:RunSeeds;result:OptimizerResult
@dataclass(slots=True)
class FailedRun:algorithm:str;run_index:int;seeds:RunSeeds;failure_type:str;message:str;traceback_text:str;evaluation_count:int=0;numerical_state:dict=field(default_factory=dict)
def failed_run_from_exception(algorithm,run_index,seeds,exc):return FailedRun(algorithm,run_index,seeds,type(exc).__name__,str(exc),''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))
def build_scenarios(config,seed):
    s=config.scenarios
    if s.mode=='deterministic':return [Scenario('base',1.0)]
    if s.mode in {'load_uncertainty','monte_carlo'}:return generate_load_scenarios(ScenarioGeneratorConfig(s.count,s.active_load_std,s.reactive_load_std),seed)
    if s.mode=='renewable_uncertainty':
        if s.renewable_bus<=0 or s.renewable_rated_mw<=0:raise ValueError('Renewable uncertainty requires a valid bus number and positive rated power')
        import numpy as np
        return renewable_scenarios(s.count,s.renewable_bus,s.renewable_rated_mw,s.renewable_mean_capacity_factor,s.renewable_std_capacity_factor,np.random.default_rng(seed))
    if s.mode=='branch_contingency':return n_minus_one_branch_scenarios(s.branch_outages)
    if s.mode=='generator_contingency':return n_minus_one_generator_scenarios(s.generator_outages)
    raise ValueError(f'Unsupported scenario mode: {s.mode}')
def build_problem(config,scenario_seed):
    case=CaseLoader.load(config.case_name)
    problem_config=ORPDProblemConfig(config.objective,config.variables,config.robust_objective)
    scenarios=build_scenarios(config,scenario_seed)
    if str(getattr(config,'scientific_backend','cpu_reference'))=='torch_fp64':
        return AcceleratedORPDProblem(
            case,problem_config,scenarios,
            device=str(getattr(config,'runtime_compute_device','cpu')),
            dtype_name='float64',
            batch_size=int(getattr(config,'tensor_batch_size',64)),
            device_resident=bool(getattr(config,'device_resident_execution',True)),
        )
    return ORPDProblem(case,problem_config,scenarios)
def run_single(config,algorithm,run_index,seeds,progress_callback=None,cancel_callback=None):
    problem=build_problem(config,seeds.scenario_seed);defaults=dict(SPECS[algorithm].default_parameters);defaults.update(config.algorithm_parameters.get(algorithm,{}))
    defaults.setdefault('execution_device',str(getattr(config,'runtime_compute_device','cpu')))
    if str(getattr(config,'scientific_backend','cpu_reference'))=='torch_fp64':defaults.setdefault('optimizer_backend','torch')
    if algorithm=='CALO':
        defaults.setdefault('ai_inference_seed',seeds.ai_inference_seed)
        defaults.setdefault('inference_device',str(getattr(config,'runtime_compute_device','cpu')))
    started=time.perf_counter();policy=config.budget.policy
    if policy is BudgetPolicy.EQUAL_WALL_CLOCK:max_evaluations=2_000_000_000;max_iterations=2_000_000_000
    elif policy is BudgetPolicy.EQUAL_EVALUATIONS:max_evaluations=config.budget.max_evaluations;max_iterations=max(config.max_iterations,config.budget.max_evaluations)
    else:max_evaluations=config.budget.max_evaluations;max_iterations=config.max_iterations
    def cancel():
        if cancel_callback and cancel_callback():return True
        return bool(policy is BudgetPolicy.EQUAL_WALL_CLOCK and config.budget.wall_clock_seconds is not None and time.perf_counter()-started>=config.budget.wall_clock_seconds)
    opt=create_optimizer(algorithm,problem,OptimizerConfig(config.population_size,max_evaluations,max_iterations,defaults),seeds.algorithm_seed,progress_callback,cancel);result=opt.run()
    if policy is BudgetPolicy.EQUAL_WALL_CLOCK and not (cancel_callback and cancel_callback()) and config.budget.wall_clock_seconds and time.perf_counter()-started>=config.budget.wall_clock_seconds:result.termination_reason='wall_clock_budget'
    return CompletedRun(algorithm,run_index,seeds,result)
def run_sequential(config,progress_callback=None,cancel_callback=None):
    config.validate();seeds=SeedManager(config.master_seed).generate(config.runs);out=[]
    for ri in range(config.runs):
        for algo in config.algorithms:
            if cancel_callback and cancel_callback():return out
            out.append(run_single(config,algo,ri,seeds[ri],progress_callback,cancel_callback))
    return out
def run_sequential_resilient(config,progress_callback=None,cancel_callback=None):
    config.validate();seeds=SeedManager(config.master_seed).generate(config.runs);done=[];failed=[]
    for ri in range(config.runs):
        for algo in config.algorithms:
            if cancel_callback and cancel_callback():return done,failed
            try:done.append(run_single(config,algo,ri,seeds[ri],progress_callback,cancel_callback))
            except Exception as exc:failed.append(failed_run_from_exception(algo,ri,seeds[ri],exc))
    return done,failed
