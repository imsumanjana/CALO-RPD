"""Common optimizer interface, budget accounting, and provenance."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any
import time
import numpy as np
from calo_rpd_studio.orpd.feasibility_rules import better,sort_key
from .result import OptimizerResult
@dataclass(slots=True)
class OptimizerConfig:
    population_size:int=50;max_evaluations:int=5000;max_iterations:int=1000;parameters:dict[str,Any]=field(default_factory=dict)
class BaseOptimizer:
    name='BASE'
    def __init__(self,problem,config=None,seed=0,progress_callback=None,cancel_callback=None):
        self.problem=problem;self.config=config or OptimizerConfig();self.seed=int(seed);self.rng=np.random.default_rng(self.seed);self.progress_callback=progress_callback;self.cancel_callback=cancel_callback;self.evaluations=0;self.iteration=0;self.best_evaluation=None;self.best_vector=None;self.history=[]
    def cancelled(self):return bool(self.cancel_callback and self.cancel_callback())
    def can_evaluate(self,n=1):return not self.cancelled() and self.evaluations+n<=self.config.max_evaluations
    def evaluate(self,x):
        if not self.can_evaluate():return None
        ev=self.problem.evaluate(np.clip(np.asarray(x,float),0,1));self.evaluations+=1
        if better(ev,self.best_evaluation):self.best_evaluation=ev;self.best_vector=np.clip(np.asarray(x,float),0,1).copy()
        return ev
    def evaluate_population(self,pop):
        out=[]
        for x in pop:
            ev=self.evaluate(x)
            if ev is None:break
            out.append(ev)
        return out
    def random_population(self,n=None):return self.rng.random((n or self.config.population_size,self.problem.dimension))
    def record(self,extra=None):
        best=float('inf') if self.best_evaluation is None else float(self.best_evaluation.value);self.history.append(best)
        if self.progress_callback:self.progress_callback({'algorithm':self.name,'iteration':self.iteration,'evaluations':self.evaluations,'best_objective':best,'feasible':False if self.best_evaluation is None else self.best_evaluation.feasible,**(extra or {})})
    def order(self,evaluations):return sorted(range(len(evaluations)),key=lambda i:sort_key(evaluations[i]))
    def run(self):raise NotImplementedError
    def finalize(self,population=None,reason='budget_or_iteration_limit',metadata=None,started=None):
        if self.best_evaluation is None or self.best_vector is None:raise RuntimeError(f'{self.name} completed without an evaluated candidate')
        runtime=0.0 if started is None else time.perf_counter()-started;md=dict(metadata or {})
        # Post-run state reconstruction is not part of the optimizer evaluation budget.
        md['solution_state']=self.problem.solution_state(self.best_vector)
        ev=self.best_evaluation
        return OptimizerResult(self.name,self.seed,dict(self.config.parameters),self.best_vector.copy(),dict(ev.physical_controls),float(ev.value),dict(ev.components),float(ev.violation),bool(ev.feasible),self.evaluations,self.iteration,list(self.history),runtime,None if population is None else np.asarray(population).copy(),reason,md)
