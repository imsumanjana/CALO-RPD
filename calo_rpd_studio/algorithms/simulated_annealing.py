"""Bounded continuous simulated annealing."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better
class SimulatedAnnealingOptimizer(BaseOptimizer):
    name='SA'
    def run(self):
        started=time.perf_counter();x=self.rng.random(self.problem.dimension);ev=self.evaluate(x);temp=float(self.config.parameters.get('temperature',1.0));cool=float(self.config.parameters.get('cooling',.995));scale=float(self.config.parameters.get('step_scale',.1))
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;c=np.clip(x+self.rng.normal(0,scale,self.problem.dimension),0,1);ce=self.evaluate(c)
            if ce is None:break
            delta=(ce.value-ev.value) if ce.feasible==ev.feasible else (ce.violation-ev.violation)
            if better(ce,ev) or self.rng.random()<np.exp(-max(delta,0)/max(temp,1e-12)):x=c;ev=ce
            temp*=cool;self.record()
        return self.finalize(np.asarray([x]),started=started)
