"""Salp Swarm Algorithm."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
class SalpSwarmOptimizer(BaseOptimizer):
    name='SSA'
    def run(self):
        started=time.perf_counter();p=self.random_population();ev=self.evaluate_population(p)
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;food=p[self.order(ev)[0]];c1=2*np.exp(-(4*self.iteration/max(self.config.max_iterations,1))**2);new=p.copy();u=self.rng.random(self.problem.dimension);sign=np.where(self.rng.random(self.problem.dimension)<.5,1,-1);new[0]=np.clip(food+sign*c1*u,0,1)
            for i in range(1,len(p)):new[i]=.5*(p[i]+new[i-1])
            p=np.clip(new,0,1);ev=self.evaluate_population(p);self.record()
        return self.finalize(p,started=started)
