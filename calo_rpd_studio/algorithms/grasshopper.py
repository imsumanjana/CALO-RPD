"""Grasshopper Optimization Algorithm."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
class GrasshopperOptimizer(BaseOptimizer):
    name='GOA'
    @staticmethod
    def _social_interaction(distance,f=.5,length_scale=1.5):return f*np.exp(-distance/length_scale)-np.exp(-distance)
    def run(self):
        started=time.perf_counter();p=self.random_population();ev=self.evaluate_population(p)
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;target=p[self.order(ev)[0]];c=1-.99999*min(self.iteration/max(self.config.max_iterations,1),1);new=np.zeros_like(p)
            for i in range(len(p)):
                social=np.zeros(self.problem.dimension)
                for j in range(len(p)):
                    if i==j:continue
                    diff=p[j]-p[i];dist=np.linalg.norm(diff)+1e-12;social+=self._social_interaction(10*dist)*(diff/dist)
                new[i]=np.clip(target+c*social/max(len(p)-1,1),0,1)
            p=new;ev=self.evaluate_population(p);self.record()
        return self.finalize(p,started=started)
