"""Multi-Verse Optimizer."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
class MultiVerseOptimizer(BaseOptimizer):
    name='MVO'
    def run(self):
        started=time.perf_counter();p=self.random_population();ev=self.evaluate_population(p)
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;order=self.order(ev);best=p[order[0]].copy();ranks=np.empty(len(p));ranks[order]=np.arange(len(p));norm=(len(p)-ranks)/max(len(p),1);wep=.2+.8*self.iteration/max(self.config.max_iterations,1);tdr=1-(self.iteration/max(self.config.max_iterations,1))**(1/6)
            new=p.copy()
            for i in range(len(p)):
                for j in range(self.problem.dimension):
                    if self.rng.random()<norm[i]:donor=int(self.rng.choice(len(p),p=norm/norm.sum()));new[i,j]=p[donor,j]
                    if self.rng.random()<wep:new[i,j]=best[j]+(-1 if self.rng.random()<.5 else 1)*tdr*self.rng.random()
            p=np.clip(new,0,1);ev=self.evaluate_population(p);self.record()
        return self.finalize(p,started=started)
