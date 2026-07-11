"""Modified teaching-learning search hybridized with DE/rand/1/bin."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better
class MTLADEOptimizer(BaseOptimizer):
    name='MTLA-DE'
    def run(self):
        started=time.perf_counter();p=self.random_population();ev=self.evaluate_population(p);n=len(p);f=float(self.config.parameters.get('f',.5));cr=float(self.config.parameters.get('cr',.9))
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;best=p[self.order(ev)[0]];mean=p.mean(0);teach=np.clip(p+self.rng.normal(0,1,p.shape)*(best-mean),0,1);te=self.evaluate_population(teach)
            for i,e in enumerate(te):
                if better(e,ev[i]):p[i]=teach[i];ev[i]=e
            for i in range(n):
                if not self.can_evaluate():break
                pool=[j for j in range(n) if j!=i];a,b,c=self.rng.choice(pool,3,replace=False);mut=np.clip(p[a]+f*(p[b]-p[c]),0,1);mask=self.rng.random(self.problem.dimension)<cr;mask[self.rng.integers(self.problem.dimension)]=True;x=np.where(mask,mut,p[i]);e=self.evaluate(x)
                if e is not None and better(e,ev[i]):p[i]=x;ev[i]=e
            self.record()
        return self.finalize(p,started=started)
