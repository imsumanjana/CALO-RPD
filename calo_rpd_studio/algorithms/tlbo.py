"""Canonical teaching-learning-based optimization."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better
class TLBOOptimizer(BaseOptimizer):
    name='TLBO'
    def run(self):
        started=time.perf_counter();pop=self.random_population();ev=self.evaluate_population(pop)
        if len(ev)<len(pop):return self.finalize(pop[:len(ev)],started=started)
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;best=pop[self.order(ev)[0]].copy();mean=pop.mean(axis=0);tf=int(self.rng.integers(1,3));cand=np.clip(pop+self.rng.random(pop.shape)*(best-tf*mean),0,1);ce=self.evaluate_population(cand)
            for i,e in enumerate(ce):
                if better(e,ev[i]):pop[i]=cand[i];ev[i]=e
            for i in range(len(pop)):
                if not self.can_evaluate():break
                j=int(self.rng.integers(0,len(pop)-1));j+=j>=i;direction=pop[i]-pop[j] if better(ev[i],ev[j]) else pop[j]-pop[i];x=np.clip(pop[i]+self.rng.random(self.problem.dimension)*direction,0,1);e=self.evaluate(x)
                if e is not None and better(e,ev[i]):pop[i]=x;ev[i]=e
            self.record()
        return self.finalize(pop,started=started)
