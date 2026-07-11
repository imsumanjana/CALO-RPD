"""Flower Pollination Algorithm."""
from __future__ import annotations
import time,numpy as np
from .base_optimizer import BaseOptimizer
from ._helpers import levy_flight
from calo_rpd_studio.orpd.feasibility_rules import better
class FlowerPollinationOptimizer(BaseOptimizer):
    name='FPA'
    def run(self):
        started=time.perf_counter();p=self.random_population();ev=self.evaluate_population(p);switch=float(self.config.parameters.get('switch_probability',.8))
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;best=p[self.order(ev)[0]].copy()
            for i in range(len(p)):
                if not self.can_evaluate():break
                if self.rng.random()<switch:x=p[i]+levy_flight(self.rng,(self.problem.dimension,))*(best-p[i])
                else:a,b=self.rng.choice(len(p),2,False);x=p[i]+self.rng.random()*(p[a]-p[b])
                x=np.clip(x,0,1);e=self.evaluate(x)
                if e is not None and better(e,ev[i]):p[i]=x;ev[i]=e
            self.record()
        return self.finalize(p,started=started)
