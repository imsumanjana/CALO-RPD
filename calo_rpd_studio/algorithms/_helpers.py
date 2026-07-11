"""Shared optimizer utility functions."""
from __future__ import annotations
import numpy as np
from calo_rpd_studio.orpd.feasibility_rules import better
def greedy_replace(pop,evals,candidates,candidate_evals):
    pop=np.asarray(pop).copy();out=list(evals)
    for i,(x,e) in enumerate(zip(candidates,candidate_evals)):
        if better(e,out[i]):pop[i]=x;out[i]=e
    return pop,out
def levy_flight(rng,shape,beta=1.5):
    from math import gamma,sin,pi
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=rng.normal(0,sigma,shape);v=rng.normal(0,1,shape);return u/(np.abs(v)**(1/beta)+1e-15)
