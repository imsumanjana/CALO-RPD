"""CALO multi-mode candidate generation operators."""
from __future__ import annotations
import numpy as np
OPERATOR_NAMES=('teacher_guided','contrastive_peer','self_reflective_memory','adaptive_exploration','feasibility_recovery','stagnation_escape')
def teacher_guided(x,best,mean,rng,alpha,beta):
    z1=np.abs(rng.normal(size=x.shape));z2=rng.normal(size=x.shape);return np.clip(x+alpha*z1*(best-x)+beta*z2*(best-mean),0,1)
def contrastive_peer(x,better_peer,diverse_peer,rng,gamma,delta):return np.clip(x+gamma*rng.random(x.shape)*(better_peer-x)+delta*rng.random(x.shape)*(x-diverse_peer),0,1)
def self_reflective_memory(x,personal,memory_direction,rng,eta,mu):return np.clip(x+eta*rng.random(x.shape)*(personal-x)+mu*memory_direction,0,1)
def adaptive_exploration(reference,rng,sigma):return np.clip(reference+sigma*rng.normal(size=reference.shape),0,1)
def feasibility_recovery(x,feasible_elite,low_violation,rng,intensity):return np.clip(x+intensity*rng.random(x.shape)*(feasible_elite-x)+.5*intensity*rng.random(x.shape)*(low_violation-x),0,1)
def stagnation_escape(elite,rng,sigma):return np.clip(elite+max(sigma,.02)*rng.normal(size=elite.shape),0,1)
