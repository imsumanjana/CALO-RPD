"""Bounded recency-weighted memory of successful CALO movements."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
@dataclass(slots=True)
class SuccessRecord: direction:np.ndarray;operator:int;step_norm:float;objective_gain:float;feasibility_gain:float
class SuccessMemory:
    def __init__(self,capacity=256,decay=.97):self.records=deque(maxlen=int(capacity));self.decay=float(decay)
    def add(self,direction,operator,objective_gain=0.0,feasibility_gain=0.0):self.records.append(SuccessRecord(np.asarray(direction,float).copy(),int(operator),float(np.linalg.norm(direction)),float(objective_gain),float(feasibility_gain)))
    def direction(self,dimension):
        if not self.records:return np.zeros(dimension)
        weights=np.asarray([self.decay**(len(self.records)-1-i) for i in range(len(self.records))]);dirs=np.asarray([x.direction for x in self.records]);return np.sum(dirs*weights[:,None],axis=0)/weights.sum()
    def success_rates(self,n_operators=6):
        rates=np.zeros(n_operators);counts=np.zeros(n_operators)
        for rec in self.records:counts[rec.operator]+=1;rates[rec.operator]+=max(rec.objective_gain,0)+max(rec.feasibility_gain,0)+1e-6
        nz=counts>0;rates[nz]/=counts[nz]
        if rates.max()>0:rates/=rates.max()
        return rates
    def __len__(self):return len(self.records)
