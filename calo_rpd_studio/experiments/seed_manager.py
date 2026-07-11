"""Central deterministic seed derivation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True,slots=True)
class RunSeeds:algorithm_seed:int;scenario_seed:int;ai_inference_seed:int
class SeedManager:
    def __init__(self,master_seed):self.master_seed=int(master_seed)
    def generate(self,count):
        ss=np.random.SeedSequence(self.master_seed);children=ss.spawn(int(count));out=[]
        for child in children:
            vals=child.generate_state(3,dtype=np.uint32);out.append(RunSeeds(*(int(v) for v in vals)))
        return out
