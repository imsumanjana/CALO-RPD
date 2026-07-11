"""Normalized CALO controller reward components."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(slots=True)
class RewardComponents:
    objective_improvement:float;feasible_ratio_improvement:float;diversity_recovery:float;constraint_penalty:float;overhead_penalty:float
    @property
    def total(self):return 1.0*self.objective_improvement+0.7*self.feasible_ratio_improvement+0.25*self.diversity_recovery-0.5*self.constraint_penalty-0.05*self.overhead_penalty
def calculate_reward(old_best,new_best,old_feasible,new_feasible,old_diversity,new_diversity,mean_violation,overhead=0.0):
    scale=max(abs(old_best),1.0);objective=max(min((old_best-new_best)/scale,1),-1) if old_best<1e100 else 0.0;diversity=max(min(new_diversity-old_diversity,.5),-.5);return RewardComponents(objective,new_feasible-old_feasible,diversity,min(mean_violation,1.0),max(overhead,0.0))
