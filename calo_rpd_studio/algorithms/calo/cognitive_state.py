"""Scale-independent CALO cognitive search-state features."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(slots=True)
class CognitiveState:
    diversity:float;best_improvement:float;median_improvement:float;stagnation:float;feasible_ratio:float;mean_violation:float;elite_spread:float;remaining_budget:float;operator_success:np.ndarray
    def vector(self):return np.r_[self.diversity,self.best_improvement,self.median_improvement,self.stagnation,self.feasible_ratio,self.mean_violation,self.elite_spread,self.remaining_budget,np.asarray(self.operator_success,float)]
def population_diversity(pop):
    x=np.asarray(pop,float);return float(np.mean(np.linalg.norm(x-x.mean(0),axis=1))/max(np.sqrt(x.shape[1]),1e-12))
def build_cognitive_state(pop,evaluations,previous_best,previous_median,stagnation,remaining_budget,operator_success):
    vals=np.asarray([e.value if e.feasible else e.violation for e in evaluations],float);finite=vals[np.isfinite(vals)];current_best=float(finite.min()) if finite.size else 1e12;current_median=float(np.median(finite)) if finite.size else 1e12
    scale=max(abs(previous_best),1.0);bi=float(np.clip((previous_best-current_best)/scale,-1,1)) if np.isfinite(previous_best) else 0.0;mi=float(np.clip((previous_median-current_median)/max(abs(previous_median),1.0),-1,1)) if np.isfinite(previous_median) else 0.0
    feasible=float(np.mean([e.feasible for e in evaluations]));viol=np.asarray([min(e.violation,1e6) for e in evaluations],float);mv=float(np.tanh(np.mean(viol)))
    order=np.argsort(vals);elite=np.asarray(pop)[order[:max(2,len(pop)//5)]];elite_spread=population_diversity(elite) if len(elite)>1 else 0.0
    return CognitiveState(population_diversity(pop),bi,mi,float(np.clip(stagnation,0,1)),feasible,mv,elite_spread,float(np.clip(remaining_budget,0,1)),np.asarray(operator_success,float))
