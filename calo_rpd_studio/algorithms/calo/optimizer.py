"""Cognitive Adaptive Learning Optimizer (CALO)."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer
from calo_rpd_studio.orpd.feasibility_rules import better
from .ai_controller import AIController
from .cognitive_state import build_cognitive_state,population_diversity
from .learning_operators import *
from .success_memory import SuccessMemory
from .reward import calculate_reward
class CALOOptimizer(BaseOptimizer):
    name='CALO'
    def _default_checkpoint(self):return Path(__file__).resolve().parents[2]/'data'/'trained_models'/'calo_policy_v1.pt'
    def run(self):
        started=time.perf_counter();params=self.config.parameters;n=self.config.population_size;pop=self.random_population();ev=self.evaluate_population(pop)
        if len(ev)<len(pop):return self.finalize(pop[:len(ev)],started=started)
        pbest=pop.copy();pbe=list(ev);memory=SuccessMemory(int(params.get('memory_capacity',256)),float(params.get('memory_decay',.97)));attempts=np.zeros(6,int);successes=np.zeros(6,int);stagnation=0;stagnation_window=max(3,int(params.get('stagnation_window',12)));previous_best=float('inf');previous_median=float('inf');rewards=[]
        use_ai=bool(params.get('use_ai',True));use_memory=bool(params.get('use_memory',True));use_diversity=bool(params.get('use_diversity',True));use_recovery=bool(params.get('use_recovery',True));deterministic=bool(params.get('deterministic_policy',False));seed=int(params.get('ai_inference_seed',self.seed+7919));checkpoint=params.get('policy_checkpoint',str(self._default_checkpoint()))
        controller=AIController(checkpoint if use_ai else None,seed=seed,deterministic=deterministic,input_dim=14)
        while self.iteration<self.config.max_iterations and self.can_evaluate():
            self.iteration+=1;order=self.order(ev);best=pop[order[0]].copy();mean=pop.mean(0);vals=np.asarray([x.value if x.feasible else x.violation for x in ev]);current_best=float(vals[order[0]]);current_median=float(np.median(vals));div=population_diversity(pop);feasible_ratio=float(np.mean([x.feasible for x in ev]));rates=memory.success_rates(6) if use_memory else np.zeros(6);remaining=max(0,1-self.evaluations/max(self.config.max_evaluations,1));state=build_cognitive_state(pop,ev,previous_best,previous_median,min(stagnation/stagnation_window,1),remaining,rates);decision=controller.decide(state.vector())
            op=decision.operator if use_ai else int(self.iteration%4);adaptive=decision.parameters
            if not use_diversity:adaptive['exploration_sigma']=.05
            # If search is severely infeasible or stagnant, prioritize the corresponding mode.
            if feasible_ratio<.15:op=4
            if use_recovery and stagnation>=stagnation_window:op=5
            attempts[op]+=n;new=[]
            feasible_indices=[i for i,e0 in enumerate(ev) if e0.feasible];low_violation=pop[min(range(n),key=lambda i:ev[i].violation)]
            for i,x in enumerate(pop):
                better_ids=order[:max(1,n//2)];bp=pop[int(self.rng.choice(better_ids))];distances=np.linalg.norm(pop-x,axis=1);dp=pop[int(np.argmax(distances))]
                if op==0:c=teacher_guided(x,best,mean,self.rng,adaptive['exploitation'],.25*adaptive['exploration_sigma'])
                elif op==1:c=contrastive_peer(x,bp,dp,self.rng,adaptive['peer_learning'],.3)
                elif op==2:c=self_reflective_memory(x,pbest[i],memory.direction(self.problem.dimension) if use_memory else np.zeros(self.problem.dimension),self.rng,.6,adaptive['memory_weight'])
                elif op==3:c=adaptive_exploration(best if self.rng.random()<.7 else x,self.rng,adaptive['exploration_sigma'])
                elif op==4:
                    elite=pop[feasible_indices[0]] if feasible_indices else low_violation;c=feasibility_recovery(x,elite,low_violation,self.rng,adaptive['recovery_intensity'])
                else:c=stagnation_escape(best,self.rng,max(adaptive['exploration_sigma'],.08)) if i in order[-max(1,int(n*adaptive['recovery_fraction'])):] else x.copy()
                new.append(c)
            new=np.asarray(new);ne=self.evaluate_population(new);accepted=0
            for i,candidate_ev in enumerate(ne):
                if better(candidate_ev,ev[i]):
                    old=pop[i].copy();old_ev=ev[i];pop[i]=new[i];ev[i]=candidate_ev;accepted+=1;successes[op]+=1
                    if better(candidate_ev,pbe[i]):pbest[i]=new[i];pbe[i]=candidate_ev
                    if use_memory:
                        og=max((old_ev.value-candidate_ev.value)/max(abs(old_ev.value),1),0) if old_ev.feasible and candidate_ev.feasible else 0;fg=max(old_ev.violation-candidate_ev.violation,0);memory.add(new[i]-old,op,og,fg)
            new_order=self.order(ev);new_best=float((ev[new_order[0]].value if ev[new_order[0]].feasible else ev[new_order[0]].violation));new_div=population_diversity(pop);new_feas=float(np.mean([x.feasible for x in ev]));reward=calculate_reward(current_best,new_best,feasible_ratio,new_feas,div,new_div,float(np.mean([min(x.violation,1) for x in ev])));rewards.append(reward.total)
            if new_best<current_best-1e-12:stagnation=0
            else:stagnation+=1
            previous_best=new_best;previous_median=current_median;self.record({'calo_operator':OPERATOR_NAMES[op],'diversity':new_div,'feasible_ratio':new_feas,'reward':reward.total,'accepted':accepted})
        md={'operator_names':list(OPERATOR_NAMES),'operator_attempts':attempts.tolist(),'operator_successes':successes.tolist(),'mean_reward':float(np.mean(rewards)) if rewards else 0.0,'success_memory_size':len(memory),'policy_checkpoint':controller.checkpoint_path,'policy_checksum':controller.checksum,'policy_metadata':controller.metadata,'ablation':{'use_ai':use_ai,'use_memory':use_memory,'use_diversity':use_diversity,'use_recovery':use_recovery}}
        return self.finalize(pop,metadata=md,started=started)
