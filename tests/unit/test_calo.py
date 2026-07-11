import numpy as np
from calo_rpd_studio.algorithms.calo.cognitive_state import build_cognitive_state
from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.success_memory import SuccessMemory
from calo_rpd_studio.orpd.problem import Evaluation

def test_cognitive_state_and_policy():
    pop=np.random.default_rng(1).random((8,5));ev=[Evaluation(float(i+1),True,0) for i in range(8)]
    state=build_cognitive_state(pop,ev,10,10,.2,.8,np.zeros(6))
    assert state.vector().shape==(14,)
    net=CALOPolicyNetwork(14)
    import torch
    logits,params,value=net(torch.tensor(state.vector(),dtype=torch.float32))
    assert logits.shape==(6,)
    assert params.shape==(6,)
    assert value.ndim==0

def test_success_memory_is_bounded():
    m=SuccessMemory(capacity=3)
    for i in range(10):m.add(np.ones(4)*i,i%6,1,0)
    assert len(m)==3
    assert m.direction(4).shape==(4,)
