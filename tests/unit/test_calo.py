import numpy as np
import torch

from calo_rpd_studio.algorithms.calo.cognitive_state import STATE_DIM, build_cognitive_state
from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.success_memory import SuccessMemory
from calo_rpd_studio.orpd.problem import Evaluation


def test_cognitive_state_and_policy():
    pop = np.random.default_rng(1).random((8, 5))
    ev = [
        Evaluation(float(i + 1), True, 0, metadata={"constraint_components": {}}) for i in range(8)
    ]
    state = build_cognitive_state(
        pop,
        ev,
        epsilon=0.0,
        previous_best_violation=1.0,
        previous_best_objective=10.0,
        constraint_stagnation=0.2,
        objective_stagnation=0.1,
        remaining_budget=0.8,
        operator_credit=np.full(6, 1 / 6),
    )
    assert state.vector().shape == (STATE_DIM,)
    net = CALOPolicyNetwork(STATE_DIM)
    regime_logits, operator_logits, alpha, beta, value = net(
        torch.tensor(state.vector(), dtype=torch.float32)
    )
    assert regime_logits.shape == (4,)
    assert operator_logits.shape == (6,)
    assert alpha.shape == (6,)
    assert beta.shape == (6,)
    assert value.ndim == 0


def test_success_memory_is_bounded():
    m = SuccessMemory(capacity=3)
    rng = np.random.default_rng(2)
    for i in range(10):
        m.add(np.ones(4) * i, i % 6, 1, 0)
    assert len(m) == 3
    assert m.direction(4).shape == (4,)
    assert m.sample_direction(4, rng).shape == (4,)
