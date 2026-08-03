"""Change-C hierarchical action, mask, replay, and device invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.topology_context import (
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    build_topology_graph_state,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import (
    GroupActionMask,
    HierarchicalPolicyAction,
    TSHCALOPolicyNetwork,
    assign_group_conditioned_learner_operators,
    hierarchical_action,
    masked_group_operator_probabilities,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import (
    N_BOUNDED_PARAMETERS,
    N_CONTROL_GROUPS,
    N_OPERATORS,
    TSH_CALO_ACTION_SCHEMA,
)
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow


def _policy_state(toy_case) -> TopologyAwarePolicyState:
    decoder = ORPDVariableDecoder(toy_case, ORPDVariableConfig())
    result = run_ac_power_flow(toy_case)
    graph = build_topology_graph_state(
        toy_case,
        decoder,
        np.full(decoder.dimension, 0.5),
        result,
        [ScenarioDescriptor(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
    )
    return TopologyAwarePolicyState(np.linspace(0.0, 1.0, 32), graph)


def test_hierarchical_policy_outputs_versioned_global_and_group_heads(toy_case):
    torch.manual_seed(91)
    network = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()
    output = network(_policy_state(toy_case))

    assert output.regime_logits.shape == (4,)
    assert output.group_operator_logits.shape == (N_CONTROL_GROUPS, N_OPERATORS)
    assert output.context_operator_logits.shape == (4, N_OPERATORS)
    assert output.group_alpha.shape == (N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS)
    assert output.group_beta.shape == (N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS)
    assert output.value.shape == ()
    assert torch.all(output.group_alpha > 1.0)
    assert torch.all(output.group_beta > 1.0)


def test_action_mask_assigns_zero_probability_to_invalid_and_unavailable_groups():
    allowed = torch.zeros((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool)
    allowed[0, 2] = True
    mask = GroupActionMask(allowed, torch.tensor([True, False, False]))
    logits = torch.full((N_CONTROL_GROUPS, N_OPERATORS), 100.0)
    probabilities = masked_group_operator_probabilities(logits, mask)

    torch.testing.assert_close(
        probabilities[0],
        torch.nn.functional.one_hot(torch.tensor(2), 6).to(probabilities.dtype),
    )
    assert torch.count_nonzero(probabilities[1:]) == 0
    with pytest.raises(ValueError, match="permit at least one"):
        GroupActionMask(
            torch.zeros((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool),
            torch.tensor([True, False, False]),
        ).validate()


def test_deterministic_action_and_per_learner_group_assignment(toy_case):
    torch.manual_seed(17)
    output = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()(_policy_state(toy_case))
    mask = GroupActionMask(
        torch.ones((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool),
        torch.ones(N_CONTROL_GROUPS, dtype=torch.bool),
    )
    action = hierarchical_action(output, mask, deterministic=True)
    learner_groups = torch.tensor([0, 1, 2, 0, 2, 1])
    learner_contexts = torch.tensor([0, 1, 2, 3, 0, 1])
    assigned = assign_group_conditioned_learner_operators(
        action, learner_groups, learner_contexts, deterministic=True
    )

    action.validate()
    assert action.schema_version == TSH_CALO_ACTION_SCHEMA
    assert assigned.shape == learner_groups.shape
    assert torch.all((action.group_parameters >= 0.0) & (action.group_parameters <= 1.0))


def test_stochastic_hierarchical_action_and_learner_sampling_replay_exactly(toy_case):
    torch.manual_seed(301)
    output = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()(_policy_state(toy_case))
    mask = GroupActionMask(
        torch.ones((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool),
        torch.ones(N_CONTROL_GROUPS, dtype=torch.bool),
    )
    first_generator = torch.Generator().manual_seed(20260803)
    second_generator = torch.Generator().manual_seed(20260803)
    first = hierarchical_action(output, mask, deterministic=False, generator=first_generator)
    second = hierarchical_action(output, mask, deterministic=False, generator=second_generator)

    assert first.regime == second.regime
    torch.testing.assert_close(first.group_operators, second.group_operators)
    torch.testing.assert_close(first.group_parameters, second.group_parameters)

    learner_groups = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    learner_contexts = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    first_ops = assign_group_conditioned_learner_operators(
        first,
        learner_groups,
        learner_contexts,
        deterministic=False,
        generator=torch.Generator().manual_seed(77),
    )
    second_ops = assign_group_conditioned_learner_operators(
        second,
        learner_groups,
        learner_contexts,
        deterministic=False,
        generator=torch.Generator().manual_seed(77),
    )
    torch.testing.assert_close(first_ops, second_ops)


def test_learner_context_conditions_independent_operator_distribution(toy_case):
    output = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()(_policy_state(toy_case))
    mask = GroupActionMask(
        torch.ones((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool),
        torch.ones(N_CONTROL_GROUPS, dtype=torch.bool),
    )
    action = hierarchical_action(output, mask, deterministic=True)
    action.context_operator_logits.zero_()
    action.context_operator_logits[0, 1] = 100.0
    action.context_operator_logits[1, 4] = 100.0

    assigned = assign_group_conditioned_learner_operators(
        action,
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        deterministic=True,
    )

    assert assigned.tolist() == [1, 4]


def test_unavailable_group_uses_explicit_sentinel_and_cannot_receive_a_learner(toy_case):
    output = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()(_policy_state(toy_case))
    allowed = torch.zeros((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool)
    allowed[0] = True
    mask = GroupActionMask(allowed, torch.tensor([True, False, False]))
    action = hierarchical_action(output, mask, deterministic=True)

    assert action.group_operators.tolist()[1:] == [-1, -1]
    with pytest.raises(ValueError, match="unavailable"):
        assign_group_conditioned_learner_operators(
            action,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            deterministic=True,
        )


def test_hierarchical_action_rejects_wrong_schema(toy_case):
    output = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()(_policy_state(toy_case))
    mask = GroupActionMask(
        torch.ones((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool),
        torch.ones(N_CONTROL_GROUPS, dtype=torch.bool),
    )
    action = hierarchical_action(output, mask, deterministic=True)
    invalid = HierarchicalPolicyAction(
        "legacy-action-schema",
        action.regime,
        action.regime_probabilities,
        action.group_operators,
        action.group_operator_probabilities,
        action.context_operator_logits,
        action.group_parameters,
        action.action_mask,
    )
    with pytest.raises(ValueError, match="schema"):
        invalid.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hierarchical_policy_cpu_cuda_numerical_agreement(toy_case):
    torch.manual_seed(812)
    cpu = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()
    cuda = TSHCALOPolicyNetwork(hidden_dim=16).double().cuda().eval()
    cuda.load_state_dict(cpu.state_dict())
    state = _policy_state(toy_case)

    cpu_output = cpu(state)
    cuda_output = cuda(state)

    torch.testing.assert_close(
        cpu_output.regime_logits,
        cuda_output.regime_logits.cpu(),
        rtol=1e-9,
        atol=1e-10,
    )
    torch.testing.assert_close(
        cpu_output.group_operator_logits,
        cuda_output.group_operator_logits.cpu(),
        rtol=1e-9,
        atol=1e-10,
    )
