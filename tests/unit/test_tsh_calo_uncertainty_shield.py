"""Change-D uncertainty, OOD, bandit, shield, and fallback invariants."""

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
    TSHCALOPolicyNetwork,
    TSHCALOPolicyOutput,
    hierarchical_action,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import (
    N_CONTROL_GROUPS,
    N_OPERATORS,
    TSH_CALO_ALGORITHM_ID,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import (
    EnsembleDisagreement,
    FallbackDisposition,
    OODCalibration,
    SafetyEnvelope,
    SlidingWindowContextualBandit,
    UncertaintySafetyShield,
    aggregate_policy_ensemble,
    resolve_policy_fallback,
    topology_ood_signature,
)
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow


def _state(toy_case) -> TopologyAwarePolicyState:
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


def _output_and_action(toy_case):
    torch.manual_seed(55)
    output = TSHCALOPolicyNetwork(hidden_dim=16).double().eval()(_state(toy_case))
    mask = GroupActionMask(
        torch.ones((N_CONTROL_GROUPS, N_OPERATORS), dtype=torch.bool),
        torch.ones(N_CONTROL_GROUPS, dtype=torch.bool),
    )
    return output, hierarchical_action(output, mask, deterministic=True)


def test_ensemble_disagreement_is_zero_for_identical_members_and_increases_on_conflict(toy_case):
    output, _action = _output_and_action(toy_case)
    identical = aggregate_policy_ensemble([output, output])
    group_conflict = torch.full((1, N_OPERATORS), -8.0)
    group_conflict[0, 0] = 8.0
    context_conflict = torch.full((1, N_OPERATORS), -8.0)
    context_conflict[0, 1] = 8.0
    conflicting = TSHCALOPolicyOutput(
        regime_logits=output.regime_logits + torch.tensor([8.0, -8.0, -8.0, -8.0]),
        group_operator_logits=output.group_operator_logits + group_conflict.expand(3, -1),
        context_operator_logits=output.context_operator_logits + context_conflict.expand(4, -1),
        group_alpha=output.group_alpha,
        group_beta=output.group_beta,
        value=output.value,
    )
    disagreement = aggregate_policy_ensemble([output, conflicting]).disagreement

    assert identical.disagreement.regime == pytest.approx(0.0)
    assert torch.count_nonzero(identical.disagreement.groups) == 0
    assert disagreement.regime > 0.0
    assert torch.all(disagreement.groups > 0.0)
    assert torch.all(disagreement.contexts > 0.0)


def test_ood_calibration_is_frozen_shape_checked_and_attenuates_distant_state(toy_case):
    signature = topology_ood_signature(_state(toy_case))
    calibration = OODCalibration(np.zeros_like(signature), np.ones_like(signature), 0.5)
    near_score, near_weight = calibration.score_and_attenuation(np.zeros_like(signature))
    far_score, far_weight = calibration.score_and_attenuation(np.full_like(signature, 8.0))

    assert near_score == 0.0
    assert near_weight == 1.0
    assert far_score > near_score
    assert 0.0 <= far_weight < near_weight
    with pytest.raises(ValueError, match="incompatible"):
        calibration.score_and_attenuation(np.zeros(len(signature) - 1))


def test_contextual_bandit_is_sliding_window_checkpoint_exact_and_reward_responsive():
    bandit = SlidingWindowContextualBandit(window_size=4, exploration=0.1)
    before = bandit.probabilities(1, 2)
    for reward in (1.0, 2.0, 3.0, 4.0, 8.0):
        bandit.update(1, 2, 4, reward)
    after = bandit.probabilities(1, 2)
    restored = SlidingWindowContextualBandit.from_state_dict(bandit.state_dict())

    assert before[4] == pytest.approx(1.0 / N_OPERATORS)
    assert after[4] == pytest.approx(max(after))
    np.testing.assert_array_equal(restored.rewards, bandit.rewards)
    np.testing.assert_array_equal(restored.cursor, bandit.cursor)
    np.testing.assert_array_equal(restored.count, bandit.count)
    np.testing.assert_allclose(restored.probabilities(1, 2), after, rtol=0.0, atol=0.0)


def test_shield_masks_invalid_actions_preserves_exploration_and_attenuates_uncertainty(toy_case):
    _output, action = _output_and_action(toy_case)
    allowed = action.action_mask.allowed.clone()
    allowed[:, 5] = False
    action.action_mask = GroupActionMask(allowed, action.action_mask.available_groups)
    bandit = SlidingWindowContextualBandit()
    groups = torch.tensor([0, 1, 2, 0])
    contexts = torch.tensor([0, 1, 2, 3])
    safe = SafetyEnvelope(remaining_evaluations=4, candidate_count=4)
    shield = UncertaintySafetyShield()
    zero = EnsembleDisagreement(0.0, torch.zeros(3), torch.zeros(4))
    high = EnsembleDisagreement(0.2, torch.full((3,), 0.2), torch.full((4,), 0.2))

    confident = shield.resolve(
        action=action,
        disagreement=zero,
        ood_score=0.0,
        ood_attenuation=1.0,
        learner_groups=groups,
        learner_contexts=contexts,
        bandit=bandit,
        safety=safe,
    )
    uncertain = shield.resolve(
        action=action,
        disagreement=high,
        ood_score=4.0,
        ood_attenuation=0.25,
        learner_groups=groups,
        learner_contexts=contexts,
        bandit=bandit,
        safety=safe,
    )

    assert torch.all(confident.probabilities[:, 5] == 0.0)
    assert torch.all(confident.probabilities[:, :5] > 0.0)
    torch.testing.assert_close(
        confident.probabilities.sum(dim=1), torch.ones(4, dtype=torch.float64)
    )
    assert torch.all(uncertain.trace.mixture_weights[:, 0] < confident.trace.mixture_weights[:, 0])
    assert "ood_attenuated_neural_weight" in uncertain.trace.intervention_reasons
    assert "invalid_operators_masked" in confident.trace.intervention_reasons


def test_shield_sampling_replays_and_budget_or_lattice_violation_fails_closed(toy_case):
    _output, action = _output_and_action(toy_case)
    shield = UncertaintySafetyShield()
    disagreement = EnsembleDisagreement(0.0, torch.zeros(3), torch.zeros(4))
    kwargs = dict(
        action=action,
        disagreement=disagreement,
        ood_score=0.0,
        ood_attenuation=1.0,
        learner_groups=torch.tensor([0, 1]),
        learner_contexts=torch.tensor([0, 1]),
        bandit=SlidingWindowContextualBandit(),
    )
    decision = shield.resolve(**kwargs, safety=SafetyEnvelope(2, 2))
    first = shield.sample(
        decision, deterministic=False, generator=torch.Generator().manual_seed(913)
    )
    second = shield.sample(
        decision, deterministic=False, generator=torch.Generator().manual_seed(913)
    )

    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="remaining FE budget"):
        shield.resolve(**kwargs, safety=SafetyEnvelope(1, 2))
    with pytest.raises(ValueError, match="mixed-variable lattice"):
        shield.resolve(**kwargs, safety=SafetyEnvelope(2, 2, False))


def test_policy_fallback_is_explicitly_blocked_or_relabelled_as_frozen_baseline():
    accepted = resolve_policy_fallback(
        policy_usable=True,
        rejection_reason="",
        baseline_fallback_permitted=False,
        tsh_algorithm_identity=TSH_CALO_ALGORITHM_ID,
        frozen_baseline_identity="CALO-v5.9",
    )
    blocked = resolve_policy_fallback(
        policy_usable=False,
        rejection_reason="checksum rejected",
        baseline_fallback_permitted=False,
        tsh_algorithm_identity=TSH_CALO_ALGORITHM_ID,
        frozen_baseline_identity="CALO-v5.9",
    )
    fallback = resolve_policy_fallback(
        policy_usable=False,
        rejection_reason="ABI incompatible",
        baseline_fallback_permitted=True,
        tsh_algorithm_identity=TSH_CALO_ALGORITHM_ID,
        frozen_baseline_identity="CALO-v5.9",
    )

    assert accepted.disposition is FallbackDisposition.EXECUTE_POLICY
    assert accepted.algorithm_identity == TSH_CALO_ALGORITHM_ID
    assert blocked.disposition is FallbackDisposition.BLOCK
    assert blocked.algorithm_identity == ""
    assert fallback.disposition is FallbackDisposition.EXPLICIT_BASELINE
    assert fallback.algorithm_identity == "CALO-v5.9"
    assert fallback.algorithm_identity != TSH_CALO_ALGORITHM_ID
