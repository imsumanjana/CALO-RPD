"""Immutable ensemble inference, admission, shield, provenance, and fallback invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry
from calo_rpd_studio.algorithms.calo.topology_context import (
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    build_topology_graph_state,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_inference import (
    TSHCALOInferenceController,
    admit_inference_device,
    ood_calibration_sha256,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import GroupActionMask, TSHCALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    IndependentTrainingProvenance,
    assemble_tsh_calo_ensemble_candidate,
    inspect_tsh_calo_candidate,
    save_tsh_calo_candidate,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    build_tsh_calo_qualification_receipt,
    qualification_config,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import (
    FallbackDisposition,
    OODCalibration,
    SafetyEnvelope,
    SlidingWindowContextualBandit,
    topology_ood_signature,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.results.database import ResultDatabase


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _device_provenance() -> dict:
    total = 1 << 30
    available = 512 << 20
    allowance = int(0.80 * available)
    estimate = 64 << 20
    return {
        "memory_estimate": {
            "estimator_version": "tsh-calo-training-memory-v1",
            "estimated_working_set_bytes": estimate,
        },
        "memory_admission": {
            "requested_device": "cpu",
            "selected_device": "cpu",
            "computation_device": "cpu",
            "estimated_working_set_bytes": estimate,
            "total_bytes": total,
            "available_bytes_at_admission": available,
            "baseline_reserved_bytes": 0,
            "allowance_bytes": allowance,
            "process_ceiling_bytes": allowance,
            "allocator_fraction_of_total": allowance / total,
            "fallback_reason": "explicit CPU training",
            "estimator_version": "tsh-calo-training-memory-v1",
        },
        "computation_semantics": "CPU computes; system RAM is admitted storage",
    }


def _member(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    provenance = IndependentTrainingProvenance(
        training_run_id=f"independent-{seed}",
        training_design_sha256=_sha("design"),
        source_commit="85c4ce4",
        development_cases=("case30", "case57"),
        seed_manifest_sha256=_sha("seeds"),
        training_device_provenance=_device_provenance(),
    )
    save_tsh_calo_candidate(path, TSHCALOPolicyNetwork(hidden_dim=16), provenance)
    return path


def _ensemble(tmp_path: Path) -> Path:
    members = [_member(tmp_path / f"member-{seed}.pt", seed) for seed in (17, 23)]
    artifact = assemble_tsh_calo_ensemble_candidate(
        tmp_path / "ensemble.pt",
        [(path, inspect_tsh_calo_candidate(path).sha256) for path in members],
    )
    return Path(artifact.path)


def _state(toy_case) -> TopologyAwarePolicyState:
    decoder = ORPDVariableDecoder(toy_case, ORPDVariableConfig())
    result = run_ac_power_flow(toy_case)
    topology = build_topology_graph_state(
        toy_case,
        decoder,
        np.full(decoder.dimension, 0.5),
        result,
        [ScenarioDescriptor(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
    )
    return TopologyAwarePolicyState(np.linspace(0.0, 1.0, 32), topology)


def _binding(tmp_path: Path, state: TopologyAwarePolicyState) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(tmp_path), name="TSH ensemble")
    signature = topology_ood_signature(state)
    calibration = OODCalibration(signature.copy(), np.ones_like(signature))
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id="qualification-001",
        source_policy_sha256=policy.sha256,
        source_commit="inference-test",
        qualification_protocol_sha256=_sha("qualification-protocol"),
        seed_manifest_sha256=_sha("qualification-seeds"),
        evidence_artifact_sha256=_sha("synthetic-evidence-fixture"),
        development_cases=("case30", "case57"),
        ood_calibration=calibration,
    )
    database.add_policy_qualification(
        qualification_id="qualification-001",
        policy_id=policy.id,
        passed=True,
        grade="A",
        score=80.0,
        qualification_status="qualified",
        config=qualification_config(receipt),
    )
    registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    config = ExperimentConfig()
    return registry.bind_to_experiment_config(
        policy.id,
        config,
        deterministic=True,
        algorithm_id=TSH_CALO_ALGORITHM_ID,
    )


def _controller(tmp_path: Path, state, **changes):
    binding = _binding(tmp_path, state)
    payload = binding["ood_calibration"]
    calibration = OODCalibration(
        np.asarray(payload["mean"]),
        np.asarray(payload["scale"]),
        payload["attenuation_start"],
        payload["minimum_neural_weight"],
    )
    calibration_sha = binding["policy_ood_calibration_sha256"]
    values = dict(
        binding=binding,
        ood_calibration=calibration,
        expected_ood_calibration_sha256=calibration_sha,
        deterministic=True,
        seed=101,
        requested_device="cpu",
    )
    values.update(changes)
    return TSHCALOInferenceController(**values)


def test_safe80_admission_uses_currently_free_vram_and_falls_back_only_when_needed(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"x" * 1024)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (2 << 20, 8 << 20))
    admitted = admit_inference_device(checkpoint, requested_device="auto")
    assert admitted.selected_device == "cuda:0"
    assert admitted.available_bytes_at_admission == 2 << 20
    assert admitted.allowance_bytes == int(0.8 * (2 << 20))

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1 << 20, 8 << 20))
    fallback = admit_inference_device(checkpoint, requested_device="auto")
    assert fallback.selected_device == "cpu"
    assert "exceeds" in fallback.fallback_reason


def test_qualified_active_immutable_ensemble_executes_through_shield(tmp_path, toy_case):
    state = _state(toy_case)
    controller = _controller(tmp_path, state)
    groups = np.asarray(state.topology.control_groups, dtype=int)
    mask = GroupActionMask.from_control_groups(groups)
    result = controller.decide(
        state,
        mask,
        groups,
        np.arange(len(groups), dtype=int) % 4,
        bandit=SlidingWindowContextualBandit(),
        safety=SafetyEnvelope(remaining_evaluations=len(groups), candidate_count=len(groups)),
    )

    assert result.fallback.disposition is FallbackDisposition.EXECUTE_POLICY
    assert result.fallback.algorithm_identity == TSH_CALO_ALGORITHM_ID
    assert result.learner_operators.shape == (len(groups),)
    assert result.operator_probabilities.shape[0] == len(groups)
    assert result.shield_trace.schema_version == "tsh-calo-shield-trace-v1"
    assert result.provenance["policy_ensemble_size"] == 2
    assert result.provenance["device_admission"]["computation_device"] == "cpu"


def test_deterministic_inference_replays_exactly(tmp_path, toy_case):
    state = _state(toy_case)
    first = _controller(tmp_path / "first", state)
    second = _controller(tmp_path / "second", state)
    groups = np.asarray(state.topology.control_groups, dtype=int)
    mask = GroupActionMask.from_control_groups(groups)
    arguments = dict(
        state=state,
        action_mask=mask,
        learner_groups=groups,
        learner_contexts=np.arange(len(groups), dtype=int) % 4,
        bandit=SlidingWindowContextualBandit(),
        safety=SafetyEnvelope(len(groups), len(groups)),
    )
    left = first.decide(**arguments)
    right = second.decide(**arguments)
    torch.testing.assert_close(left.learner_operators, right.learner_operators, rtol=0, atol=0)
    torch.testing.assert_close(left.group_parameters, right.group_parameters, rtol=0, atol=0)
    torch.testing.assert_close(
        left.operator_probabilities, right.operator_probabilities, rtol=0, atol=0
    )


def test_unavailable_or_mismatched_policy_blocks_or_explicitly_relabels_baseline(
    tmp_path, toy_case
):
    state = _state(toy_case)
    binding = _binding(tmp_path, state)
    binding["policy_sha256"] = _sha("wrong")
    signature = topology_ood_signature(state)
    calibration = OODCalibration(signature, np.ones_like(signature))
    calibration_sha = ood_calibration_sha256(calibration)
    binding["policy_ood_calibration_sha256"] = calibration_sha
    blocked = TSHCALOInferenceController(
        binding,
        ood_calibration=calibration,
        expected_ood_calibration_sha256=calibration_sha,
        deterministic=True,
        seed=1,
        requested_device="cpu",
    )
    assert blocked.fallback_decision().disposition is FallbackDisposition.BLOCK
    fallback = TSHCALOInferenceController(
        binding,
        ood_calibration=calibration,
        expected_ood_calibration_sha256=calibration_sha,
        deterministic=True,
        seed=1,
        requested_device="cpu",
        baseline_fallback_permitted=True,
    ).fallback_decision()
    assert fallback.disposition is FallbackDisposition.EXPLICIT_BASELINE
    assert fallback.algorithm_identity == "CALO-v5.9"
    assert fallback.algorithm_identity != TSH_CALO_ALGORITHM_ID


def test_runtime_rejects_a_binding_whose_frozen_qualification_receipt_was_mutated(
    tmp_path, toy_case
):
    state = _state(toy_case)
    binding = _binding(tmp_path, state)
    calibration_payload = binding["ood_calibration"]
    calibration = OODCalibration(
        np.asarray(calibration_payload["mean"]),
        np.asarray(calibration_payload["scale"]),
        calibration_payload["attenuation_start"],
        calibration_payload["minimum_neural_weight"],
    )
    binding["policy_qualification_receipt"]["ood_calibration"]["mean"][0] += 1.0

    controller = TSHCALOInferenceController(
        binding,
        ood_calibration=calibration,
        expected_ood_calibration_sha256=binding["policy_ood_calibration_sha256"],
        deterministic=True,
        seed=1,
        requested_device="cpu",
    )

    assert controller.fallback_decision().disposition is FallbackDisposition.BLOCK
    assert "checksum mismatch" in controller.rejection_reason


def test_runtime_safety_rejection_never_returns_a_partial_policy_action(tmp_path, toy_case):
    state = _state(toy_case)
    controller = _controller(tmp_path, state)
    groups = np.asarray(state.topology.control_groups, dtype=int)
    result = controller.decide(
        state,
        GroupActionMask.from_control_groups(groups),
        groups,
        np.zeros(len(groups), dtype=int),
        bandit=SlidingWindowContextualBandit(),
        safety=SafetyEnvelope(remaining_evaluations=0, candidate_count=len(groups)),
    )
    assert result.fallback.disposition is FallbackDisposition.BLOCK
    assert result.regime is result.learner_operators is result.operator_probabilities is None
    assert "remaining FE budget" in result.provenance["runtime_rejection"]


def test_tsh_binding_requires_activation_and_inference_has_no_solver_or_training_authority(
    tmp_path,
):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(tmp_path))
    database.add_policy_qualification(
        qualification_id="qualification-001",
        policy_id=policy.id,
        passed=True,
        grade="A",
        qualification_status="qualified",
    )
    with pytest.raises(ValueError, match="explicitly activated"):
        registry.bind_to_experiment_config(
            policy.id,
            ExperimentConfig(),
            deterministic=True,
            algorithm_id=TSH_CALO_ALGORITHM_ID,
        )
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_inference.py").read_text(
        encoding="utf-8"
    )
    assert "run_ac_power_flow" not in source
    assert "tsh_calo_training" not in source
    assert "export_unqualified_candidate" not in source
    assert "PolicyRegistry" not in source
