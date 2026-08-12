"""Independent counted TSH-CALO training-session and receipt invariants."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    inspect_tsh_calo_candidate,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training import (
    IndependentTSHCALOTrainer,
    TSHCALOTrainingConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_environment import (
    IndependentTSHCALOTrainingEnvironment,
    TSHCALOTrainingEnvironmentConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_receipt import (
    build_tsh_calo_training_episode_receipt,
    canonical_reward_sequence_sha256,
    load_tsh_calo_training_episode_receipt,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_session import (
    IndependentTSHCALOTrainingSession,
    TSHCALOTrainingSessionConfig,
)
from calo_rpd_studio.orpd.problem import ORPDProblem


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _training(*, rollout_capacity: int = 2) -> TSHCALOTrainingConfig:
    return TSHCALOTrainingConfig(
        training_run_id="fresh-member-training-001",
        development_cases=("toy-development",),
        seed_manifest_sha256=_sha("fresh-member-seeds"),
        resource_envelope=TSHCALOTrainingResourceEnvelope(rollout_capacity, 8, 16, 32, 16, 8),
        development_freeze_commit="e" * 40,
        development_freeze_sha256="f" * 64,
        phase4_acceptance_sha256="a" * 64,
        seed=211,
        hidden_dim=16,
        graph_steps=1,
        ppo_epochs=1,
        device="cpu",
    )


def _environment(*, max_evaluations: int = 16) -> TSHCALOTrainingEnvironmentConfig:
    return TSHCALOTrainingEnvironmentConfig(
        case_identity="toy-development",
        population_size=4,
        max_evaluations=max_evaluations,
        seed=307,
        environment_deterministic=False,
    )


def _session(
    toy_case,
    training: TSHCALOTrainingConfig,
    environment: TSHCALOTrainingEnvironmentConfig,
    session: TSHCALOTrainingSessionConfig,
) -> IndependentTSHCALOTrainingSession:
    problem = ORPDProblem(toy_case)
    trainer = IndependentTSHCALOTrainer(training)
    rollout = IndependentTSHCALOTrainingEnvironment(problem, training, environment)
    return IndependentTSHCALOTrainingSession(trainer, rollout, session)


def test_completed_counted_session_earns_receipt_and_only_unqualified_export(tmp_path, toy_case):
    training = _training(rollout_capacity=1)
    environment = _environment(max_evaluations=12)
    config = TSHCALOTrainingSessionConfig(
        session_id="fresh-member-session-001", deterministic_policy=True
    )
    session = _session(toy_case, training, environment, config)

    result = session.advance()

    assert result is not None
    receipt = result.receipt
    assert receipt.candidate_evaluations == 12
    assert receipt.scenario_power_flow_calls == 12
    assert receipt.canonical_transition_count == 2
    assert receipt.ppo_update_count == session.trainer.update_steps == 2
    assert receipt.canonical_reward_sha256 == canonical_reward_sequence_sha256(
        result.canonical_rewards
    )
    assert (
        receipt.receipt_sha256
        == load_tsh_calo_training_episode_receipt(receipt.to_dict()).receipt_sha256
    )
    assert len(session.trainer.training_episode_receipts) == 1
    replay_values = receipt.unsigned_payload()
    replay_values.pop("schema_version")
    replay_values["seed"] += 1
    replay = build_tsh_calo_training_episode_receipt(**replay_values)
    with pytest.raises(ValueError, match="session ID was already recorded"):
        session.trainer.record_training_episode_receipt(replay.to_dict())

    artifact = session.trainer.export_unqualified_candidate(
        tmp_path / "candidate.pt", source_commit="e" * 40
    )
    inspected = inspect_tsh_calo_candidate(artifact.path, expected_sha256=artifact.sha256)
    provenance = inspected.training_provenance
    assert provenance["training_episode_receipts"][0]["receipt_sha256"] == (receipt.receipt_sha256)
    assert provenance["training_device_provenance"]["memory_admission"]["selected_device"] == "cpu"
    payload = torch.load(artifact.path, map_location="cpu", weights_only=True)
    assert payload["metadata"]["lifecycle_status"] == "candidate_unqualified"
    with pytest.raises(RuntimeError, match="already complete"):
        session.advance()


def test_partial_session_trusted_resume_replays_rewards_updates_and_receipt_exactly(
    tmp_path, toy_case
):
    training = _training(rollout_capacity=2)
    environment = _environment(max_evaluations=16)
    config = TSHCALOTrainingSessionConfig(session_id="resume-session-001")
    original = _session(toy_case, training, environment, config)
    assert original.advance(max_transitions=1) is None
    path = tmp_path / "training-session.resume"
    original.save_resume(path)
    restored = IndependentTSHCALOTrainingSession.load_resume(
        path,
        problem=ORPDProblem(toy_case),
        training_config=training,
        environment_config=environment,
        session_config=config,
    )

    left = original.advance()
    right = restored.advance()

    assert left is not None and right is not None
    assert left.receipt.receipt_sha256 == right.receipt.receipt_sha256
    np.testing.assert_array_equal(left.canonical_rewards, right.canonical_rewards)
    assert left.update_metrics == right.update_metrics
    for name, tensor in original.trainer.network.state_dict().items():
        torch.testing.assert_close(
            restored.trainer.network.state_dict()[name], tensor, rtol=0.0, atol=0.0
        )


def test_session_resume_and_receipt_are_design_and_integrity_bound(toy_case):
    training = _training()
    environment = _environment()
    config = TSHCALOTrainingSessionConfig(session_id="bound-session-001")
    session = _session(toy_case, training, environment, config)
    session.advance(max_transitions=1)
    checkpoint = session.state_dict()

    with pytest.raises(ValueError, match="scientific design changed"):
        IndependentTSHCALOTrainingSession.from_state_dict(
            ORPDProblem(toy_case),
            training,
            environment,
            replace(config, deterministic_policy=True),
            checkpoint,
        )
    session.advance()
    assert session.receipt is not None
    changed = session.receipt.to_dict()
    changed["seed"] += 1
    with pytest.raises(ValueError, match="integrity"):
        load_tsh_calo_training_episode_receipt(changed)
    changed = session.receipt.to_dict()
    changed["deterministic_policy"] = "false"
    with pytest.raises(ValueError, match="must be Boolean"):
        load_tsh_calo_training_episode_receipt(changed)


def test_failed_session_cannot_checkpoint_or_issue_a_receipt(toy_case):
    training = _training()
    environment = _environment()
    config = TSHCALOTrainingSessionConfig(session_id="failed-session-001")
    session = _session(toy_case, training, environment, config)

    def failed_evaluator(_values, **_kwargs):
        raise RuntimeError("synthetic counted evaluator failure")

    session.environment.problem.evaluate_with_context = failed_evaluator
    with pytest.raises(RuntimeError, match="synthetic counted evaluator failure"):
        session.advance()

    assert session.failed is True
    assert session.completed is False
    assert session.receipt is None
    assert session.trainer.training_episode_receipts == []
    with pytest.raises(RuntimeError, match="cannot be checkpointed"):
        session.state_dict()
    with pytest.raises(RuntimeError, match="failed and cannot continue"):
        session.advance()


def test_training_session_module_has_no_experiment_or_lifecycle_authority():
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_training_session.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "experiments.experiment_runner",
        "PolicyRegistry",
        "activate(",
        "bind_to_experiment",
        "create_experiment",
        "TSHCALOInferenceController",
        "export_unqualified_candidate",
    ):
        assert forbidden not in source
