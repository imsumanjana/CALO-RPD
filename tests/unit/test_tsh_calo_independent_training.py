"""Independent TSH-CALO PPO, resume, export, and leakage invariants."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.topology_context import (
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    build_topology_graph_state,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import GroupActionMask
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import inspect_tsh_calo_candidate
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSHCALOFeatureFlags
from calo_rpd_studio.algorithms.calo.tsh_calo_training import (
    IndependentTSHCALOTrainer,
    TSHCALORolloutBatch,
    TSHCALOTrainingConfig,
)
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _config(**changes) -> TSHCALOTrainingConfig:
    values = dict(
        training_run_id="independent-training-001",
        development_cases=("case30", "case57"),
        seed_manifest_sha256=_sha("seed-manifest"),
        seed=29,
        hidden_dim=16,
        graph_steps=1,
        ppo_epochs=1,
        device="cpu",
    )
    values.update(changes)
    return TSHCALOTrainingConfig(**values)


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


def _mask_and_learners(state: TopologyAwarePolicyState):
    groups = np.asarray(state.topology.control_groups, dtype=int)
    mask = GroupActionMask.from_control_groups(groups)
    contexts = np.arange(len(groups), dtype=int) % 4
    return mask, groups, contexts


def _rollout(trainer: IndependentTSHCALOTrainer, state: TopologyAwarePolicyState):
    mask, groups, contexts = _mask_and_learners(state)
    first, first_logp, first_value = trainer.sample_action(state, mask, groups, contexts)
    second, second_logp, second_value = trainer.sample_action(state, mask, groups, contexts)
    return TSHCALORolloutBatch(
        states=(state, state),
        actions=(first, second),
        old_log_probabilities=np.asarray([first_logp, second_logp]),
        old_values=np.asarray([first_value, second_value]),
        advantages=np.asarray([1.0, -0.4]),
        returns=np.asarray([0.8, -0.2]),
    )


def test_training_config_is_independent_hashed_and_rejects_protected_cases():
    config = _config()
    config.validate()
    assert (
        config.scientific_design_hash() == replace(config, device="auto").scientific_design_hash()
    )
    assert (
        config.scientific_design_hash() != replace(config, clip_ratio=0.25).scientific_design_hash()
    )
    with pytest.raises(ValueError, match="Protected holdout"):
        _config(development_cases=("case118",)).validate()
    with pytest.raises(ValueError, match="experimental"):
        _config(feature_flags=TSHCALOFeatureFlags(population_schedule=True)).validate()


def test_independent_ppo_update_is_finite_and_changes_policy(toy_case):
    trainer = IndependentTSHCALOTrainer(_config())
    state = _state(toy_case)
    before = {
        name: tensor.detach().clone() for name, tensor in trainer.network.state_dict().items()
    }

    metrics = trainer.update(_rollout(trainer, state))

    assert trainer.update_steps == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in trainer.network.state_dict().items()
    )


def test_training_action_rejects_mask_bypass(toy_case):
    trainer = IndependentTSHCALOTrainer(_config())
    state = _state(toy_case)
    mask, groups, contexts = _mask_and_learners(state)
    action, _logp, _value = trainer.sample_action(state, mask, groups, contexts, deterministic=True)
    unavailable = np.flatnonzero(~np.asarray(mask.available_groups, dtype=bool))
    if unavailable.size:
        changed = action.group_operators.copy()
        changed[unavailable[0]] = 0
        with pytest.raises(ValueError, match="sentinel"):
            replace(action, group_operators=changed).validate()
    changed_learners = action.learner_operators.copy()
    changed_learners[0] = 6
    with pytest.raises(ValueError, match="mask"):
        replace(action, learner_operators=changed_learners).validate()


def test_training_resume_restores_exact_model_optimizer_and_action_rng(tmp_path, toy_case):
    trainer = IndependentTSHCALOTrainer(_config())
    state = _state(toy_case)
    trainer.update(_rollout(trainer, state))
    resume_path = tmp_path / "training.resume.pt"
    digest = trainer.save_resume(resume_path)
    restored = IndependentTSHCALOTrainer.load_resume(
        resume_path, expected_sha256=digest, expected_config=trainer.config
    )

    assert restored.update_steps == trainer.update_steps
    for name, tensor in trainer.network.state_dict().items():
        torch.testing.assert_close(restored.network.state_dict()[name], tensor, rtol=0.0, atol=0.0)
    mask, groups, contexts = _mask_and_learners(state)
    original_action, original_logp, original_value = trainer.sample_action(
        state, mask, groups, contexts
    )
    restored_action, restored_logp, restored_value = restored.sample_action(
        state, mask, groups, contexts
    )
    np.testing.assert_array_equal(original_action.group_operators, restored_action.group_operators)
    np.testing.assert_allclose(original_action.group_parameters, restored_action.group_parameters)
    np.testing.assert_array_equal(
        original_action.learner_operators, restored_action.learner_operators
    )
    assert original_logp == pytest.approx(restored_logp, rel=0.0, abs=0.0)
    assert original_value == pytest.approx(restored_value, rel=0.0, abs=0.0)


def test_exact_resume_rejects_hash_and_scientific_design_drift(tmp_path):
    trainer = IndependentTSHCALOTrainer(_config())
    path = tmp_path / "training.resume.pt"
    digest = trainer.save_resume(path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        IndependentTSHCALOTrainer.load_resume(
            path, expected_sha256=_sha("wrong"), expected_config=trainer.config
        )
    with pytest.raises(ValueError, match="scientific design changed"):
        IndependentTSHCALOTrainer.load_resume(
            path,
            expected_sha256=digest,
            expected_config=replace(trainer.config, clip_ratio=0.3),
        )


def test_training_can_only_export_unqualified_candidate_after_update(tmp_path, toy_case):
    trainer = IndependentTSHCALOTrainer(_config())
    with pytest.raises(ValueError, match="at least one"):
        trainer.export_unqualified_candidate(tmp_path / "early.pt", source_commit="380e7a7")
    trainer.update(_rollout(trainer, _state(toy_case)))

    artifact = trainer.export_unqualified_candidate(
        tmp_path / "candidate.pt", source_commit="380e7a7"
    )
    inspected = inspect_tsh_calo_candidate(artifact.path, expected_sha256=artifact.sha256)

    assert inspected.training_provenance["training_run_id"] == "independent-training-001"
    assert (
        inspected.training_provenance["training_design_sha256"]
        == trainer.config.scientific_design_hash()
    )
    payload = torch.load(artifact.path, map_location="cpu", weights_only=True)
    assert payload["metadata"]["lifecycle_status"] == "candidate_unqualified"


def test_training_module_has_no_experiment_registry_or_activation_authority():
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_training.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "app.experiment_manager",
        "experiments.experiment_runner",
        "PolicyRegistry",
        "activate(",
        "bind_to_experiment",
        "create_experiment",
    ):
        assert forbidden not in source
