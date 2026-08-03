"""TSH-CALO candidate artifact and lifecycle separation invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import numpy as np

from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry
from calo_rpd_studio.algorithms.calo.policy_schema import infer_checkpoint_schema
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import TSHCALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    IndependentTrainingProvenance,
    assemble_tsh_calo_ensemble_candidate,
    inspect_tsh_calo_candidate,
    load_tsh_calo_ensemble,
    load_tsh_calo_candidate,
    save_tsh_calo_candidate,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import (
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    build_tsh_calo_qualification_receipt,
    qualification_config,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import OODCalibration
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.results.database import ResultDatabase


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _provenance(*cases: str) -> IndependentTrainingProvenance:
    return IndependentTrainingProvenance(
        training_run_id="training-run-001",
        training_design_sha256=_sha("design"),
        source_commit="0a8989f",
        development_cases=tuple(cases or ("case30", "case57")),
        seed_manifest_sha256=_sha("seeds"),
    )


def _candidate(path: Path, seed: int = 17) -> Path:
    torch.manual_seed(seed)
    save_tsh_calo_candidate(path, TSHCALOPolicyNetwork(hidden_dim=16), _provenance())
    return path


def _ensemble(tmp_path: Path) -> Path:
    first = _candidate(tmp_path / "member-1.pt", seed=17)
    second = _candidate(tmp_path / "member-2.pt", seed=23)
    return Path(
        assemble_tsh_calo_ensemble_candidate(
            tmp_path / "ensemble.pt",
            [
                (first, inspect_tsh_calo_candidate(first).sha256),
                (second, inspect_tsh_calo_candidate(second).sha256),
            ],
        ).path
    )


def _qualification_config(policy_sha256: str) -> dict:
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id="qualification-001",
        source_policy_sha256=policy_sha256,
        source_commit="lifecycle-test",
        qualification_protocol_sha256=_sha("qualification-protocol"),
        seed_manifest_sha256=_sha("qualification-seeds"),
        evidence_artifact_sha256=_sha("synthetic-evidence-fixture"),
        development_cases=("case30", "case57"),
        ood_calibration=OODCalibration(np.zeros(2), np.ones(2)),
    )
    return qualification_config(receipt)


def test_candidate_export_is_exact_versioned_unqualified_and_loadable(tmp_path):
    path = _candidate(tmp_path / "candidate.pt")
    artifact = inspect_tsh_calo_candidate(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    schema = infer_checkpoint_schema(payload)

    assert artifact.algorithm_id == TSH_CALO_ALGORITHM_ID
    assert schema["native_tsh_calo"] is True
    assert schema["native_v59"] is False
    assert payload["metadata"]["lifecycle_status"] == "candidate_unqualified"
    assert artifact.feature_flags["population_schedule"] is False
    restored, loaded = load_tsh_calo_candidate(path, expected_sha256=artifact.sha256, device="cpu")
    assert restored.training is False
    assert loaded == artifact
    for name, tensor in payload["model_state_dict"].items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0.0, atol=0.0)


def test_protected_holdout_is_rejected_before_candidate_export(tmp_path):
    with pytest.raises(ValueError, match="Protected holdout"):
        save_tsh_calo_candidate(
            tmp_path / "leaked.pt", TSHCALOPolicyNetwork(hidden_dim=16), _provenance("case118")
        )
    assert not (tmp_path / "leaked.pt").exists()


def test_ensemble_assembly_preserves_independent_member_provenance(tmp_path):
    path = _ensemble(tmp_path)
    artifact = inspect_tsh_calo_candidate(path)
    networks, loaded = load_tsh_calo_ensemble(path, expected_sha256=artifact.sha256, device="cpu")

    assert artifact.artifact_kind == "ensemble_policy"
    assert artifact.ensemble_size == len(networks) == 2
    assert loaded == artifact
    assert artifact.training_provenance["source_kind"] == "independent_policy_training_ensemble"
    assert len(artifact.training_provenance["members"]) == 2


def test_registry_keeps_tsh_candidate_separate_from_frozen_calo_runtime(tmp_path):
    registry = PolicyRegistry(ResultDatabase(tmp_path / "results.sqlite"))
    policy = registry.register(_ensemble(tmp_path), name="TSH ensemble")

    assert policy.algorithm_id == TSH_CALO_ALGORITHM_ID
    assert policy.qualification_status == "candidate"
    assert policy.runtime_compatible is False
    assert policy.compatible_with(TSH_CALO_ALGORITHM_ID) is True
    with pytest.raises(ValueError, match="current CALO|CALO runtime"):
        registry.activate(policy.id)
    with pytest.raises(ValueError, match="before qualification"):
        registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID, allow_unqualified=True)


def test_qualified_tsh_policy_activation_and_binding_are_explicit_and_immutable(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(tmp_path), name="TSH ensemble")
    database.add_policy_qualification(
        qualification_id="qualification-001",
        policy_id=policy.id,
        passed=True,
        grade="A",
        score=80.0,
        qualification_status="qualified",
        config=_qualification_config(policy.sha256),
    )

    active = registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    assert active.active is True
    config = ExperimentConfig()
    binding = registry.bind_to_experiment_config(
        policy.id,
        config,
        deterministic=True,
        algorithm_id=TSH_CALO_ALGORITHM_ID,
    )
    assert binding["policy_algorithm_id"] == TSH_CALO_ALGORITHM_ID
    assert binding["policy_sha256"] == policy.sha256
    assert binding["policy_feature_flags"]["population_schedule"] is False
    assert (
        binding["policy_training_provenance"]["source_kind"]
        == "independent_policy_training_ensemble"
    )
    assert binding["policy_ensemble_size"] == 2
    assert binding["policy_qualification_id"] == "qualification-001"
    assert binding["policy_qualification_receipt_sha256"]
    assert binding["policy_ood_calibration_sha256"]
    assert binding["ood_calibration"]["mean"] == [0.0, 0.0]
    assert "policy_id" not in config.algorithm_parameters.get("CALO", {})
    assert config.algorithm_parameters[TSH_CALO_ALGORITHM_ID]["policy_id"] == policy.id


def test_tsh_registration_cannot_self_qualify_or_accept_an_incompatible_abi(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    candidate = _candidate(tmp_path / "candidate.pt")
    with pytest.raises(ValueError, match="candidates only"):
        registry.register(candidate, status="qualified")

    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    payload["metadata"]["action_schema_version"] = TSH_CALO_ACTION_SCHEMA + "-changed"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)
    record = registry.register(incompatible)
    assert record.compatible_with(TSH_CALO_ALGORITHM_ID) is False


def test_registered_artifact_mutation_blocks_tsh_activation(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(tmp_path))
    database.add_policy_qualification(
        qualification_id="qualification-001",
        policy_id=policy.id,
        passed=True,
        grade="A",
        score=80.0,
        qualification_status="qualified",
    )
    path = Path(policy.checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    first = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][first] = payload["model_state_dict"][first] + 1.0
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="checksum changed"):
        registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)


def test_generic_qualified_row_without_calibration_receipt_cannot_activate(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(tmp_path))
    database.add_policy_qualification(
        qualification_id="qualification-without-receipt",
        policy_id=policy.id,
        passed=True,
        grade="A",
        qualification_status="qualified",
    )

    with pytest.raises(ValueError, match="calibration receipt"):
        registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)


def test_candidate_artifact_module_has_no_experiment_workflow_dependency():
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_policy_artifact.py").read_text(
        encoding="utf-8"
    )
    assert "app.experiment_manager" not in source
    assert "create_experiment" not in source
    assert "activate(" not in source
