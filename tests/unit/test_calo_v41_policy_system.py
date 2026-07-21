from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.policy_qualification import PolicyQualificationConfig
from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry
from calo_rpd_studio.algorithms.calo.policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_DIM,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
    PolicyRuntimeContext,
    build_policy_vector,
    infer_checkpoint_schema,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.results.database import ResultDatabase


def _write_native_policy(path: Path) -> Path:
    network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": 16},
            "metadata": {
                "calo_core": "v4.1",
                "state_dimension": POLICY_STATE_DIM,
                "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
                "state_schema_version": POLICY_STATE_SCHEMA,
                "action_schema_version": POLICY_ACTION_SCHEMA,
                "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
            },
        },
        path,
    )
    return path


def test_native_policy_schema_is_explicit_and_32_dimensional(tmp_path):
    path = _write_native_policy(tmp_path / "native.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    schema = infer_checkpoint_schema(payload)
    assert schema["native_v41"] is True
    assert schema["input_dim"] == 32
    assert schema["state_schema_version"] == POLICY_STATE_SCHEMA

    class State:
        def vector(self):
            return np.arange(24, dtype=np.float32) / 24.0

    vector = build_policy_vector(
        State(),
        PolicyRuntimeContext(
            hpem_occupancy=0.7,
            memory_consensus=0.6,
            memory_readiness=0.5,
            success_memory_density=0.4,
            learning_lane_fraction=0.3,
            precision_active=1.0,
            precision_radius=0.2,
            variable_group_concentration=0.1,
        ),
        input_dim=POLICY_STATE_DIM,
    )
    assert vector.shape == (32,)
    np.testing.assert_allclose(vector[-8:], [0.7, 0.6, 0.5, 0.4, 0.3, 1.0, 0.2, 0.1])


def test_policy_registry_requires_qualification_and_preserves_experiment_references(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(
        _write_native_policy(tmp_path / "candidate.pt"), name="Candidate P01"
    )
    assert policy.qualification_status == "candidate"
    assert policy.grade == "U"

    with pytest.raises(ValueError, match="Only qualified policies"):
        registry.activate(policy.id)

    database.add_policy_qualification(
        qualification_id="q1",
        policy_id=policy.id,
        config={"cases": ["case30", "case57"]},
        metrics={"participants": {}},
        passed=True,
        grade="A",
        score=82.0,
        qualification_status="qualified",
    )
    activated = registry.activate(policy.id)
    assert activated.active is True
    assert activated.grade == "A"

    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    binding = registry.bind_to_experiment_config(policy.id, config, deterministic=True)
    assert binding["policy_sha256"] == policy.sha256
    assert binding["strict_policy_binding"] is True
    assert config.algorithm_parameters["CALO"]["policy_id"] == policy.id

    experiment_id = database.create_experiment(config, collect_provenance())
    database.bind_policy_to_experiment(experiment_id, binding)
    assert database.policy_reference_count(policy.id, policy.sha256) == 1

    # Move the default-active marker to another qualified policy so the deletion guard reaches
    # the stronger experiment-provenance reference check.
    replacement = registry.register(
        _write_native_policy(tmp_path / "replacement.pt"), name="Replacement"
    )
    database.add_policy_qualification(
        qualification_id="q2",
        policy_id=replacement.id,
        passed=True,
        grade="A",
        score=80.0,
        qualification_status="qualified",
    )
    registry.activate(replacement.id)
    with pytest.raises(ValueError, match="referenced"):
        registry.delete(policy.id, delete_artifact=True)
    assert Path(policy.checkpoint_path).is_file()


def test_workspace_state_round_trip_and_delete_cleanup(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    experiment_id = database.create_experiment(ExperimentConfig(), collect_provenance())
    database.save_workspace_state(
        experiment_id,
        workflow={"completed": ["power_system", "orpd"], "experiment_started": True},
        ui={"workspace_index": 8, "live_optimization": {"run_selector": 3}},
    )
    restored = database.get_workspace_state(experiment_id)
    assert restored["workflow"]["completed"] == ["power_system", "orpd"]
    assert restored["ui"]["workspace_index"] == 8
    assert restored["ui"]["live_optimization"]["run_selector"] == 3
    database.delete_experiment(experiment_id)
    assert database.get_workspace_state(experiment_id) is None


def test_policy_qualification_protects_holdout_cases_by_default():
    PolicyQualificationConfig(cases=("case30", "case57"), runs=5).validate()
    with pytest.raises(ValueError, match="protected holdout"):
        PolicyQualificationConfig(cases=("case118",), runs=5).validate()
