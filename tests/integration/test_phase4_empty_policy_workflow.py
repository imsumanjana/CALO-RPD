from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from calo_rpd_studio.algorithms.calo.policy_readiness import (
    GoverningPolicyStatus,
    evaluate_governing_policy,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def _complete_tsh_binding(*, policy_sha256: str = "b" * 64) -> dict:
    freeze_commit = "f" * 40
    members = [
        {
            "source_candidate_sha256": str(index + 1) * 64,
            "training_provenance": {
                "source_kind": "independent_policy_training",
                "source_commit": freeze_commit,
                "development_freeze_commit": freeze_commit,
                "development_freeze_sha256": "d" * 64,
                "phase4_acceptance_sha256": "a" * 64,
                "initialization_policy_sha256": "",
            },
        }
        for index in range(2)
    ]
    return {
        "policy_id": "post-freeze-policy",
        "policy_checkpoint": "immutable.pt",
        "policy_sha256": policy_sha256,
        "policy_architecture_version": "architecture",
        "policy_state_schema_version": "state",
        "policy_action_schema_version": "action",
        "policy_training_environment_version": "environment",
        "policy_algorithm_id": TSH_CALO_ALGORITHM_ID,
        "policy_qualification_status": "qualified",
        "policy_active_at_binding": True,
        "policy_artifact_kind": "ensemble_policy",
        "policy_ensemble_size": 2,
        "policy_ensemble_members": members,
        "policy_training_provenance": {
            "source_kind": "independent_policy_training_ensemble",
            "members": members,
        },
        "policy_qualification_id": "qualification-001",
        "policy_qualification_receipt_sha256": "e" * 64,
        "policy_qualification_receipt": {"schema": "test"},
        "policy_ood_calibration_sha256": "c" * 64,
        "ood_calibration": {"mean": [0.0], "std": [1.0]},
        "strict_policy_binding": True,
        "allow_unqualified_policy": False,
        "allow_cpu_fallback": False,
        "baseline_fallback_permitted": False,
        "policy_feature_flags": {
            "population_schedule": False,
            "allow_experimental_components": False,
        },
    }


def test_empty_policy_state_forces_rule_only_calo_and_disables_tsh_policy_use(tmp_path):
    state = AppState(tmp_path / "empty-policy.sqlite")
    state.config.algorithm_parameters["CALO"] = {
        "use_ai": True,
        "policy_id": "stale-calo-policy",
        "policy_sha256": "1" * 64,
    }
    state.config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = {
        "use_ai": True,
        "policy_id": "stale-tsh-policy",
        "policy_sha256": "2" * 64,
        "policy_training_provenance": {"unsafe": "stale"},
        "ood_calibration": {"mean": [0.0], "std": [1.0]},
    }

    changed = state.synchronize_governing_policy_binding(
        GoverningPolicyStatus(False, "missing", "No policy is registered")
    )

    assert changed is True
    calo = state.config.algorithm_parameters["CALO"]
    tsh = state.config.algorithm_parameters[TSH_CALO_ALGORITHM_ID]
    assert calo["use_ai"] is False
    assert calo["allow_unqualified_policy"] is False
    assert tsh["use_ai"] is False
    assert tsh["strict_policy_binding"] is True
    assert tsh["allow_unqualified_policy"] is False
    assert tsh["allow_cpu_fallback"] is False
    assert tsh["baseline_fallback_permitted"] is False
    assert not any(key.startswith("policy_") for key in tsh)
    assert "ood_calibration" not in tsh


def test_rejected_active_binding_cannot_crash_startup_or_leave_policy_state(tmp_path, monkeypatch):
    state = AppState(tmp_path / "rejected-policy.sqlite")
    state.config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = {
        "use_ai": True,
        "policy_id": "malformed-active-policy",
        "policy_sha256": "a" * 64,
    }

    def reject_binding(*_args, **_kwargs):
        raise ValueError("malformed qualification receipt")

    monkeypatch.setattr(state.policy_registry, "bind_to_experiment_config", reject_binding)
    status = GoverningPolicyStatus(
        True,
        "ready",
        "lightweight readiness passed",
        "malformed-active-policy",
        "Malformed active policy",
        "a" * 64,
        "qualified",
        "A",
        TSH_CALO_ALGORITHM_ID,
    )

    assert state.synchronize_governing_policy_binding(status) is True
    tsh = state.config.algorithm_parameters[TSH_CALO_ALGORITHM_ID]
    assert tsh["use_ai"] is False
    assert tsh["allow_cpu_fallback"] is False
    assert tsh["baseline_fallback_permitted"] is False
    assert not any(key.startswith("policy_") for key in tsh)


def test_pre_freeze_active_policy_readiness_fails_closed_without_exception():
    record = SimpleNamespace(
        active=True,
        archived=False,
        post_development_eligible=False,
        id="historical-policy",
        name="Historical policy",
        sha256="a" * 64,
        qualification_status="legacy_qualified",
        grade="A",
        algorithm_id="CALO",
    )
    registry = SimpleNamespace(list=lambda include_archived: [record])

    status = evaluate_governing_policy(registry)

    assert status.ready is False
    assert status.state == "development_only"
    assert status.algorithm_id == "CALO"


def test_tsh_calo_is_executable_only_with_an_immutable_qualified_binding():
    config = ExperimentConfig()
    config.algorithms = [TSH_CALO_ALGORITHM_ID]
    config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = {
        "use_ai": False,
        "strict_policy_binding": True,
        "allow_unqualified_policy": False,
    }

    with pytest.raises(ValueError, match="immutable qualified policy binding"):
        config.validate()


def test_default_and_loaded_primary_calo_are_fail_closed_to_rule_only_mode():
    default = ExperimentConfig()
    assert default.algorithm_parameters["CALO"]["use_ai"] is False
    default.validate()

    stale = ExperimentConfig.from_dict(
        {
            "algorithms": ["CALO"],
            "algorithm_parameters": {
                "CALO": {
                    "use_ai": True,
                    "policy_id": "old-policy",
                    "policy_checkpoint": "old.pt",
                    "policy_sha256": "c" * 64,
                }
            },
        }
    )
    with pytest.raises(ValueError, match="rule-only"):
        stale.validate()


def test_tsh_calo_rejects_experimental_change_f_even_with_a_complete_binding():
    config = ExperimentConfig()
    config.algorithms = [TSH_CALO_ALGORITHM_ID]
    config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = _complete_tsh_binding(
        policy_sha256="a" * 64
    )
    config.algorithm_parameters[TSH_CALO_ALGORITHM_ID]["policy_feature_flags"][
        "population_schedule"
    ] = True

    with pytest.raises(ValueError, match="Change F"):
        config.validate()


def test_tsh_calo_accepts_only_a_complete_a_to_e_binding_without_internal_fallback():
    config = ExperimentConfig()
    config.algorithms = [TSH_CALO_ALGORITHM_ID]
    config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = _complete_tsh_binding()

    config.validate()


def test_tsh_calo_config_rejects_forged_pre_freeze_provenance_before_execution():
    config = ExperimentConfig()
    config.algorithms = [TSH_CALO_ALGORITHM_ID]
    binding = _complete_tsh_binding()
    binding["policy_training_provenance"]["members"][0]["training_provenance"][
        "development_freeze_sha256"
    ] = ""
    config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = binding

    with pytest.raises(ValueError, match="development-freeze-bound ensemble"):
        config.validate()

    binding = _complete_tsh_binding()
    binding["policy_training_provenance"]["members"][0]["training_provenance"][
        "phase4_acceptance_sha256"
    ] = ""
    config.algorithm_parameters[TSH_CALO_ALGORITHM_ID] = binding

    with pytest.raises(ValueError, match="development-freeze-bound ensemble"):
        config.validate()


def test_gui_exposes_retirement_plan_and_blocks_historical_training_surfaces():
    intelligence = Path("calo_rpd_studio/gui/panels/calo_intelligence_panel.py").read_text(
        encoding="utf-8"
    )
    algorithms = Path("calo_rpd_studio/gui/panels/algorithms_panel.py").read_text(encoding="utf-8")

    assert "discover_bundled" not in intelligence
    assert 'QPushButton("Review removal")' in intelligence
    assert 'QPushButton("Legacy training unavailable")' not in intelligence
    assert "self.train_button" not in intelligence
    assert 'QPushButton("Train policy")' in intelligence
    assert intelligence.count('QPushButton("Import policy")') == 1
    assert "independent_training_requested" in intelligence
    assert "TrainingWorker" not in intelligence
    assert "resume_task_by_id" not in intelligence
    assert "POLICY_GATED_SPECS" in algorithms
    assert "def _safe_defaults" in algorithms
    assert 'if name == "CALO"' in algorithms
    assert "Primary CALO is rule-only in v12" in algorithms
