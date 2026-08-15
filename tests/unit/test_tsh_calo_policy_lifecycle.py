"""TSH-CALO candidate artifact and lifecycle separation invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import numpy as np

from calo_rpd_studio.ai.model_io import checkpoint_sha256
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
    TSH_CALO_TRAINING_ENVIRONMENT,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    build_tsh_calo_qualification_receipt,
    qualification_config,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    LEGACY_TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
    QUALIFICATION_CELL_INDEX_FILE,
    QUALIFICATION_COMPLETION_FILE,
    QUALIFICATION_EVENT_LOG_FILE,
    QUALIFICATION_STATUS_FILE,
    TSH_CALO_QUALIFICATION_CELL_INDEX_SCHEMA,
    TSH_CALO_QUALIFICATION_CELL_SUCCESS_SCHEMA,
    TSH_CALO_QUALIFICATION_COMPLETION_SCHEMA,
    TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
    TSH_CALO_QUALIFICATION_STATUS_SCHEMA,
    TSHCALOQualificationPlan,
    grade_tsh_calo_qualification_evidence,
    qualification_candidate_contract,
    tsh_calo_qualification_cell_identity,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_feasibility_assessment import (
    TSH_CALO_FEASIBILITY_ASSESSMENT_SCHEMA,
    TSH_CALO_FEASIBILITY_COMPLETION_SCHEMA,
    build_tsh_calo_feasibility_assessment,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import OODCalibration
from calo_rpd_studio.algorithms.calo.tsh_calo_training_receipt import (
    build_tsh_calo_training_episode_receipt,
    canonical_reward_sequence_sha256,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
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


def _episode_receipts(run_id: str, design: str) -> tuple[dict, ...]:
    return (
        build_tsh_calo_training_episode_receipt(
            session_id=run_id + "-session",
            training_run_id=run_id,
            training_design_sha256=design,
            session_design_sha256=_sha("session"),
            environment_design_sha256=_sha("environment"),
            case_identity="case30",
            case_checksum=_sha("case30"),
            problem_fingerprint=_sha("problem"),
            seed=17,
            deterministic_policy=True,
            candidate_evaluations=8,
            scenario_power_flow_calls=8,
            canonical_transition_count=1,
            ppo_update_count=1,
            canonical_reward_sha256=canonical_reward_sequence_sha256((0.25,)),
            accounting_complete=True,
            terminal=True,
        ).to_dict(),
    )


def _provenance(*cases: str, run_id: str = "training-run-001") -> IndependentTrainingProvenance:
    design = _sha("design")
    freeze_commit = "a" * 40
    return IndependentTrainingProvenance(
        training_run_id=run_id,
        training_design_sha256=design,
        source_commit=freeze_commit,
        development_cases=tuple(cases or ("case30", "case57")),
        seed_manifest_sha256=_sha("seeds"),
        training_device_provenance=_device_provenance(),
        training_episode_receipts=_episode_receipts(run_id, design),
        development_freeze_commit=freeze_commit,
        development_freeze_sha256=_sha("development-freeze"),
        phase4_acceptance_sha256=_sha("phase4-acceptance"),
        initialization_policy_sha256="",
    )


def _candidate(path: Path, seed: int = 17) -> Path:
    torch.manual_seed(seed)
    save_tsh_calo_candidate(
        path,
        TSHCALOPolicyNetwork(hidden_dim=16),
        _provenance(run_id=f"training-run-{seed}"),
    )
    return path


def _ensemble(tmp_path: Path, *, seed_offset: int = 0) -> Path:
    first = _candidate(tmp_path / "member-1.pt", seed=17 + seed_offset)
    second = _candidate(tmp_path / "member-2.pt", seed=23 + seed_offset)
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


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _formal_evidence(
    directory: Path,
    policy_path: Path,
    policy_sha256: str,
    *,
    strength=1.0,
    current_candidate_contract: bool = False,
):
    directory.mkdir(parents=True)
    component_evidence = {
        component: {"path": f"component-{component}.json", "sha256": _sha(component)}
        for component in "ABCDE"
    }
    candidate_contract = (
        qualification_candidate_contract(
            inspect_tsh_calo_candidate(policy_path, expected_sha256=policy_sha256)
        )
        if current_candidate_contract
        else {}
    )
    plan = TSHCALOQualificationPlan(
        qualification_run_id=f"qualification-{policy_sha256[:12]}",
        source_commit="b" * 40,
        source_tracked_clean=True,
        candidate_path=str(policy_path),
        candidate_sha256=policy_sha256,
        development_cases=("case30", "case57"),
        runs=30,
        master_seed=1907,
        population_size=4,
        max_evaluations=8,
        mode="formal",
        calibration_samples_per_case=4,
        calibration_population_size=4,
        bootstrap_resamples=1_000,
        component_evidence={} if current_candidate_contract else component_evidence,
        candidate_contract=candidate_contract,
    )
    plan.validate()
    case_evidence = []
    for case_name in plan.development_cases:
        case_evidence.append(
            {
                "case": case_name,
                "equal_exact_fe": True,
                "all_candidate_independently_validated": True,
                "all_baseline_independently_validated": True,
                "candidate_feasible_probability": min(1.0, 0.96 + 0.01 * strength),
                "feasible_probability_difference_ci95": [0.0 + 0.01 * strength, 0.1],
                "paired_feasible_objective_fraction": 1.0,
                "median_relative_objective_improvement": 0.01 + 0.01 * strength,
                "objective_win_rate": min(1.0, 0.65 + 0.05 * strength),
                "paired_rank_biserial": min(1.0, 0.30 + 0.10 * strength),
                "holm_p": max(0.001, 0.02 - 0.005 * strength),
                "anytime": {
                    str(fraction): {
                        "feasible_probability_difference": 0.0 + 0.01 * strength,
                        "median_relative_objective_improvement": 0.005 * strength,
                    }
                    for fraction in plan.anytime_fractions
                },
            }
        )
    decision = grade_tsh_calo_qualification_evidence(plan, case_evidence, [])
    assert decision["passed"] is True
    _write_json(directory / "qualification_plan.json", plan.to_dict())
    _write_json(directory / "seed_manifest.json", plan.seed_manifest())
    evidence = {
        "schema_version": LEGACY_TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA,
        "analysis_schema_version": plan.analysis_schema_version,
        "relative_improvement_version": plan.relative_improvement_version,
        "objective_scale_floor": plan.objective_scale_floor,
        "qualification_run_id": plan.qualification_run_id,
        "source_commit": plan.source_commit,
        "source_tracked_clean": True,
        "source_policy_sha256": policy_sha256,
        "qualification_plan_sha256": plan.execution_plan_sha256(),
        "scientific_design_sha256": plan.scientific_design_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
        "ood_calibration_sha256": "",
        "development_cases": list(plan.development_cases),
        "protected_cases_opened": False,
        "candidate_contract": candidate_contract,
        "component_evidence": (
            {}
            if current_candidate_contract
            else {key: {**value, "accepted": True} for key, value in component_evidence.items()}
        ),
        "records": {
            "expected": len(plan.development_cases) * plan.runs * 2,
            "completed": len(plan.development_cases) * plan.runs * 2,
            "failed": 0,
        },
        "case_evidence": case_evidence,
        "decision": decision,
        "authority_boundary": "independent_qualification_only_no_registration_or_activation",
    }
    evidence_sha256 = _write_json(directory / "qualification_evidence.json", evidence)
    calibration = OODCalibration(np.zeros(2), np.ones(2))
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id=plan.qualification_run_id,
        source_policy_sha256=policy_sha256,
        source_commit=plan.source_commit,
        qualification_protocol_sha256=plan.scientific_design_sha256(),
        seed_manifest_sha256=plan.seed_manifest_sha256(),
        evidence_artifact_sha256=evidence_sha256,
        development_cases=plan.development_cases,
        ood_calibration=calibration,
    )
    evidence["ood_calibration_sha256"] = receipt["ood_calibration_sha256"]
    evidence_sha256 = _write_json(directory / "qualification_evidence.json", evidence)
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id=plan.qualification_run_id,
        source_policy_sha256=policy_sha256,
        source_commit=plan.source_commit,
        qualification_protocol_sha256=plan.scientific_design_sha256(),
        seed_manifest_sha256=plan.seed_manifest_sha256(),
        evidence_artifact_sha256=evidence_sha256,
        development_cases=plan.development_cases,
        ood_calibration=calibration,
    )
    _write_json(directory / "qualification_receipt.json", receipt)
    return directory


def _add_transactional_completion(directory: Path) -> Path:
    plan = TSHCALOQualificationPlan.from_dict(
        json.loads((directory / "qualification_plan.json").read_text(encoding="utf-8"))
    )
    paired_runs = list(plan.seed_manifest()["paired_runs"])
    records_directory = directory / "records"
    records_directory.mkdir()
    entries = []
    cell_index = 0
    for case_name in plan.development_cases:
        for run_index, seeds in enumerate(paired_runs):
            for label in ("baseline", "candidate"):
                cell_index += 1
                identity = tsh_calo_qualification_cell_identity(
                    plan,
                    case_name=case_name,
                    run_index=run_index,
                    label=label,
                    seeds=seeds,
                )
                artifact_name = f"{case_name}-{run_index:03d}-{label}.json"
                artifact_path = records_directory / artifact_name
                artifact_sha256 = _write_json(
                    artifact_path,
                    {
                        "schema_version": TSH_CALO_QUALIFICATION_CELL_SUCCESS_SCHEMA,
                        "terminal_state": "committed_success",
                        "cell_identity": identity,
                        "cell_index": cell_index,
                        "total_cells": len(plan.development_cases) * plan.runs * 2,
                        "qualification_run_id": plan.qualification_run_id,
                        "qualification_plan_sha256": plan.execution_plan_sha256(),
                        "source_policy_sha256": plan.candidate_sha256,
                        "case": case_name,
                        "run_index": run_index,
                        "label": label,
                        "seeds": seeds,
                        "evaluations": plan.max_evaluations,
                    },
                )
                entries.append(
                    {
                        "cell_identity": identity,
                        "cell_index": cell_index,
                        "case": case_name,
                        "run_index": run_index,
                        "label": label,
                        "terminal_state": "committed_success",
                        "artifact_path": f"records/{artifact_name}",
                        "artifact_sha256": artifact_sha256,
                    }
                )
    expected = len(entries)
    index_sha256 = _write_json(
        directory / QUALIFICATION_CELL_INDEX_FILE,
        {
            "schema_version": TSH_CALO_QUALIFICATION_CELL_INDEX_SCHEMA,
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan.execution_plan_sha256(),
            "source_policy_sha256": plan.candidate_sha256,
            "expected_cells": expected,
            "committed_unique_cells": expected,
            "entries": entries,
        },
    )
    evidence_path = directory / "qualification_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["schema_version"] = TSH_CALO_QUALIFICATION_EVIDENCE_SCHEMA
    evidence["records"]["committed_unique"] = expected
    evidence["terminal_cell_index"] = {
        "schema_version": TSH_CALO_QUALIFICATION_CELL_INDEX_SCHEMA,
        "path": str(directory / QUALIFICATION_CELL_INDEX_FILE),
        "sha256": index_sha256,
        "committed_unique_cells": expected,
    }
    evidence["infrastructure_incidents"] = []
    evidence_sha256 = _write_json(evidence_path, evidence)
    calibration = OODCalibration(np.zeros(2), np.ones(2))
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id=plan.qualification_run_id,
        source_policy_sha256=plan.candidate_sha256,
        source_commit=plan.source_commit,
        qualification_protocol_sha256=plan.scientific_design_sha256(),
        seed_manifest_sha256=plan.seed_manifest_sha256(),
        evidence_artifact_sha256=evidence_sha256,
        development_cases=plan.development_cases,
        ood_calibration=calibration,
    )
    receipt_sha256 = _write_json(directory / "qualification_receipt.json", receipt)
    event_sha256 = _write_json(
        directory / QUALIFICATION_EVENT_LOG_FILE,
        {
            "schema_version": "tsh-calo-qualification-progress-event-v1",
            "event": "campaign_completed",
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan.execution_plan_sha256(),
        },
    )
    status_sha256 = _write_json(
        directory / QUALIFICATION_STATUS_FILE,
        {
            "schema_version": TSH_CALO_QUALIFICATION_STATUS_SCHEMA,
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan.execution_plan_sha256(),
            "state": "completed_qualified",
            "evidence_sha256": evidence_sha256,
            "receipt_sha256": receipt_sha256,
            "qualification_receipt_permitted": True,
        },
    )
    _write_json(
        directory / QUALIFICATION_COMPLETION_FILE,
        {
            "schema_version": TSH_CALO_QUALIFICATION_COMPLETION_SCHEMA,
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan.execution_plan_sha256(),
            "seed_manifest_sha256": plan.seed_manifest_sha256(),
            "source_policy_sha256": plan.candidate_sha256,
            "terminal_cell_index_sha256": index_sha256,
            "qualification_event_log_sha256": event_sha256,
            "qualification_status_sha256": status_sha256,
            "evidence_artifact_sha256": evidence_sha256,
            "receipt_sha256": receipt_sha256,
            "passed": True,
            "committed_unique_cells": expected,
            "failed_scientific_cells": 0,
            "infrastructure_incident_count": 0,
            "authority_boundary": (
                "completion_only_no_registration_activation_or_experiment_binding"
            ),
        },
    )
    return directory


def _convert_to_feasibility_assessment(directory: Path) -> Path:
    """Convert the synthetic current-contract fixture to the new measurement authority."""

    _add_transactional_completion(directory)
    plan = TSHCALOQualificationPlan.from_dict(
        json.loads((directory / "qualification_plan.json").read_text(encoding="utf-8"))
    )
    evidence_path = directory / "qualification_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for case in evidence["case_evidence"]:
        candidate_probability = float(case["candidate_feasible_probability"])
        case.update(
            {
                "n_pairs": plan.runs,
                "baseline_feasible_probability": max(0.0, candidate_probability - 0.05),
                "candidate_first_feasible_reached_probability": candidate_probability,
                "candidate_first_feasible_efficiency": 0.75,
                "candidate_first_feasible_evaluation_median": 4.0,
                "candidate_independent_validation_probability": 1.0,
            }
        )
    assessment = build_tsh_calo_feasibility_assessment(
        cases=evidence["case_evidence"], expected_case_order=plan.development_cases
    )
    evidence["schema_version"] = TSH_CALO_FEASIBILITY_ASSESSMENT_SCHEMA
    evidence.pop("decision", None)
    evidence["feasibility_assessment"] = assessment
    evidence["authority_boundary"] = (
        "measurement_only_scientist_decides_no_selection_activation_or_experiment_binding"
    )
    evidence_sha256 = _write_json(evidence_path, evidence)
    calibration = OODCalibration(np.zeros(2), np.ones(2))
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id=plan.qualification_run_id,
        source_policy_sha256=plan.candidate_sha256,
        source_commit=plan.source_commit,
        qualification_protocol_sha256=plan.scientific_design_sha256(),
        seed_manifest_sha256=plan.seed_manifest_sha256(),
        evidence_artifact_sha256=evidence_sha256,
        development_cases=plan.development_cases,
        ood_calibration=calibration,
    )
    receipt_sha256 = _write_json(directory / "qualification_receipt.json", receipt)
    event_sha256 = hashlib.sha256(
        (directory / QUALIFICATION_EVENT_LOG_FILE).read_bytes()
    ).hexdigest()
    status_sha256 = _write_json(
        directory / QUALIFICATION_STATUS_FILE,
        {
            "schema_version": TSH_CALO_QUALIFICATION_STATUS_SCHEMA,
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan.execution_plan_sha256(),
            "state": "completed_assessed",
            "evidence_sha256": evidence_sha256,
            "receipt_sha256": receipt_sha256,
            "qualification_receipt_permitted": False,
            "feasibility_receipt_permitted": True,
            "scientist_decision": "not_recorded",
        },
    )
    index_sha256 = hashlib.sha256(
        (directory / QUALIFICATION_CELL_INDEX_FILE).read_bytes()
    ).hexdigest()
    _write_json(
        directory / QUALIFICATION_COMPLETION_FILE,
        {
            "schema_version": TSH_CALO_FEASIBILITY_COMPLETION_SCHEMA,
            "qualification_run_id": plan.qualification_run_id,
            "qualification_plan_sha256": plan.execution_plan_sha256(),
            "seed_manifest_sha256": plan.seed_manifest_sha256(),
            "source_policy_sha256": plan.candidate_sha256,
            "terminal_cell_index_sha256": index_sha256,
            "qualification_event_log_sha256": event_sha256,
            "qualification_status_sha256": status_sha256,
            "evidence_artifact_sha256": evidence_sha256,
            "receipt_sha256": receipt_sha256,
            "assessment_complete": True,
            "automated_suitability_decision": None,
            "overall_feasibility_score": assessment["overall_feasibility_score"],
            "committed_unique_cells": len(plan.development_cases) * plan.runs * 2,
            "failed_scientific_cells": 0,
            "infrastructure_incident_count": 0,
            "authority_boundary": (
                "assessment_completion_only_scientist_decides_no_selection_activation_or_binding"
            ),
        },
    )
    return directory


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


def test_ensemble_rejects_duplicate_candidate_or_training_run(tmp_path):
    first = _candidate(tmp_path / "member-1.pt", seed=17)
    first_sha = inspect_tsh_calo_candidate(first).sha256
    with pytest.raises(ValueError, match="duplicate a source candidate"):
        assemble_tsh_calo_ensemble_candidate(
            tmp_path / "duplicate.pt",
            [(first, first_sha), (first, first_sha)],
        )

    torch.manual_seed(23)
    same_run = tmp_path / "same-run.pt"
    save_tsh_calo_candidate(
        same_run,
        TSHCALOPolicyNetwork(hidden_dim=16),
        _provenance(run_id="training-run-17"),
    )
    with pytest.raises(ValueError, match="independent training-run IDs"):
        assemble_tsh_calo_ensemble_candidate(
            tmp_path / "same-run-ensemble.pt",
            [
                (first, first_sha),
                (same_run, inspect_tsh_calo_candidate(same_run).sha256),
            ],
        )


def test_registry_keeps_tsh_candidate_separate_from_frozen_calo_runtime(tmp_path):
    registry = PolicyRegistry(ResultDatabase(tmp_path / "results.sqlite"))
    policy = registry.register(_ensemble(tmp_path), name="TSH ensemble")

    assert policy.algorithm_id == TSH_CALO_ALGORITHM_ID
    assert policy.qualification_status == "candidate"
    assert registry.training_evaluation_count(policy.id) == 16
    assert policy.metadata["training_candidate_evaluations"] == 16
    assert policy.metadata["training_evaluation_count_scope"] == (
        "cumulative_exact_training_candidate_evaluations"
    )
    assert policy.runtime_compatible is False
    assert policy.compatible_with(TSH_CALO_ALGORITHM_ID) is True
    with pytest.raises(ValueError, match="only.*TSH-CALO"):
        registry.activate(policy.id)
    with pytest.raises(ValueError, match="before qualification"):
        registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID, allow_unqualified=True)


def test_legacy_phase_receipts_are_not_a_qualification_compatibility_gate(tmp_path):
    path = _candidate(tmp_path / "legacy-authority-member.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    provenance = payload["metadata"]["training_provenance"]
    provenance["development_freeze_commit"] = ""
    provenance["development_freeze_sha256"] = ""
    provenance["phase4_acceptance_sha256"] = ""
    torch.save(payload, path)
    second = _candidate(tmp_path / "legacy-authority-member-2.pt", seed=23)
    second_payload = torch.load(second, map_location="cpu", weights_only=False)
    second_provenance = second_payload["metadata"]["training_provenance"]
    second_provenance["development_freeze_commit"] = ""
    second_provenance["development_freeze_sha256"] = ""
    second_provenance["phase4_acceptance_sha256"] = ""
    torch.save(second_payload, second)
    ensemble = assemble_tsh_calo_ensemble_candidate(
        tmp_path / "legacy-authority-ensemble.pt",
        ((path, checkpoint_sha256(path)), (second, checkpoint_sha256(second))),
    )
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(ensemble.path)

    inspected = registry.inspect_qualification_candidate(policy.id)

    assert policy.compatible_with(TSH_CALO_ALGORITHM_ID) is True
    assert inspected.sha256 == policy.sha256
    assert inspected.artifact_kind == "ensemble_policy"
    assert inspected.ensemble_size == 2

    database.add_policy_qualification(
        qualification_id="qualification-legacy-software-001",
        policy_id=policy.id,
        passed=True,
        grade="A",
        score=80.0,
        qualification_status="qualified",
        config=_qualification_config(policy.sha256),
    )
    active = registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    config = ExperimentConfig()
    binding = registry.bind_to_experiment_config(
        active.id,
        config,
        deterministic=True,
        algorithm_id=TSH_CALO_ALGORITHM_ID,
    )

    assert active.active is True
    assert binding["policy_sha256"] == policy.sha256
    assert (
        binding["policy_training_provenance"]["members"][0]["training_provenance"][
            "development_freeze_commit"
        ]
        == ""
    )


def test_qualification_contract_rejects_unsupported_experimental_architecture(tmp_path):
    path = _ensemble(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["feature_flags"]["population_schedule"] = True
    payload["metadata"]["feature_flags"]["allow_experimental_components"] = True
    torch.save(payload, path)
    registry = PolicyRegistry(ResultDatabase(tmp_path / "results.sqlite"))
    policy = registry.register(path)

    with pytest.raises(ValueError, match="unsupported experimental architecture options"):
        registry.inspect_qualification_candidate(policy.id)


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
    assert binding["allow_cpu_fallback"] is False
    assert binding["baseline_fallback_permitted"] is False
    assert binding["policy_qualification_id"] == "qualification-001"
    assert binding["policy_qualification_receipt_sha256"]
    assert binding["policy_ood_calibration_sha256"]
    assert binding["ood_calibration"]["mean"] == [0.0, 0.0]
    assert "policy_id" not in config.algorithm_parameters.get("CALO", {})
    assert config.algorithm_parameters[TSH_CALO_ALGORITHM_ID]["policy_id"] == policy.id


def test_formal_evidence_admission_is_explicit_integrity_bound_and_does_not_activate(tmp_path):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(model_directory), name="Evidence candidate")
    evidence_directory = _formal_evidence(
        tmp_path / "qualification", Path(policy.checkpoint_path), policy.sha256
    )

    verified = registry.inspect_qualification_evidence(policy.id, evidence_directory)
    assert registry.get(policy.id).qualification_status == "candidate"
    admitted = registry.admit_qualification_evidence(policy.id, evidence_directory)

    assert verified.policy_sha256 == policy.sha256
    assert admitted.qualification_status == "qualified"
    assert admitted.grade == "A"
    assert admitted.active is False
    assert database.list_policy_qualifications(policy.id)[0]["passed"] == 1
    summary = registry.qualification_evidence_summaries()[0]
    assert summary["recommendation"] == "Only policy in this evidence design"
    assert summary["summary"]["minimum_candidate_feasible_probability"] >= 0.95
    activated = registry.activate(admitted.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    assert activated.active is True


def test_verified_feasibility_requires_scientist_selection_before_separate_activation(tmp_path):
    model_directory = tmp_path / "feasibility-model"
    model_directory.mkdir()
    database = ResultDatabase(tmp_path / "feasibility-results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(_ensemble(model_directory), name="Feasibility candidate")
    evidence_directory = _convert_to_feasibility_assessment(
        _formal_evidence(
            tmp_path / "feasibility-assessment",
            Path(policy.checkpoint_path),
            policy.sha256,
            current_candidate_contract=True,
        )
    )

    verified = registry.inspect_feasibility_assessment(policy.id, evidence_directory)
    assessed = registry.admit_feasibility_assessment(policy.id, evidence_directory)

    assert verified.score == pytest.approx(97.0)
    assert assessed.qualification_status == "assessed"
    assert assessed.active is False
    with pytest.raises(ValueError, match="scientist selection"):
        registry.activate(assessed.id, algorithm_id=TSH_CALO_ALGORITHM_ID)

    selected = registry.select_assessed_policy(assessed.id)
    assert selected.qualification_status == "scientist_selected"
    assert selected.active is False
    active = registry.activate(selected.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    assert active.active is True
    config = ExperimentConfig()
    binding = registry.bind_to_experiment_config(
        active.id,
        config,
        deterministic=True,
        algorithm_id=TSH_CALO_ALGORITHM_ID,
    )
    assert binding["policy_assessment_id"] == verified.assessment_id
    assert binding["policy_scientist_selection"]["candidate_sha256"] == policy.sha256
    assert binding["policy_scientist_selection"]["activation_performed"] is True


def test_feasibility_admission_selection_and_activation_are_three_distinct_states(tmp_path):
    database = ResultDatabase(tmp_path / "assessment-results.sqlite")
    registry = PolicyRegistry(database)
    model_directory = tmp_path / "assessment-model"
    model_directory.mkdir()
    policy = registry.register(_ensemble(model_directory), name="Measured")
    assessment_id = "assessment-001"
    evidence_sha256 = _sha("assessment-evidence")
    metrics = {
        "admission_schema_version": "tsh-calo-policy-feasibility-admission-v1",
        "candidate_sha256": policy.sha256,
        "evidence_artifact_sha256": evidence_sha256,
        "evidence_directory": str(tmp_path / "retained-assessment"),
        "feasibility_assessment": {
            "overall_feasibility_score": 76.0,
            "automated_suitability_decision": None,
            "decision_authority": "scientist_only",
        },
    }

    admitted = database.admit_verified_policy_assessment(
        assessment_id=assessment_id,
        policy_id=policy.id,
        expected_sha256=policy.sha256,
        config=_qualification_config(policy.sha256),
        metrics=metrics,
        score=76.0,
    )

    assert admitted is True
    assessed = registry.get(policy.id)
    assert assessed.qualification_status == "assessed"
    assert assessed.active is False
    retained = database.list_policy_qualifications(policy.id)[0]
    assert retained["passed"] == 0
    assert retained["grade"] == "N/A"

    selected = database.record_scientist_policy_selection(
        policy_id=policy.id,
        assessment_id=assessment_id,
        expected_sha256=policy.sha256,
        evidence_sha256=evidence_sha256,
    )

    assert selected is True
    scientist_selected = registry.get(policy.id)
    assert scientist_selected.qualification_status == "scientist_selected"
    assert scientist_selected.active is False
    assert scientist_selected.metadata["scientist_selection"] == {
        "schema_version": "tsh-calo-scientist-policy-selection-v1",
        "assessment_id": assessment_id,
        "candidate_sha256": policy.sha256,
        "evidence_sha256": evidence_sha256,
        "selected_at": scientist_selected.metadata["scientist_selection"]["selected_at"],
        "activation_performed": False,
    }
    assert database.list_policy_qualifications(policy.id)[0]["passed"] == 1
    with pytest.raises((OSError, ValueError)):
        registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    assert registry.get(policy.id).active is False


def test_transactional_evidence_requires_the_final_completion_authority(tmp_path):
    model_directory = tmp_path / "model-transactional"
    model_directory.mkdir()
    registry = PolicyRegistry(ResultDatabase(tmp_path / "transactional-results.sqlite"))
    policy = registry.register(_ensemble(model_directory), name="Transactional candidate")
    evidence_directory = _add_transactional_completion(
        _formal_evidence(
            tmp_path / "transactional-qualification",
            Path(policy.checkpoint_path),
            policy.sha256,
        )
    )

    verified = registry.inspect_qualification_evidence(policy.id, evidence_directory)
    assert verified.policy_sha256 == policy.sha256
    (evidence_directory / QUALIFICATION_COMPLETION_FILE).unlink()

    with pytest.raises(ValueError, match="completion"):
        registry.admit_qualification_evidence(policy.id, evidence_directory)
    assert registry.get(policy.id).qualification_status == "candidate"
    assert registry.database.list_policy_qualifications(policy.id) == []


def test_policy_comparison_names_a_leader_only_within_one_matching_design(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    designs = (("conservative", 0.0), ("stronger", 2.0))
    for index, (label, strength) in enumerate(designs):
        model_directory = tmp_path / f"model-{label}"
        model_directory.mkdir()
        policy = registry.register(_ensemble(model_directory, seed_offset=index * 100), name=label)
        evidence = _formal_evidence(
            tmp_path / f"qualification-{label}",
            Path(policy.checkpoint_path),
            policy.sha256,
            strength=strength,
        )
        registry.admit_qualification_evidence(policy.id, evidence)

    summaries = {item["policy_name"]: item for item in registry.qualification_evidence_summaries()}
    assert (
        summaries["conservative"]["comparison_protocol_sha256"]
        == summaries["stronger"]["comparison_protocol_sha256"]
    )
    assert summaries["stronger"]["recommendation"] == "Strongest comparable evidence"
    assert summaries["conservative"]["recommendation"] == (
        "Dominated by strongest comparable evidence"
    )


def test_tampered_qualification_decision_is_rejected_without_registry_mutation(tmp_path):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    registry = PolicyRegistry(ResultDatabase(tmp_path / "results.sqlite"))
    policy = registry.register(_ensemble(model_directory))
    directory = _formal_evidence(
        tmp_path / "qualification", Path(policy.checkpoint_path), policy.sha256
    )
    evidence_path = directory / "qualification_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["case_evidence"][0]["median_relative_objective_improvement"] = -1.0
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="canonical frozen gates"):
        registry.admit_qualification_evidence(policy.id, directory)
    assert registry.get(policy.id).qualification_status == "candidate"
    assert registry.database.list_policy_qualifications(policy.id) == []


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

    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    assert payload["metadata"]["training_environment_version"] == TSH_CALO_TRAINING_ENVIRONMENT
    payload["metadata"]["training_environment_version"] = "tsh-calo-training-v2-counted-safe80"
    legacy_environment = tmp_path / "legacy-training-environment.pt"
    torch.save(payload, legacy_environment)
    record = registry.register(legacy_environment)
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

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
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
