"""Independent TSH-CALO qualification campaign and non-promotion invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from calo_rpd_studio.algorithms.calo.tsh_calo_policy import TSHCALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    IndependentTrainingProvenance,
    assemble_tsh_calo_ensemble_candidate,
    inspect_tsh_calo_candidate,
    save_tsh_calo_candidate,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    QUALIFICATION_CELL_INDEX_FILE,
    QUALIFICATION_COMPLETION_FILE,
    QUALIFICATION_CONTROL_FILE,
    QUALIFICATION_EVENT_LOG_FILE,
    QUALIFICATION_INFRASTRUCTURE_DIRECTORY,
    QUALIFICATION_STATUS_FILE,
    QualificationCampaignLeaseUnavailable,
    QualificationEvidenceIntegrityError,
    QualificationInfrastructureError,
    TSH_CALO_COMPONENT_EVIDENCE_SCHEMA,
    TSH_CALO_QUALIFICATION_EVENT_SCHEMA,
    TSHCALOQualificationPauseRequested,
    TSHCALOQualificationCampaign,
    TSHCALOQualificationPlan,
    _ExclusiveQualificationCampaignLease,
    _verify_component_evidence,
    inspect_tsh_calo_qualification_resume_state,
    qualification_candidate_contract,
    request_tsh_calo_qualification_pause,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_receipt import (
    build_tsh_calo_training_episode_receipt,
    canonical_reward_sequence_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    run_id = f"qualification-test-member-{seed}"
    design = _sha("qualification-test-training-design")
    receipt = build_tsh_calo_training_episode_receipt(
        session_id=run_id + "-session",
        training_run_id=run_id,
        training_design_sha256=design,
        session_design_sha256=_sha(f"session-{seed}"),
        environment_design_sha256=_sha(f"environment-{seed}"),
        case_identity="case30",
        case_checksum=_sha("case30"),
        problem_fingerprint=_sha("problem"),
        seed=seed,
        deterministic_policy=True,
        candidate_evaluations=8,
        scenario_power_flow_calls=8,
        canonical_transition_count=1,
        ppo_update_count=1,
        canonical_reward_sha256=canonical_reward_sequence_sha256((0.1,)),
        accounting_complete=True,
        terminal=True,
    )
    provenance = IndependentTrainingProvenance(
        training_run_id=run_id,
        training_design_sha256=design,
        source_commit="qualification-test",
        development_cases=("case30",),
        seed_manifest_sha256=_sha("training-seeds"),
        training_device_provenance=_device_provenance(),
        training_episode_receipts=(receipt.to_dict(),),
    )
    save_tsh_calo_candidate(path, TSHCALOPolicyNetwork(hidden_dim=16), provenance)
    return path


def _ensemble(tmp_path: Path) -> tuple[Path, str]:
    members = [_member(tmp_path / f"member-{seed}.pt", seed) for seed in (17, 23)]
    artifact = assemble_tsh_calo_ensemble_candidate(
        tmp_path / "ensemble.pt",
        [(path, inspect_tsh_calo_candidate(path).sha256) for path in members],
    )
    return Path(artifact.path), artifact.sha256


def _plan(path: Path, sha256: str, **changes) -> TSHCALOQualificationPlan:
    values = {
        "qualification_run_id": "qualification-screening-001",
        "source_commit": "1" * 40,
        "candidate_path": str(path),
        "candidate_sha256": sha256,
        "development_cases": ("case30",),
        "runs": 2,
        "master_seed": 1907,
        "population_size": 4,
        "max_evaluations": 8,
        "calibration_samples_per_case": 4,
        "calibration_population_size": 4,
        "bootstrap_resamples": 1_000,
        "inference_device": "cpu",
    }
    values.update(changes)
    return TSHCALOQualificationPlan(**values)


def test_screening_campaign_retains_evidence_but_cannot_emit_receipt(tmp_path):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256)
    result = TSHCALOQualificationCampaign(plan, tmp_path / "screening").start()

    assert result["passed"] is False
    assert result["receipt"] is None
    assert result["registration_performed"] is False
    assert result["activation_performed"] is False
    output = Path(result["evidence_path"])
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["records"] == {
        "expected": 4,
        "completed": 4,
        "failed": 0,
        "committed_unique": 4,
        "directory": str(output.parent / "records"),
        "failures_directory": str(output.parent / "failures"),
    }
    assert "screening campaigns cannot qualify" in " ".join(evidence["decision"]["reasons"])
    assert evidence["protected_cases_opened"] is False
    assert not (output.parent / "qualification_receipt.json").exists()
    candidate_records = [
        json.loads(item.read_text(encoding="utf-8"))
        for item in (output.parent / "records").glob("*-candidate.json")
    ]
    assert candidate_records
    assert all(item["evaluations"] == 8 for item in candidate_records)
    assert all(item["source_policy_sha256"] == sha256 for item in candidate_records)
    assert (output.parent / QUALIFICATION_CELL_INDEX_FILE).is_file()
    assert (output.parent / QUALIFICATION_COMPLETION_FILE).is_file()


def test_completion_event_failure_preserves_one_success_and_aborts_infrastructure(tmp_path):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256, runs=1)
    output = tmp_path / "completion-event-failure"

    def fail_after_commit(event):
        if event["event"] == "cell_completed":
            raise RuntimeError("injected completion telemetry failure")

    with pytest.raises(QualificationInfrastructureError, match="cell completion event"):
        TSHCALOQualificationCampaign(plan, output, event_callback=fail_after_commit).start()

    records = list((output / "records").glob("*.json"))
    failures = list((output / "failures").glob("*.json"))
    index = json.loads((output / QUALIFICATION_CELL_INDEX_FILE).read_text(encoding="utf-8"))
    status = json.loads((output / QUALIFICATION_STATUS_FILE).read_text(encoding="utf-8"))
    assert len(records) == 1
    assert failures == []
    assert index["committed_unique_cells"] == 1
    assert index["entries"][0]["terminal_state"] == "committed_success"
    assert status["state"] == "infrastructure_aborted"
    assert status["fresh_run_required"] is True
    assert list((output / QUALIFICATION_INFRASTRUCTURE_DIRECTORY).glob("*.json"))
    assert not (output / "qualification_evidence.json").exists()
    assert not (output / "qualification_receipt.json").exists()
    assert not (output / QUALIFICATION_COMPLETION_FILE).exists()
    with pytest.raises(QualificationEvidenceIntegrityError, match="fresh source-bound"):
        TSHCALOQualificationCampaign(plan, output).resume()


def test_progress_telemetry_failure_is_not_a_scientific_cell_failure(
    tmp_path, monkeypatch
):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256, runs=1, max_evaluations=12)
    output = tmp_path / "progress-event-failure"
    monkeypatch.setattr(
        "calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign."
        "QUALIFICATION_CHECKPOINT_INTERVAL",
        4,
    )

    def fail_during_live_work(event):
        if event["event"] == "cell_progress":
            raise RuntimeError("injected live telemetry failure")

    with pytest.raises(QualificationInfrastructureError, match="cell progress event"):
        TSHCALOQualificationCampaign(
            plan, output, event_callback=fail_during_live_work
        ).start()

    assert list((output / "records").glob("*.json")) == []
    assert list((output / "failures").glob("*.json")) == []
    status = json.loads((output / QUALIFICATION_STATUS_FILE).read_text(encoding="utf-8"))
    assert status["state"] == "infrastructure_aborted"
    assert status["qualification_receipt_permitted"] is False


def test_result_record_construction_failure_is_infrastructure_not_scientific(
    tmp_path, monkeypatch
):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256, runs=1)
    output = tmp_path / "result-record-failure"

    def fail_result_record(**_kwargs):
        raise RuntimeError("injected result-record construction failure")

    monkeypatch.setattr(
        "calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign._result_record",
        fail_result_record,
    )
    with pytest.raises(QualificationInfrastructureError, match="result record construction"):
        TSHCALOQualificationCampaign(plan, output).start()

    assert list((output / "records").glob("*.json")) == []
    assert list((output / "failures").glob("*.json")) == []
    status = json.loads((output / QUALIFICATION_STATUS_FILE).read_text(encoding="utf-8"))
    assert status["state"] == "infrastructure_aborted"
    assert status["qualification_receipt_permitted"] is False


def test_conflicting_terminal_artifacts_are_read_only_and_require_fresh_run(tmp_path):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256, runs=1)
    output = tmp_path / "contradictory-retained"
    (output / "records").mkdir(parents=True)
    (output / "failures").mkdir()
    (output / "qualification_plan.json").write_text(
        json.dumps(plan.to_dict(), sort_keys=True), encoding="utf-8"
    )
    artifact_name = "case30-000-baseline.json"
    (output / "records" / artifact_name).write_text("{}", encoding="utf-8")
    (output / "failures" / artifact_name).write_text("{}", encoding="utf-8")
    before = {
        str(item.relative_to(output)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in output.rglob("*")
        if item.is_file()
    }

    disposition = inspect_tsh_calo_qualification_resume_state(output)
    assert disposition["classification"] == "infrastructure_aborted"
    assert disposition["fresh_run_required"] is True
    assert disposition["conflicting_terminal_artifacts"] == [artifact_name]
    with pytest.raises(QualificationEvidenceIntegrityError, match="Contradictory success"):
        TSHCALOQualificationCampaign(plan, output).resume()

    after = {
        str(item.relative_to(output)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in output.rglob("*")
        if item.is_file()
    }
    assert after == before


def test_qualification_micro_events_pause_inside_cell_and_exactly_resume(
    tmp_path, monkeypatch
):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256, max_evaluations=12)
    output = tmp_path / "resumable-screening"
    events = []
    pause_requested = False
    monkeypatch.setattr(
        "calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign."
        "QUALIFICATION_CHECKPOINT_INTERVAL",
        4,
    )

    def capture(event):
        nonlocal pause_requested
        events.append(dict(event))
        if event["event"] == "cell_progress" and not pause_requested:
            pause_requested = True
            request_tsh_calo_qualification_pause(output)

    campaign = TSHCALOQualificationCampaign(plan, output, event_callback=capture)
    with pytest.raises(TSHCALOQualificationPauseRequested):
        campaign.start()

    status = json.loads((output / QUALIFICATION_STATUS_FILE).read_text(encoding="utf-8"))
    control = json.loads((output / QUALIFICATION_CONTROL_FILE).read_text(encoding="utf-8"))
    event_log = [
        json.loads(line)
        for line in (output / QUALIFICATION_EVENT_LOG_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert status["state"] == "paused"
    assert status["pause"]["boundary"] == "optimizer_checkpoint"
    assert status["pause"]["resumable"] is True
    assert control["state"] == "acknowledged"
    assert control["evaluations"] == 8
    assert Path(control["durable_path"]).is_file()
    assert any(item["event"] == "cell_progress" for item in events)
    assert event_log[-1]["schema_version"] == TSH_CALO_QUALIFICATION_EVENT_SCHEMA
    assert event_log[-1]["event"] == "campaign_paused"

    resumed_events = []
    result = TSHCALOQualificationCampaign(
        plan,
        output,
        event_callback=lambda event: resumed_events.append(dict(event)),
    ).resume()

    assert result["registration_performed"] is False
    assert result["activation_performed"] is False
    assert any(
        item["event"] == "cell_started" and item["resumed_checkpoint"] is True
        for item in resumed_events
    )
    assert any(item["event"] == "campaign_completed" for item in resumed_events)
    assert json.loads((output / QUALIFICATION_STATUS_FILE).read_text(encoding="utf-8"))[
        "state"
    ] == "completed_not_qualified"

    uninterrupted_output = tmp_path / "uninterrupted-screening"
    TSHCALOQualificationCampaign(plan, uninterrupted_output).start()
    for resumed_path in sorted((output / "records").glob("*.json")):
        resumed_record = json.loads(resumed_path.read_text(encoding="utf-8"))
        uninterrupted_record = json.loads(
            (uninterrupted_output / "records" / resumed_path.name).read_text(
                encoding="utf-8"
            )
        )
        for field in (
            "seeds",
            "feasible",
            "objective",
            "violation",
            "evaluations",
            "iterations",
            "first_feasible_evaluation",
            "best_vector",
            "anytime",
        ):
            assert resumed_record[field] == uninterrupted_record[field]


def test_formal_plan_requires_frozen_candidate_architecture_contract(tmp_path):
    path, sha256 = _ensemble(tmp_path)
    plan = _plan(path, sha256, mode="formal", runs=30)

    with pytest.raises(ValueError, match="candidate architecture contract"):
        plan.validate()

    artifact = inspect_tsh_calo_candidate(path, expected_sha256=sha256)
    contracted = _plan(
        path,
        sha256,
        mode="formal",
        runs=30,
        candidate_contract=qualification_candidate_contract(artifact),
    )
    contracted.validate()


def test_formal_component_evidence_requires_frozen_direct_analysis(tmp_path):
    path, sha256 = _ensemble(tmp_path)
    references = {}
    for component in "ABCDE":
        evidence_path = tmp_path / f"component-{component}.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "schema_version": TSH_CALO_COMPONENT_EVIDENCE_SCHEMA,
                    "component": component,
                    "accepted": True,
                    "source_policy_sha256": sha256,
                    "development_cases": ["case30"],
                }
            ),
            encoding="utf-8",
        )
        references[component] = {
            "path": str(evidence_path),
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        }
    plan = _plan(path, sha256, mode="formal", runs=30, component_evidence=references)

    with pytest.raises(ValueError, match="another source"):
        _verify_component_evidence(plan)


def test_qualification_plan_rejects_protected_cases_and_mixed_contract_authorities(tmp_path):
    path, sha256 = _ensemble(tmp_path)
    protected = _plan(path, sha256, development_cases=("case30", "case118"))
    with pytest.raises(ValueError, match="Protected holdouts"):
        protected.validate()

    artifact = inspect_tsh_calo_candidate(path, expected_sha256=sha256)
    mixed = _plan(
        path,
        sha256,
        mode="formal",
        runs=30,
        candidate_contract=qualification_candidate_contract(artifact),
        component_evidence={component: {} for component in "ABCDE"},
    )
    with pytest.raises(ValueError, match="cannot mix"):
        mixed.validate()


def test_qualification_campaign_has_no_registry_or_activation_authority():
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_qualification_campaign.py").read_text(
        encoding="utf-8"
    )

    assert "PolicyRegistry" not in source
    assert ".activate(" not in source
    assert "add_policy_qualification" not in source
    assert "tsh_calo_training_session" not in source


def test_qualification_campaign_has_one_OS_released_writer(tmp_path):
    with _ExclusiveQualificationCampaignLease(tmp_path):
        with pytest.raises(QualificationCampaignLeaseUnavailable, match="already owns"):
            _ExclusiveQualificationCampaignLease(tmp_path)

    with _ExclusiveQualificationCampaignLease(tmp_path):
        pass


def test_integrity_failed_campaign_cannot_resume(tmp_path):
    output = tmp_path / "failed"
    output.mkdir()
    (output / "campaign_integrity_failure.json").write_text("{}", encoding="utf-8")
    plan = _plan(tmp_path / "missing.pt", _sha("missing"))

    with pytest.raises(QualificationEvidenceIntegrityError, match="integrity failure marker"):
        TSHCALOQualificationCampaign(plan, output).resume()
