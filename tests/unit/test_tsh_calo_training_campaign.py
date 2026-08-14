"""Frozen-plan and exact-resume tests for independent TSH-CALO member campaigns."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from calo_rpd_studio.ai.model_io import checkpoint_sha256
import calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign as campaign_module
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    count_tsh_calo_candidate_training_evaluations,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    CUDA_DURABLE_EVALUATION_WINDOW,
    IndependentTSHCALOTrainingCampaign,
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingEpisodePlan,
    TSHCALOTrainingMemberPlan,
    TSHCALOTrainingPauseRequested,
    TSH_CALO_TRAINING_EVENT_SCHEMA,
    request_tsh_calo_training_pause,
    tsh_calo_training_compatibility_contract,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_extension import (
    IndependentTSHCALOTrainingExtension,
    extension_plan_summary,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.scripts import accept_development_freeze as acceptance
from calo_rpd_studio.scripts import train_tsh_calo
from calo_rpd_studio.scripts import create_development_freeze_candidate as freeze


SOURCE_COMMIT = "a4329c3a39ae2646da134c4d3219b957c7c3c2bc"
DEVELOPMENT_FREEZE_SHA256 = "f" * 64
PHASE4_ACCEPTANCE_SHA256 = "e" * 64


def _plan() -> TSHCALOTrainingCampaignPlan:
    return TSHCALOTrainingCampaignPlan(
        campaign_id="fresh-campaign-001",
        source_commit=SOURCE_COMMIT,
        development_freeze_commit=SOURCE_COMMIT,
        development_freeze_sha256=DEVELOPMENT_FREEZE_SHA256,
        phase4_acceptance_sha256=PHASE4_ACCEPTANCE_SHA256,
        development_cases=("toy-development",),
        members=(
            TSHCALOTrainingMemberPlan(
                "member-001",
                101,
                (TSHCALOTrainingEpisodePlan("member-001-episode-001", "toy-development", 201),),
            ),
            TSHCALOTrainingMemberPlan(
                "member-002",
                102,
                (TSHCALOTrainingEpisodePlan("member-002-episode-001", "toy-development", 202),),
            ),
        ),
        resource_envelope=TSHCALOTrainingResourceEnvelope(1, 4, 8, 16, 16, 4),
        population_size=4,
        max_evaluations=8,
        requested_device="cpu",
    )


def test_builtin_architecture_plan_needs_no_legacy_governance_receipts():
    plan = replace(
        _plan(),
        development_freeze_commit="",
        development_freeze_sha256="",
        phase4_acceptance_sha256="",
    )

    plan.validate()
    assert plan.training_config(plan.members[0]).development_freeze_commit == ""

    with pytest.raises(ValueError, match="complete or absent"):
        replace(plan, development_freeze_commit=SOURCE_COMMIT).validate()


def _factory(toy_case):
    return lambda _identity: ORPDProblem(toy_case)


def _model_states(result):
    states = []
    for candidate in result.member_candidates:
        payload = torch.load(candidate.path, map_location="cpu", weights_only=True)
        states.append(payload["model_state_dict"])
    return states


def test_campaign_freezes_plan_and_exports_only_unqualified_candidates(tmp_path, toy_case):
    plan = _plan()
    streamed_events = []
    restored = TSHCALOTrainingCampaignPlan.from_dict(plan.to_dict())
    assert restored.execution_plan_sha256() == plan.execution_plan_sha256()
    assert restored.seed_manifest_sha256() == plan.seed_manifest_sha256()

    result = IndependentTSHCALOTrainingCampaign(
        plan,
        tmp_path / "campaign",
        problem_factory=_factory(toy_case),
        event_callback=streamed_events.append,
    ).start()

    assert len(result.member_candidates) == 2
    assert result.ensemble_candidate.artifact_kind == "ensemble_policy"
    assert result.ensemble_candidate.ensemble_size == 2
    for index, candidate in enumerate(result.member_candidates):
        provenance = candidate.training_provenance
        assert provenance["training_run_id"] == f"fresh-campaign-001:member-00{index + 1}"
        assert provenance["development_freeze_commit"] == SOURCE_COMMIT
        assert provenance["development_freeze_sha256"] == DEVELOPMENT_FREEZE_SHA256
        assert provenance["phase4_acceptance_sha256"] == PHASE4_ACCEPTANCE_SHA256
        assert provenance["initialization_policy_sha256"] == ""
        assert provenance["seed_manifest_sha256"] == plan.seed_manifest_sha256()
        assert len(provenance["training_episode_receipts"]) == 1
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["state"] == "completed_unqualified"
    assert manifest["authority_boundary"] == {
        "training_only": True,
        "registered": False,
        "qualified": False,
        "activated": False,
        "experiment_bound": False,
    }
    assert manifest["extension_contract"]["repeatable_finite_segments"] is True
    assert manifest["training_compatibility_contract"]["algorithm_version"]
    assert manifest["training_compatibility_contract"]["policy_parameter_layout_sha256"]
    assert manifest["training_compatibility_contract"][
        "training_parameter_schema_sha256"
    ]
    assert manifest["training_compatibility_contract"]["plan_field_schema_sha256"]
    assert len(manifest["continuation_checkpoints"]) == len(plan.members)
    for checkpoint in manifest["continuation_checkpoints"]:
        assert checkpoint_sha256(
            tmp_path / "campaign" / checkpoint["path"]
        ) == checkpoint["sha256"]
    event_names = [event["event"] for event in streamed_events]
    assert event_names[0] == "campaign_started"
    assert event_names[-1] == "campaign_completed"
    assert event_names.count("checkpoint_committed") == 2
    assert event_names.count("member_completed") == 2
    assert [event["event_sequence"] for event in streamed_events] == list(
        range(1, len(streamed_events) + 1)
    )
    with pytest.raises(FileExistsError, match="explicit resume"):
        IndependentTSHCALOTrainingCampaign(
            plan,
            tmp_path / "campaign",
            problem_factory=_factory(toy_case),
        ).start()


def test_completed_campaign_can_add_repeatable_finite_authenticated_extensions(
    tmp_path, toy_case
):
    plan = _plan()
    root = tmp_path / "extendable-campaign"
    base = IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()
    base_manifest_bytes = Path(base.manifest_path).read_bytes()
    base_states = _model_states(base)

    readiness = extension_plan_summary(plan, root)
    assert readiness["completed_extension_count"] == 0
    assert readiness["source_revision_is_compatibility_identity"] is False
    assert readiness["architecture_and_parameter_schema_required"] is True
    assert readiness["segment_candidate_evaluations"] == 16
    assert readiness["next_cumulative_candidate_evaluations"] == 32
    with pytest.raises(ValueError, match="plan changed"):
        extension_plan_summary(replace(plan, max_evaluations=16), root)

    extension_source_commit = "b" * 40
    first = IndependentTSHCALOTrainingExtension(
        plan,
        root,
        problem_factory=_factory(toy_case),
        execution_source_commit=extension_source_commit,
    ).start()
    second = IndependentTSHCALOTrainingExtension(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()

    assert Path(base.manifest_path).read_bytes() == base_manifest_bytes
    first_manifest = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    second_manifest = json.loads(Path(second.manifest_path).read_text(encoding="utf-8"))
    assert first_manifest["segment_number"] == 1
    assert first_manifest["source_commit"] == SOURCE_COMMIT
    assert first_manifest["execution_source_commit"] == extension_source_commit
    assert first_manifest["cumulative_candidate_evaluations"] == 32
    assert second_manifest["segment_number"] == 2
    assert second_manifest["cumulative_candidate_evaluations"] == 48
    assert count_tsh_calo_candidate_training_evaluations(base.ensemble_candidate) == 16
    assert count_tsh_calo_candidate_training_evaluations(first.ensemble_candidate) == 32
    assert count_tsh_calo_candidate_training_evaluations(second.ensemble_candidate) == 48
    assert second_manifest["parent_manifest_sha256"] == first.manifest_sha256
    for candidate in first.member_candidates:
        assert candidate.training_provenance["source_commit"] == SOURCE_COMMIT
        assert (
            candidate.training_provenance["execution_source_commit"]
            == extension_source_commit
        )
        receipts = candidate.training_provenance["training_episode_receipts"]
        assert len(receipts) == 2
        assert receipts[-1]["session_id"].endswith(":extension:000001")
        assert receipts[-1]["candidate_evaluations"] == plan.max_evaluations
    for candidate in second.member_candidates:
        receipts = candidate.training_provenance["training_episode_receipts"]
        assert len(receipts) == 3
        assert receipts[-1]["session_id"].endswith(":extension:000002")
        assert receipts[-1]["candidate_evaluations"] == plan.max_evaluations
    for initial, extended in zip(base_states, _model_states(second)):
        assert any(not torch.equal(initial[name], extended[name]) for name in initial)


def test_extension_blocks_added_or_removed_training_parameter_fields(tmp_path, toy_case):
    plan = _plan()
    root = tmp_path / "parameter-schema-campaign"
    IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()
    stored_plan_path = root / "training_plan.json"
    stored_plan = json.loads(stored_plan_path.read_text(encoding="utf-8"))
    stored_plan["training"].pop("learning_rate")
    stored_plan_path.write_text(json.dumps(stored_plan), encoding="utf-8")

    with pytest.raises(ValueError, match="parameter schema changed.*added fields"):
        extension_plan_summary(plan, root)


def test_extension_ignores_reserved_plan_writer_metadata(tmp_path, toy_case):
    plan = _plan()
    root = tmp_path / "writer-metadata-campaign"
    IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()
    stored_plan_path = root / "training_plan.json"
    stored_plan = json.loads(stored_plan_path.read_text(encoding="utf-8"))
    stored_plan["writer_metadata"] = {
        "product_version": "99.0.0",
        "source_writer": "future-calo-studio",
    }
    stored_plan_path.write_text(json.dumps(stored_plan), encoding="utf-8")

    readiness = extension_plan_summary(plan, root)

    assert readiness["completed_extension_count"] == 0
    assert readiness["training_compatibility_contract"][
        "training_parameter_schema_sha256"
    ]


def test_precontract_completed_model_remains_extendable_when_checkpoint_architecture_matches(
    tmp_path,
    toy_case,
):
    plan = _plan()
    root = tmp_path / "precontract-compatible-campaign"
    result = IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()
    manifest_path = Path(result.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("training_compatibility_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status_path = root / "training_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["manifest_sha256"] = checkpoint_sha256(manifest_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    readiness = extension_plan_summary(plan, root)

    assert readiness["completed_extension_count"] == 0
    assert readiness["training_compatibility_contract"]["algorithm_version"]


def test_extension_ignores_writer_metadata_but_blocks_changed_frozen_architecture(
    tmp_path,
    toy_case,
):
    plan = _plan()
    root = tmp_path / "architecture-contract-campaign"
    result = IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()
    manifest_path = Path(result.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_compatibility_contract"].pop("policy_parameter_layout_sha256")
    manifest["training_compatibility_contract"]["writer_product_version"] = "99.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status_path = root / "training_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["manifest_sha256"] = checkpoint_sha256(manifest_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    assert extension_plan_summary(plan, root)["completed_extension_count"] == 0

    manifest["training_compatibility_contract"][
        "training_parameter_schema_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture or parameter schema changed"):
        extension_plan_summary(plan, root)

    manifest["training_compatibility_contract"][
        "training_parameter_schema_sha256"
    ] = tsh_calo_training_compatibility_contract(plan)[
        "training_parameter_schema_sha256"
    ]
    manifest["training_compatibility_contract"]["algorithm_version"] = "changed-architecture"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="architecture or parameter schema changed"):
        extension_plan_summary(plan, root)


def test_training_extension_pauses_and_resumes_at_authenticated_checkpoint(
    tmp_path, toy_case
):
    plan = _plan()
    root = tmp_path / "paused-extension"
    IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()
    extension = IndependentTSHCALOTrainingExtension(
        plan,
        root,
        problem_factory=_factory(toy_case),
    )

    def pause_extension(_event):
        assert extension.segment_directory is not None
        request_tsh_calo_training_pause(extension.segment_directory)

    extension.transition_callback = pause_extension
    with pytest.raises(TSHCALOTrainingPauseRequested):
        extension.start()
    paused_segment = extension.segment_directory
    assert paused_segment is not None
    paused_status = json.loads(
        (paused_segment / "training_status.json").read_text(encoding="utf-8")
    )
    assert paused_status["state"] == "interrupted"
    assert paused_status["extension"]["segment_number"] == 1
    assert paused_status["uncommitted_cuda_window"] is None

    completed = IndependentTSHCALOTrainingExtension(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start_or_resume()
    manifest = json.loads(Path(completed.manifest_path).read_text(encoding="utf-8"))
    assert manifest["segment_number"] == 1
    assert manifest["cumulative_candidate_evaluations"] == 32


def test_interrupted_campaign_resumes_exactly_from_authenticated_session(tmp_path, toy_case):
    plan = _plan()
    interrupted_path = tmp_path / "interrupted"
    calls = 0

    def interrupt_once(_status):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        IndependentTSHCALOTrainingCampaign(
            plan,
            interrupted_path,
            problem_factory=_factory(toy_case),
            transition_callback=interrupt_once,
        ).start()
    interrupted_status = json.loads(
        (interrupted_path / "training_status.json").read_text(encoding="utf-8")
    )
    assert interrupted_status["state"] == "interrupted"
    assert interrupted_status["session_checkpoint"]["transition_count"] == 1

    resumed = IndependentTSHCALOTrainingCampaign(
        plan,
        interrupted_path,
        problem_factory=_factory(toy_case),
    ).resume()
    uninterrupted = IndependentTSHCALOTrainingCampaign(
        plan,
        tmp_path / "uninterrupted",
        problem_factory=_factory(toy_case),
    ).start()

    for resumed_state, expected_state in zip(_model_states(resumed), _model_states(uninterrupted)):
        assert resumed_state.keys() == expected_state.keys()
        for name, tensor in expected_state.items():
            torch.testing.assert_close(resumed_state[name], tensor, rtol=0.0, atol=0.0)
    resumed_receipts = [
        item.training_provenance["training_episode_receipts"] for item in resumed.member_candidates
    ]
    expected_receipts = [
        item.training_provenance["training_episode_receipts"]
        for item in uninterrupted.member_candidates
    ]
    assert resumed_receipts == expected_receipts


def test_checkpoint_safe_pause_can_repeat_without_changing_finite_plan(tmp_path, toy_case):
    plan = _plan()
    output = tmp_path / "repeated-safe-pause"

    def pause_at_next_checkpoint(event):
        assert event["schema_version"] == TSH_CALO_TRAINING_EVENT_SCHEMA
        assert event["event"] == "checkpoint_committed"
        request = request_tsh_calo_training_pause(output)
        assert request["plan_sha256"] == plan.execution_plan_sha256()

    with pytest.raises(TSHCALOTrainingPauseRequested):
        IndependentTSHCALOTrainingCampaign(
            plan,
            output,
            problem_factory=_factory(toy_case),
            transition_callback=pause_at_next_checkpoint,
        ).start()
    first_pause = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
    assert first_pause["state"] == "interrupted"
    assert first_pause["pause"]["resumable"] is True
    assert first_pause["uncommitted_cuda_window"] is None
    first_control = json.loads(
        (output / "training_control.json").read_text(encoding="utf-8")
    )
    assert first_control["state"] == "acknowledged"
    assert first_control["request_id"] == first_pause["pause"]["request_id"]
    assert first_control["checkpoint_sha256"] == first_pause["session_checkpoint"][
        "sha256"
    ]

    with pytest.raises(TSHCALOTrainingPauseRequested):
        IndependentTSHCALOTrainingCampaign(
            plan,
            output,
            problem_factory=_factory(toy_case),
            transition_callback=pause_at_next_checkpoint,
        ).resume()
    second_pause = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
    assert second_pause["state"] == "interrupted"
    assert second_pause["pause"]["request_id"] != first_pause["pause"]["request_id"]
    assert second_pause["progress"]["committed_candidate_evaluations"] > first_pause[
        "progress"
    ]["committed_candidate_evaluations"]

    resumed = IndependentTSHCALOTrainingCampaign(
        plan,
        output,
        problem_factory=_factory(toy_case),
    ).resume()
    uninterrupted = IndependentTSHCALOTrainingCampaign(
        plan,
        tmp_path / "repeated-safe-pause-reference",
        problem_factory=_factory(toy_case),
    ).start()

    for resumed_state, expected_state in zip(_model_states(resumed), _model_states(uninterrupted)):
        assert resumed_state.keys() == expected_state.keys()
        for name, tensor in expected_state.items():
            torch.testing.assert_close(resumed_state[name], tensor, rtol=0.0, atol=0.0)
    completed = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
    assert completed["state"] == "completed"
    assert completed["progress"]["progress_percent"] == 100
    assert completed["progress"]["total_candidate_evaluations"] == 16
    events = [
        json.loads(line)
        for line in (output / "training_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events].count("campaign_paused") == 2
    assert [event["event_sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )


def test_pause_request_is_idempotent_and_bound_to_running_campaign(tmp_path, toy_case):
    plan = _plan()
    output = tmp_path / "idempotent-safe-pause"

    def request_twice(_event):
        first = request_tsh_calo_training_pause(output)
        second = request_tsh_calo_training_pause(output)
        assert second["request_id"] == first["request_id"]

    with pytest.raises(TSHCALOTrainingPauseRequested):
        IndependentTSHCALOTrainingCampaign(
            plan,
            output,
            problem_factory=_factory(toy_case),
            transition_callback=request_twice,
        ).start()
    with pytest.raises(RuntimeError, match="Only a running"):
        request_tsh_calo_training_pause(output)


def test_training_cli_emits_machine_readable_checkpoint_progress(capsys):
    event = {
        "schema_version": TSH_CALO_TRAINING_EVENT_SCHEMA,
        "event": "checkpoint_committed",
        "progress_percent": 25,
    }

    train_tsh_calo.emit_training_event(event)

    line = capsys.readouterr().out.strip()
    assert line.startswith(train_tsh_calo.TRAINING_EVENT_PREFIX)
    assert json.loads(line.removeprefix(train_tsh_calo.TRAINING_EVENT_PREFIX)) == event


def test_failed_campaign_retains_accounting_and_cannot_retry(tmp_path, toy_case):
    plan = _plan()

    def failing_factory(_identity):
        problem = ORPDProblem(toy_case)

        def failed_evaluator(_values, **_kwargs):
            raise RuntimeError("synthetic campaign evaluator failure")

        problem.evaluate_with_context = failed_evaluator
        return problem

    campaign = IndependentTSHCALOTrainingCampaign(
        plan,
        tmp_path / "failed",
        problem_factory=failing_factory,
    )
    with pytest.raises(RuntimeError, match="synthetic campaign evaluator failure"):
        campaign.start()

    status = json.loads((tmp_path / "failed" / "training_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["failure"]["environment_provenance"]["candidate_evaluations"] == 1
    assert status["failure"]["environment_provenance"]["accounting_complete"] is False
    assert not list((tmp_path / "failed").glob("*.candidate.pt"))
    with pytest.raises(RuntimeError, match="cannot retry"):
        IndependentTSHCALOTrainingCampaign(
            plan,
            tmp_path / "failed",
            problem_factory=failing_factory,
        ).resume()


def test_campaign_resume_rejects_checkpoint_status_mismatch(tmp_path, toy_case):
    plan = _plan()
    output = tmp_path / "tampered"

    def interrupt(_status):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        IndependentTSHCALOTrainingCampaign(
            plan,
            output,
            problem_factory=_factory(toy_case),
            transition_callback=interrupt,
        ).start()
    status_path = output / "training_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["session_checkpoint"]["sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        IndependentTSHCALOTrainingCampaign(
            plan,
            output,
            problem_factory=_factory(toy_case),
        ).resume()
    failed = json.loads(status_path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"


def test_campaign_refuses_resume_inside_uncommitted_cuda_window(tmp_path, toy_case):
    plan = _plan()
    output = tmp_path / "uncommitted-window"
    output.mkdir()
    campaign_module._write_json(output / "training_plan.json", plan.to_dict())
    campaign = IndependentTSHCALOTrainingCampaign(
        plan,
        output,
        problem_factory=_factory(toy_case),
    )
    campaign._write_status(
        {
            "state": "interrupted",
            "current_member_index": 0,
            "current_episode_index": 0,
            "session_checkpoint": None,
            "uncommitted_cuda_window": {
                "member_index": 0,
                "episode_index": 0,
                "starting_transition_count": 0,
                "maximum_transitions": 25,
                "target_candidate_evaluations": CUDA_DURABLE_EVALUATION_WINDOW,
            },
            "member_candidates": [],
            "failure": None,
        }
    )

    with pytest.raises(RuntimeError, match="uncommitted CUDA evaluation window"):
        campaign.resume()
    failed = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert failed["failure"]["resumable"] is False


def test_campaign_can_resume_safe_infrastructure_write_interruption(
    tmp_path, toy_case, monkeypatch
):
    plan = _plan()
    output = tmp_path / "infrastructure-interruption"
    original_write_json = campaign_module._write_json
    interrupted = False

    def interrupt_status_write(path, payload):
        nonlocal interrupted
        if (
            not interrupted
            and Path(path).name == "training_status.json"
            and payload.get("session_checkpoint") is not None
        ):
            interrupted = True
            raise PermissionError("synthetic Windows status replacement lock")
        return original_write_json(path, payload)

    monkeypatch.setattr(campaign_module, "_write_json", interrupt_status_write)
    with pytest.raises(PermissionError, match="synthetic Windows"):
        IndependentTSHCALOTrainingCampaign(
            plan,
            output,
            problem_factory=_factory(toy_case),
        ).start()
    status = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted"
    assert status["failure"]["resumable"] is True
    assert status["failure"]["category"] == "resumable_infrastructure_interruption"
    assert status["failure"]["environment_provenance"]["accounting_complete"] is True

    monkeypatch.setattr(campaign_module, "_write_json", original_write_json)
    result = IndependentTSHCALOTrainingCampaign(
        plan,
        output,
        problem_factory=_factory(toy_case),
    ).resume()
    assert result.ensemble_candidate.ensemble_size == 2


def test_campaign_rejects_leakage_reused_seeds_and_lifecycle_authority():
    plan = _plan()
    leaked_members = tuple(
        replace(
            member,
            episodes=(replace(member.episodes[0], case_identity="case118"),),
        )
        for member in plan.members
    )
    with pytest.raises(ValueError, match="Protected holdout"):
        replace(plan, development_cases=("case118",), members=leaked_members).validate()
    with pytest.raises(ValueError, match="member training seeds must be unique"):
        replace(
            plan,
            members=(plan.members[0], replace(plan.members[1], training_seed=101)),
        ).validate()
    with pytest.raises(ValueError, match="Change F"):
        replace(
            plan,
            feature_flags=replace(
                plan.feature_flags,
                population_schedule=True,
                allow_experimental_components=True,
            ),
        ).validate()
    cpu = replace(plan, requested_device="cpu")
    auto = replace(plan, requested_device="auto")
    assert cpu.scientific_design_hash() == auto.scientific_design_hash()
    assert cpu.execution_plan_sha256() != auto.execution_plan_sha256()

    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "experiments.experiment_runner",
        "PolicyRegistry",
        "activate(",
        "bind_to_experiment",
        "create_experiment",
        "TSHCALOInferenceController",
    ):
        assert forbidden not in source


def test_default_cuda_campaign_builds_no_fallback_accelerated_problem(
    tmp_path, toy_case, monkeypatch
):
    plan = replace(_plan(), requested_device="cuda", allow_cpu_fallback=False)
    captured = {}

    class FakeAcceleratedProblem:
        dimension = 3

        def __init__(self, case, **kwargs):
            captured["case"] = case
            captured.update(kwargs)

        def evaluate_with_context(self, _row, **_kwargs):
            raise AssertionError("construction test must not evaluate")

        def evaluate_population_with_context(self, _rows, **_kwargs):
            raise AssertionError("construction test must not evaluate")

    monkeypatch.setattr(campaign_module.CaseLoader, "load", lambda _identity: toy_case)
    monkeypatch.setattr(campaign_module, "AcceleratedORPDProblem", FakeAcceleratedProblem)
    campaign = IndependentTSHCALOTrainingCampaign(plan, tmp_path / "cuda-campaign")

    problem = campaign._build_problem("toy-development", device_hint="cuda:0")

    assert isinstance(problem, FakeAcceleratedProblem)
    assert captured["case"] is toy_case
    assert captured["device"] == "cuda:0"
    assert captured["batch_size"] == CUDA_DURABLE_EVALUATION_WINDOW
    assert captured["device_resident"] is True
    assert captured["cuda_resident_hot_loop"] is True
    assert captured["cuda_cpu_fallback_enabled"] is False


def test_explicit_training_command_requires_frozen_clean_source(tmp_path, monkeypatch):
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    loaded = train_tsh_calo.load_plan(plan_path)
    assert loaded.execution_plan_sha256() == plan.execution_plan_sha256()
    plan_with_writer_metadata = plan.to_dict()
    plan_with_writer_metadata["writer_metadata"] = {"product_version": "99.0.0"}
    plan_path.write_text(json.dumps(plan_with_writer_metadata), encoding="utf-8")
    extension_loaded = train_tsh_calo.load_plan(
        plan_path,
        compatible_extension=True,
    )
    assert extension_loaded.execution_plan_sha256() == plan.execution_plan_sha256()
    with pytest.raises(ValueError, match="fields are incomplete"):
        train_tsh_calo.load_plan(plan_path)

    monkeypatch.setattr(
        train_tsh_calo,
        "repository_state",
        lambda _root: (SOURCE_COMMIT, ""),
    )
    train_tsh_calo.validate_repository_for_plan(loaded, root=tmp_path)
    monkeypatch.setattr(
        train_tsh_calo,
        "repository_state",
        lambda _root: ("0" * 40, ""),
    )
    with pytest.raises(RuntimeError, match="source commit"):
        train_tsh_calo.validate_repository_for_plan(loaded, root=tmp_path)
    assert train_tsh_calo.validate_repository_for_plan(
        loaded,
        root=tmp_path,
        compatible_extension=True,
    ) == "0" * 40
    monkeypatch.setattr(
        train_tsh_calo,
        "repository_state",
        lambda _root: (SOURCE_COMMIT, " M tracked.py"),
    )
    with pytest.raises(RuntimeError, match="clean non-ignored source tree"):
        train_tsh_calo.validate_repository_for_plan(loaded, root=tmp_path)
    with pytest.raises(RuntimeError, match="clean non-ignored source tree"):
        train_tsh_calo.validate_repository_for_plan(
            loaded,
            root=tmp_path,
            compatible_extension=True,
        )


def test_readiness_preflights_the_same_training_resource_guard(monkeypatch):
    plan = _plan()
    captured = {}

    def preflight(config):
        captured["config"] = config
        return {
            "memory_estimate": {"estimated_working_set_bytes": 123},
            "memory_admission": {"selected_device": "cpu", "allowance_bytes": 456},
        }

    monkeypatch.setattr(train_tsh_calo, "preflight_tsh_calo_training_resources", preflight)

    result = train_tsh_calo.validate_training_resources(plan)

    assert captured["config"] == plan.training_config(plan.members[0])
    assert result["memory_estimate"]["estimated_working_set_bytes"] == 123
    assert result["memory_admission"]["selected_device"] == "cpu"


def test_extension_check_uses_authenticated_campaign_without_requiring_legacy_paths(
    tmp_path,
    monkeypatch,
    capsys,
):
    plan = _plan()
    monkeypatch.setattr(train_tsh_calo, "load_plan", lambda _path, **_kwargs: plan)
    monkeypatch.setattr(
        train_tsh_calo,
        "validate_repository_for_plan",
        lambda _plan, **_kwargs: "b" * 40,
    )
    monkeypatch.setattr(train_tsh_calo, "validate_training_resources", lambda _plan: {})
    monkeypatch.setattr(
        train_tsh_calo,
        "extension_plan_summary",
        lambda _plan, _output: {"authenticated": True},
    )

    exit_code = train_tsh_calo.main(
        [
            str(tmp_path / "legacy-plan.json"),
            "--check",
            "--extend",
            "--output",
            str(tmp_path / "completed-campaign"),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "extension_validated_not_started"
    assert summary["extension"] == {"authenticated": True}


def test_readiness_resource_failure_is_not_reported_as_validated(monkeypatch):
    plan = _plan()

    def reject(_config):
        raise MemoryError(
            "TSH-CALO training working set exceeds 80% of currently available CPU RAM"
        )

    monkeypatch.setattr(train_tsh_calo, "preflight_tsh_calo_training_resources", reject)

    with pytest.raises(MemoryError, match="currently available CPU RAM"):
        train_tsh_calo.validate_training_resources(plan)

    source = Path("calo_rpd_studio/scripts/train_tsh_calo.py").read_text(encoding="utf-8")
    assert "experiments.experiment_runner" not in source
    assert "PolicyRegistry" not in source
    assert "activate(" not in source


def test_explicit_training_command_requires_exact_training_eligible_freeze_report(tmp_path):
    plan = _plan()

    def record(path: str) -> dict:
        return {"path": path, "size_bytes": 1, "sha256": "1" * 64}

    complete_paths = sorted(
        set(freeze.INTERFACE_FILES.values())
        | set(freeze.DEPENDENCY_FILES)
        | set(freeze.CONTAINER_FILES)
        | set(freeze.EXCLUSION_FILES)
    )
    stable = {
        "schema": freeze.DEVELOPMENT_FREEZE_SCHEMA,
        "status": "development_freeze_candidate",
        "source_identity": {
            "source_commit": SOURCE_COMMIT,
            "tracked_source_clean": True,
            "durable_evidence_eligible": True,
        },
        "validator": {"path": "validator.ps1", "size_bytes": 1, "sha256": "2" * 64},
        "interfaces": {name: record(path) for name, path in freeze.INTERFACE_FILES.items()},
        "dependencies": [record(path) for path in freeze.DEPENDENCY_FILES],
        "containers": [record(path) for path in freeze.CONTAINER_FILES],
        "distribution_exclusions": [record(path) for path in freeze.EXCLUSION_FILES],
        "complete_source_manifest": {
            "schema": freeze.COMPLETE_SOURCE_MANIFEST_SCHEMA,
            "enumeration": "test_fixture",
            "source_status_sha256": "3" * 64,
            "source_status_clean": True,
            "file_count": len(complete_paths),
            "files": [record(path) for path in complete_paths],
        },
        "policy_inventory": {
            "inventory_sha256": "4" * 64,
            "removable_development_file_count": 0,
            "external_existing_artifact_count": 0,
            "database_row_counts": {},
            "release_scope_policy_count": 0,
            "old_policy_removal_executed": False,
            "old_policy_transition_pending": False,
        },
        "policy_scope": {
            "qualified_policy_in_development_freeze": False,
            "active_policy_in_development_freeze": False,
            "final_policy_in_development_freeze": False,
            "future_policy_initialization_policy_sha256": "",
        },
        "commands_not_executed": [
            "policy_training",
            "policy_evaluation",
            "policy_qualification",
            "policy_registration",
            "policy_activation",
            "policy_deletion",
            "protected_case_campaign",
            "release_publication",
        ],
        "post_transition_training_eligible": True,
    }
    report = {
        **stable,
        "created_at": "2026-08-12T00:00:00+00:00",
        "development_freeze_payload_sha256": freeze._canonical_sha256(stable),
    }
    plan = replace(
        plan,
        development_freeze_sha256=report["development_freeze_payload_sha256"],
    )
    path = tmp_path / "accepted-freeze.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    contract = freeze.development_source_contract(report)
    acceptance_stable = {
        "schema": acceptance.ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "decision_id": "phase4-test-acceptance",
        "validation_run_id": "phase4-test-run",
        "validated_source_commit": SOURCE_COMMIT,
        "validated_source_dirty": False,
        "claim_boundary": acceptance.ACCEPTANCE_CLAIM_BOUNDARY,
        "validation_summary_sha256": "5" * 64,
        "validation_log_manifest_sha256": "6" * 64,
        "development_freeze_candidate_sha256": report["development_freeze_payload_sha256"],
        "development_source_contract_sha256": contract["files_sha256"],
        "development_source_file_count": contract["file_count"],
        "validator_sha256": "7" * 64,
    }
    acceptance_report = {
        **acceptance_stable,
        "created_at": "2026-08-12T00:00:00+00:00",
        "acceptance_receipt_sha256": acceptance._canonical_sha256(acceptance_stable),
    }
    acceptance_path = tmp_path / "phase4-acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance_report), encoding="utf-8")
    plan = replace(
        plan,
        phase4_acceptance_sha256=acceptance_report["acceptance_receipt_sha256"],
    )

    validated = train_tsh_calo.validate_development_freeze_for_plan(
        plan,
        path,
        acceptance_path,
    )
    assert validated["development_freeze_payload_sha256"] == plan.development_freeze_sha256

    report["post_transition_training_eligible"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        train_tsh_calo.validate_development_freeze_for_plan(plan, path, acceptance_path)
