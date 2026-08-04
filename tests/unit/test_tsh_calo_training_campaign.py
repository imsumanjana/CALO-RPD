"""Frozen-plan and exact-resume tests for independent TSH-CALO member campaigns."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    IndependentTSHCALOTrainingCampaign,
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingEpisodePlan,
    TSHCALOTrainingMemberPlan,
)
import calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign as campaign_module
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.scripts import train_tsh_calo


SOURCE_COMMIT = "a4329c3a39ae2646da134c4d3219b957c7c3c2bc"


def _plan() -> TSHCALOTrainingCampaignPlan:
    return TSHCALOTrainingCampaignPlan(
        campaign_id="fresh-campaign-001",
        source_commit=SOURCE_COMMIT,
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
    restored = TSHCALOTrainingCampaignPlan.from_dict(plan.to_dict())
    assert restored.execution_plan_sha256() == plan.execution_plan_sha256()
    assert restored.seed_manifest_sha256() == plan.seed_manifest_sha256()

    result = IndependentTSHCALOTrainingCampaign(
        plan,
        tmp_path / "campaign",
        problem_factory=_factory(toy_case),
    ).start()

    assert len(result.member_candidates) == 2
    assert result.ensemble_candidate.artifact_kind == "ensemble_policy"
    assert result.ensemble_candidate.ensemble_size == 2
    for index, candidate in enumerate(result.member_candidates):
        provenance = candidate.training_provenance
        assert provenance["training_run_id"] == f"fresh-campaign-001:member-00{index + 1}"
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
    with pytest.raises(FileExistsError, match="explicit resume"):
        IndependentTSHCALOTrainingCampaign(
            plan,
            tmp_path / "campaign",
            problem_factory=_factory(toy_case),
        ).start()


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


def test_explicit_training_command_requires_frozen_clean_source(tmp_path, monkeypatch):
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    loaded = train_tsh_calo.load_plan(plan_path)
    assert loaded.execution_plan_sha256() == plan.execution_plan_sha256()

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
    monkeypatch.setattr(
        train_tsh_calo,
        "repository_state",
        lambda _root: (SOURCE_COMMIT, " M tracked.py"),
    )
    with pytest.raises(RuntimeError, match="clean tracked working tree"):
        train_tsh_calo.validate_repository_for_plan(loaded, root=tmp_path)

    source = Path("calo_rpd_studio/scripts/train_tsh_calo.py").read_text(encoding="utf-8")
    assert "experiments.experiment_runner" not in source
    assert "PolicyRegistry" not in source
    assert "activate(" not in source
