from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from calo_rpd_studio.ai.model_io import checkpoint_sha256
from calo_rpd_studio.algorithms.calo.tsh_calo_automatic_qualification import (
    AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST,
    AutomaticQualificationRejected,
    AutomaticQualificationWorkspace,
    accepted_component_references,
    automatic_qualification_workload,
    automatic_qualification_workflow_payload,
    build_automatic_component_ablation_plan,
    build_automatic_formal_qualification_plan,
    freeze_plan,
    frozen_qualification_restart_design,
    frozen_qualification_restart_design_sha256,
    prepare_automatic_source_snapshot,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_component_ablation import (
    TSH_CALO_COMPONENT_ABLATION_CAMPAIGN_SCHEMA,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    TSH_CALO_COMPONENT_EVIDENCE_SCHEMA,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import TSHCALOCandidateArtifact
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import (
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_STATE_SCHEMA,
    TSH_CALO_TRAINING_ENVIRONMENT,
)
from calo_rpd_studio.statistics.paired import (
    DEFAULT_OBJECTIVE_SCALE_FLOOR,
    PAIRED_ANALYSIS_SCHEMA_VERSION,
    RELATIVE_IMPROVEMENT_VERSION,
)


_SHA = "a" * 64
_COMMIT = "b" * 40


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return checkpoint_sha256(path)


def _accepted_component_campaign(tmp_path: Path, *, rejected: str = ""):
    candidate = tmp_path / "ensemble.candidate.pt"
    candidate.write_bytes(b"candidate")
    plan = build_automatic_component_ablation_plan(
        candidate_path=candidate,
        candidate_sha256=_SHA,
        source_commit=_COMMIT,
    )
    output = tmp_path / "component-ablation-evidence"
    references = {}
    for component in "ABCDE":
        accepted = component != rejected
        evidence = {
            "schema_version": TSH_CALO_COMPONENT_EVIDENCE_SCHEMA,
            "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
            "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
            "objective_scale_floor": DEFAULT_OBJECTIVE_SCALE_FLOOR,
            "component": component,
            "accepted": accepted,
            "source_policy_sha256": _SHA,
            "source_commit": _COMMIT,
            "source_tracked_clean": True,
            "campaign_id": plan.campaign_id,
            "component_ablation_plan_sha256": plan.execution_plan_sha256(),
            "scientific_design_sha256": plan.scientific_design_sha256(),
            "seed_manifest_sha256": plan.seed_manifest_sha256(),
            "development_cases": list(plan.development_cases),
            "protected_cases_opened": False,
            "analysis": [{"direct": True}],
            "authority_boundary": "component_ablation_only_no_qualification_or_lifecycle",
        }
        path = output / f"component-{component}.evidence.json"
        references[component] = {
            "path": str(path),
            "sha256": _write_json(path, evidence),
            "accepted": accepted,
        }
    _write_json(
        output / "campaign_evidence.json",
        {
            "schema_version": TSH_CALO_COMPONENT_ABLATION_CAMPAIGN_SCHEMA,
            "source_commit": _COMMIT,
            "source_policy_sha256": _SHA,
            "execution_plan_sha256": plan.execution_plan_sha256(),
            "component_evidence": references,
            "all_A_E_accepted": not rejected,
        },
    )
    return candidate, plan, output


def _candidate_artifact(candidate: Path) -> TSHCALOCandidateArtifact:
    members = [
        {
            "source_candidate_sha256": character * 64,
            "training_provenance": {"training_design_sha256": str(index) * 64},
        }
        for index, character in ((1, "c"), (2, "d"))
    ]
    return TSHCALOCandidateArtifact(
        path=str(candidate),
        sha256=_SHA,
        algorithm_id=TSH_CALO_ALGORITHM_ID,
        algorithm_version=TSH_CALO_ALGORITHM_VERSION,
        state_schema_version=TSH_CALO_STATE_SCHEMA,
        action_schema_version=TSH_CALO_ACTION_SCHEMA,
        training_environment_version=TSH_CALO_TRAINING_ENVIRONMENT,
        artifact_kind="ensemble_policy",
        ensemble_size=2,
        feature_flags={"population_schedule": False},
        training_provenance={
            "source_kind": "independent_policy_training_ensemble",
            "members": members,
        },
    )


def test_one_action_plans_are_deterministic_finite_and_source_bound(tmp_path):
    candidate = tmp_path / "ensemble.candidate.pt"
    candidate.write_bytes(b"candidate")
    formal = build_automatic_formal_qualification_plan(
        candidate_path=candidate,
        candidate_sha256=_SHA,
        source_commit=_COMMIT,
        candidate_artifact=_candidate_artifact(candidate),
    )

    assert formal.development_cases == ("case30", "case57")
    assert formal.runs == 30
    assert formal.population_size == 20
    assert formal.max_evaluations == 10_000
    assert formal.mode == "formal"
    assert formal.component_evidence == {}
    assert formal.candidate_contract["candidate_sha256"] == _SHA
    workflow = automatic_qualification_workflow_payload(qualification_plan=formal)
    assert workflow["source_commit"] == _COMMIT
    assert workflow["formal_qualification_plan"] == formal.to_dict()
    assert workflow["candidate_contract"] == formal.candidate_contract
    assert workflow["resume_count_limit"] is None
    assert workflow["automatic_activation"] is False
    assert workflow["automatic_suitability_decision"] is False
    assert workflow["schema_version"] == ("tsh-calo-one-action-feasibility-v1-transactional-cells")
    assert automatic_qualification_workload() == {
        "cases": 2,
        "runs_per_case": 30,
        "evaluations_per_cell": 10_000,
        "qualification_cells": 120,
        "total_cells": 120,
    }


def test_corrected_source_restart_changes_only_provenance_not_frozen_design(tmp_path):
    candidate = tmp_path / "ensemble.candidate.pt"
    candidate.write_bytes(b"candidate")
    artifact = _candidate_artifact(candidate)
    retained = build_automatic_formal_qualification_plan(
        candidate_path=candidate,
        candidate_sha256=_SHA,
        source_commit=_COMMIT,
        candidate_artifact=artifact,
    )
    corrected = build_automatic_formal_qualification_plan(
        candidate_path=candidate,
        candidate_sha256=_SHA,
        source_commit="c" * 40,
        candidate_artifact=artifact,
    )

    assert retained.qualification_run_id != corrected.qualification_run_id
    assert retained.source_commit != corrected.source_commit
    assert retained.execution_plan_sha256() != corrected.execution_plan_sha256()
    assert frozen_qualification_restart_design(retained) == (
        frozen_qualification_restart_design(corrected)
    )
    assert frozen_qualification_restart_design_sha256(retained) == (
        frozen_qualification_restart_design_sha256(corrected)
    )

    same_source_retry = build_automatic_formal_qualification_plan(
        candidate_path=candidate,
        candidate_sha256=_SHA,
        source_commit="c" * 40,
        candidate_artifact=artifact,
        restart_ordinal=1,
    )
    assert same_source_retry.qualification_run_id.endswith("-restart-001")
    assert frozen_qualification_restart_design_sha256(same_source_retry) == (
        frozen_qualification_restart_design_sha256(corrected)
    )


def test_one_action_rejects_completed_ablation_when_any_A_E_gate_fails(tmp_path):
    _candidate, plan, output = _accepted_component_campaign(tmp_path, rejected="C")

    with pytest.raises(AutomaticQualificationRejected, match="did not accept C"):
        accepted_component_references(output, plan=plan)


def test_workspace_identity_and_frozen_plan_refuse_silent_changes(tmp_path):
    workspace = AutomaticQualificationWorkspace.create(
        tmp_path, candidate_sha256=_SHA, source_commit=_COMMIT
    )
    same = AutomaticQualificationWorkspace.create(
        tmp_path, candidate_sha256=_SHA, source_commit=_COMMIT
    )
    retry = AutomaticQualificationWorkspace.create(
        tmp_path,
        candidate_sha256=_SHA,
        source_commit=_COMMIT,
        restart_ordinal=1,
    )
    assert workspace == same
    assert retry.root != workspace.root
    assert retry.root.name.endswith("-restart-001")
    assert workspace.workflow_plan.name == "automatic_qualification_workflow.json"
    assert _SHA[:16] in workspace.root.name
    assert _COMMIT[:12] in workspace.root.name

    first_hash = freeze_plan(workspace.qualification_plan, {"frozen": 1})
    assert freeze_plan(workspace.qualification_plan, {"frozen": 1}) == first_hash
    with pytest.raises(ValueError, match="different content"):
        freeze_plan(workspace.qualification_plan, {"frozen": 2})


def test_dirty_worktree_is_frozen_as_a_separate_clean_deterministic_commit(tmp_path):
    repository = tmp_path / "live-source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True
    )
    tracked = repository / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repository, check=True)
    tracked.write_text("value = 2\n", encoding="utf-8")
    (repository / "new_module.py").write_text("new_value = 3\n", encoding="utf-8")
    status_before = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    first = prepare_automatic_source_snapshot(repository, tmp_path / "snapshots")
    second = prepare_automatic_source_snapshot(repository, tmp_path / "snapshots")

    assert first == second
    assert first.file_count == 2
    assert (first.root / "tracked.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (first.root / "new_module.py").read_text(encoding="utf-8") == "new_value = 3\n"
    snapshot_manifest = json.loads(
        (first.root / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST).read_text(encoding="utf-8")
    )
    assert snapshot_manifest["worktree_sha256"] == first.worktree_sha256
    assert checkpoint_sha256(first.root / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST) == (
        first.manifest_sha256
    )
    assert len(snapshot_manifest["files"]) == first.file_count
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=first.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    assert (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=first.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == first.source_commit
    )
    status_after = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status_after == status_before
    assert " M tracked.py" in status_after
    assert "?? new_module.py" in status_after
