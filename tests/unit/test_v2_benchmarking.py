from __future__ import annotations

import json
from pathlib import Path

import pytest

from calo_rpd_studio.algorithms.registry import primary_algorithm_names
from calo_rpd_studio.benchmarking.campaign import BenchmarkCampaignConfig, build_campaign
from calo_rpd_studio.benchmarking.freeze import create_freeze_manifest, verify_freeze_manifest
from calo_rpd_studio.benchmarking.suite import standard_benchmark_suite
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader


def test_v2_standard_suite_contains_30_57_118_and_larger_300_case():
    suite = standard_benchmark_suite()
    assert suite.cases == ("case30", "case57", "case118", "case300")
    assert "load_mean_risk" in {study.key for study in suite.studies}
    assert "renewable_cvar" in {study.key for study in suite.studies}
    assert "branch_worst_case" in {study.key for study in suite.studies}


def test_case300_is_exposed_by_case_loader():
    assert "case300" in CaseLoader.available_cases()


def test_final_campaign_requires_30_to_50_runs():
    campaign = BenchmarkCampaignConfig(cases=("case30",), study_keys=("deterministic",), runs=29)
    with pytest.raises(ValueError, match="30–50"):
        campaign.validate(verify_freeze=False)
    campaign.runs = 51
    with pytest.raises(ValueError, match="30–50"):
        campaign.validate(verify_freeze=False)


def test_final_campaign_requires_exactly_twenty_primary_algorithms():
    campaign = BenchmarkCampaignConfig(
        cases=("case30",),
        study_keys=("deterministic",),
        runs=30,
        algorithms=("CALO", "TLBO"),
    )
    with pytest.raises(ValueError, match="exactly the 20 primary algorithms"):
        campaign.validate(verify_freeze=False)
    assert len(primary_algorithm_names()) == 20


def test_campaign_plan_uses_all_algorithms_and_equal_job_count():
    campaign = BenchmarkCampaignConfig(
        cases=("case30", "case57"),
        study_keys=("deterministic", "mixed"),
        runs=30,
        max_evaluations=1234,
    )
    tasks = build_campaign(campaign, base_config=ExperimentConfig(), verify_freeze=False)
    assert len(tasks) == 4
    assert all(task.planned_jobs == 20 * 30 for task in tasks)
    assert all(task.config.algorithms == list(primary_algorithm_names()) for task in tasks)
    assert all(task.config.budget.max_evaluations == 1234 for task in tasks)
    assert all(task.config.runs == 30 for task in tasks)


def test_freeze_manifest_detects_frozen_source_change(tmp_path):
    root = tmp_path / "repo"
    (root / "calo_rpd_studio/algorithms/calo").mkdir(parents=True)
    source = root / "calo_rpd_studio/algorithms/calo/optimizer.py"
    source.write_text("frozen = 1\n", encoding="utf-8")
    manifest = create_freeze_manifest(
        tmp_path / "freeze.json",
        project_root=root,
        relative_paths=("calo_rpd_studio/algorithms/calo/optimizer.py",),
    )
    assert verify_freeze_manifest(manifest, project_root=root).passed
    source.write_text("frozen = 2\n", encoding="utf-8")
    verification = verify_freeze_manifest(manifest, project_root=root)
    assert not verification.passed
    assert verification.changed_files == ("calo_rpd_studio/algorithms/calo/optimizer.py",)


def test_freeze_manifest_captures_scope(tmp_path):
    root = tmp_path / "repo"
    path = root / "file.txt"
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    manifest = create_freeze_manifest(
        tmp_path / "freeze.json",
        project_root=root,
        relative_paths=("file.txt",),
    )
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    scope = payload["frozen_scope"]
    # The freeze manifest records both completed capabilities and explicit scientific
    # limitations. False values are deliberate disclosures, not missing freeze coverage.
    assert scope["mathematical_equations"] is True
    assert scope["policy_lineage_latest_vs_best"] is True
    assert scope["exact_calo_run_state_continuation"] is True
    assert scope["calo_control_fully_device_resident"] is False
    assert scope["baseline_exact_optimizer_state_continuation"] is False
    assert scope["automatic_periodic_policy_qualification"] is False
    assert payload["benchmark_rule"].startswith("No CALO tuning")
