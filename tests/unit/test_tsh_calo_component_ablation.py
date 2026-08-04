"""Development-only TSH-CALO A–E ablation plan and capability invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_component_ablation import (
    TSHCALOComponentAblationCampaign,
    TSHCALOComponentAblationPlan,
    _PROFILES,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_inference import (
    ComponentAblationAuthority,
    TSHCALOComponentAblationProfile,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan(**changes) -> TSHCALOComponentAblationPlan:
    values = {
        "campaign_id": "tsh-calo-v4-a-e-ablation-001",
        "source_commit": "1" * 40,
        "candidate_path": "candidate.pt",
        "candidate_sha256": _sha("candidate"),
        "development_cases": ("case30", "case57"),
        "runs": 30,
        "master_seed": 1907,
        "population_size": 4,
        "max_evaluations": 8,
        "calibration_samples_per_case": 4,
        "calibration_population_size": 4,
        "bootstrap_resamples": 1_000,
    }
    values.update(changes)
    return TSHCALOComponentAblationPlan(**values)


def test_ablation_plan_is_frozen_powered_and_development_only():
    plan = _plan()
    plan.validate()

    assert plan.execution_plan_sha256() != plan.scientific_design_sha256()
    assert plan.seed_manifest_sha256() == plan.seed_manifest_sha256()
    assert len(plan.seed_manifest()["paired_runs"]) == 30
    with pytest.raises(ValueError, match="at least 30"):
        _plan(runs=29).validate()
    with pytest.raises(ValueError, match="Protected holdouts"):
        _plan(development_cases=("case30", "case118")).validate()


def test_ablation_profiles_cover_A_E_and_cannot_smuggle_experimental_F():
    assert set(_PROFILES) == {
        "graph_context_only",
        "hierarchical_actions_only",
        "graph_plus_hierarchy",
        "uncertainty_shield",
        "contextual_bandit_residual",
        "full_approved_A_E",
    }
    for profile in _PROFILES.values():
        profile.validate()

    with pytest.raises(ValueError, match="Unknown"):
        TSHCALOComponentAblationProfile(
            "population_schedule", True, True, True, True, False
        ).validate()
    with pytest.raises(ValueError, match="preceding uncertainty"):
        TSHCALOComponentAblationProfile(
            "contextual_bandit_residual", True, True, False, True, False
        ).validate()


def test_component_authority_rejects_protected_identity_and_invalid_policy():
    profile = _PROFILES["graph_plus_hierarchy"]
    authority = ComponentAblationAuthority(
        "campaign",
        _sha("plan"),
        _sha("policy"),
        "1" * 40,
        ("case30",),
        _sha("calibration"),
        profile,
    )
    authority.validate()

    with pytest.raises(ValueError, match="development-only"):
        ComponentAblationAuthority(
            "campaign",
            _sha("plan"),
            _sha("policy"),
            "1" * 40,
            ("case118",),
            _sha("calibration"),
            profile,
        ).validate()


def test_component_campaign_has_no_lifecycle_or_training_authority():
    path = Path(
        __import__(
            "calo_rpd_studio.algorithms.calo.tsh_calo_component_ablation",
            fromlist=["sentinel"],
        ).__file__
    )
    source = path.read_text(encoding="utf-8")

    assert "PolicyRegistry" not in source
    assert ".activate(" not in source
    assert "add_policy_qualification" not in source
    assert "IndependentTSHCALOTrainingCampaign" not in source
    assert TSHCALOComponentAblationCampaign.PLAN_FILE == "component_ablation_plan.json"
