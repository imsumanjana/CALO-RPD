from dataclasses import replace

import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_feasibility_assessment import (
    build_tsh_calo_feasibility_assessment,
    validate_tsh_calo_feasibility_assessment,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingEpisodePlan,
    TSHCALOTrainingHyperparameters,
    TSHCALOTrainingMemberPlan,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_influence import (
    build_training_parameter_influence,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)


def _case(name: str, feasibility: float, first_efficiency: float) -> dict:
    return {
        "case": name,
        "n_pairs": 2,
        "candidate_feasible_probability": feasibility,
        "baseline_feasible_probability": 0.5,
        "candidate_first_feasible_reached_probability": feasibility,
        "candidate_first_feasible_efficiency": first_efficiency,
        "candidate_first_feasible_evaluation_median": 2500.0 if feasibility else None,
        "candidate_independent_validation_probability": 1.0,
        "paired_feasible_objective_fraction": feasibility,
    }


def _ratings(score_a: float, score_b: float) -> dict:
    return build_tsh_calo_feasibility_assessment(
        cases=[_case("case30", score_a, score_a), _case("case57", score_b, score_b)],
        expected_case_order=("case30", "case57"),
    )


def _training_plan(*, campaign: str, learning_rate: float) -> TSHCALOTrainingCampaignPlan:
    episodes = (
        TSHCALOTrainingEpisodePlan(f"{campaign}:case30", "case30", 11),
        TSHCALOTrainingEpisodePlan(f"{campaign}:case57", "case57", 12),
    )
    members = (
        TSHCALOTrainingMemberPlan("member-1", 101, episodes),
        TSHCALOTrainingMemberPlan(
            "member-2",
            202,
            tuple(
                replace(episode, session_id=episode.session_id.replace(campaign, campaign + ":m2"))
                for episode in episodes
            ),
        ),
    )
    return TSHCALOTrainingCampaignPlan(
        campaign_id=campaign,
        source_commit="a" * 40,
        development_freeze_commit="",
        development_freeze_sha256="",
        phase4_acceptance_sha256="",
        development_cases=("case30", "case57"),
        members=members,
        resource_envelope=TSHCALOTrainingResourceEnvelope(32, 20, 60, 80, 40, 4),
        population_size=20,
        max_evaluations=10_000,
        training=replace(TSHCALOTrainingHyperparameters(), learning_rate=learning_rate),
        requested_device="cpu",
    )


def test_feasibility_assessment_is_measurement_not_binary_qualification():
    assessment = _ratings(0.5, 1.0)

    assert assessment["overall_feasibility_score"] == pytest.approx(75.0)
    assert assessment["automated_suitability_decision"] is None
    assert assessment["decision_authority"] == "scientist_only"
    assert [item["ratings"]["full_feasibility"] for item in assessment["case_ratings"]] == [
        50.0,
        100.0,
    ]
    validate_tsh_calo_feasibility_assessment(assessment)


def test_feasibility_assessment_checksum_rejects_posthoc_score_change():
    assessment = _ratings(0.5, 1.0)
    assessment["overall_feasibility_score"] = 99.0

    with pytest.raises(ValueError, match="checksum"):
        validate_tsh_calo_feasibility_assessment(assessment)


def test_training_influence_requires_comparable_campaigns_and_never_changes_training():
    plans = [
        _training_plan(campaign="campaign-low", learning_rate=1e-4),
        _training_plan(campaign="campaign-mid", learning_rate=3e-4),
        _training_plan(campaign="campaign-high", learning_rate=5e-4),
    ]
    cohort = [
        {"candidate_sha256": "1" * 64, "plan": plans[0], "ratings": _ratings(0.4, 0.6)},
        {"candidate_sha256": "2" * 64, "plan": plans[1], "ratings": _ratings(0.6, 0.8)},
        {"candidate_sha256": "3" * 64, "plan": plans[2], "ratings": _ratings(0.8, 1.0)},
    ]

    report = build_training_parameter_influence(
        selected_candidate_sha256="2" * 64,
        selected_plan=plans[1],
        cohort=cohort,
    )

    learning_rate = next(
        item for item in report["parameters"] if item["parameter"] == "training.learning_rate"
    )
    assert learning_rate["selected_value"] == pytest.approx(3e-4)
    assert learning_rate["evidence_classification"] == "observational_association"
    assert learning_rate["affected_rating"] == "overall_full_feasibility"
    assert {item["rating"] for item in learning_rate["rating_effects"]} == {
        "overall_full_feasibility",
        "first_feasible_reached",
        "first_feasible_efficiency",
        "independent_validation",
        "paired_feasible_objective_coverage",
    }
    assert learning_rate["direction"] == "positive"
    assert report["automatic_training_or_parameter_change"] is False
    assert plans[1].training.learning_rate == pytest.approx(3e-4)


def test_single_model_influence_is_truthfully_insufficient():
    plan = _training_plan(campaign="campaign-one", learning_rate=3e-4)
    report = build_training_parameter_influence(
        selected_candidate_sha256="4" * 64,
        selected_plan=plan,
        cohort=[{"candidate_sha256": "4" * 64, "plan": plan, "ratings": _ratings(1.0, 1.0)}],
    )

    assert report["evidence_classification"] == "insufficient_comparative_evidence"
    assert all(item["standardized_effect"] is None for item in report["parameters"])


def test_influence_excludes_a_rating_payload_that_no_longer_matches_its_checksum():
    plan = _training_plan(campaign="campaign-tampered", learning_rate=3e-4)
    ratings = _ratings(1.0, 1.0)
    ratings["overall_feasibility_score"] = 12.0

    report = build_training_parameter_influence(
        selected_candidate_sha256="5" * 64,
        selected_plan=plan,
        cohort=[{"candidate_sha256": "5" * 64, "plan": plan, "ratings": ratings}],
    )

    assert report["compatible_campaign_count"] == 0
    assert report["evidence_classification"] == "insufficient_comparative_evidence"
    assert "checksum" in report["excluded_campaigns"][0]["reason"]


def test_duplicate_candidate_cannot_inflate_the_comparative_influence_cohort():
    plan = _training_plan(campaign="campaign-one", learning_rate=3e-4)
    row = {"candidate_sha256": "6" * 64, "plan": plan, "ratings": _ratings(1.0, 1.0)}

    report = build_training_parameter_influence(
        selected_candidate_sha256="6" * 64,
        selected_plan=plan,
        cohort=[row, row, row],
    )

    assert report["compatible_campaign_count"] == 1
    assert len(report["excluded_campaigns"]) == 2
    assert all("duplicated" in item["reason"] for item in report["excluded_campaigns"])
