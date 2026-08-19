"""Deterministic learning-health guard regressions for independent TSH-CALO training."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest

import calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign as campaign_module
from calo_rpd_studio.algorithms.calo.tsh_calo_generalization_guard import (
    TSH_CALO_GENERALIZATION_EVIDENCE_SCHEMA,
    TSHCALOGeneralizationGuardConfig,
    build_generalization_guard_provenance,
    compare_generalization_evidence,
    generalization_guard_design_sha256,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    IndependentTrainingProvenance,
    _training_provenance_payload,
    inspect_tsh_calo_candidate,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training import (
    IndependentTSHCALOTrainer,
    TSHCALOTrainingConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    IndependentTSHCALOTrainingCampaign,
    TSHCALOEnvironmentHyperparameters,
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingEpisodePlan,
    TSHCALOTrainingMemberPlan,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_environment import (
    TSHCALOTrainingEnvironmentConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_TRAINING_ENVIRONMENT
from calo_rpd_studio.algorithms.calo.tsh_calo_training_extension import (
    IndependentTSHCALOTrainingExtension,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_receipt import (
    build_tsh_calo_training_episode_receipt,
    canonical_reward_sequence_sha256,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
from calo_rpd_studio.orpd.problem import ORPDProblem


SOURCE_COMMIT = "a4329c3a39ae2646da134c4d3219b957c7c3c2bc"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _guard() -> TSHCALOGeneralizationGuardConfig:
    return TSHCALOGeneralizationGuardConfig(
        validation_batches_per_case=2,
        validation_seed=900_001,
        final_audit_seed_offset=100_003,
        degradation_patience=2,
        feasible_ratio_tolerance=0.10,
        reward_component_tolerance=0.10,
        minimum_learning_gain=0.02,
        minimum_acceptable_feasible_ratio=0.80,
    )


def _environment_template() -> dict:
    return asdict(TSHCALOEnvironmentHyperparameters())


def _guard_design(guard: TSHCALOGeneralizationGuardConfig) -> str:
    return generalization_guard_design_sha256(
        guard,
        development_cases=("toy-development",),
        population_size=4,
        environment_template=_environment_template(),
    )


def _plan(*, guarded: bool = True) -> TSHCALOTrainingCampaignPlan:
    return TSHCALOTrainingCampaignPlan(
        campaign_id="guard-campaign-001",
        source_commit=SOURCE_COMMIT,
        development_freeze_commit="",
        development_freeze_sha256="",
        phase4_acceptance_sha256="",
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
        generalization_guard=_guard() if guarded else None,
    )


def _factory(toy_case):
    return lambda _identity: ORPDProblem(toy_case)


def _evidence_payload(
    *,
    level: float,
    training_design_sha256: str,
    guard: TSHCALOGeneralizationGuardConfig,
    development_cases=("toy-development",),
    population_size=4,
    final=False,
    observation_index=0,
    ppo_update_steps=0,
    guard_design_sha256=None,
    problem_identities=None,
    environment_template=None,
):
    seeds = guard.seed_block(tuple(development_cases), final=final)
    guard_design_sha256 = guard_design_sha256 or _guard_design(guard)
    problem_identities = dict(problem_identities or {})
    environment_template = dict(environment_template or _environment_template())
    per_case = guard.validation_evaluations_per_case(population_size)
    rows = []
    for case_identity, seed in zip(development_cases, seeds, strict=True):
        case_checksum, problem_fingerprint = problem_identities.get(
            case_identity, (_sha("case"), _sha("problem"))
        )
        rows.append(
            {
                "case_identity": case_identity,
                "seed": seed,
                "candidate_evaluations": per_case,
                "scenario_power_flow_calls": per_case,
                "case_checksum": case_checksum,
                "problem_fingerprint": problem_fingerprint,
                "environment_design_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "schema_version": TSH_CALO_TRAINING_ENVIRONMENT,
                            "training_design_sha256": training_design_sha256,
                            "environment": asdict(
                                TSHCALOTrainingEnvironmentConfig(
                                    **{
                                        **environment_template,
                                        "case_identity": case_identity,
                                        "population_size": population_size,
                                        "max_evaluations": per_case,
                                        "seed": seed,
                                        "environment_deterministic": True,
                                    }
                                )
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "transition_count": max(1, guard.validation_batches_per_case - 1),
                "final_feasible_ratio": float(level),
                "best_violation": float(max(0.0, 1.0 - level)),
                "best_feasible_objective": 1.0 if level >= 0.8 else None,
                "mean_canonical_reward": float(level),
                "mean_objective_improvement": float(level),
                "mean_constraint_improvement": float(level),
                "mean_feasible_ratio_improvement": float(level),
                "mean_diversity_recovery": 0.0,
            }
        )
    return {
        "schema_version": TSH_CALO_GENERALIZATION_EVIDENCE_SCHEMA,
        "evidence_kind": "final_audit" if final else "monitor",
        "observation_index": int(observation_index),
        "evaluation_backend": "deterministic_test_fixture",
        "guard_config_sha256": guard.scientific_design_hash(),
        "guard_design_sha256": guard_design_sha256,
        "training_design_sha256": training_design_sha256,
        "development_cases": list(development_cases),
        "validation_seed_block": list(seeds),
        "case_rows": rows,
        "candidate_evaluations": per_case * len(rows),
        "scenario_power_flow_calls": per_case * len(rows),
        "mean_final_feasible_ratio": float(level),
        "mean_canonical_reward": float(level),
        "mean_objective_improvement": float(level),
        "mean_constraint_improvement": float(level),
        "mean_feasible_ratio_improvement": float(level),
        "ppo_update_steps_observed": int(ppo_update_steps),
    }


def _fake_bundle(
    trainer,
    training_config,
    guard_config,
    *,
    development_cases,
    population_size,
    environment_template,
    problem_factory,
    final,
    observation_index,
    evaluation_backend,
):
    del evaluation_backend
    level = 0.20 if trainer.update_steps == 0 else (0.82 if trainer.update_steps == 1 else 0.95)
    identities = {}
    for case_identity in development_cases:
        problem = problem_factory(case_identity)
        identities[case_identity] = (
            str(problem.case.checksum()).lower(),
            scientific_problem_fingerprint(problem),
        )
    design_sha = generalization_guard_design_sha256(
        guard_config,
        development_cases=tuple(development_cases),
        population_size=population_size,
        environment_template=environment_template,
    )
    return _evidence_payload(
        level=level,
        training_design_sha256=training_config.scientific_design_hash(),
        guard=guard_config,
        development_cases=development_cases,
        population_size=population_size,
        final=final,
        observation_index=observation_index,
        ppo_update_steps=trainer.update_steps,
        guard_design_sha256=design_sha,
        problem_identities=identities,
        environment_template=environment_template,
    )


def _training_config(guard: TSHCALOGeneralizationGuardConfig) -> TSHCALOTrainingConfig:
    return TSHCALOTrainingConfig(
        training_run_id="guard-export-member",
        development_cases=("toy-development",),
        seed_manifest_sha256=_sha("guard-seed-manifest"),
        resource_envelope=TSHCALOTrainingResourceEnvelope(1, 4, 8, 16, 16, 4),
        seed=101,
        hidden_dim=16,
        graph_steps=1,
        ppo_epochs=1,
        device="cpu",
        generalization_guard_sha256=_guard_design(guard),
    )


def _receipt(config: TSHCALOTrainingConfig, *, seed=201):
    return build_tsh_calo_training_episode_receipt(
        session_id="guard-export-session",
        training_run_id=config.training_run_id,
        training_design_sha256=config.scientific_design_hash(),
        session_design_sha256=_sha("session"),
        environment_design_sha256=_sha("environment"),
        case_identity="toy-development",
        case_checksum=_sha("case"),
        problem_fingerprint=_sha("problem"),
        seed=seed,
        deterministic_policy=False,
        candidate_evaluations=8,
        scenario_power_flow_calls=8,
        canonical_transition_count=1,
        ppo_update_count=1,
        canonical_reward_sha256=canonical_reward_sequence_sha256([0.25]),
        accounting_complete=True,
        terminal=True,
    )


def _guard_payload(trainer, config, guard, *, monitor_level=0.90, final_level=0.90):
    baseline_monitor = _evidence_payload(
        level=0.20,
        training_design_sha256=config.scientific_design_hash(),
        guard=guard,
        final=False,
        observation_index=0,
        ppo_update_steps=0,
    )
    baseline_final = _evidence_payload(
        level=0.20,
        training_design_sha256=config.scientific_design_hash(),
        guard=guard,
        final=True,
        observation_index=0,
        ppo_update_steps=0,
    )
    monitor = _evidence_payload(
        level=monitor_level,
        training_design_sha256=config.scientific_design_hash(),
        guard=guard,
        final=False,
        observation_index=1,
        ppo_update_steps=1,
    )
    final = _evidence_payload(
        level=final_level,
        training_design_sha256=config.scientific_design_hash(),
        guard=guard,
        final=True,
        observation_index=1,
        ppo_update_steps=1,
    )
    return build_generalization_guard_provenance(
        config=guard,
        training_config=config,
        development_cases=("toy-development",),
        population_size=4,
        environment_template=_environment_template(),
        training_episode_receipts=tuple(trainer.training_episode_receipts),
        baseline_monitor_evidence=baseline_monitor,
        baseline_final_evidence=baseline_final,
        monitor_evidence=[monitor],
        final_evidence=final,
    )


def test_guard_rejects_protected_or_training_seed_leakage():
    guard = _guard()
    with pytest.raises(ValueError, match="Protected holdout"):
        guard.validate(
            development_cases=("case118",), population_size=4, training_episode_seeds=()
        )
    collision = guard.seed_block(("toy-development",), final=False)[0]
    with pytest.raises(ValueError, match="disjoint from policy-training"):
        guard.validate(
            development_cases=("toy-development",),
            population_size=4,
            training_episode_seeds=(collision,),
        )


def test_learning_health_comparison_is_lexicographic_not_hidden_scalar():
    guard = _guard()
    row = {
        "case_identity": "toy-development",
        "final_feasible_ratio": 0.50,
        "mean_constraint_improvement": 0.0,
        "mean_feasible_ratio_improvement": 0.0,
        "mean_objective_improvement": 0.0,
        "mean_canonical_reward": 0.0,
    }
    reference = {"development_cases": ["toy-development"], "case_rows": [row]}
    improved = {
        "development_cases": ["toy-development"],
        "case_rows": [{**row, "final_feasible_ratio": 0.70}],
    }
    degraded = {
        "development_cases": ["toy-development"],
        "case_rows": [{**row, "final_feasible_ratio": 0.20}],
    }
    assert compare_generalization_evidence(improved, reference, guard).verdict == "improved"
    assert compare_generalization_evidence(degraded, reference, guard).verdict == "degraded"


def test_legacy_training_hash_and_plan_shape_remain_unchanged_when_guard_absent():
    plan = _plan(guarded=False)
    assert "generalization_guard" not in plan.to_dict()
    restored = TSHCALOTrainingCampaignPlan.from_dict(plan.to_dict())
    assert restored.generalization_guard is None
    assert restored.execution_plan_sha256() == plan.execution_plan_sha256()

    config = plan.training_config(plan.members[0])
    payload = asdict(config)
    payload.pop("device", None)
    payload.pop("allow_cpu_fallback", None)
    payload.pop("generalization_guard_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    expected_pre_guard_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert config.scientific_design_hash() == expected_pre_guard_hash

    legacy_provenance = IndependentTrainingProvenance(
        training_run_id="legacy-run",
        training_design_sha256=_sha("legacy-design"),
        source_commit=SOURCE_COMMIT,
        development_cases=("toy-development",),
        seed_manifest_sha256=_sha("legacy-seeds"),
        training_device_provenance={},
        training_episode_receipts=(),
    )
    serialized = _training_provenance_payload(legacy_provenance)
    assert "generalization_guard_sha256" not in serialized
    assert "generalization_guard" not in serialized

    # Resume metadata also preserves the pre-guard training-config field shape when the guard is absent.
    trainer = IndependentTSHCALOTrainer(config)
    try:
        assert "generalization_guard_sha256" not in trainer.resume_state_dict()["training_config"]
    finally:
        trainer.close()


def test_guarded_candidate_export_fails_closed_without_passed_bound_evidence(tmp_path):
    guard = _guard()
    config = _training_config(guard)
    trainer = IndependentTSHCALOTrainer(config)
    try:
        trainer.update_steps = 1
        trainer.record_training_episode_receipt(_receipt(config).to_dict())
        with pytest.raises(ValueError, match="requires the configured generalization"):
            trainer.export_unqualified_candidate(
                tmp_path / "missing-guard.pt", source_commit=SOURCE_COMMIT
            )

        passed = _guard_payload(trainer, config, guard)
        rejected = _guard_payload(
            trainer, config, guard, monitor_level=0.10, final_level=0.05
        )
        assert rejected["status"] == "generalization_risk"
        assert rejected["promotion_allowed"] is False
        with pytest.raises(ValueError, match="blocked by the generalization guard"):
            trainer.export_unqualified_candidate(
                tmp_path / "rejected-guard.pt",
                source_commit=SOURCE_COMMIT,
                generalization_guard=rejected,
            )

        artifact = trainer.export_unqualified_candidate(
            tmp_path / "passed-guard.pt",
            source_commit=SOURCE_COMMIT,
            generalization_guard=passed,
        )
        inspected = inspect_tsh_calo_candidate(artifact.path, expected_sha256=artifact.sha256)
        assert inspected.training_provenance["generalization_guard"]["status"] == "passed"
        assert (
            inspected.training_provenance["generalization_guard_sha256"]
            == _guard_design(guard)
        )
    finally:
        trainer.close()


def test_fresh_campaign_binds_guard_evidence_before_candidate_export(
    tmp_path, toy_case, monkeypatch
):
    monkeypatch.setattr(campaign_module, "evaluate_generalization_bundle", _fake_bundle)
    plan = _plan(guarded=True)
    result = IndependentTSHCALOTrainingCampaign(
        plan,
        tmp_path / "guarded-campaign",
        problem_factory=_factory(toy_case),
    ).start()

    for candidate in result.member_candidates:
        payload = candidate.training_provenance["generalization_guard"]
        assert payload["status"] == "passed"
        assert payload["promotion_allowed"] is True
        assert payload["training_episode_count"] == 1
        assert payload["additional_candidate_evaluations"] == 32
        assert candidate.training_provenance["generalization_guard_sha256"] == _guard_design(_guard())


def test_completed_extension_cannot_bypass_fresh_generalization_evidence(
    tmp_path, toy_case, monkeypatch
):
    monkeypatch.setattr(campaign_module, "evaluate_generalization_bundle", _fake_bundle)
    plan = _plan(guarded=True)
    root = tmp_path / "guarded-extension"
    IndependentTSHCALOTrainingCampaign(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()

    extended = IndependentTSHCALOTrainingExtension(
        plan,
        root,
        problem_factory=_factory(toy_case),
    ).start()

    for candidate in extended.member_candidates:
        payload = candidate.training_provenance["generalization_guard"]
        assert payload["status"] == "passed"
        assert payload["training_episode_count"] == 2
        assert payload["baseline_monitor_evidence"]["ppo_update_steps_observed"] == 1
        assert payload["final_evidence"]["ppo_update_steps_observed"] == 2
