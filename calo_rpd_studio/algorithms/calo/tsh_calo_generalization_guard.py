"""Deterministic development-only learning-health guard for TSH-CALO policy training.

The guard never opens protected publication holdouts, never updates PPO state, never changes
scientific hyperparameters, never qualifies or activates a policy, and never hides its additional
counted ORPD evaluations. Monitor evidence may diagnose repeated development degradation during a
training segment. A disjoint final seed block is evaluated at the frozen pre-training baseline and
again only after the segment completes; its baseline result is never used for monitor decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable

import numpy as np

from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

from .diagnostics import population_diagnostics
from .tsh_calo_training import IndependentTSHCALOTrainer, TSHCALOTrainingConfig
from .tsh_calo_training_environment import (
    IndependentTSHCALOTrainingEnvironment,
    TSHCALOTrainingEnvironmentConfig,
)
from .tsh_calo_training_receipt import load_tsh_calo_training_episode_receipt
from .tsh_calo_schema import TSH_CALO_TRAINING_ENVIRONMENT


TSH_CALO_GENERALIZATION_GUARD_SCHEMA = "tsh-calo-generalization-guard-v1"
TSH_CALO_GENERALIZATION_EVIDENCE_SCHEMA = "tsh-calo-generalization-evidence-v1"


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_or_none(value: float):
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(rows: list[dict], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and math.isfinite(float(row.get(key, math.nan)))
    ]
    return float(np.mean(values)) if values else math.nan


def _receipt_seed_sha256(receipts: tuple[dict, ...] | list[dict]) -> str:
    rows = []
    for payload in receipts:
        receipt = load_tsh_calo_training_episode_receipt(payload)
        rows.append(
            {
                "case_identity": receipt.case_identity,
                "seed": int(receipt.seed),
                "receipt_sha256": receipt.receipt_sha256,
            }
        )
    return _canonical_sha256({"receipts": rows})


@dataclass(frozen=True, slots=True)
class TSHCALOGeneralizationGuardConfig:
    """Frozen scientific design for development-only learning-health evidence."""

    enabled: bool = True
    validation_batches_per_case: int = 3
    validation_seed: int = 20_260_001
    final_audit_seed_offset: int = 1_000_003
    degradation_patience: int = 2
    feasible_ratio_tolerance: float = 0.10
    reward_component_tolerance: float = 0.10
    minimum_learning_gain: float = 0.02
    minimum_acceptable_feasible_ratio: float = 0.80
    schema_version: str = TSH_CALO_GENERALIZATION_GUARD_SCHEMA

    def validate(
        self,
        *,
        development_cases: tuple[str, ...],
        population_size: int,
        training_episode_seeds: tuple[int, ...] = (),
    ) -> None:
        if self.schema_version != TSH_CALO_GENERALIZATION_GUARD_SCHEMA:
            raise ValueError("TSH-CALO generalization-guard schema is incompatible")
        if not isinstance(self.enabled, bool):
            raise ValueError("TSH-CALO generalization-guard enabled flag must be Boolean")
        if not self.enabled:
            return
        if not development_cases:
            raise ValueError("TSH-CALO generalization guard requires development cases")
        leaked = protected_holdout_matches(development_cases)
        if leaked:
            raise ValueError(
                "Protected holdout cases cannot enter the TSH-CALO generalization guard: "
                + ", ".join(leaked)
            )
        if int(population_size) < 2:
            raise ValueError("TSH-CALO generalization guard requires a valid population")
        if not 2 <= int(self.validation_batches_per_case) <= 64:
            raise ValueError(
                "TSH-CALO generalization validation batches per case must be within [2, 64]"
            )
        if int(self.validation_seed) < 0 or int(self.final_audit_seed_offset) < 1:
            raise ValueError("TSH-CALO generalization validation seeds are invalid")
        if not 1 <= int(self.degradation_patience) <= 16:
            raise ValueError("TSH-CALO generalization degradation patience is invalid")
        for label, value in (
            ("feasible-ratio tolerance", self.feasible_ratio_tolerance),
            ("reward-component tolerance", self.reward_component_tolerance),
            ("minimum learning gain", self.minimum_learning_gain),
            ("minimum acceptable feasible ratio", self.minimum_acceptable_feasible_ratio),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"TSH-CALO generalization {label} must be within [0, 1]")
        monitor = set(self.seed_block(development_cases, final=False))
        final = set(self.seed_block(development_cases, final=True))
        training = {int(seed) for seed in training_episode_seeds}
        if monitor & final:
            raise ValueError("TSH-CALO monitor and final generalization seed blocks must be disjoint")
        if training & (monitor | final):
            raise ValueError(
                "TSH-CALO generalization seeds must be disjoint from policy-training episode seeds"
            )

    def seed_block(self, development_cases: tuple[str, ...], *, final: bool) -> tuple[int, ...]:
        base = int(self.validation_seed) + (int(self.final_audit_seed_offset) if final else 0)
        return tuple(base + index * 1009 for index, _case in enumerate(development_cases))

    def validation_evaluations_per_case(self, population_size: int) -> int:
        return int(self.validation_batches_per_case) * int(population_size)

    def scientific_design_hash(self) -> str:
        return _canonical_sha256(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "TSHCALOGeneralizationGuardConfig | None":
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("TSH-CALO generalization-guard configuration must be an object")
        return cls(**dict(payload))


def generalization_guard_design_sha256(
    config: TSHCALOGeneralizationGuardConfig,
    *,
    development_cases: tuple[str, ...],
    population_size: int,
    environment_template: dict,
) -> str:
    """Bind guard controls to the exact development/environment design used for evaluation."""

    config.validate(
        development_cases=development_cases,
        population_size=population_size,
        training_episode_seeds=(),
    )
    return _canonical_sha256(
        {
            "guard_config": config.to_dict(),
            "development_cases": list(development_cases),
            "population_size": int(population_size),
            "environment_template": dict(environment_template),
        }
    )


@dataclass(frozen=True, slots=True)
class GeneralizationComparison:
    verdict: str
    reason: str

    @property
    def improved(self) -> bool:
        return self.verdict == "improved"

    @property
    def degraded(self) -> bool:
        return self.verdict == "degraded"


def _guard_environment_config(
    *,
    case_identity: str,
    seed: int,
    population_size: int,
    max_evaluations: int,
    environment_template: dict,
) -> TSHCALOTrainingEnvironmentConfig:
    values = {
        **dict(environment_template),
        "case_identity": str(case_identity),
        "seed": int(seed),
        "population_size": int(population_size),
        "max_evaluations": int(max_evaluations),
        # Guard evidence must not introduce stochastic policy-independent lane/group choices.
        "environment_deterministic": True,
    }
    return TSHCALOTrainingEnvironmentConfig(**values)


def evaluate_generalization_bundle(
    trainer: IndependentTSHCALOTrainer,
    training_config: TSHCALOTrainingConfig,
    guard_config: TSHCALOGeneralizationGuardConfig,
    *,
    development_cases: tuple[str, ...],
    population_size: int,
    environment_template: dict,
    problem_factory: Callable[[str], object],
    final: bool,
    observation_index: int,
    evaluation_backend: str = "declared_problem_factory",
) -> dict:
    """Evaluate one deterministic development seed block without changing PPO/RNG state."""

    guard_config.validate(
        development_cases=development_cases,
        population_size=population_size,
        training_episode_seeds=tuple(
            int(item.get("seed", -1)) for item in trainer.training_episode_receipts
        ),
    )
    if not guard_config.enabled:
        raise ValueError("Disabled TSH-CALO generalization guard cannot evaluate evidence")
    before_updates = int(trainer.update_steps)
    before_numpy = _canonical_sha256(trainer.numpy_rng.bit_generator.state)
    before_torch = bytes(trainer.torch_generator.get_state().cpu().numpy().tobytes())
    seeds = guard_config.seed_block(development_cases, final=final)
    rows: list[dict] = []
    total_evaluations = 0
    total_scenario_calls = 0

    for case_identity, seed in zip(development_cases, seeds, strict=True):
        problem = problem_factory(case_identity)
        environment_config = _guard_environment_config(
            case_identity=case_identity,
            seed=seed,
            population_size=population_size,
            max_evaluations=guard_config.validation_evaluations_per_case(population_size),
            environment_template=environment_template,
        )
        environment_config.validate(training_config)
        environment = IndependentTSHCALOTrainingEnvironment(
            problem,
            training_config,
            environment_config,
        )
        observation = environment.reset()
        reward_total = 0.0
        objective_reward = 0.0
        constraint_reward = 0.0
        feasible_reward = 0.0
        diversity_reward = 0.0
        transition_count = 0
        while not environment.terminal:
            action, _log_probability, _value = trainer.sample_action(
                observation.policy_state,
                observation.action_mask,
                observation.learner_groups,
                observation.learner_contexts,
                deterministic=True,
            )
            step = environment.step(action)
            reward = step.transition.reward
            reward_total += float(reward.total)
            objective_reward += float(reward.objective_improvement)
            constraint_reward += float(reward.constraint_improvement)
            feasible_reward += float(reward.feasible_ratio_improvement)
            diversity_reward += float(reward.diversity_recovery)
            transition_count += 1
            if step.next_observation is not None:
                observation = step.next_observation
        if environment.state is None:
            raise RuntimeError(
                "TSH-CALO generalization evaluation ended without an environment state"
            )
        provenance = environment.scientific_provenance()
        candidate_evaluations = int(provenance.get("candidate_evaluations", 0))
        expected = guard_config.validation_evaluations_per_case(population_size)
        if candidate_evaluations != expected or not bool(
            provenance.get("accounting_complete", False)
        ):
            raise RuntimeError(
                "TSH-CALO generalization evaluation did not retain complete exact FE accounting"
            )
        diagnostics = population_diagnostics(environment.state.evaluations, 0.0)
        divisor = max(transition_count, 1)
        scenario_calls = int(provenance.get("scenario_power_flow_calls", 0))
        total_evaluations += candidate_evaluations
        total_scenario_calls += scenario_calls
        rows.append(
            {
                "case_identity": str(case_identity),
                "seed": int(seed),
                "candidate_evaluations": candidate_evaluations,
                "scenario_power_flow_calls": scenario_calls,
                "case_checksum": str(provenance.get("case_checksum", "")),
                "problem_fingerprint": str(provenance.get("problem_fingerprint", "")),
                "environment_design_sha256": str(
                    provenance.get("environment_design_sha256", "")
                ),
                "transition_count": int(transition_count),
                "final_feasible_ratio": float(diagnostics.feasible_ratio),
                "best_violation": _finite_or_none(diagnostics.best_violation),
                # Raw objective is retained per case only and never pooled across unlike systems.
                "best_feasible_objective": _finite_or_none(diagnostics.best_feasible_objective),
                "mean_canonical_reward": float(reward_total / divisor),
                "mean_objective_improvement": float(objective_reward / divisor),
                "mean_constraint_improvement": float(constraint_reward / divisor),
                "mean_feasible_ratio_improvement": float(feasible_reward / divisor),
                "mean_diversity_recovery": float(diversity_reward / divisor),
            }
        )

    after_numpy = _canonical_sha256(trainer.numpy_rng.bit_generator.state)
    after_torch = bytes(trainer.torch_generator.get_state().cpu().numpy().tobytes())
    if int(trainer.update_steps) != before_updates:
        raise RuntimeError(
            "TSH-CALO generalization evaluation unexpectedly changed PPO update state"
        )
    if after_numpy != before_numpy or after_torch != before_torch:
        raise RuntimeError(
            "TSH-CALO generalization evaluation unexpectedly changed trainer RNG state"
        )
    return {
        "schema_version": TSH_CALO_GENERALIZATION_EVIDENCE_SCHEMA,
        "evidence_kind": "final_audit" if final else "monitor",
        "observation_index": int(observation_index),
        "evaluation_backend": str(evaluation_backend),
        "guard_config_sha256": guard_config.scientific_design_hash(),
        "guard_design_sha256": generalization_guard_design_sha256(
            guard_config,
            development_cases=development_cases,
            population_size=population_size,
            environment_template=environment_template,
        ),
        "training_design_sha256": training_config.scientific_design_hash(),
        "development_cases": list(development_cases),
        "validation_seed_block": list(seeds),
        "case_rows": rows,
        "candidate_evaluations": int(total_evaluations),
        "scenario_power_flow_calls": int(total_scenario_calls),
        "mean_final_feasible_ratio": _mean(rows, "final_feasible_ratio"),
        "mean_canonical_reward": _mean(rows, "mean_canonical_reward"),
        "mean_objective_improvement": _mean(rows, "mean_objective_improvement"),
        "mean_constraint_improvement": _mean(rows, "mean_constraint_improvement"),
        "mean_feasible_ratio_improvement": _mean(rows, "mean_feasible_ratio_improvement"),
        "ppo_update_steps_observed": int(before_updates),
    }



def validate_generalization_evidence(
    payload: dict,
    *,
    config: TSHCALOGeneralizationGuardConfig,
    training_design_sha256: str,
    development_cases: tuple[str, ...],
    population_size: int,
    final: bool,
    expected_guard_design_sha256: str = "",
    expected_problem_identities: dict[str, tuple[str, str]] | None = None,
    expected_environment_template: dict | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("TSH-CALO generalization evidence must be an object")
    if payload.get("schema_version") != TSH_CALO_GENERALIZATION_EVIDENCE_SCHEMA:
        raise ValueError("TSH-CALO generalization evidence schema is incompatible")
    expected_kind = "final_audit" if final else "monitor"
    if payload.get("evidence_kind") != expected_kind:
        raise ValueError("TSH-CALO generalization evidence kind is inconsistent")
    if payload.get("guard_config_sha256") != config.scientific_design_hash():
        raise ValueError("TSH-CALO generalization evidence guard configuration changed")
    guard_design = str(payload.get("guard_design_sha256", ""))
    if expected_guard_design_sha256 and guard_design != expected_guard_design_sha256:
        raise ValueError("TSH-CALO generalization evidence guard design changed")
    if not guard_design or len(guard_design) != 64 or any(
        character not in "0123456789abcdef" for character in guard_design.lower()
    ):
        raise ValueError("TSH-CALO generalization evidence guard design SHA-256 is invalid")
    if payload.get("training_design_sha256") != training_design_sha256:
        raise ValueError("TSH-CALO generalization evidence training design changed")
    if tuple(payload.get("development_cases", ())) != tuple(development_cases):
        raise ValueError("TSH-CALO generalization evidence development cases changed")
    expected_seeds = config.seed_block(development_cases, final=final)
    if tuple(int(value) for value in payload.get("validation_seed_block", ())) != expected_seeds:
        raise ValueError("TSH-CALO generalization evidence seed block changed")
    rows = list(payload.get("case_rows", []) or [])
    if len(rows) != len(development_cases):
        raise ValueError("TSH-CALO generalization evidence case rows are incomplete")
    expected_per_case = config.validation_evaluations_per_case(population_size)
    expected_total = expected_per_case * len(development_cases)
    total_scenario_calls = 0
    for row, case_identity, seed in zip(rows, development_cases, expected_seeds, strict=True):
        if row.get("case_identity") != case_identity or int(row.get("seed", -1)) != int(seed):
            raise ValueError("TSH-CALO generalization evidence case/seed binding changed")
        if int(row.get("candidate_evaluations", -1)) != expected_per_case:
            raise ValueError("TSH-CALO generalization evidence FE accounting changed")
        scenario_calls = int(row.get("scenario_power_flow_calls", -1))
        if scenario_calls < expected_per_case:
            raise ValueError("TSH-CALO generalization scenario accounting is impossible")
        total_scenario_calls += scenario_calls
        case_checksum = str(row.get("case_checksum", "")).lower()
        problem_fingerprint = str(row.get("problem_fingerprint", "")).lower()
        environment_design = str(row.get("environment_design_sha256", "")).lower()
        for label, digest in (
            ("case checksum", case_checksum),
            ("problem fingerprint", problem_fingerprint),
            ("environment design", environment_design),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"TSH-CALO generalization {label} SHA-256 is invalid")
        if expected_problem_identities is not None:
            expected_identity = expected_problem_identities.get(case_identity)
            if expected_identity is None or (case_checksum, problem_fingerprint) != expected_identity:
                raise ValueError(
                    "TSH-CALO generalization problem identity differs from authenticated training"
                )
        if expected_environment_template is not None:
            expected_environment = _guard_environment_config(
                case_identity=case_identity,
                seed=seed,
                population_size=population_size,
                max_evaluations=expected_per_case,
                environment_template=expected_environment_template,
            )
            expected_environment_design = _canonical_sha256(
                {
                    "schema_version": TSH_CALO_TRAINING_ENVIRONMENT,
                    "training_design_sha256": training_design_sha256,
                    "environment": asdict(expected_environment),
                }
            )
            if environment_design != expected_environment_design:
                raise ValueError(
                    "TSH-CALO generalization environment design differs from the frozen guard plan"
                )
        feasible = float(row.get("final_feasible_ratio", math.nan))
        if not math.isfinite(feasible) or not 0.0 <= feasible <= 1.0:
            raise ValueError("TSH-CALO generalization feasible-ratio evidence is invalid")
        for key in (
            "mean_canonical_reward",
            "mean_objective_improvement",
            "mean_constraint_improvement",
            "mean_feasible_ratio_improvement",
            "mean_diversity_recovery",
        ):
            if not math.isfinite(float(row.get(key, math.nan))):
                raise ValueError(f"TSH-CALO generalization {key} is non-finite")
    if int(payload.get("candidate_evaluations", -1)) != expected_total:
        raise ValueError("TSH-CALO generalization total FE accounting changed")
    if int(payload.get("scenario_power_flow_calls", -1)) != total_scenario_calls:
        raise ValueError("TSH-CALO generalization total scenario accounting changed")
    if int(payload.get("observation_index", -1)) < 0:
        raise ValueError("TSH-CALO generalization observation index is invalid")
    if not str(payload.get("evaluation_backend", "")).strip():
        raise ValueError("TSH-CALO generalization evaluation backend is missing")
    if int(payload.get("ppo_update_steps_observed", -1)) < 0:
        raise ValueError("TSH-CALO generalization PPO update provenance is invalid")
    for key, row_key in (
        ("mean_final_feasible_ratio", "final_feasible_ratio"),
        ("mean_canonical_reward", "mean_canonical_reward"),
        ("mean_objective_improvement", "mean_objective_improvement"),
        ("mean_constraint_improvement", "mean_constraint_improvement"),
        ("mean_feasible_ratio_improvement", "mean_feasible_ratio_improvement"),
    ):
        expected_mean = _mean(rows, row_key)
        actual = float(payload.get(key, math.nan))
        if not math.isfinite(actual) or not math.isclose(
            actual, expected_mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"TSH-CALO generalization aggregate {key} changed")

def compare_generalization_evidence(
    candidate: dict,
    reference: dict,
    config: TSHCALOGeneralizationGuardConfig,
) -> GeneralizationComparison:
    """Compare case-aligned evidence without a hidden scalar or cross-case compensation."""

    if candidate.get("development_cases") != reference.get("development_cases"):
        return GeneralizationComparison("invalid", "development-case bundle changed")
    candidate_rows = list(candidate.get("case_rows", []) or [])
    reference_rows = list(reference.get("case_rows", []) or [])
    if len(candidate_rows) != len(reference_rows) or not candidate_rows:
        return GeneralizationComparison("invalid", "generalization evidence is incomplete")
    if [row.get("case_identity") for row in candidate_rows] != [
        row.get("case_identity") for row in reference_rows
    ]:
        return GeneralizationComparison("invalid", "generalization case order changed")

    improvement_reasons: list[str] = []
    for candidate_row, reference_row in zip(candidate_rows, reference_rows, strict=True):
        case_identity = str(candidate_row.get("case_identity", "development case"))
        c_feasible = float(candidate_row.get("final_feasible_ratio", math.nan))
        r_feasible = float(reference_row.get("final_feasible_ratio", math.nan))
        if not math.isfinite(c_feasible) or not math.isfinite(r_feasible):
            return GeneralizationComparison(
                "invalid", f"generalization feasibility evidence is non-finite on {case_identity}"
            )
        if c_feasible < r_feasible - float(config.feasible_ratio_tolerance):
            return GeneralizationComparison(
                "degraded",
                f"held-out development feasibility materially deteriorated on {case_identity}",
            )
        if c_feasible > r_feasible + float(config.minimum_learning_gain):
            improvement_reasons.append(
                f"held-out development feasibility improved on {case_identity}"
            )

    for key, label in (
        ("mean_constraint_improvement", "constraint progress"),
        ("mean_feasible_ratio_improvement", "feasibility progress"),
        ("mean_objective_improvement", "objective progress"),
        ("mean_canonical_reward", "canonical reward"),
    ):
        for candidate_row, reference_row in zip(candidate_rows, reference_rows, strict=True):
            case_identity = str(candidate_row.get("case_identity", "development case"))
            c_value = float(candidate_row.get(key, math.nan))
            r_value = float(reference_row.get(key, math.nan))
            if not math.isfinite(c_value) or not math.isfinite(r_value):
                return GeneralizationComparison(
                    "invalid", f"generalization {label} evidence is non-finite on {case_identity}"
                )
            if c_value < r_value - float(config.reward_component_tolerance):
                return GeneralizationComparison(
                    "degraded",
                    f"held-out development {label} materially deteriorated on {case_identity}",
                )
            if c_value > r_value + float(config.minimum_learning_gain):
                improvement_reasons.append(
                    f"held-out development {label} improved on {case_identity}"
                )

    if improvement_reasons:
        return GeneralizationComparison("improved", improvement_reasons[0])
    return GeneralizationComparison(
        "stable", "held-out development evidence remained within predeclared tolerances"
    )


def _classify_generalization_evidence(
    *,
    baseline_monitor_evidence: dict,
    baseline_final_evidence: dict,
    monitor_evidence: list[dict],
    final_evidence: dict,
    config: TSHCALOGeneralizationGuardConfig,
) -> dict:
    best = baseline_monitor_evidence
    degradation_streak = 0
    maximum_degradation_streak = 0
    comparisons: list[dict] = []
    for evidence in monitor_evidence:
        comparison = compare_generalization_evidence(evidence, best, config)
        comparisons.append(
            {
                "observation_index": int(evidence.get("observation_index", -1)),
                "verdict": comparison.verdict,
                "reason": comparison.reason,
            }
        )
        if comparison.verdict == "invalid":
            return {
                "status": "generalization_risk",
                "promotion_allowed": False,
                "reason": comparison.reason,
                "monitor_comparisons": comparisons,
                "maximum_monitor_degradation_streak": maximum_degradation_streak,
            }
        if comparison.improved:
            best = evidence
            degradation_streak = 0
        elif comparison.degraded:
            degradation_streak += 1
            maximum_degradation_streak = max(maximum_degradation_streak, degradation_streak)
        else:
            degradation_streak = 0

    final_comparison = compare_generalization_evidence(
        final_evidence, baseline_final_evidence, config
    )
    if final_comparison.verdict in {"invalid", "degraded"}:
        return {
            "status": "generalization_risk",
            "promotion_allowed": False,
            "reason": final_comparison.reason,
            "monitor_comparisons": comparisons,
            "maximum_monitor_degradation_streak": maximum_degradation_streak,
        }
    if maximum_degradation_streak >= int(config.degradation_patience):
        return {
            "status": "generalization_risk",
            "promotion_allowed": False,
            "reason": (
                "held-out development evidence deteriorated for "
                f"{maximum_degradation_streak} consecutive monitor observations"
            ),
            "monitor_comparisons": comparisons,
            "maximum_monitor_degradation_streak": maximum_degradation_streak,
        }
    final_rows = list(final_evidence.get("case_rows", []) or [])
    feasible_floor_met = bool(
        final_rows
        and all(
            math.isfinite(float(row.get("final_feasible_ratio", math.nan)))
            and float(row.get("final_feasible_ratio", math.nan))
            >= float(config.minimum_acceptable_feasible_ratio)
            for row in final_rows
        )
    )
    if final_comparison.improved and feasible_floor_met:
        return {
            "status": "passed",
            "promotion_allowed": True,
            "reason": (
                "development-only final audit established improvement and met the predeclared "
                "minimum feasible-rate floor"
            ),
            "monitor_comparisons": comparisons,
            "maximum_monitor_degradation_streak": maximum_degradation_streak,
        }
    return {
        "status": "insufficient_learning_evidence",
        "promotion_allowed": False,
        "reason": (
            "training completed without both independent final-audit improvement and the "
            "predeclared feasible-rate floor; this does not prove architectural underfitting"
        ),
        "monitor_comparisons": comparisons,
        "maximum_monitor_degradation_streak": maximum_degradation_streak,
    }


def build_generalization_guard_provenance(
    *,
    config: TSHCALOGeneralizationGuardConfig,
    training_config: TSHCALOTrainingConfig,
    development_cases: tuple[str, ...],
    population_size: int,
    environment_template: dict,
    training_episode_receipts: tuple[dict, ...] | list[dict],
    segment_receipt_offset: int = 0,
    baseline_monitor_evidence: dict,
    baseline_final_evidence: dict,
    monitor_evidence: list[dict],
    final_evidence: dict,
) -> dict:
    """Classify learning health after a finite training segment and gate candidate export."""

    receipts = tuple(training_episode_receipts)
    offset = int(segment_receipt_offset)
    if offset < 0 or offset >= len(receipts):
        raise ValueError("TSH-CALO generalization segment receipt offset is invalid")
    segment_receipts = receipts[offset:]
    if len(monitor_evidence) != len(segment_receipts):
        raise ValueError(
            "TSH-CALO generalization monitor evidence must cover every segment training episode"
        )
    training_seeds = tuple(int(load_tsh_calo_training_episode_receipt(item).seed) for item in receipts)
    config.validate(
        development_cases=development_cases,
        population_size=population_size,
        training_episode_seeds=training_seeds,
    )
    expected_config_sha = config.scientific_design_hash()
    expected_guard_design_sha = str(training_config.generalization_guard_sha256)
    if not expected_guard_design_sha:
        raise ValueError("TSH-CALO training design did not declare a generalization guard")
    calculated_guard_design_sha = generalization_guard_design_sha256(
        config,
        development_cases=development_cases,
        population_size=population_size,
        environment_template=environment_template,
    )
    if expected_guard_design_sha != calculated_guard_design_sha:
        raise ValueError("TSH-CALO training guard design differs from its frozen campaign plan")
    expected_training_sha = training_config.scientific_design_hash()
    problem_identities: dict[str, tuple[str, str]] = {}
    for receipt_payload in receipts:
        receipt = load_tsh_calo_training_episode_receipt(receipt_payload)
        identity = (str(receipt.case_checksum).lower(), str(receipt.problem_fingerprint).lower())
        existing = problem_identities.setdefault(receipt.case_identity, identity)
        if existing != identity:
            raise ValueError(
                "TSH-CALO authenticated training receipts disagree on development problem identity"
            )
    validate_generalization_evidence(
        baseline_monitor_evidence,
        config=config,
        training_design_sha256=expected_training_sha,
        development_cases=development_cases,
        population_size=population_size,
        final=False,
        expected_guard_design_sha256=expected_guard_design_sha,
        expected_problem_identities=problem_identities,
        expected_environment_template=environment_template,
    )
    validate_generalization_evidence(
        baseline_final_evidence,
        config=config,
        training_design_sha256=expected_training_sha,
        development_cases=development_cases,
        population_size=population_size,
        final=True,
        expected_guard_design_sha256=expected_guard_design_sha,
        expected_problem_identities=problem_identities,
        expected_environment_template=environment_template,
    )
    for evidence in monitor_evidence:
        validate_generalization_evidence(
            evidence,
            config=config,
            training_design_sha256=expected_training_sha,
            development_cases=development_cases,
            population_size=population_size,
            final=False,
            expected_guard_design_sha256=expected_guard_design_sha,
            expected_problem_identities=problem_identities,
            expected_environment_template=environment_template,
        )
    validate_generalization_evidence(
        final_evidence,
        config=config,
        training_design_sha256=expected_training_sha,
        development_cases=development_cases,
        population_size=population_size,
        final=True,
        expected_guard_design_sha256=expected_guard_design_sha,
        expected_problem_identities=problem_identities,
        expected_environment_template=environment_template,
    )

    classification = _classify_generalization_evidence(
        baseline_monitor_evidence=baseline_monitor_evidence,
        baseline_final_evidence=baseline_final_evidence,
        monitor_evidence=list(monitor_evidence),
        final_evidence=final_evidence,
        config=config,
    )

    payload = {
        "schema_version": TSH_CALO_GENERALIZATION_GUARD_SCHEMA,
        "status": classification["status"],
        "promotion_allowed": bool(classification["promotion_allowed"]),
        "reason": classification["reason"],
        "guard_config": config.to_dict(),
        "guard_config_sha256": expected_config_sha,
        "guard_design_sha256": expected_guard_design_sha,
        "training_design_sha256": expected_training_sha,
        "development_cases": list(development_cases),
        "population_size": int(population_size),
        "environment_template": dict(environment_template),
        "training_episode_seed_receipt_sha256": _receipt_seed_sha256(receipts),
        "training_episode_count": len(receipts),
        "segment_receipt_offset": offset,
        "segment_training_episode_count": len(segment_receipts),
        "segment_training_receipt_sha256": _receipt_seed_sha256(segment_receipts),
        "baseline_monitor_evidence": baseline_monitor_evidence,
        "baseline_final_evidence": baseline_final_evidence,
        "monitor_evidence": list(monitor_evidence),
        "monitor_comparisons": list(classification["monitor_comparisons"]),
        "maximum_monitor_degradation_streak": int(
            classification["maximum_monitor_degradation_streak"]
        ),
        "final_evidence": final_evidence,
        "additional_candidate_evaluations": int(
            sum(
                int(item.get("candidate_evaluations", 0))
                for item in [
                    baseline_monitor_evidence,
                    baseline_final_evidence,
                    *monitor_evidence,
                    final_evidence,
                ]
            )
        ),
        "additional_scenario_power_flow_calls": int(
            sum(
                int(item.get("scenario_power_flow_calls", 0))
                for item in [
                    baseline_monitor_evidence,
                    baseline_final_evidence,
                    *monitor_evidence,
                    final_evidence,
                ]
            )
        ),
    }
    validate_generalization_guard_provenance(
        payload,
        training_episode_receipts=receipts,
        expected_training_design_sha256=expected_training_sha,
    )
    return payload


def validate_generalization_guard_provenance(
    payload: dict,
    *,
    training_episode_receipts: tuple[dict, ...] | list[dict] = (),
    expected_training_design_sha256: str = "",
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("TSH-CALO candidate generalization-guard provenance must be an object")
    if payload.get("schema_version") != TSH_CALO_GENERALIZATION_GUARD_SCHEMA:
        raise ValueError("TSH-CALO candidate generalization-guard provenance is incompatible")
    status = str(payload.get("status", ""))
    if status not in {"passed", "generalization_risk", "insufficient_learning_evidence"}:
        raise ValueError("TSH-CALO candidate generalization-guard status is invalid")
    promotion_allowed = payload.get("promotion_allowed")
    if promotion_allowed is not (status == "passed"):
        raise ValueError("TSH-CALO generalization-guard promotion state is inconsistent")
    raw_config = dict(payload.get("guard_config", {}) or {})
    config = TSHCALOGeneralizationGuardConfig.from_dict(raw_config)
    if config is None or not config.enabled:
        raise ValueError("TSH-CALO candidate generalization guard must be enabled")
    if payload.get("guard_config_sha256") != config.scientific_design_hash():
        raise ValueError("TSH-CALO candidate generalization-guard configuration hash changed")
    guard_design_sha = str(payload.get("guard_design_sha256", ""))
    if len(guard_design_sha) != 64 or any(
        character not in "0123456789abcdef" for character in guard_design_sha.lower()
    ):
        raise ValueError("TSH-CALO candidate generalization-guard design SHA-256 is invalid")
    development_cases = tuple(payload.get("development_cases", ()))
    population_size = int(payload.get("population_size", 0) or 0)
    environment_template = payload.get("environment_template")
    if not isinstance(environment_template, dict):
        raise ValueError("TSH-CALO candidate generalization environment template is invalid")
    calculated_guard_design_sha = generalization_guard_design_sha256(
        config,
        development_cases=development_cases,
        population_size=population_size,
        environment_template=environment_template,
    )
    if guard_design_sha != calculated_guard_design_sha:
        raise ValueError("TSH-CALO candidate generalization-guard design payload changed")
    receipt_rows = tuple(training_episode_receipts)
    validated_receipts = tuple(
        load_tsh_calo_training_episode_receipt(item) for item in receipt_rows
    )
    training_seeds = tuple(int(item.seed) for item in validated_receipts)
    problem_identities: dict[str, tuple[str, str]] = {}
    for receipt in validated_receipts:
        identity = (str(receipt.case_checksum).lower(), str(receipt.problem_fingerprint).lower())
        existing = problem_identities.setdefault(receipt.case_identity, identity)
        if existing != identity:
            raise ValueError(
                "TSH-CALO authenticated training receipts disagree on development problem identity"
            )
    config.validate(
        development_cases=development_cases,
        population_size=population_size,
        training_episode_seeds=training_seeds,
    )
    if expected_training_design_sha256 and payload.get(
        "training_design_sha256"
    ) != expected_training_design_sha256:
        raise ValueError("TSH-CALO candidate generalization evidence training design changed")
    if receipt_rows:
        if int(payload.get("training_episode_count", -1)) != len(receipt_rows):
            raise ValueError("TSH-CALO candidate generalization evidence episode count changed")
        if payload.get("training_episode_seed_receipt_sha256") != _receipt_seed_sha256(receipt_rows):
            raise ValueError("TSH-CALO candidate generalization evidence no longer binds its receipts")
        offset = int(payload.get("segment_receipt_offset", -1))
        if offset < 0 or offset >= len(validated_receipts):
            raise ValueError("TSH-CALO candidate generalization segment receipt offset changed")
        segment_receipts = validated_receipts[offset:]
        if int(payload.get("segment_training_episode_count", -1)) != len(segment_receipts):
            raise ValueError("TSH-CALO candidate generalization segment episode count changed")
        if payload.get("segment_training_receipt_sha256") != _receipt_seed_sha256(
            tuple(receipt_rows[offset:])
        ):
            raise ValueError("TSH-CALO candidate generalization segment receipt binding changed")
    else:
        offset = 0
        segment_receipts = ()
    nested_training_sha = str(payload.get("training_design_sha256", ""))
    validate_generalization_evidence(
        dict(payload.get("baseline_monitor_evidence", {}) or {}),
        config=config,
        training_design_sha256=nested_training_sha,
        development_cases=development_cases,
        population_size=population_size,
        final=False,
        expected_guard_design_sha256=guard_design_sha,
        expected_problem_identities=problem_identities if receipt_rows else None,
        expected_environment_template=environment_template,
    )
    validate_generalization_evidence(
        dict(payload.get("baseline_final_evidence", {}) or {}),
        config=config,
        training_design_sha256=nested_training_sha,
        development_cases=development_cases,
        population_size=population_size,
        final=True,
        expected_guard_design_sha256=guard_design_sha,
        expected_problem_identities=problem_identities if receipt_rows else None,
        expected_environment_template=environment_template,
    )
    for evidence in list(payload.get("monitor_evidence", []) or []):
        validate_generalization_evidence(
            dict(evidence or {}),
            config=config,
            training_design_sha256=nested_training_sha,
            development_cases=development_cases,
            population_size=population_size,
            final=False,
            expected_guard_design_sha256=guard_design_sha,
            expected_problem_identities=problem_identities if receipt_rows else None,
            expected_environment_template=environment_template,
        )
    validate_generalization_evidence(
        dict(payload.get("final_evidence", {}) or {}),
        config=config,
        training_design_sha256=nested_training_sha,
        development_cases=development_cases,
        population_size=population_size,
        final=True,
        expected_guard_design_sha256=guard_design_sha,
        expected_problem_identities=problem_identities if receipt_rows else None,
        expected_environment_template=environment_template,
    )
    baseline_monitor = dict(payload.get("baseline_monitor_evidence", {}) or {})
    baseline_final = dict(payload.get("baseline_final_evidence", {}) or {})
    monitors = list(payload.get("monitor_evidence", []) or [])
    final_evidence = dict(payload.get("final_evidence", {}) or {})
    if receipt_rows:
        if len(monitors) != len(segment_receipts):
            raise ValueError(
                "TSH-CALO candidate generalization monitor history omits a segment episode"
            )
        baseline_updates = (
            int(validated_receipts[offset - 1].ppo_update_count) if offset > 0 else 0
        )
        if int(baseline_monitor.get("observation_index", -1)) != 0 or int(
            baseline_final.get("observation_index", -1)
        ) != 0:
            raise ValueError("TSH-CALO candidate generalization baseline observation index changed")
        if int(baseline_monitor.get("ppo_update_steps_observed", -1)) != baseline_updates or int(
            baseline_final.get("ppo_update_steps_observed", -1)
        ) != baseline_updates:
            raise ValueError("TSH-CALO candidate generalization baseline PPO boundary changed")
        for index, (evidence, receipt) in enumerate(
            zip(monitors, segment_receipts, strict=True), start=1
        ):
            if int(evidence.get("observation_index", -1)) != index:
                raise ValueError("TSH-CALO candidate generalization monitor index changed")
            if int(evidence.get("ppo_update_steps_observed", -1)) != int(receipt.ppo_update_count):
                raise ValueError("TSH-CALO candidate generalization monitor PPO boundary changed")
        if int(final_evidence.get("observation_index", -1)) != len(segment_receipts) or int(
            final_evidence.get("ppo_update_steps_observed", -1)
        ) != int(segment_receipts[-1].ppo_update_count):
            raise ValueError("TSH-CALO candidate final generalization PPO boundary changed")
    classification = _classify_generalization_evidence(
        baseline_monitor_evidence=baseline_monitor,
        baseline_final_evidence=baseline_final,
        monitor_evidence=monitors,
        final_evidence=final_evidence,
        config=config,
    )
    for key in ("status", "promotion_allowed", "reason", "monitor_comparisons"):
        if payload.get(key) != classification[key]:
            raise ValueError("TSH-CALO candidate generalization-guard classification changed")
    if int(payload.get("maximum_monitor_degradation_streak", -1)) != int(
        classification["maximum_monitor_degradation_streak"]
    ):
        raise ValueError("TSH-CALO candidate generalization degradation history changed")
    evidence_rows = [baseline_monitor, baseline_final, *monitors, final_evidence]
    expected_additional_evaluations = sum(
        int(item.get("candidate_evaluations", 0)) for item in evidence_rows
    )
    expected_additional_scenarios = sum(
        int(item.get("scenario_power_flow_calls", 0)) for item in evidence_rows
    )
    if int(payload.get("additional_candidate_evaluations", -1)) != expected_additional_evaluations:
        raise ValueError("TSH-CALO candidate generalization FE ledger changed")
    if int(payload.get("additional_scenario_power_flow_calls", -1)) != expected_additional_scenarios:
        raise ValueError("TSH-CALO candidate generalization scenario ledger changed")
    for key in ("baseline_monitor_evidence", "baseline_final_evidence", "final_evidence"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"TSH-CALO candidate {key.replace('_', ' ')} is invalid")
    if not isinstance(payload.get("monitor_evidence", []), list):
        raise ValueError("TSH-CALO candidate monitor generalization evidence is invalid")
    if not isinstance(payload.get("monitor_comparisons", []), list):
        raise ValueError("TSH-CALO candidate monitor comparison evidence is invalid")


def candidate_generalization_status(training_provenance: dict) -> tuple[bool, str]:
    """Return whether explicit guard evidence permits downstream promotion workflows."""

    provenance = dict(training_provenance or {})
    if provenance.get("source_kind") == "independent_policy_training_ensemble":
        members = list(provenance.get("members", []) or [])
        if not members:
            return False, "TSH-CALO ensemble training provenance has no members."
        explicit_guard_seen = False
        for index, member in enumerate(members):
            member_provenance = dict(member.get("training_provenance", {}) or {})
            payload = dict(member_provenance.get("generalization_guard", {}) or {})
            if payload:
                explicit_guard_seen = True
            allowed, reason = candidate_generalization_status(member_provenance)
            if not allowed:
                return False, f"Ensemble member {index + 1} learning guard rejected promotion: {reason}"
        return (
            True,
            "passed" if explicit_guard_seen else "legacy_candidate_without_generalization_guard",
        )

    payload = dict(provenance.get("generalization_guard", {}) or {})
    declared_guard = str(provenance.get("generalization_guard_sha256", "") or "")
    if declared_guard and not payload:
        return False, "candidate declares a generalization guard but its evidence is missing"
    if payload and not declared_guard:
        return False, "candidate contains generalization evidence without a declared guard design"
    if not payload:
        return True, "legacy_candidate_without_generalization_guard"
    if payload.get("guard_design_sha256") != declared_guard:
        return False, "candidate generalization guard design binding changed"
    validate_generalization_guard_provenance(
        payload,
        training_episode_receipts=tuple(provenance.get("training_episode_receipts", ()) or ()),
        expected_training_design_sha256=str(provenance.get("training_design_sha256", "")),
    )
    if payload.get("status") != "passed" or payload.get("promotion_allowed") is not True:
        return False, str(payload.get("reason", "Policy training learning guard did not pass."))
    return True, "passed"
