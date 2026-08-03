"""Counted, independent TSH-CALO development-environment invariants."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSHCALOFeatureFlags
from calo_rpd_studio.algorithms.calo.tsh_calo_training import (
    IndependentTSHCALORolloutCollector,
    IndependentTSHCALOTrainer,
    TSHCALOTrainingConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_environment import (
    IndependentTSHCALOTrainingEnvironment,
    TSHCALOTrainingEnvironmentConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)
from calo_rpd_studio.orpd.problem import ORPDProblem


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _training_config(**changes) -> TSHCALOTrainingConfig:
    values = dict(
        training_run_id="development-rollout-001",
        development_cases=("toy-development",),
        seed_manifest_sha256=_sha("development-seeds"),
        resource_envelope=TSHCALOTrainingResourceEnvelope(4, 8, 16, 32, 16, 8),
        seed=41,
        hidden_dim=16,
        graph_steps=1,
        ppo_epochs=1,
        device="cpu",
    )
    values.update(changes)
    return TSHCALOTrainingConfig(**values)


def _environment_config(**changes) -> TSHCALOTrainingEnvironmentConfig:
    values = dict(
        case_identity="toy-development",
        population_size=4,
        max_evaluations=12,
        seed=107,
        environment_deterministic=True,
    )
    values.update(changes)
    return TSHCALOTrainingEnvironmentConfig(**values)


def _sample(trainer, observation):
    action, _log_probability, _value = trainer.sample_action(
        observation.policy_state,
        observation.action_mask,
        observation.learner_groups,
        observation.learner_contexts,
        deterministic=True,
    )
    return action


def test_counted_environment_collects_only_canonical_transitions(toy_case):
    problem = ORPDProblem(toy_case)
    training = _training_config()
    trainer = IndependentTSHCALOTrainer(training)
    collector = IndependentTSHCALORolloutCollector(trainer)
    environment = IndependentTSHCALOTrainingEnvironment(problem, training, _environment_config())

    observation = environment.reset()

    assert observation.candidate_evaluations == 4
    assert observation.scenario_power_flow_calls == 4
    assert not np.asarray(observation.action_mask.allowed, dtype=bool)[:, 6].any()
    assert observation.physics_repair_status.startswith("masked")
    with pytest.raises(RuntimeError, match="one hash-bound episode"):
        environment.reset()
    assert environment.candidate_evaluations == 4
    pending = collector.sample(
        observation.policy_state,
        observation.action_mask,
        observation.learner_groups,
        observation.learner_contexts,
        deterministic=True,
    )
    step = environment.step(pending.action)
    collector.commit(step.transition, terminal=step.terminal)

    assert step.candidate_evaluations == 8
    assert step.scenario_power_flow_calls == 8
    assert not step.terminal
    assert step.next_observation is not None
    assert collector.rewards == [step.transition.reward.total]
    assert environment.scientific_provenance()["lifecycle_authority"] == "none"
    assert environment.scientific_provenance()["accounting_complete"] is True


def test_environment_exhausts_exact_batch_budget_and_collector_builds_rollout(toy_case):
    problem = ORPDProblem(toy_case)
    training = _training_config()
    trainer = IndependentTSHCALOTrainer(training)
    collector = IndependentTSHCALORolloutCollector(trainer)
    environment = IndependentTSHCALOTrainingEnvironment(problem, training, _environment_config())
    observation = environment.reset()

    while True:
        pending = collector.sample(
            observation.policy_state,
            observation.action_mask,
            observation.learner_groups,
            observation.learner_contexts,
            deterministic=True,
        )
        step = environment.step(pending.action)
        collector.commit(step.transition, terminal=step.terminal)
        if step.terminal:
            break
        assert step.next_observation is not None
        observation = step.next_observation

    batch = collector.build_batch()
    assert environment.candidate_evaluations == 12
    assert environment.scenario_power_flow_calls == 12
    assert len(batch.states) == 2
    assert step.next_observation is None
    with pytest.raises(RuntimeError, match="exhausted"):
        environment.step(batch.actions[-1])


def test_environment_resume_replays_pending_observation_and_transition_exactly(toy_case):
    problem = ORPDProblem(toy_case)
    training = _training_config()
    config = _environment_config(max_evaluations=16, environment_deterministic=False)
    trainer = IndependentTSHCALOTrainer(training)
    original = IndependentTSHCALOTrainingEnvironment(problem, training, config)
    observation = original.reset()
    checkpoint = original.state_dict()
    restored = IndependentTSHCALOTrainingEnvironment.from_state_dict(
        problem, training, config, checkpoint
    )
    action = _sample(trainer, observation)

    left = original.step(action)
    right = restored.step(action)

    assert left.transition.reward.total == pytest.approx(
        right.transition.reward.total, rel=0.0, abs=0.0
    )
    np.testing.assert_array_equal(
        left.transition.selected_indices, right.transition.selected_indices
    )
    np.testing.assert_array_equal(
        left.transition.selected_population, right.transition.selected_population
    )
    np.testing.assert_array_equal(left.executed_operators, right.executed_operators)
    assert left.candidate_evaluations == right.candidate_evaluations == 8
    assert left.scenario_power_flow_calls == right.scenario_power_flow_calls == 8
    assert left.next_observation is not None and right.next_observation is not None
    np.testing.assert_array_equal(
        left.next_observation.learner_groups,
        right.next_observation.learner_groups,
    )
    with pytest.raises(ValueError, match="environment_design_sha256"):
        IndependentTSHCALOTrainingEnvironment.from_state_dict(
            problem,
            training,
            replace(config, recovery_fraction=0.25),
            checkpoint,
        )
    checkpoint["runtime"]["iteration"] = 99
    with pytest.raises(ValueError, match="iteration and FE count"):
        IndependentTSHCALOTrainingEnvironment.from_state_dict(problem, training, config, checkpoint)


def test_action_assignment_mismatch_fails_before_an_evaluation(toy_case):
    problem = ORPDProblem(toy_case)
    training = _training_config()
    trainer = IndependentTSHCALOTrainer(training)
    environment = IndependentTSHCALOTrainingEnvironment(problem, training, _environment_config())
    observation = environment.reset()
    action = _sample(trainer, observation)
    changed = action.learner_groups.copy()
    changed[0] = (int(changed[0]) + 1) % 3

    with pytest.raises(ValueError, match="learner groups"):
        environment.step(replace(action, learner_groups=changed))

    assert environment.candidate_evaluations == 4
    assert environment.scenario_power_flow_calls == 4


def test_experimental_population_schedule_is_rejected_before_any_solve(toy_case):
    problem = ORPDProblem(toy_case)
    calls = 0
    original = problem.evaluate_with_context

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    problem.evaluate_with_context = counted
    training = _training_config(
        feature_flags=TSHCALOFeatureFlags(
            population_schedule=True, allow_experimental_components=True
        )
    )

    with pytest.raises(ValueError, match="Experimental Change F"):
        IndependentTSHCALOTrainingEnvironment(problem, training, _environment_config())

    assert calls == 0
    bounded = replace(
        _training_config(),
        resource_envelope=replace(
            _training_config().resource_envelope,
            maximum_topology_nodes=2,
        ),
    )
    with pytest.raises(MemoryError, match="before solve"):
        IndependentTSHCALOTrainingEnvironment(problem, bounded, _environment_config())
    assert calls == 0


def test_solver_failure_is_retained_and_poisoned_environment_cannot_reset(toy_case):
    problem = ORPDProblem(toy_case)

    def failed_evaluator(_values):
        raise RuntimeError("synthetic counted evaluator failure")

    problem.evaluate_with_context = failed_evaluator
    environment = IndependentTSHCALOTrainingEnvironment(
        problem, _training_config(), _environment_config()
    )

    with pytest.raises(RuntimeError, match="synthetic counted evaluator failure"):
        environment.reset()

    provenance = environment.scientific_provenance()
    assert provenance["candidate_evaluations"] == 1
    assert provenance["scenario_power_flow_calls"] == 0
    assert provenance["accounting_complete"] is False
    with pytest.raises(RuntimeError, match="retain its failure provenance"):
        environment.reset()


def test_loaded_protected_case_identity_is_rejected_even_under_alias(monkeypatch, toy_case):
    problem = ORPDProblem(toy_case)
    monkeypatch.setattr(
        "calo_rpd_studio.algorithms.calo.tsh_calo_training_environment."
        "canonical_protected_holdout_checksums",
        lambda: {"case118": problem.case.checksum()},
    )

    with pytest.raises(ValueError, match="Protected holdout case118"):
        IndependentTSHCALOTrainingEnvironment(problem, _training_config(), _environment_config())


def test_training_environment_module_has_no_experiment_or_lifecycle_authority():
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_training_environment.py").read_text(
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
