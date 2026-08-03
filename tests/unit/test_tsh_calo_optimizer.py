"""End-to-end TSH-CALO policy, accounting, fallback, and resume invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.registry import (
    create_optimizer,
    policy_gated_algorithm_names,
    primary_algorithm_names,
)
from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry
from calo_rpd_studio.algorithms.calo.tsh_calo_optimizer import (
    TSHCALOBaselineFallbackRequired,
    TSHCALOOptimizer,
    TSHCALOPolicyRejected,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import TSHCALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    IndependentTrainingProvenance,
    assemble_tsh_calo_ensemble_candidate,
    inspect_tsh_calo_candidate,
    save_tsh_calo_candidate,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_runtime_context import (
    build_runtime_topology_policy_context,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    build_tsh_calo_qualification_receipt,
    qualification_config,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import (
    OODCalibration,
    topology_ood_signature,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.results.database import ResultDatabase


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _member(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    provenance = IndependentTrainingProvenance(
        training_run_id=f"independent-{seed}",
        training_design_sha256=_sha("design"),
        source_commit="optimizer-test",
        development_cases=("case30", "case57"),
        seed_manifest_sha256=_sha("seeds"),
    )
    save_tsh_calo_candidate(path, TSHCALOPolicyNetwork(hidden_dim=16), provenance)
    return path


def _parameters(tmp_path: Path, problem: ORPDProblem, *, deterministic: bool) -> dict:
    members = [_member(tmp_path / f"member-{seed}.pt", seed) for seed in (17, 23)]
    ensemble = assemble_tsh_calo_ensemble_candidate(
        tmp_path / "ensemble.pt",
        [(path, inspect_tsh_calo_candidate(path).sha256) for path in members],
    )
    database = ResultDatabase(tmp_path / "results.sqlite")
    registry = PolicyRegistry(database)
    policy = registry.register(ensemble.path)
    _evaluation, counted = problem.evaluate_with_context(np.full(problem.dimension, 0.5))
    state = build_runtime_topology_policy_context(np.zeros(32), problem, counted).policy_state
    signature = topology_ood_signature(state)
    calibration = OODCalibration(np.zeros_like(signature), np.ones_like(signature), 100.0)
    receipt = build_tsh_calo_qualification_receipt(
        qualification_run_id="qualification-001",
        source_policy_sha256=policy.sha256,
        source_commit="optimizer-test",
        qualification_protocol_sha256=_sha("qualification-protocol"),
        seed_manifest_sha256=_sha("qualification-seeds"),
        evidence_artifact_sha256=_sha("synthetic-evidence-fixture"),
        development_cases=("case30", "case57"),
        ood_calibration=calibration,
    )
    database.add_policy_qualification(
        qualification_id="qualification-001",
        policy_id=policy.id,
        passed=True,
        grade="A",
        qualification_status="qualified",
        config=qualification_config(receipt),
    )
    registry.activate(policy.id, algorithm_id=TSH_CALO_ALGORITHM_ID)
    binding = registry.bind_to_experiment_config(
        policy.id,
        ExperimentConfig(),
        deterministic=deterministic,
        algorithm_id=TSH_CALO_ALGORITHM_ID,
    )
    binding.update(
        {
            "inference_device": "cpu",
            "deterministic_policy": deterministic,
        }
    )
    return binding


def _run(problem, parameters, *, evaluations: int, checkpoint: str = "", max_iterations: int = 10):
    values = dict(parameters)
    if checkpoint:
        values["run_checkpoint_path"] = checkpoint
        values["checkpoint_interval_evaluations"] = 4
    optimizer = TSHCALOOptimizer(
        problem,
        OptimizerConfig(
            population_size=4,
            max_evaluations=evaluations,
            max_iterations=max_iterations,
            parameters=values,
        ),
        seed=103,
    )
    return optimizer.run(), optimizer


def test_qualified_activated_ensemble_drives_counted_end_to_end_run(tmp_path, toy_case):
    problem = ORPDProblem(toy_case)
    parameters = _parameters(tmp_path, problem, deterministic=True)

    result, optimizer = _run(problem, parameters, evaluations=8)

    assert result.algorithm == TSH_CALO_ALGORITHM_ID
    assert result.evaluations == optimizer.evaluations == 8
    assert result.metadata["candidate_evaluations"] == 8
    assert result.metadata["scenario_power_flow_calls"] == 8
    assert result.metadata["device_admission"]["computation_device"] == "cpu"
    assert result.metadata["computation_semantics"]["trusted_orpd_evaluator"] == "cpu"
    assert result.metadata["physics_repair_runtime"].startswith("disabled")
    assert len(result.metadata["runtime_trajectory"]) == 1
    generation = result.metadata["runtime_trajectory"][0]
    assert len(generation["executed_operators"]) == 4
    assert len(generation["operator_probabilities"]) == 4
    assert len(generation["shield_mixture_weights"]) == 4
    assert len(generation["shield_action_mask"]) == 4
    assert generation["policy_sha256"] == parameters["policy_sha256"]


def test_registry_exposes_distinct_tsh_algorithm_without_redefining_calo(toy_case):
    problem = ORPDProblem(toy_case)
    optimizer = create_optimizer(
        TSH_CALO_ALGORITHM_ID,
        problem,
        OptimizerConfig(
            population_size=4,
            max_evaluations=8,
            parameters={"optimizer_backend": "torch"},
        ),
        seed=3,
    )

    assert isinstance(optimizer, TSHCALOOptimizer)
    assert TSH_CALO_ALGORITHM_ID in policy_gated_algorithm_names()
    assert TSH_CALO_ALGORITHM_ID not in primary_algorithm_names()
    assert "CALO" in primary_algorithm_names()


@pytest.mark.parametrize(
    ("baseline_permitted", "error"),
    [
        (False, TSHCALOPolicyRejected),
        (True, TSHCALOBaselineFallbackRequired),
    ],
)
def test_policy_preflight_blocks_before_any_power_experiment(
    tmp_path, toy_case, baseline_permitted, error
):
    problem = ORPDProblem(toy_case)
    parameters = _parameters(tmp_path, problem, deterministic=True)
    parameters["policy_sha256"] = _sha("mutated-binding")
    parameters["baseline_fallback_permitted"] = baseline_permitted
    optimizer = TSHCALOOptimizer(
        problem,
        OptimizerConfig(population_size=4, max_evaluations=8, parameters=parameters),
        seed=103,
    )

    with pytest.raises(error):
        optimizer.run()

    assert optimizer.evaluations == 0


def test_mid_batch_cancellation_records_every_completed_scenario_call(tmp_path, toy_case):
    problem = ORPDProblem(toy_case)
    parameters = _parameters(tmp_path, problem, deterministic=True)
    holder = {}
    optimizer = TSHCALOOptimizer(
        problem,
        OptimizerConfig(population_size=4, max_evaluations=8, parameters=parameters),
        seed=103,
        cancel_callback=lambda: holder["optimizer"].evaluations >= 6,
    )
    holder["optimizer"] = optimizer

    result = optimizer.run()

    assert result.evaluations == 6
    assert result.metadata["candidate_evaluations"] == 6
    assert result.metadata["scenario_power_flow_calls"] == 6


def test_experimental_population_schedule_cannot_enter_fixed_production_path(tmp_path, toy_case):
    problem = ORPDProblem(toy_case)
    parameters = _parameters(tmp_path, problem, deterministic=True)
    parameters["policy_feature_flags"] = {
        **parameters["policy_feature_flags"],
        "population_schedule": True,
        "allow_experimental_components": True,
    }
    optimizer = TSHCALOOptimizer(
        problem,
        OptimizerConfig(population_size=4, max_evaluations=8, parameters=parameters),
        seed=103,
    )

    with pytest.raises(ValueError, match="Change F"):
        optimizer.run()

    assert optimizer.evaluations == 0


def test_optimizer_has_no_training_registry_or_activation_authority():
    source = Path("calo_rpd_studio/algorithms/calo/tsh_calo_optimizer.py").read_text(
        encoding="utf-8"
    )

    assert "tsh_calo_training" not in source
    assert "PolicyRegistry" not in source
    assert ".activate(" not in source
    assert "save_tsh_calo_candidate" not in source


def test_exact_resume_matches_uninterrupted_stochastic_policy_run(tmp_path, toy_case):
    problem = ORPDProblem(toy_case)
    parameters = _parameters(tmp_path, problem, deterministic=False)
    uninterrupted, _optimizer = _run(problem, parameters, evaluations=12)

    checkpoint = str(tmp_path / "runtime.resume.pt")
    segment, _optimizer = _run(
        problem,
        parameters,
        evaluations=12,
        checkpoint=checkpoint,
        max_iterations=1,
    )
    assert segment.evaluations == 8
    resumed_parameters = dict(parameters)
    resumed_parameters["resume_run_checkpoint"] = checkpoint
    resumed, resumed_optimizer = _run(problem, resumed_parameters, evaluations=12)

    assert resumed_optimizer.evaluations == uninterrupted.evaluations == 12
    np.testing.assert_allclose(resumed.best_vector, uninterrupted.best_vector, rtol=0, atol=0)
    assert resumed.best_objective == uninterrupted.best_objective
    assert resumed.total_constraint_violation == uninterrupted.total_constraint_violation
    assert resumed.metadata["runtime_trajectory"] == uninterrupted.metadata["runtime_trajectory"]
