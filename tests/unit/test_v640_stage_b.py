from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.device_resident_synthetic import (
    DeviceResidentCurriculumProblem,
    SyntheticCrossEpisodeBatchBroker,
)
from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
    _environment_for_episode,
)
from calo_rpd_studio.algorithms.calo.training import CurriculumProblem, SyntheticCALOEnvironment
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.objectives import ObjectiveKind


def test_device_resident_curriculum_matches_numpy_reference_all_stages():
    for stage in range(4):
        rng = np.random.default_rng(9000 + stage)
        reference = CurriculumProblem(rng, stage)
        population = rng.random((23, reference.dimension))
        wrapped = DeviceResidentCurriculumProblem(
            reference,
            device="cpu",
            require_startup_parity=True,
            parity_tolerance=1e-9,
        )
        accelerated = wrapped.evaluate_population(population)
        expected = [reference.evaluate(candidate) for candidate in population]
        assert wrapped.parity_verified
        assert wrapped.parity_max_error < 1e-10
        for a, b in zip(expected, accelerated):
            assert a.feasible == b.feasible
            assert a.value == pytest.approx(b.value, rel=1e-12, abs=1e-12)
            assert a.violation == pytest.approx(b.violation, rel=1e-12, abs=1e-12)
            assert a.metadata["constraint_components"] == pytest.approx(
                b.metadata["constraint_components"], rel=1e-12, abs=1e-12
            )


def test_cross_episode_broker_merges_heterogeneous_dimensions():
    with SyntheticCrossEpisodeBatchBroker(
        device="cpu", batch_window_ms=10.0, max_candidates=4096
    ) as broker:
        wrapped = []
        populations = []
        for stage in range(4):
            rng = np.random.default_rng(100 + stage)
            reference = CurriculumProblem(rng, stage)
            wrapped.append(
                DeviceResidentCurriculumProblem(
                    reference,
                    device="cpu",
                    broker=broker,
                    require_startup_parity=True,
                )
            )
            populations.append(rng.random((20, reference.dimension)))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(problem.evaluate_population, population)
                for problem, population in zip(wrapped, populations)
            ]
            [future.result() for future in futures]
        metrics = broker.metrics()
        assert metrics["request_count"] == 4
        assert metrics["candidate_count"] == 80
        assert metrics["max_batch_candidates"] == 80
        assert metrics["batch_count"] == 1


def test_stage_b_wrapper_does_not_change_curriculum_rng_or_initial_population():
    seed = 123456
    rng_reference = np.random.default_rng(seed)
    env_reference = SyntheticCALOEnvironment(rng_reference, 3, 20)

    rng_wrapped = np.random.default_rng(seed)
    problem = CurriculumProblem(rng_wrapped, 3)
    wrapped_problem = DeviceResidentCurriculumProblem(
        problem, device="cpu", require_startup_parity=True
    )
    env_wrapped = SyntheticCALOEnvironment(rng_wrapped, 3, 20, problem=wrapped_problem)

    assert np.array_equal(env_reference.population, env_wrapped.population)
    assert np.array_equal(env_reference.personal_best, env_wrapped.personal_best)
    for a, b in zip(env_reference.evaluations, env_wrapped.evaluations):
        assert a.value == pytest.approx(b.value, rel=1e-12, abs=1e-12)
        assert a.violation == pytest.approx(b.violation, rel=1e-12, abs=1e-12)
        assert a.feasible == b.feasible


def test_heterogeneous_stage_b_defaults_are_fail_closed_and_enabled():
    config = HeterogeneousTrainingConfig()
    assert config.device_resident_synthetic_rollouts is True
    assert config.synthetic_cross_episode_batching is True
    assert config.require_synthetic_startup_parity is True
    assert config.synthetic_parity_tolerance <= 1e-9


def test_heterogeneous_real_orpd_environment_uses_declared_experiment_formulation(
    tmp_path, toy_case, monkeypatch
):
    # The heterogeneous actor path used to construct AcceleratedORPDProblem(case) with defaults.
    # v6.4 must preserve the exact ExperimentConfig formulation, matching the CPU training path.
    experiment = ExperimentConfig()
    experiment.case_name = "toy3"
    experiment.objective.kind = ObjectiveKind.VOLTAGE_DEVIATION
    experiment.objective.weight_loss = 0.0
    experiment.objective.weight_voltage_deviation = 1.0
    experiment.power_flow.tolerance = 2.5e-7
    config_path = tmp_path / "development.yaml"
    experiment.save(config_path)

    from calo_rpd_studio.power_system.case_loader import CaseLoader

    monkeypatch.setattr(CaseLoader, "load", classmethod(lambda cls, source: toy_case.clone()))
    config = HeterogeneousTrainingConfig(
        development_cases=("toy3",),
        development_experiment_config_path=str(config_path),
        use_accelerated_orpd_rollouts=False,
        population_size=6,
    )
    _seed, environment = _environment_for_episode(
        config,
        epoch=20,
        stage=4,
        episode=0,
        compute_device="cpu",
    )
    assert environment.problem.config.objective.kind is ObjectiveKind.VOLTAGE_DEVIATION
    assert environment.problem.config.power_flow.tolerance == pytest.approx(2.5e-7)


def test_stage_b_gui_exposes_real_development_suite_and_no_hardcoded_empty_cases():
    source = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "gui"
        / "panels"
        / "calo_intelligence_panel.py"
    ).read_text(encoding="utf-8")
    assert "device_resident_synthetic" in source
    assert "Development cases" in source
    assert "development_experiment_config_path=development_config_path" in source
    assert "development_cases=()," not in source


def test_stage_b_multi_transition_trajectory_matches_reference_on_torch_cpu():
    seed = 778899
    rng_a = np.random.default_rng(seed)
    env_a = SyntheticCALOEnvironment(rng_a, 3, 20)

    rng_b = np.random.default_rng(seed)
    ref_problem = CurriculumProblem(rng_b, 3)
    wrapped = DeviceResidentCurriculumProblem(
        ref_problem,
        device="cpu",
        require_startup_parity=True,
        parity_recheck_interval=1,
    )
    env_b = SyntheticCALOEnvironment(rng_b, 3, 20, problem=wrapped)

    actions = [
        (0, 0, np.asarray([0.2, 0.4, 0.3, 0.6, 0.5, 0.2])),
        (1, 2, np.asarray([0.7, 0.2, 0.8, 0.4, 0.3, 0.6])),
        (2, 3, np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])),
        (3, 4, np.asarray([0.3, 0.6, 0.4, 0.7, 0.2, 0.8])),
        (1, 1, np.asarray([0.8, 0.3, 0.2, 0.6, 0.7, 0.1])),
    ]
    for regime, operator, params in actions:
        state_a = env_a.policy_state(12)
        state_b = env_b.policy_state(12)
        assert state_a == pytest.approx(state_b, rel=1e-12, abs=1e-12)
        reward_a = env_a.step(regime, operator, params, 12)
        reward_b = env_b.step(regime, operator, params, 12)
        assert reward_a == pytest.approx(reward_b, rel=1e-10, abs=1e-10)
        assert env_a.population == pytest.approx(env_b.population, rel=1e-10, abs=1e-10)
        assert env_a.last_step_trace["executed_operators"] == env_b.last_step_trace["executed_operators"]
        assert env_a.last_step_trace["forced_recovery_indices"] == env_b.last_step_trace["forced_recovery_indices"]


def test_stage_b_host_controller_thread_pool_is_capped_by_protected_rollout_budget():
    source = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "algorithms"
        / "calo"
        / "heterogeneous_training.py"
    ).read_text(encoding="utf-8")
    assert "step_workers = max(1, min(len(environments), int(config.rollout_workers)))" in source
    assert "ThreadPoolExecutor(max_workers=step_workers)" in source
