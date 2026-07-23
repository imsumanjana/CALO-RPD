from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from calo_rpd_studio.algorithms.base_optimizer import BaseOptimizer, OptimizerConfig
from calo_rpd_studio.algorithms.calo.evaluation_cache import ExactEvaluationCache
from calo_rpd_studio.algorithms.calo.policy_qualification import PolicyQualificationConfig, _apply_holm, _grade
from calo_rpd_studio.algorithms.calo.competitive_training import recover_competitive_session
from calo_rpd_studio.experiments.experiment_config import RobustScenarioSettings
from calo_rpd_studio.orpd.constraints import ConstraintToleranceConfig, branch_angle_limit_violation
from calo_rpd_studio.orpd.objectives import ObjectiveConfig, ObjectiveKind
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableDecoder, ORPDVariableConfig
from calo_rpd_studio.power_system.ac_power_flow import PowerFlowOptions
from calo_rpd_studio.power_system.case_model import ANGMAX, ANGMIN, BUS_TYPE, GEN_BUS, GEN_STATUS, PV, REF
from calo_rpd_studio.power_system.ybus import build_ybus
from calo_rpd_studio.robustness.contingencies import n_minus_one_branch_scenarios, n_minus_one_generator_scenarios
from calo_rpd_studio.robustness.robust_objectives import RobustObjectiveConfig
from calo_rpd_studio.robustness.scenario import Scenario


def test_invalid_objective_config_fails_fast():
    with pytest.raises(ValueError):
        ObjectiveConfig(ObjectiveKind.MULTI_OBJECTIVE, weight_loss=-1.0, weight_voltage_deviation=1.0)
    with pytest.raises(ValueError):
        ObjectiveConfig(ObjectiveKind.MULTI_OBJECTIVE, weight_loss=1.0, loss_scale=0.0)
    with pytest.raises(ValueError):
        ObjectiveConfig(ObjectiveKind.MULTI_OBJECTIVE, weight_loss=float("nan"), loss_scale=1.0)


def test_invalid_power_flow_and_robust_configs_fail_fast():
    with pytest.raises(ValueError):
        PowerFlowOptions(tolerance=0.0)
    with pytest.raises(ValueError):
        PowerFlowOptions(q_limit_tolerance_mvar=float("nan"))
    with pytest.raises(ValueError):
        RobustObjectiveConfig(risk_lambda=float("nan"))
    with pytest.raises(ValueError):
        RobustScenarioSettings(mode="load_uncertainty", active_load_std=float("nan")).validate()


def test_n_minus_one_bundles_include_intact_base_by_default():
    branch = n_minus_one_branch_scenarios([0, 2])
    generator = n_minus_one_generator_scenarios([1])
    assert [s.name for s in branch] == ["base", "branch_out_0", "branch_out_2"]
    assert [s.name for s in generator] == ["base", "generator_out_1"]


def test_branch_angle_limits_are_enforced_with_matpower_semantics(toy_case):
    case = toy_case.clone()
    case.branch[0, ANGMIN] = -5.0
    case.branch[0, ANGMAX] = 5.0
    # angle(Vf)-angle(Vt) = 0-(-10)=+10 deg => 5 deg high-side violation.
    violation = branch_angle_limit_violation(
        case,
        np.asarray([0.0, -10.0, 0.0]),
        ConstraintToleranceConfig(branch_angle_deg=1e-9),
    )
    assert violation > 0.0
    case.branch[0, ANGMIN] = -360.0
    case.branch[0, ANGMAX] = 360.0
    assert branch_angle_limit_violation(case, np.asarray([0.0, -10.0, 0.0])) == 0.0


def test_generator_voltage_controls_exclude_pq_bus_generators(toy_case):
    case = toy_case.clone()
    # Add an online generator row at the PQ bus. Its VG is not an ORPD voltage-control degree of freedom.
    extra = case.gen[0].copy()
    extra[GEN_BUS] = 3
    extra[GEN_STATUS] = 1
    case.gen = np.vstack([case.gen, extra])
    decoder = ORPDVariableDecoder(case, ORPDVariableConfig(transformer_taps=False, shunt_compensation=False))
    voltage_names = [v.name for v in decoder.variables if v.name.startswith("Vg@")]
    assert "Vg@1" in voltage_names and "Vg@2" in voltage_names
    assert "Vg@3" not in voltage_names
    assert {int(row[0]): int(row[BUS_TYPE]) for row in case.bus}[1] == REF
    assert {int(row[0]): int(row[BUS_TYPE]) for row in case.bus}[2] == PV


def test_yf_yt_are_direct_sparse_branch_incidence_matrices(toy_case):
    y = build_ybus(toy_case)
    assert hasattr(y.y_from, "nnz") and hasattr(y.y_to, "nnz")
    active = int(np.count_nonzero(toy_case.branch[:, 10] > 0))
    assert y.y_from.nnz <= 2 * active
    assert y.y_to.nnz <= 2 * active


class _BoxProblem:
    dimension = 2
    def evaluate(self, x):
        x = np.asarray(x, float)
        return Evaluation(float(np.sum(x * x)), True, 0.0, {})


class _BoxOptimizer(BaseOptimizer):
    name = "BOX"
    def run(self):  # pragma: no cover
        raise NotImplementedError


def test_repair_telemetry_is_identical_with_and_without_exact_cache():
    pop = np.asarray([[-0.2, 0.4], [0.5, 1.3]], dtype=float)
    direct = _BoxOptimizer(_BoxProblem(), OptimizerConfig(2, 2, 1, {}), seed=1)
    direct.evaluate_population(pop)
    cached = _BoxOptimizer(_BoxProblem(), OptimizerConfig(2, 2, 1, {}), seed=1)
    cache = ExactEvaluationCache(_BoxProblem(), capacity=16, adaptive=False)
    cache.evaluate_requests(cached, pop)
    assert direct.repair_candidate_count == cached.repair_candidate_count == 2
    assert direct.repair_coordinate_count == cached.repair_coordinate_count == 2
    assert direct.repair_total_coordinates == cached.repair_total_coordinates == 4


def test_stale_competitive_recovery_refuses_newer_authority(tmp_path):
    output = tmp_path / "policy.pt"
    manifest = output.with_suffix(".branches.json")
    old = {"schema_version": 3, "generation_id": "G1", "branches": []}
    manifest.write_text(json.dumps(old), encoding="utf-8")
    old_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    recovery_dir = tmp_path / "policy_branches" / "recovery"
    recovery_dir.mkdir(parents=True)
    session = "stale-session"
    (recovery_dir / f"{session}.json").write_text(
        json.dumps({
            "session_id": session,
            "output_path": str(output),
            "scratch_root": str(tmp_path / "scratch"),
            "prior_manifest_sha256": old_sha,
            "prior_generation_id": "G1",
            "latest_common_safe_epoch": 0,
            "branches": [],
        }),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"schema_version": 3, "generation_id": "G2", "branches": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stale competitive recovery refused"):
        recover_competitive_session(output, session)


def test_noninferiority_uses_one_sided_holm_adjusted_evidence():
    paired = {
        "no_ai": {
            "wilcoxon_p_two_sided": 1.0,
            "paired_relative_differences": [-0.03] * 12,
        },
        "reference": {
            "wilcoxon_p_two_sided": 1.0,
            "paired_relative_differences": [-0.02] * 12,
        },
    }
    out = _apply_holm(paired, non_inferiority_margin=0.01)
    assert all(np.isfinite(v["holm_noninferiority_p"]) for v in out.values())
    assert all(v["holm_noninferiority_p"] <= 0.05 for v in out.values())


def test_native_v59_training_transition_matches_deployed_calo_one_step(tmp_path):
    """One seeded native-policy transition must match deployed CALO exactly."""
    import torch

    from calo_rpd_studio.ai.model_io import checkpoint_sha256, load_trusted_resume
    from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
    from calo_rpd_studio.algorithms.calo.ai_controller import AIController
    from calo_rpd_studio.algorithms.calo.optimizer import CALOOptimizer
    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.policy_schema import (
        POLICY_ACTION_SCHEMA,
        POLICY_STATE_DIM,
        POLICY_STATE_SCHEMA,
    )
    from calo_rpd_studio.algorithms.calo.training import (
        SyntheticCALOEnvironment,
        TrainingConfig,
        save_deployable_policy_snapshot,
    )
    from calo_rpd_studio.orpd.problem import Evaluation

    class _Case:
        name = "parity_sphere"

        @staticmethod
        def checksum():
            return "parity-sphere-v59"

    class _Sphere:
        dimension = 4
        variables = []
        case = _Case()

        @staticmethod
        def evaluate(x):
            vector = np.asarray(x, dtype=float)
            value = float(np.sum((vector - 0.25) ** 2))
            components = {
                "bus_voltage": 0.0,
                "generator_q": 0.0,
                "generator_p": 0.0,
                "branch_thermal": 0.0,
                "power_flow": 0.0,
            }
            return Evaluation(
                value,
                True,
                0.0,
                {},
                {"constraint_components": components},
            )

        @classmethod
        def evaluate_population(cls, population):
            return [cls.evaluate(row) for row in population]

        @staticmethod
        def solution_state(x):
            return {"normalized_decision_vector": np.asarray(x).tolist(), "scenarios": []}

    torch.manual_seed(42)
    network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=16)
    training_config = TrainingConfig(
        epochs=1,
        episodes_per_epoch=1,
        horizon=3,
        population_size=4,
        hidden_dim=16,
        ppo_epochs=1,
        minibatch_size=4,
    )
    artifact = save_deployable_policy_snapshot(
        tmp_path / "policy.pt",
        network,
        training_config,
        [],
        {},
        1,
        device="cpu",
        rollout_workers=1,
    )
    digest = checkpoint_sha256(artifact)
    seed = 17
    environment = SyntheticCALOEnvironment(
        np.random.default_rng(seed), 0, 4, problem=_Sphere()
    )
    controller = AIController(
        artifact,
        seed=seed + 7919,
        deterministic=True,
        device="cpu",
        expected_checksum=digest,
        expected_state_schema=POLICY_STATE_SCHEMA,
        expected_action_schema=POLICY_ACTION_SCHEMA,
    )
    training_rewards = []
    for _ in range(3):
        decision = controller.decide(environment.state(3), environment._last_context)
        training_rewards.append(
            environment.step(decision.regime, decision.operator, decision.raw_parameter_action, 3)
        )
    training_reward = training_rewards[-1]

    resume = tmp_path / "runtime.resume.pt"
    parameters = {
        "use_ai": True,
        "strict_policy_binding": True,
        "policy_checkpoint": str(artifact),
        "policy_sha256": digest,
        "policy_state_schema_version": POLICY_STATE_SCHEMA,
        "policy_action_schema_version": POLICY_ACTION_SCHEMA,
        "deterministic_policy": True,
        "use_exact_evaluation_cache": False,
        "run_checkpoint_path": str(resume),
        "checkpoint_interval_evaluations": 16,
        "ai_inference_seed": seed + 7919,
    }
    optimizer = CALOOptimizer(
        _Sphere(),
        OptimizerConfig(
            population_size=4,
            max_evaluations=16,
            max_iterations=3,
            parameters=parameters,
        ),
        seed=seed,
    )
    optimizer.run()
    runtime = load_trusted_resume(resume)["runtime_state"]
    runtime_state = runtime["state"]

    np.testing.assert_array_equal(environment.population, runtime_state.population)
    np.testing.assert_array_equal(environment.personal_best, runtime_state.personal_best)
    assert environment.rng.bit_generator.state == optimizer.rng.bit_generator.state
    assert environment.epsilon_controller.current == runtime["epsilon_controller"].current
    np.testing.assert_array_equal(
        environment.credit.operator_credit, runtime["credit"].operator_credit
    )
    np.testing.assert_array_equal(
        environment.credit.memory_credit, runtime["credit"].memory_credit
    )
    assert training_reward == runtime["reward_history"][-1]
    assert environment.last_step_trace["executed_operators"] == runtime["policy_trajectory"][-1][
        "executed_controller"
    ]["executed_operators"]



def test_scenario_transform_rejects_structural_identity_changes(toy_case):
    def change_bus_identity(case):
        case.bus[0, 0] = 999
        return case

    def change_dimensions(case):
        case.bus = case.bus[:-1].copy()
        return case

    def introduce_nonfinite(case):
        case.branch[0, 2] = np.nan
        return case

    with pytest.raises(ValueError, match="bus identities/order"):
        Scenario("bad_bus_identity", transform=change_bus_identity).apply(toy_case)
    with pytest.raises(ValueError, match="changed bus matrix dimensions"):
        Scenario("bad_dimensions", transform=change_dimensions).apply(toy_case)
    with pytest.raises(ValueError, match="non-finite branch data"):
        Scenario("bad_nonfinite", transform=introduce_nonfinite).apply(toy_case)


def test_variable_config_fails_fast_for_invalid_tap_science():
    with pytest.raises(ValueError, match="Transformer tap bounds"):
        ORPDVariableConfig(transformer_minimum=1.10, transformer_maximum=0.90)
    with pytest.raises(ValueError, match="Transformer tap step"):
        ORPDVariableConfig(transformer_step=0.0)
    with pytest.raises(ValueError, match="formulation_profile"):
        ORPDVariableConfig(formulation_profile="")



def test_policy_qualification_scientific_thresholds_fail_fast():
    with pytest.raises(ValueError, match="minimum_feasible_probability"):
        PolicyQualificationConfig(minimum_feasible_probability=float("nan")).validate()
    with pytest.raises(ValueError, match="minimum_rank_biserial"):
        PolicyQualificationConfig(minimum_rank_biserial=2.0).validate()
    with pytest.raises(ValueError, match="non_inferiority_margin"):
        PolicyQualificationConfig(non_inferiority_margin=-0.01).validate()



def test_formal_qualification_requires_each_case_to_pass_paired_gate():
    config = PolicyQualificationConfig(cases=("case30", "case57"), runs=30, minimum_promotion_runs=30)
    config.validate()
    candidate = {
        "feasible_probability": 1.0,
        "independent_validation_probability": 1.0,
        "median_auc": 1.0,
    }
    comparator = {
        "feasible_probability": 1.0,
        "independent_validation_probability": 1.0,
        "median_auc": 1.1,
    }
    case_summaries = {
        "candidate": {"case30": {"median_objective": 0.9}, "case57": {"median_objective": 0.9}},
        "no_ai": {"case30": {"median_objective": 1.0}, "case57": {"median_objective": 1.0}},
    }
    favorable = {
        "n_pairs": 30, "median_difference": -0.1, "win_rate": 0.8,
        "rank_biserial": 0.6, "holm_p": 0.01, "holm_noninferiority_p": 0.01,
    }
    bad_case = dict(favorable, median_difference=0.02, win_rate=0.4, rank_biserial=-0.2, holm_p=0.5)
    aggregate = {"vs_no_ai": favorable}
    case_paired = {"vs_no_ai::case30": favorable, "vs_no_ai::case57": bad_case}
    passed, _grade_name, _score, reasons = _grade(
        candidate, None, comparator, config, aggregate, case_summaries, case_paired
    )
    assert passed is False
    assert any("formal superiority promotion requires" in reason for reason in reasons)
