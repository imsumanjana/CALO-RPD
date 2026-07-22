from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.ai.model_io import (
    checkpoint_sha256,
    durable_torch_save,
    load_trusted_resume,
    trusted_resume_sha_path,
    write_trusted_resume_hash,
)
from calo_rpd_studio.algorithms.calo.optimizer import CALOOptimizer
from calo_rpd_studio.algorithms.calo.policy_qualification import PolicyQualificationConfig, _convergence_auc
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.constraints import generator_limit_violation
from calo_rpd_studio.power_system.case_model import PowerSystemCase
from calo_rpd_studio.power_system.case_model import PG, PMAX, PMIN, QG, QMAX, QMIN
from calo_rpd_studio.robustness.robust_objectives import (
    ConstraintAggregation,
    RobustObjectiveConfig,
    aggregate_constraint_violation,
    normalize_scenario_weights,
)
from calo_rpd_studio.robustness.scenario import Scenario
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.statistics.effect_sizes import cliffs_delta
from calo_rpd_studio.statistics.confidence_intervals import mean_confidence_interval


def test_robust_feasibility_defaults_to_all_scenario_max_and_cannot_dilute_low_weight_violation():
    cfg = RobustObjectiveConfig()
    assert cfg.constraint_aggregation is ConstraintAggregation.ALL_SCENARIO_MAX
    assert aggregate_constraint_violation([0.0, 10.0], [0.999, 0.001], cfg) == pytest.approx(10.0)


def test_low_level_scenario_weights_reject_negative_nan_and_zero_sum():
    for weights in ([1.0, -0.1], [float("nan"), 1.0], [0.0, 0.0]):
        with pytest.raises(ValueError):
            normalize_scenario_weights(weights)


def test_scenario_is_immutable_and_validates_weight():
    scenario = Scenario("base", 1.0)
    with pytest.raises(FrozenInstanceError):
        scenario.weight = 2.0
    with pytest.raises(ValueError):
        Scenario("bad", -1.0)


def test_callable_compatibility_fingerprint_includes_closure_values_and_code():
    def make_transform(scale):
        def transform(case):
            case.bus[:, 2] *= scale
            return case
        return transform
    a = CALOOptimizer._compatibility_jsonable(make_transform(0.95))
    b = CALOOptimizer._compatibility_jsonable(make_transform(1.05))
    assert a != b
    assert a["closure"] != b["closure"]
    assert a["code_identity_sha256"] == b["code_identity_sha256"]


def test_generator_limits_are_enforced_per_unit_not_aggregated_by_bus():
    # Two co-located online units. One violates PMAX/QMAX while the second has spare capacity.
    first = np.zeros(10, dtype=float)
    first[0] = 1.0
    first[7] = 1.0
    second = first.copy()
    first[PMIN], first[PMAX], first[PG] = 0.0, 50.0, 60.0
    first[QMIN], first[QMAX], first[QG] = -10.0, 10.0, 20.0
    second[PMIN], second[PMAX], second[PG] = 0.0, 100.0, 0.0
    second[QMIN], second[QMAX], second[QG] = -100.0, 100.0, 0.0
    case = PowerSystemCase(
        "two_gen", 100.0,
        bus=np.zeros((1, 13), dtype=float),
        gen=np.vstack([first, second]),
        branch=np.zeros((0, 13), dtype=float),
    )
    qv, pv = generator_limit_violation(case)
    assert qv > 0.0
    assert pv > 0.0


def test_equal_fe_budget_must_be_divisible_by_population():
    cfg = ExperimentConfig()
    cfg.population_size = 40
    cfg.budget.max_evaluations = 1001
    with pytest.raises(ValueError, match="divisible"):
        cfg.validate()


def test_policy_qualification_defaults_are_formal_not_screening_and_auc_penalizes_no_feasibility():
    cfg = PolicyQualificationConfig()
    assert cfg.runs >= 30
    assert cfg.minimum_promotion_runs >= 30

    class Result:
        feasible = False
        best_objective = float("inf")
        evaluations = 1000
        metadata = {
            "convergence_evaluations": [100, 200, 500, 1000],
            "best_feasible_objective_history": [float("inf")] * 4,
        }
    assert np.isinf(_convergence_auc(Result()))


def test_policy_suppression_is_project_database_scoped(tmp_path):
    sha = "a" * 64
    a = ResultDatabase(tmp_path / "a.sqlite")
    b = ResultDatabase(tmp_path / "b.sqlite")
    a.suppress_policy_sha256(sha, reason="test")
    assert sha in a.list_suppressed_policy_sha256()
    assert sha not in b.list_suppressed_policy_sha256()


def test_exact_resume_requires_authenticated_local_sidecar_not_self_asserted_hash(tmp_path):
    path = tmp_path / "resume.pt"
    durable_torch_save({"state": {"epoch": 10}}, path)
    write_trusted_resume_hash(path)
    loaded = load_trusted_resume(path)
    assert loaded["state"]["epoch"] == 10

    # A downloaded pickle plus its own bare SHA is not authentication and must be rejected.
    trusted_resume_sha_path(path).write_text(checkpoint_sha256(path) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not trusted|Legacy|unauthenticated"):
        load_trusted_resume(path)


def test_generic_stats_guards_handle_empty_and_nonfinite_inputs():
    assert np.isnan(cliffs_delta([], []))
    lo, hi = mean_confidence_interval([np.nan, np.inf])
    assert np.isnan(lo) and np.isnan(hi)
