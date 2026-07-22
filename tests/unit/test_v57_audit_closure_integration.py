from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from calo_rpd_studio.accelerated.throughput_engine import CrossRunBatchBroker
from calo_rpd_studio.algorithms.calo.policy_qualification import (
    PolicyQualificationConfig,
    _grade,
)
from calo_rpd_studio.algorithms.calo.training import TrainingConfig
from calo_rpd_studio.orpd.objectives import ObjectiveConfig, ObjectiveKind, calculate_objective
from calo_rpd_studio.portfolio.exporter import PortfolioExporter
from calo_rpd_studio.results.comparison_engine import summarize_runs


class _FakePortfolioDatabase:
    def __init__(self, rows):
        self.rows = list(rows)

    def get_experiment(self, _experiment_id):
        return {
            "config_json": json.dumps(
                {
                    "portfolio": {
                        "require_independent_validation": True,
                        "requested_outputs": [],
                    }
                }
            )
        }

    def list_experiment_horizons(self, _experiment_id):
        return [1000]

    def list_experiment_revisions(self, _experiment_id):
        return []

    def experiment_horizon_status(self, _experiment_id, _evaluation_horizon):
        return {
            "rows": self.rows,
            "publication_eligible": True,
            "complete": True,
            "available_count": len(self.rows),
            "expected_count": len(self.rows),
            "revision": {
                "id": "r1",
                "revision_number": 1,
                "status": "completed",
                "publication_eligible": True,
            },
        }


def _portfolio_row(*, validation="unverified", feasible=True, objective=1.0):
    return {
        "id": "run-1",
        "algorithm": "CALO",
        "run_index": 0,
        "validation_status": validation,
        "result_json": json.dumps(
            {
                "feasible": feasible,
                "best_objective": objective,
                "total_constraint_violation": 0.0 if feasible else 1.0,
                "runtime_seconds": 1.0,
                "evaluations": 1000,
                "metadata": {},
            }
        ),
    }


def test_publication_export_fails_closed_when_validation_required_and_zero_verified(tmp_path):
    exporter = PortfolioExporter(_FakePortfolioDatabase([_portfolio_row(validation="unverified")]))
    with pytest.raises(ValueError, match="verified 0/1|zero verified"):
        exporter.export("exp", tmp_path, evaluation_horizon=1000)



def test_publication_export_blocks_partial_verified_subset(tmp_path):
    rows = [
        _portfolio_row(validation="verified", objective=1.0),
        {**_portfolio_row(validation="unverified", objective=1.1), "id": "run-2", "run_index": 1},
    ]
    exporter = PortfolioExporter(_FakePortfolioDatabase(rows))
    with pytest.raises(ValueError, match="verified 1/2|Partial verified subsets"):
        exporter.export("exp", tmp_path, evaluation_horizon=1000)

def test_scenario_matrix_uses_constraint_feasibility_and_total_loss_key():
    row = _portfolio_row(validation="verified")
    payload = json.loads(row["result_json"])
    payload["metadata"] = {
        "solution_state": {
            "scenarios": [
                {"converged": True, "total_constraint_violation": 0.25, "total_loss_mw": 12.5},
                {"converged": True, "total_constraint_violation": 0.0, "total_loss_mw": 10.0},
            ]
        }
    }
    row["result_json"] = json.dumps(payload)
    exporter = PortfolioExporter(None)
    _, feasible = exporter._scenario_matrix([row], "feasible")
    _, losses = exporter._scenario_matrix([row], "loss")
    assert feasible.tolist() == [[0.0, 1.0]]
    assert losses.tolist() == [[12.5, 10.0]]


def test_comparison_summary_excludes_unverified_and_infeasible_objectives():
    rows = [
        _portfolio_row(validation="verified", feasible=True, objective=5.0),
        {**_portfolio_row(validation="unverified", feasible=True, objective=1.0), "id": "run-2"},
        {**_portfolio_row(validation="verified", feasible=False, objective=0.5), "id": "run-3"},
    ]
    summary = summarize_runs(rows)
    assert summary.loc["CALO", "median"] == pytest.approx(5.0)
    assert int(summary.loc["CALO", "eligible_objective_runs"]) == 1


def test_voltage_deviation_uses_original_formulation_partition(toy_case):
    # Original bus 2 is PV and bus 3 is PQ.  Simulate Q-limit switching bus 2 to PQ in the solved case.
    formulation = toy_case.clone()
    solved = toy_case.clone()
    solved.bus[1, 1] = 1  # dynamic PV->PQ conversion in numerical solve

    class _PF:
        converged = True
        case = solved
        total_loss_mw = 1.0
        vm_pu = np.asarray([1.04, 0.70, 1.02], dtype=float)
        voltage = vm_pu.astype(complex)

    result = calculate_objective(
        _PF(),
        ObjectiveConfig(kind=ObjectiveKind.VOLTAGE_DEVIATION),
        formulation_case=formulation,
    )
    # Only the originally declared PQ/load bus contributes; switched bus 2 must not redefine the objective.
    assert result.components["voltage_deviation_pu"] == pytest.approx(0.02)


def test_cross_run_broker_propagates_worker_evaluator_failure_without_hanging():
    class BadEvaluator:
        batch_signature = "same-science"

        def _evaluate_population_direct(self, _candidates):
            raise RuntimeError("forced evaluator failure")

    broker = CrossRunBatchBroker(batch_window_ms=0.1)
    try:
        with pytest.raises(RuntimeError, match="forced evaluator failure"):
            broker.submit(BadEvaluator(), np.zeros((2, 3), dtype=float))
    finally:
        broker.close()


def test_formal_periodic_qualification_interval_is_retired_by_design():
    cfg = TrainingConfig()
    assert cfg.qualification_interval_epochs == 0


def test_policy_grade_handles_nonfinite_comparator_medians_without_min_empty_failure():
    cfg = PolicyQualificationConfig(
        cases=("case30",),
        runs=30,
        minimum_promotion_runs=30,
        require_independent_validation=False,
    )
    candidate = {"feasible_probability": 1.0, "median_auc": 1.0}
    no_ai = {"feasible_probability": 0.0, "median_auc": float("inf")}
    case_summaries = {
        "candidate": {"case30": {"median_objective": 10.0}},
        "no_ai": {"case30": {"median_objective": float("nan")}},
    }
    passed, grade, _score, reasons = _grade(candidate, None, no_ai, cfg, {}, case_summaries)
    # v5.8 keeps the original edge-case guarantee (no min(empty) crash) while correctly refusing
    # formal superiority promotion when there is no complete paired statistical evidence.
    assert passed is False
    assert grade == "U"
    assert any("superiority" in reason.lower() for reason in reasons)


def test_no_broad_exception_pass_remains_in_scientific_source_tree():
    root = Path(__file__).resolve().parents[2] / "calo_rpd_studio"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not isinstance(node.type, ast.Name):
                continue
            if node.type.id not in {"Exception", "BaseException"}:
                continue
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_release_source_does_not_bundle_generated_publication_export_or_v500_default_names():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "publication_export").exists()
    script = (root / "calo_rpd_studio" / "scripts" / "run_final_benchmark.py").read_text(encoding="utf-8")
    panel = (root / "calo_rpd_studio" / "gui" / "panels" / "benchmark_campaign_panel.py").read_text(encoding="utf-8")
    assert "benchmark_v500" not in script
    assert "benchmark_v500" not in panel
