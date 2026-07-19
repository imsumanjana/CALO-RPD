from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
    plan_training_lanes,
)
from calo_rpd_studio.algorithms.registry import SPECS, create_optimizer
from calo_rpd_studio.benchmarking.campaign import BenchmarkCampaignConfig, build_campaign, write_campaign_plan
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.results.publication_export import PublicationExporter


@pytest.mark.parametrize("runs", [30, 31, 35, 50])
def test_campaign_preserves_requested_run_count_exactly(runs):
    campaign = BenchmarkCampaignConfig(
        cases=("case30",), study_keys=("deterministic",), runs=runs
    )
    task = build_campaign(campaign, base_config=ExperimentConfig(), verify_freeze=False)[0]
    assert task.config.runs == runs
    assert task.config.portfolio.custom_runs == runs
    assert task.planned_jobs == runs * len(SPECS)


def test_campaign_plan_persists_exact_formulation_manifest(tmp_path):
    campaign = BenchmarkCampaignConfig(
        cases=("case118",), study_keys=("deterministic",), runs=30
    )
    tasks = build_campaign(campaign, verify_freeze=False)
    path = write_campaign_plan(campaign, tasks, tmp_path / "campaign.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload["tasks"][0]["formulation_manifest"]
    assert manifest["case_name"] == "case118"
    assert manifest["dimension"] == 75
    assert {row["bus_number"] for row in manifest["fixed_shunts"]} >= {5, 37}


def _insert_verified_run(db, experiment_id, *, run_id, objective, feasible, violation):
    result = {
        "algorithm": "CALO",
        "seed": 1,
        "parameters": {},
        "best_vector": [0.5],
        "decoded_controls": {},
        "best_objective": objective,
        "objective_components": {},
        "total_constraint_violation": violation,
        "feasible": feasible,
        "evaluations": 10,
        "iterations": 1,
        "convergence_history": [objective],
        "runtime_seconds": 0.1,
        "termination_reason": "budget",
        "metadata": {},
    }
    with db.connect() as con:
        con.execute(
            "INSERT INTO runs(id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path,validation_status) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, experiment_id, "CALO", 0, "{}", json.dumps(result), "", "verified"),
        )


def test_publication_export_excludes_infeasible_objective_from_statistics(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    experiment_id = db.create_experiment(ExperimentConfig(), collect_provenance())
    _insert_verified_run(
        db, experiment_id, run_id=str(uuid.uuid4()), objective=1.0, feasible=True, violation=0.0
    )
    _insert_verified_run(
        db, experiment_id, run_id=str(uuid.uuid4()), objective=-999.0, feasible=False, violation=0.2
    )
    output = PublicationExporter(db).export(experiment_id, tmp_path / "publication")
    stats = pd.read_csv(output / "descriptive_statistics_verified_feasible.csv")
    assert stats.loc[0, "verified_runs"] == 2
    assert stats.loc[0, "verified_feasible_runs"] == 1
    assert stats.loc[0, "objective_best"] == pytest.approx(1.0)
    metadata = json.loads((output / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert metadata["objective_statistics_basis"].startswith("independently verified AND feasible")




def test_gpu_maximum_training_defaults_and_accelerator_fallback_order():
    config = HeterogeneousTrainingConfig()
    assert (
        config.cuda_rollout_share,
        config.xpu_rollout_share,
        config.cpu_rollout_share,
    ) == (100, 0, 0)
    cuda_plan = plan_training_lanes(12, cuda_available=True, xpu_available=True)
    assert cuda_plan.episode_counts == {"cuda": 12, "xpu": 0, "cpu": 0}
    xpu_plan = plan_training_lanes(12, cuda_available=False, xpu_available=True)
    assert xpu_plan.episode_counts == {"cuda": 0, "xpu": 12, "cpu": 0}
    cpu_plan = plan_training_lanes(12, cuda_available=False, xpu_available=False)
    assert cpu_plan.episode_counts == {"cuda": 0, "xpu": 0, "cpu": 12}

class SphereProblem:
    dimension = 4

    def evaluate(self, x):
        vector = np.asarray(x, dtype=float)
        value = float(np.sum((vector - 0.25) ** 2))
        return Evaluation(value, True, 0.0, {"sphere": value}, {})

    def evaluate_population(self, population):
        return [self.evaluate(row) for row in population]

    def solution_state(self, x):
        return {"normalized_decision_vector": np.asarray(x).tolist(), "scenarios": []}


_OPTIMIZER_SNAPSHOTS = json.loads(
    (Path(__file__).parents[1] / "data" / "v4_optimizer_seed_snapshots.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("name", list(SPECS))
def test_every_optimizer_matches_seeded_release_snapshot_and_budget(name):
    parameters = dict(SPECS[name].default_parameters)
    if name == "CALO":
        parameters["use_ai"] = False
    config = OptimizerConfig(8, 32, 32, parameters)
    result = create_optimizer(name, SphereProblem(), config, seed=340).run()
    expected = _OPTIMIZER_SNAPSHOTS[name]
    assert result.evaluations == expected["evaluations"] <= 32
    assert result.best_objective == pytest.approx(expected["best_objective"], rel=0, abs=1e-14)
    np.testing.assert_allclose(result.best_vector, expected["best_vector"], rtol=0, atol=1e-14)
    assert np.all(np.asarray(result.best_vector) >= 0.0)
    assert np.all(np.asarray(result.best_vector) <= 1.0)
