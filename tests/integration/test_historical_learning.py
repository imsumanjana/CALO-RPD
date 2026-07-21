from __future__ import annotations

import json
import uuid

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.learning.experience_repository import (
    build_experience_repository,
    load_experience_repository,
)
from calo_rpd_studio.results.database import ResultDatabase


def _insert_run(
    db: ResultDatabase, experiment_id: str, *, algorithm="CALO", verified=True, trajectory=True
):
    result = {
        "algorithm": algorithm,
        "seed": 7,
        "parameters": {"epsilon_quantile": 0.7, "ai_credit_blend": 0.6},
        "best_vector": [0.1, 0.2, 0.3],
        "decoded_controls": {},
        "best_objective": 4.2,
        "objective_components": {},
        "total_constraint_violation": 0.0,
        "feasible": True,
        "evaluations": 100,
        "iterations": 4,
        "convergence_history": [5.0, 4.2],
        "runtime_seconds": 0.5,
        "termination_reason": "budget",
        "metadata": {
            "solution_state": {"case_checksum": "checksum-30"},
            "policy_trajectory": [
                {
                    "state": [0.0] * 24,
                    "regime": 1,
                    "operator": 2,
                    "parameter": [0.5] * 6,
                    "reward": 1.0,
                    "evaluations": 50,
                    "source_policy": "ai",
                }
            ]
            if trajectory
            else [],
        },
    }
    with db.connect() as con:
        con.execute(
            """INSERT INTO runs(
                id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path,validation_status
            ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                experiment_id,
                algorithm,
                0,
                json.dumps({"algorithm_seed": 7}),
                json.dumps(result),
                "",
                "verified" if verified else "unverified",
            ),
        )


def test_existing_and_new_experiments_are_excluded_from_learning_by_default(tmp_path):
    db = ResultDatabase(tmp_path / "history.sqlite")
    experiment_id = db.create_experiment(ExperimentConfig(), collect_provenance())
    row = db.get_experiment(experiment_id)
    assert row["data_role"] == "excluded"
    assert row["learning_eligible"] == 0


def test_repository_includes_only_explicit_eligible_training_experiments(tmp_path):
    db = ResultDatabase(tmp_path / "history.sqlite")
    train_id = db.create_experiment(ExperimentConfig(), collect_provenance())
    test_id = db.create_experiment(ExperimentConfig(name="held-out"), collect_provenance())
    _insert_run(db, train_id, verified=True, trajectory=True)
    _insert_run(db, test_id, verified=True, trajectory=True)
    db.set_experiment_learning_role(train_id, "train", eligible=True)
    db.set_experiment_learning_role(test_id, "test", eligible=True)

    path = tmp_path / "experience.json"
    repository = build_experience_repository(db, path, verified_only=True)
    loaded = load_experience_repository(path)
    assert repository.summary["eligible_training_experiments"] == 1
    assert loaded.summary["policy_transitions"] == 1
    assert len(loaded.cross_algorithm_solutions) == 1
    assert db.get_experiment(test_id)["learning_eligible"] == 0


def test_legacy_calo_summaries_are_reconstructed_with_lower_confidence(tmp_path):
    db = ResultDatabase(tmp_path / "history.sqlite")
    experiment_id = db.create_experiment(ExperimentConfig(), collect_provenance())
    _insert_run(db, experiment_id, verified=True, trajectory=False)
    db.set_experiment_learning_role(experiment_id, "train", eligible=True)
    repository = build_experience_repository(db, tmp_path / "experience.json")
    assert repository.summary["summary_only_calo_runs"] == 1
    assert repository.summary["reconstructed_legacy_calo_trajectories"] == 0
    # This synthetic fixture intentionally omits the v1.2 diagnostic histories needed for reconstruction.
    assert repository.summary["policy_transitions"] == 0
    assert repository.summary["cross_algorithm_solutions"] == 1
    assert repository.summary["parameter_prior_groups"] == 1


def test_v12_diagnostic_histories_can_be_reconstructed_for_lower_weight_policy_pretraining(
    tmp_path,
):
    db = ResultDatabase(tmp_path / "history.sqlite")
    experiment_id = db.create_experiment(ExperimentConfig(), collect_provenance())
    result = {
        "algorithm": "CALO",
        "seed": 9,
        "parameters": {"epsilon_quantile": 0.75},
        "best_vector": [0.1, 0.2, 0.3],
        "best_objective": 4.0,
        "total_constraint_violation": 0.0,
        "feasible": True,
        "evaluations": 100,
        "iterations": 2,
        "runtime_seconds": 1.0,
        "metadata": {
            "solution_state": {"case_checksum": "checksum-30"},
            "operator_names": ["o0", "o1", "o2", "o3", "o4", "o5"],
            "regime_history": ["feasibility", "transition"],
            "operator_usage_history": [{"o0": 3, "o1": 1}, {"o2": 4}],
            "operator_success_history": [{"o0": 0.5}, {"o2": 0.8}],
            "reward_history": [0.2, 0.8],
            "convergence_evaluations": [50, 100],
            "best_feasible_objective_history": [None, 4.0],
            "diagnostics_history": {
                "population_diversity": [0.2, 0.15],
                "elite_diversity": [0.1, 0.08],
                "feasible_ratio": [0.0, 0.2],
                "epsilon_feasible_ratio": [0.3, 0.5],
                "mean_total_violation": [0.2, 0.1],
                "best_total_violation": [0.05, 0.0],
                "best_bus_voltage": [0.01, 0.0],
                "best_generator_q": [0.04, 0.0],
                "best_generator_p": [0.0, 0.0],
                "best_branch_thermal": [0.0, 0.0],
                "best_power_flow": [0.0, 0.0],
            },
        },
    }
    with db.connect() as con:
        con.execute(
            """INSERT INTO runs(
                id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path,validation_status
            ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                experiment_id,
                "CALO",
                0,
                json.dumps({"algorithm_seed": 9}),
                json.dumps(result),
                "",
                "verified",
            ),
        )
    db.set_experiment_learning_role(experiment_id, "train", eligible=True)
    repository = build_experience_repository(db, tmp_path / "legacy.json")
    assert repository.summary["reconstructed_legacy_calo_trajectories"] == 1
    transitions = repository.policy_trajectories[0]["transitions"]
    assert len(transitions) == 2
    assert transitions[0]["source_policy"] == "legacy_reconstructed"
    assert transitions[0]["parameter_supervision"] is False
    assert transitions[0]["quality_weight"] < 1.0
