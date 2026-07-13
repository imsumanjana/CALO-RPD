from __future__ import annotations

import json

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.results.database import ResultDatabase


def _insert_run(db: ResultDatabase, experiment_id: str, run_id: str, trace_path, *, verified=False):
    payload = {
        "best_objective": 1.0,
        "feasible": True,
        "total_constraint_violation": 0.0,
        "runtime_seconds": 0.1,
    }
    with db.connect() as con:
        con.execute(
            "INSERT INTO runs(id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path,validation_status) VALUES(?,?,?,?,?,?,?,?)",
            (
                run_id,
                experiment_id,
                "CALO",
                0,
                json.dumps({"algorithm_seed": 1}),
                json.dumps(payload),
                str(trace_path),
                "verified" if verified else "unverified",
            ),
        )
        if verified:
            con.execute(
                "INSERT INTO validations(id,run_id,created_at,validation_json,passed) VALUES(?,?,?,?,?)",
                ("validation-" + run_id, run_id, "2026-07-13T00:00:00+00:00", "{}", 1),
            )


def test_delete_experiment_removes_records_and_trace_file(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    experiment_id = db.create_experiment(ExperimentConfig(), {})
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"trace-data")
    _insert_run(db, experiment_id, "run-1", trace, verified=True)
    with db.connect() as con:
        con.execute(
            "INSERT INTO run_failures VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("failure-1", experiment_id, "PSO", 0, "{}", "Error", "failed", "tb", 1, "{}"),
        )

    summary = db.experiment_storage_summary(experiment_id)
    assert summary["runs"] == 1
    assert summary["failures"] == 1
    assert summary["validations"] == 1
    assert summary["trace_files"] == 1

    deleted = db.delete_experiment(experiment_id)
    assert deleted["experiments_deleted"] == 1
    assert deleted["runs_deleted"] == 1
    assert deleted["failures_deleted"] == 1
    assert deleted["validations_deleted"] == 1
    assert deleted["trace_files_deleted"] == 1
    assert not trace.exists()
    assert db.get_experiment(experiment_id) is None
    assert db.list_runs(experiment_id) == []
    assert db.list_failures(experiment_id) == []


def test_delete_run_preserves_experiment_and_other_runs(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    experiment_id = db.create_experiment(ExperimentConfig(), {})
    trace_a = tmp_path / "a.npz"
    trace_b = tmp_path / "b.npz"
    trace_a.write_bytes(b"a")
    trace_b.write_bytes(b"b")
    _insert_run(db, experiment_id, "run-a", trace_a, verified=True)
    _insert_run(db, experiment_id, "run-b", trace_b, verified=False)

    deleted = db.delete_run("run-a")
    assert deleted["runs_deleted"] == 1
    assert deleted["validations_deleted"] == 1
    assert not trace_a.exists()
    assert trace_b.exists()
    assert db.get_experiment(experiment_id) is not None
    assert [row["id"] for row in db.list_runs(experiment_id)] == ["run-b"]


def test_clear_history_removes_all_experiments_and_referenced_traces(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    traces = []
    for index in range(2):
        experiment_id = db.create_experiment(ExperimentConfig(name=f"exp-{index}"), {})
        trace = tmp_path / f"trace-{index}.npz"
        trace.write_bytes(b"payload")
        traces.append(trace)
        _insert_run(db, experiment_id, f"run-{index}", trace)

    summary = db.clear_history()
    assert summary["experiments_deleted"] == 2
    assert summary["runs_deleted"] == 2
    assert summary["trace_files_deleted"] == 2
    assert db.list_experiments() == []
    assert all(not path.exists() for path in traces)
