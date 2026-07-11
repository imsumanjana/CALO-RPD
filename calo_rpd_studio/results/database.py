"""SQLite experiment, run, validation, and failure repository."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
import threading
import uuid

class ResultDatabase:
    def __init__(self, path="calo_rpd_results.sqlite"):
        self.path = str(path)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _initialize(self):
        schema = """
        CREATE TABLE IF NOT EXISTS experiments(
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, name TEXT NOT NULL,
            config_json TEXT NOT NULL, provenance_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, algorithm TEXT NOT NULL,
            run_index INTEGER NOT NULL, seed_json TEXT NOT NULL, result_json TEXT NOT NULL,
            arrays_path TEXT NOT NULL, validation_status TEXT NOT NULL DEFAULT 'unverified',
            FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        );
        CREATE TABLE IF NOT EXISTS validations(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created_at TEXT NOT NULL,
            validation_json TEXT NOT NULL, passed INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS run_failures(
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, algorithm TEXT NOT NULL,
            run_index INTEGER NOT NULL, seed_json TEXT NOT NULL, failure_type TEXT NOT NULL,
            message TEXT NOT NULL, traceback_text TEXT NOT NULL, evaluation_count INTEGER NOT NULL,
            numerical_state_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_failures_experiment ON run_failures(experiment_id);
        """
        with self.connect() as con:
            con.executescript(schema)

    def create_experiment(self, config, provenance):
        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO experiments VALUES(?,?,?,?,?)",
                (experiment_id, now, config.name, json.dumps(config.to_dict(), allow_nan=True), json.dumps(provenance, allow_nan=True)),
            )
        return experiment_id

    @staticmethod
    def _result_dict(result):
        return {
            "algorithm": result.algorithm,
            "seed": result.seed,
            "parameters": result.parameters,
            "best_vector": result.best_vector.tolist(),
            "decoded_controls": result.decoded_controls,
            "best_objective": result.best_objective,
            "objective_components": result.objective_components,
            "total_constraint_violation": result.total_constraint_violation,
            "feasible": result.feasible,
            "evaluations": result.evaluations,
            "iterations": result.iterations,
            "convergence_history": result.convergence_history,
            "runtime_seconds": result.runtime_seconds,
            "termination_reason": result.termination_reason,
            "metadata": result.metadata,
        }

    def add_run(self, experiment_id, completed, arrays_path):
        run_id = str(uuid.uuid4())
        seeds = {
            "algorithm_seed": completed.seeds.algorithm_seed,
            "scenario_seed": completed.seeds.scenario_seed,
            "ai_inference_seed": completed.seeds.ai_inference_seed,
        }
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO runs(id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path) VALUES(?,?,?,?,?,?,?)",
                (run_id, experiment_id, completed.algorithm, completed.run_index, json.dumps(seeds), json.dumps(self._result_dict(completed.result), allow_nan=True), str(arrays_path)),
            )
        return run_id

    def add_failure(self, experiment_id, failure):
        failure_id = str(uuid.uuid4())
        seeds = {
            "algorithm_seed": failure.seeds.algorithm_seed,
            "scenario_seed": failure.seeds.scenario_seed,
            "ai_inference_seed": failure.seeds.ai_inference_seed,
        }
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO run_failures VALUES(?,?,?,?,?,?,?,?,?,?)",
                (failure_id, experiment_id, failure.algorithm, failure.run_index, json.dumps(seeds), failure.failure_type, failure.message, failure.traceback_text, failure.evaluation_count, json.dumps(failure.numerical_state, allow_nan=True)),
            )
        return failure_id

    def add_validation(self, run_id, validation):
        validation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        passed = bool(validation.get("passed"))
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO validations VALUES(?,?,?,?,?)",
                (validation_id, run_id, now, json.dumps(validation, allow_nan=True), int(passed)),
            )
            con.execute("UPDATE runs SET validation_status=? WHERE id=?", ("verified" if passed else "failed", run_id))
        return validation_id

    def get_experiment(self, experiment_id):
        with self.connect() as con:
            row = con.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        return None if row is None else dict(row)

    def get_run(self, run_id):
        with self.connect() as con:
            row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return None if row is None else dict(row)

    def list_runs(self, experiment_id=None, verified_only=False):
        query = "SELECT * FROM runs"
        args = []
        where = []
        if experiment_id:
            where.append("experiment_id=?")
            args.append(experiment_id)
        if verified_only:
            where.append("validation_status='verified'")
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY run_index, algorithm"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    def list_failures(self, experiment_id=None):
        query = "SELECT * FROM run_failures"
        args = []
        if experiment_id:
            query += " WHERE experiment_id=?"
            args = [experiment_id]
        query += " ORDER BY run_index, algorithm"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    def list_experiments(self):
        with self.connect() as con:
            return [dict(row) for row in con.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()]
