"""SQLite experiment, run, validation, failure, and trace repository."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid


class ResultDatabase:
    """Persist experiment metadata and provide safe history deletion operations.

    Deletion methods remove the selected database records together with the referenced
    compressed ``.npz`` run-array files.  External publication export directories are
    intentionally not touched because they may contain user-managed copies.
    """

    def __init__(self, path="calo_rpd_results.sqlite"):
        self.path = str(path)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
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
            config_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
            data_role TEXT NOT NULL DEFAULT 'excluded',
            learning_eligible INTEGER NOT NULL DEFAULT 0,
            learning_locked INTEGER NOT NULL DEFAULT 0
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
        CREATE INDEX IF NOT EXISTS idx_validations_run ON validations(run_id);
        """
        with self.connect() as con:
            con.executescript(schema)
            # Forward-compatible migration for repositories created before v1.3.0. Existing
            # experiments are deliberately excluded from learning until the user classifies them.
            columns = {row["name"] for row in con.execute("PRAGMA table_info(experiments)").fetchall()}
            if "data_role" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN data_role TEXT NOT NULL DEFAULT 'excluded'")
            if "learning_eligible" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN learning_eligible INTEGER NOT NULL DEFAULT 0")
            if "learning_locked" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN learning_locked INTEGER NOT NULL DEFAULT 0")

    def create_experiment(self, config, provenance):
        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO experiments(
                    id,created_at,name,config_json,provenance_json,
                    data_role,learning_eligible,learning_locked
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    now,
                    config.name,
                    json.dumps(config.to_dict(), allow_nan=True),
                    json.dumps(provenance, allow_nan=True),
                    "excluded",
                    0,
                    0,
                ),
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
                (
                    run_id,
                    experiment_id,
                    completed.algorithm,
                    completed.run_index,
                    json.dumps(seeds),
                    json.dumps(self._result_dict(completed.result), allow_nan=True),
                    str(arrays_path),
                ),
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
                (
                    failure_id,
                    experiment_id,
                    failure.algorithm,
                    failure.run_index,
                    json.dumps(seeds),
                    failure.failure_type,
                    failure.message,
                    failure.traceback_text,
                    failure.evaluation_count,
                    json.dumps(failure.numerical_state, allow_nan=True),
                ),
            )
        return failure_id

    def add_validation(self, run_id, validation):
        validation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        passed = bool(validation.get("passed"))
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO validations VALUES(?,?,?,?,?)",
                (
                    validation_id,
                    run_id,
                    now,
                    json.dumps(validation, allow_nan=True),
                    int(passed),
                ),
            )
            con.execute(
                "UPDATE runs SET validation_status=? WHERE id=?",
                ("verified" if passed else "failed", run_id),
            )
        return validation_id

    def get_experiment(self, experiment_id):
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def get_run(self, run_id):
        with self.connect() as con:
            row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return None if row is None else dict(row)

    def list_validations(self, run_id: str | None = None):
        query = "SELECT * FROM validations"
        args = []
        if run_id:
            query += " WHERE run_id=?"
            args.append(run_id)
        query += " ORDER BY created_at"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

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
            return [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC"
                ).fetchall()
            ]


    # ------------------------------------------------------------------
    # Historical-learning classification
    # ------------------------------------------------------------------

    def set_experiment_learning_role(
        self,
        experiment_id: str,
        role: str,
        *,
        eligible: bool = False,
        locked: bool | None = None,
    ) -> dict:
        """Classify one experiment for leakage-aware historical learning.

        Only ``train`` experiments may be learning-eligible. Validation and test experiments are
        always excluded from model/algorithm updates. A locked experiment cannot be reclassified
        until it is explicitly unlocked.
        """
        role = str(role).strip().lower()
        allowed = {"train", "validation", "test", "excluded"}
        if role not in allowed:
            raise ValueError(f"Unsupported experiment data role: {role}")
        eligible = bool(eligible and role == "train")
        with self._lock, self.connect() as con:
            row = con.execute(
                "SELECT learning_locked FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown experiment: {experiment_id}")
            current_locked = bool(row["learning_locked"])
            if current_locked and locked is not False:
                raise RuntimeError(
                    "This experiment learning classification is locked. Unlock it before changing the role."
                )
            next_locked = current_locked if locked is None else bool(locked)
            con.execute(
                "UPDATE experiments SET data_role=?,learning_eligible=?,learning_locked=? WHERE id=?",
                (role, int(eligible), int(next_locked), experiment_id),
            )
        return self.get_experiment(experiment_id)

    def list_learning_experiments(
        self, *, role: str | None = None, eligible_only: bool = False
    ) -> list[dict]:
        query = "SELECT * FROM experiments"
        where = []
        args: list = []
        if role is not None:
            where.append("data_role=?")
            args.append(str(role).strip().lower())
        if eligible_only:
            where.append("learning_eligible=1")
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY created_at DESC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    # ------------------------------------------------------------------
    # History and trace management
    # ------------------------------------------------------------------

    def _resolve_array_path(self, value: str) -> Path | None:
        if not value:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        # Existing repositories commonly store paths relative to the application working
        # directory.  Also try a path relative to the database directory for moved projects.
        if path.exists():
            return path.resolve()
        database_relative = Path(self.path).expanduser().resolve().parent / path
        return database_relative

    def _trace_file_stats(self, array_paths: list[str]) -> tuple[int, int, int]:
        existing = 0
        missing = 0
        total_bytes = 0
        for value in dict.fromkeys(array_paths):
            path = self._resolve_array_path(value)
            if path is None:
                continue
            try:
                if path.is_file():
                    existing += 1
                    total_bytes += path.stat().st_size
                else:
                    missing += 1
            except OSError:
                missing += 1
        return existing, missing, total_bytes

    def experiment_storage_summary(self, experiment_id: str) -> dict:
        """Return record and referenced trace-storage counts for one experiment."""
        with self.connect() as con:
            experiment = con.execute(
                "SELECT id,name,created_at FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if experiment is None:
                return {
                    "experiment_id": experiment_id,
                    "name": "",
                    "created_at": "",
                    "runs": 0,
                    "failures": 0,
                    "validations": 0,
                    "verified_runs": 0,
                    "trace_files": 0,
                    "missing_trace_files": 0,
                    "trace_bytes": 0,
                }
            run_rows = con.execute(
                "SELECT id,arrays_path,validation_status FROM runs WHERE experiment_id=?",
                (experiment_id,),
            ).fetchall()
            failures = con.execute(
                "SELECT COUNT(*) FROM run_failures WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()[0]
            run_ids = [row["id"] for row in run_rows]
            validations = 0
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                validations = con.execute(
                    f"SELECT COUNT(*) FROM validations WHERE run_id IN ({placeholders})",
                    run_ids,
                ).fetchone()[0]
        existing, missing, trace_bytes = self._trace_file_stats(
            [row["arrays_path"] for row in run_rows]
        )
        return {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "created_at": experiment["created_at"],
            "runs": len(run_rows),
            "failures": int(failures),
            "validations": int(validations),
            "verified_runs": sum(
                1 for row in run_rows if row["validation_status"] == "verified"
            ),
            "trace_files": existing,
            "missing_trace_files": missing,
            "trace_bytes": trace_bytes,
        }

    def history_storage_summary(self) -> dict:
        experiments = self.list_experiments()
        summaries = [self.experiment_storage_summary(row["id"]) for row in experiments]
        return {
            "experiments": len(summaries),
            "runs": sum(item["runs"] for item in summaries),
            "failures": sum(item["failures"] for item in summaries),
            "validations": sum(item["validations"] for item in summaries),
            "verified_runs": sum(item["verified_runs"] for item in summaries),
            "trace_files": sum(item["trace_files"] for item in summaries),
            "missing_trace_files": sum(
                item["missing_trace_files"] for item in summaries
            ),
            "trace_bytes": sum(item["trace_bytes"] for item in summaries),
        }

    def _delete_trace_files(self, array_paths: list[str]) -> dict:
        deleted = 0
        missing = 0
        failed = 0
        reclaimed_bytes = 0
        for value in dict.fromkeys(array_paths):
            path = self._resolve_array_path(value)
            if path is None:
                continue
            try:
                if not path.is_file():
                    missing += 1
                    continue
                size = path.stat().st_size
                path.unlink()
                deleted += 1
                reclaimed_bytes += size
            except OSError:
                failed += 1
        return {
            "trace_files_deleted": deleted,
            "trace_files_missing": missing,
            "trace_files_failed": failed,
            "trace_bytes_reclaimed": reclaimed_bytes,
        }

    def _compact_database(self) -> None:
        """Checkpoint WAL and compact free pages after destructive history operations."""
        with self._lock:
            con = sqlite3.connect(self.path, timeout=30)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.execute("VACUUM")
            finally:
                con.close()

    def delete_run(self, run_id: str, *, compact: bool = True) -> dict:
        """Delete one completed run, its validation records, and referenced array trace."""
        with self._lock, self.connect() as con:
            row = con.execute(
                "SELECT id,experiment_id,arrays_path FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                return {
                    "experiments_deleted": 0,
                    "runs_deleted": 0,
                    "failures_deleted": 0,
                    "validations_deleted": 0,
                    "trace_files_deleted": 0,
                    "trace_files_missing": 0,
                    "trace_files_failed": 0,
                    "trace_bytes_reclaimed": 0,
                }
            validations = con.execute(
                "SELECT COUNT(*) FROM validations WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            con.execute("DELETE FROM validations WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM runs WHERE id=?", (run_id,))
            array_paths = [row["arrays_path"]]
        trace_summary = self._delete_trace_files(array_paths)
        if compact:
            self._compact_database()
        return {
            "experiments_deleted": 0,
            "runs_deleted": 1,
            "failures_deleted": 0,
            "validations_deleted": int(validations),
            **trace_summary,
        }

    def delete_experiment(self, experiment_id: str, *, compact: bool = True) -> dict:
        """Delete one experiment and all database/array traces owned by it."""
        with self._lock, self.connect() as con:
            exists = con.execute(
                "SELECT 1 FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if exists is None:
                return {
                    "experiments_deleted": 0,
                    "runs_deleted": 0,
                    "failures_deleted": 0,
                    "validations_deleted": 0,
                    "trace_files_deleted": 0,
                    "trace_files_missing": 0,
                    "trace_files_failed": 0,
                    "trace_bytes_reclaimed": 0,
                }
            run_rows = con.execute(
                "SELECT id,arrays_path FROM runs WHERE experiment_id=?",
                (experiment_id,),
            ).fetchall()
            run_ids = [row["id"] for row in run_rows]
            validations = 0
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                validations = con.execute(
                    f"SELECT COUNT(*) FROM validations WHERE run_id IN ({placeholders})",
                    run_ids,
                ).fetchone()[0]
                con.execute(
                    f"DELETE FROM validations WHERE run_id IN ({placeholders})", run_ids
                )
            failures = con.execute(
                "SELECT COUNT(*) FROM run_failures WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()[0]
            con.execute("DELETE FROM runs WHERE experiment_id=?", (experiment_id,))
            con.execute(
                "DELETE FROM run_failures WHERE experiment_id=?", (experiment_id,)
            )
            con.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
            array_paths = [row["arrays_path"] for row in run_rows]
        trace_summary = self._delete_trace_files(array_paths)
        if compact:
            self._compact_database()
        return {
            "experiments_deleted": 1,
            "runs_deleted": len(run_rows),
            "failures_deleted": int(failures),
            "validations_deleted": int(validations),
            **trace_summary,
        }

    def clear_history(self) -> dict:
        """Delete all experiment history and all referenced run-array traces."""
        with self._lock, self.connect() as con:
            experiment_count = con.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]
            run_rows = con.execute("SELECT id,arrays_path FROM runs").fetchall()
            failure_count = con.execute(
                "SELECT COUNT(*) FROM run_failures"
            ).fetchone()[0]
            validation_count = con.execute(
                "SELECT COUNT(*) FROM validations"
            ).fetchone()[0]
            con.execute("DELETE FROM validations")
            con.execute("DELETE FROM runs")
            con.execute("DELETE FROM run_failures")
            con.execute("DELETE FROM experiments")
            array_paths = [row["arrays_path"] for row in run_rows]
        trace_summary = self._delete_trace_files(array_paths)
        self._compact_database()
        return {
            "experiments_deleted": int(experiment_count),
            "runs_deleted": len(run_rows),
            "failures_deleted": int(failure_count),
            "validations_deleted": int(validation_count),
            **trace_summary,
        }
