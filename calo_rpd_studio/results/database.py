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
            evaluation_horizon INTEGER NOT NULL DEFAULT 0, revision_id TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS run_failures(
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, algorithm TEXT NOT NULL,
            run_index INTEGER NOT NULL, seed_json TEXT NOT NULL, failure_type TEXT NOT NULL,
            message TEXT NOT NULL, traceback_text TEXT NOT NULL, evaluation_count INTEGER NOT NULL,
            numerical_state_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS portfolios(
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            name TEXT NOT NULL, config_json TEXT NOT NULL, plan_json TEXT NOT NULL,
            fingerprint TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'planned'
        );
        CREATE TABLE IF NOT EXISTS campaigns(
            id TEXT PRIMARY KEY, experiment_id TEXT, portfolio_id TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            mode TEXT NOT NULL, status TEXT NOT NULL,
            config_json TEXT NOT NULL, total_tasks INTEGER NOT NULL,
            completed_tasks INTEGER NOT NULL DEFAULT 0,
            last_message TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(experiment_id) REFERENCES experiments(id),
            FOREIGN KEY(portfolio_id) REFERENCES portfolios(id)
        );
        CREATE TABLE IF NOT EXISTS campaign_tasks(
            id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, job_index INTEGER NOT NULL,
            algorithm TEXT NOT NULL, run_index INTEGER NOT NULL,
            seed_json TEXT NOT NULL, fingerprint TEXT NOT NULL,
            required_outputs_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'planned', attempts INTEGER NOT NULL DEFAULT 0,
            checkpoint_path TEXT NOT NULL DEFAULT '', checkpoint_sha256 TEXT NOT NULL DEFAULT '',
            run_id TEXT, failure_id TEXT, last_activity TEXT NOT NULL,
            UNIQUE(campaign_id, job_index),
            FOREIGN KEY(campaign_id) REFERENCES campaigns(id),
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS task_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            created_at TEXT NOT NULL, event_type TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS resumable_tasks(
            id TEXT PRIMARY KEY, task_type TEXT NOT NULL, title TEXT NOT NULL,
            status TEXT NOT NULL, progress_current INTEGER NOT NULL DEFAULT 0,
            progress_total INTEGER NOT NULL DEFAULT 0, state_json TEXT NOT NULL DEFAULT '{}',
            resumable INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS policies(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, checkpoint_path TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE, architecture_version TEXT NOT NULL DEFAULT '',
            state_schema_version TEXT NOT NULL DEFAULT '', action_schema_version TEXT NOT NULL DEFAULT '',
            training_environment_version TEXT NOT NULL DEFAULT '',
            qualification_status TEXT NOT NULL DEFAULT 'candidate', grade TEXT NOT NULL DEFAULT 'U',
            active INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS policy_qualifications(
            id TEXT PRIMARY KEY, policy_id TEXT NOT NULL, created_at TEXT NOT NULL,
            reference_policy_id TEXT NOT NULL DEFAULT '', config_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}', passed INTEGER NOT NULL DEFAULT 0,
            grade TEXT NOT NULL DEFAULT 'U', score REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY(policy_id) REFERENCES policies(id)
        );
        CREATE TABLE IF NOT EXISTS experiment_policy_bindings(
            experiment_id TEXT PRIMARY KEY, policy_id TEXT NOT NULL DEFAULT '', policy_name TEXT NOT NULL DEFAULT '',
            checkpoint_path TEXT NOT NULL DEFAULT '', sha256 TEXT NOT NULL DEFAULT '',
            binding_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        );
        CREATE TABLE IF NOT EXISTS experiment_workspace_state(
            experiment_id TEXT PRIMARY KEY, workflow_json TEXT NOT NULL DEFAULT '{}',
            ui_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        );
        CREATE TABLE IF NOT EXISTS policy_lineages(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            parent_lineage_id TEXT NOT NULL DEFAULT '', forked_from_checkpoint_id TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '', archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS policy_checkpoints(
            id TEXT PRIMARY KEY, lineage_id TEXT NOT NULL, cumulative_epoch INTEGER NOT NULL DEFAULT 0,
            phase_index INTEGER NOT NULL DEFAULT 1, checkpoint_path TEXT NOT NULL, resume_path TEXT NOT NULL DEFAULT '',
            sha256 TEXT NOT NULL, qualification_status TEXT NOT NULL DEFAULT 'candidate', grade TEXT NOT NULL DEFAULT 'U',
            is_latest INTEGER NOT NULL DEFAULT 0, is_best INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
            FOREIGN KEY(lineage_id) REFERENCES policy_lineages(id)
        );
        CREATE TABLE IF NOT EXISTS experiment_revisions(
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, revision_number INTEGER NOT NULL,
            parent_revision_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
            run_target INTEGER NOT NULL, evaluation_target INTEGER NOT NULL,
            extension_mode TEXT NOT NULL DEFAULT 'original', publication_eligible INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'planned', protocol_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(experiment_id, revision_number), FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        );
        CREATE TABLE IF NOT EXISTS run_segments(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, segment_index INTEGER NOT NULL,
            start_evaluations INTEGER NOT NULL, end_evaluations INTEGER NOT NULL,
            checkpoint_path TEXT NOT NULL DEFAULT '', checkpoint_sha256 TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'completed', publication_eligible INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
            UNIQUE(run_id, segment_index), FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS run_horizon_snapshots(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, evaluation_horizon INTEGER NOT NULL,
            revision_id TEXT NOT NULL DEFAULT '', result_json TEXT NOT NULL, arrays_path TEXT NOT NULL DEFAULT '',
            validation_status TEXT NOT NULL DEFAULT 'unverified', scientific_fingerprint TEXT NOT NULL DEFAULT '',
            validations_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
            UNIQUE(run_id, evaluation_horizon), FOREIGN KEY(run_id) REFERENCES runs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_failures_experiment ON run_failures(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_validations_run ON validations(run_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_status ON campaigns(status);
        CREATE INDEX IF NOT EXISTS idx_campaign_tasks_status ON campaign_tasks(campaign_id,status);
        CREATE INDEX IF NOT EXISTS idx_campaign_tasks_fingerprint ON campaign_tasks(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_resumable_status ON resumable_tasks(status,resumable);
        CREATE INDEX IF NOT EXISTS idx_policies_active ON policies(active,archived);
        CREATE INDEX IF NOT EXISTS idx_policy_qualifications_policy ON policy_qualifications(policy_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_policy_bindings_sha ON experiment_policy_bindings(sha256);
        CREATE INDEX IF NOT EXISTS idx_policy_checkpoint_lineage ON policy_checkpoints(lineage_id,cumulative_epoch);
        CREATE INDEX IF NOT EXISTS idx_policy_checkpoint_sha ON policy_checkpoints(sha256);
        CREATE INDEX IF NOT EXISTS idx_experiment_revisions ON experiment_revisions(experiment_id,revision_number);
        CREATE INDEX IF NOT EXISTS idx_run_segments ON run_segments(run_id,segment_index);
        CREATE INDEX IF NOT EXISTS idx_run_horizon_snapshots ON run_horizon_snapshots(run_id,evaluation_horizon);
        """
        with self.connect() as con:
            con.executescript(schema)
            # Forward-compatible migration for repositories created before v1.3.0. Existing
            # experiments are deliberately excluded from learning until the user classifies them.
            columns = {
                row["name"] for row in con.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "data_role" not in columns:
                con.execute(
                    "ALTER TABLE experiments ADD COLUMN data_role TEXT NOT NULL DEFAULT 'excluded'"
                )
            if "learning_eligible" not in columns:
                con.execute(
                    "ALTER TABLE experiments ADD COLUMN learning_eligible INTEGER NOT NULL DEFAULT 0"
                )
            if "learning_locked" not in columns:
                con.execute(
                    "ALTER TABLE experiments ADD COLUMN learning_locked INTEGER NOT NULL DEFAULT 0"
                )
            if "scientific_fingerprint" not in columns:
                con.execute(
                    "ALTER TABLE experiments ADD COLUMN scientific_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            if "portfolio_id" not in columns:
                con.execute(
                    "ALTER TABLE experiments ADD COLUMN portfolio_id TEXT NOT NULL DEFAULT ''"
                )
            if "campaign_status" not in columns:
                con.execute(
                    "ALTER TABLE experiments ADD COLUMN campaign_status TEXT NOT NULL DEFAULT 'completed'"
                )
            run_columns = {row["name"] for row in con.execute("PRAGMA table_info(runs)").fetchall()}
            if "scientific_fingerprint" not in run_columns:
                con.execute(
                    "ALTER TABLE runs ADD COLUMN scientific_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            validation_columns = {
                row["name"] for row in con.execute("PRAGMA table_info(validations)").fetchall()
            }
            if "evaluation_horizon" not in validation_columns:
                con.execute(
                    "ALTER TABLE validations ADD COLUMN evaluation_horizon INTEGER NOT NULL DEFAULT 0"
                )
            if "revision_id" not in validation_columns:
                con.execute(
                    "ALTER TABLE validations ADD COLUMN revision_id TEXT NOT NULL DEFAULT ''"
                )
            snapshot_columns = {
                row["name"]
                for row in con.execute("PRAGMA table_info(run_horizon_snapshots)").fetchall()
            }
            if "scientific_fingerprint" not in snapshot_columns:
                con.execute(
                    "ALTER TABLE run_horizon_snapshots ADD COLUMN scientific_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            if "validations_json" not in snapshot_columns:
                con.execute(
                    "ALTER TABLE run_horizon_snapshots ADD COLUMN validations_json TEXT NOT NULL DEFAULT '[]'"
                )

    def create_experiment(
        self,
        config,
        provenance,
        *,
        scientific_fingerprint: str = "",
        portfolio_id: str = "",
        campaign_status: str = "running",
    ):
        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO experiments(
                    id,created_at,name,config_json,provenance_json,
                    data_role,learning_eligible,learning_locked,
                    scientific_fingerprint,portfolio_id,campaign_status
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    now,
                    config.name,
                    json.dumps(config.to_dict(), allow_nan=True),
                    json.dumps(provenance, allow_nan=True),
                    "excluded",
                    0,
                    0,
                    str(scientific_fingerprint),
                    str(portfolio_id),
                    str(campaign_status),
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

    def add_run(self, experiment_id, completed, arrays_path, *, scientific_fingerprint: str = ""):
        run_id = str(uuid.uuid4())
        seeds = {
            "algorithm_seed": completed.seeds.algorithm_seed,
            "scenario_seed": completed.seeds.scenario_seed,
            "ai_inference_seed": completed.seeds.ai_inference_seed,
        }
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO runs(id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path,scientific_fingerprint) VALUES(?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    experiment_id,
                    completed.algorithm,
                    completed.run_index,
                    json.dumps(seeds),
                    json.dumps(self._result_dict(completed.result), allow_nan=True),
                    str(arrays_path),
                    str(scientific_fingerprint),
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

    def add_validation(
        self, run_id, validation, *, evaluation_horizon: int | None = None, revision_id: str = ""
    ):
        """Attach validation to exactly one FE horizon without corrupting another horizon's status."""
        validation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        passed = bool(validation.get("passed"))
        row = self.get_run(run_id)
        if row is None:
            raise KeyError(run_id)
        try:
            current_horizon = int(
                json.loads(str(row.get("result_json", "{}") or "{}")).get("evaluations", 0) or 0
            )
        except Exception:
            current_horizon = 0
        if evaluation_horizon is None:
            evaluation_horizon = current_horizon
        horizon = int(evaluation_horizon or 0)
        status = "verified" if passed else "failed"
        validation_record = {
            "id": validation_id,
            "created_at": now,
            "validation_json": json.dumps(validation, allow_nan=True),
            "passed": int(passed),
            "evaluation_horizon": horizon,
            "revision_id": str(revision_id),
        }
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO validations(id,run_id,created_at,validation_json,passed,evaluation_horizon,revision_id)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    validation_id,
                    run_id,
                    now,
                    json.dumps(validation, allow_nan=True),
                    int(passed),
                    horizon,
                    str(revision_id),
                ),
            )
            if horizon == current_horizon:
                con.execute("UPDATE runs SET validation_status=? WHERE id=?", (status, run_id))
            else:
                snapshot = con.execute(
                    "SELECT validations_json FROM run_horizon_snapshots WHERE run_id=? AND evaluation_horizon=?",
                    (str(run_id), horizon),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        f"Cannot attach validation at {horizon} FE because no preserved evidence exists for that run horizon"
                    )
                records = json.loads(str(snapshot["validations_json"] or "[]"))
                records.append(validation_record)
                con.execute(
                    "UPDATE run_horizon_snapshots SET validation_status=?,validations_json=? WHERE run_id=? AND evaluation_horizon=?",
                    (status, json.dumps(records, allow_nan=True), str(run_id), horizon),
                )
        return validation_id

    def get_experiment(self, experiment_id):
        with self.connect() as con:
            row = con.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
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
    # Portfolio, campaign, fingerprint reuse, and universal resume
    # ------------------------------------------------------------------

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_portfolio(self, name: str, config: dict, plan: dict, fingerprint: str) -> str:
        portfolio_id = str(uuid.uuid4())
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO portfolios VALUES(?,?,?,?,?,?,?,?)",
                (
                    portfolio_id,
                    now,
                    now,
                    str(name),
                    json.dumps(config, allow_nan=True),
                    json.dumps(plan, allow_nan=True),
                    str(fingerprint),
                    "planned",
                ),
            )
        return portfolio_id

    def update_portfolio(
        self,
        portfolio_id: str,
        *,
        status: str | None = None,
        config: dict | None = None,
        plan: dict | None = None,
    ) -> None:
        values = []
        clauses = ["updated_at=?"]
        values.append(self._utcnow())
        if status is not None:
            clauses.append("status=?")
            values.append(str(status))
        if config is not None:
            clauses.append("config_json=?")
            values.append(json.dumps(config, allow_nan=True))
        if plan is not None:
            clauses.append("plan_json=?")
            values.append(json.dumps(plan, allow_nan=True))
        values.append(portfolio_id)
        with self._lock, self.connect() as con:
            con.execute(f"UPDATE portfolios SET {','.join(clauses)} WHERE id=?", values)

    def list_portfolios(self) -> list[dict]:
        with self.connect() as con:
            return [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM portfolios ORDER BY updated_at DESC"
                ).fetchall()
            ]

    def create_campaign(
        self, experiment_id: str, portfolio_id: str, mode: str, config: dict, total_tasks: int
    ) -> str:
        campaign_id = str(uuid.uuid4())
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO campaigns VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    campaign_id,
                    experiment_id,
                    portfolio_id,
                    now,
                    now,
                    str(mode),
                    "planned",
                    json.dumps(config, allow_nan=True),
                    int(total_tasks),
                    0,
                    "",
                ),
            )
        return campaign_id

    def get_campaign(self, campaign_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        return None if row is None else dict(row)

    def list_campaigns(self, unfinished_only: bool = False) -> list[dict]:
        query = "SELECT * FROM campaigns"
        if unfinished_only:
            query += (
                " WHERE status IN ('planned','running','pausing','paused','interrupted','failed')"
            )
        query += " ORDER BY updated_at DESC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query).fetchall()]

    def update_campaign(
        self,
        campaign_id: str,
        *,
        status: str | None = None,
        completed_tasks: int | None = None,
        message: str | None = None,
    ) -> None:
        clauses = ["updated_at=?"]
        values = [self._utcnow()]
        if status is not None:
            clauses.append("status=?")
            values.append(str(status))
        if completed_tasks is not None:
            clauses.append("completed_tasks=?")
            values.append(int(completed_tasks))
        if message is not None:
            clauses.append("last_message=?")
            values.append(str(message))
        values.append(campaign_id)
        with self._lock, self.connect() as con:
            con.execute(f"UPDATE campaigns SET {','.join(clauses)} WHERE id=?", values)
            row = con.execute(
                "SELECT experiment_id FROM campaigns WHERE id=?", (campaign_id,)
            ).fetchone()
            if row and status is not None and row["experiment_id"]:
                con.execute(
                    "UPDATE experiments SET campaign_status=? WHERE id=?",
                    (str(status), row["experiment_id"]),
                )

    def add_campaign_task(
        self,
        campaign_id: str,
        job_index: int,
        algorithm: str,
        run_index: int,
        seeds: dict,
        fingerprint: str,
        required_outputs: list[str],
    ) -> str:
        task_id = str(uuid.uuid4())
        with self._lock, self.connect() as con:
            existing = con.execute(
                "SELECT id FROM campaign_tasks WHERE campaign_id=? AND job_index=?",
                (campaign_id, int(job_index)),
            ).fetchone()
            if existing:
                return str(existing["id"])
            con.execute(
                """INSERT INTO campaign_tasks(
                    id,campaign_id,job_index,algorithm,run_index,seed_json,fingerprint,
                    required_outputs_json,status,attempts,checkpoint_path,checkpoint_sha256,
                    run_id,failure_id,last_activity
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task_id,
                    campaign_id,
                    int(job_index),
                    str(algorithm),
                    int(run_index),
                    json.dumps(seeds),
                    str(fingerprint),
                    json.dumps(required_outputs),
                    "planned",
                    0,
                    "",
                    "",
                    None,
                    None,
                    self._utcnow(),
                ),
            )
        return task_id

    def list_campaign_tasks(
        self, campaign_id: str, statuses: list[str] | None = None
    ) -> list[dict]:
        query = "SELECT * FROM campaign_tasks WHERE campaign_id=?"
        args: list = [campaign_id]
        if statuses:
            query += " AND status IN (" + ",".join("?" for _ in statuses) + ")"
            args.extend(statuses)
        query += " ORDER BY job_index"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    def update_campaign_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
        run_id: str | None = None,
        failure_id: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        clauses = ["last_activity=?"]
        values = [self._utcnow()]
        if status is not None:
            clauses.append("status=?")
            values.append(str(status))
        if checkpoint_path is not None:
            clauses.append("checkpoint_path=?")
            values.append(str(checkpoint_path))
        if checkpoint_sha256 is not None:
            clauses.append("checkpoint_sha256=?")
            values.append(str(checkpoint_sha256))
        if run_id is not None:
            clauses.append("run_id=?")
            values.append(str(run_id))
        if failure_id is not None:
            clauses.append("failure_id=?")
            values.append(str(failure_id))
        if increment_attempts:
            clauses.append("attempts=attempts+1")
        values.append(task_id)
        with self._lock, self.connect() as con:
            con.execute(f"UPDATE campaign_tasks SET {','.join(clauses)} WHERE id=?", values)

    def append_task_event(self, task_id: str, event_type: str, payload: dict | None = None) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO task_events(task_id,created_at,event_type,payload_json) VALUES(?,?,?,?)",
                (
                    task_id,
                    self._utcnow(),
                    str(event_type),
                    json.dumps(payload or {}, allow_nan=True),
                ),
            )

    def clone_run_to_experiment(self, source_run_id: str, experiment_id: str) -> str:
        """Link a scientifically identical completed run into a new portfolio experiment.

        Numeric trace files are intentionally shared read-only; history deletion keeps the file
        while another run record still references it.
        """
        source = self.get_run(source_run_id)
        if source is None:
            raise KeyError(f"Unknown source run: {source_run_id}")
        run_id = str(uuid.uuid4())
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO runs(
                    id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path,
                    validation_status,scientific_fingerprint
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    experiment_id,
                    source["algorithm"],
                    source["run_index"],
                    source["seed_json"],
                    source["result_json"],
                    source["arrays_path"],
                    source["validation_status"],
                    source.get("scientific_fingerprint", ""),
                ),
            )
        return run_id

    def find_reusable_run(self, fingerprint: str, verified_only: bool = False) -> dict | None:
        query = "SELECT * FROM runs WHERE scientific_fingerprint=?"
        args = [str(fingerprint)]
        if verified_only:
            query += " AND validation_status='verified'"
        query += (
            " ORDER BY CASE validation_status WHEN 'verified' THEN 0 ELSE 1 END, rowid DESC LIMIT 1"
        )
        with self.connect() as con:
            row = con.execute(query, args).fetchone()
        return None if row is None else dict(row)

    def upsert_resumable_task(
        self,
        task_id: str,
        task_type: str,
        title: str,
        status: str,
        progress_current: int,
        progress_total: int,
        state: dict,
        resumable: bool = True,
    ) -> None:
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO resumable_tasks(id,task_type,title,status,progress_current,progress_total,state_json,resumable,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET task_type=excluded.task_type,title=excluded.title,status=excluded.status,
                progress_current=excluded.progress_current,progress_total=excluded.progress_total,state_json=excluded.state_json,
                resumable=excluded.resumable,updated_at=excluded.updated_at""",
                (
                    task_id,
                    task_type,
                    title,
                    status,
                    int(progress_current),
                    int(progress_total),
                    json.dumps(state, allow_nan=True),
                    int(bool(resumable)),
                    now,
                    now,
                ),
            )

    def update_resumable_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        state: dict | None = None,
        resumable: bool | None = None,
    ) -> None:
        clauses = ["updated_at=?"]
        values = [self._utcnow()]
        if status is not None:
            clauses.append("status=?")
            values.append(str(status))
        if progress_current is not None:
            clauses.append("progress_current=?")
            values.append(int(progress_current))
        if progress_total is not None:
            clauses.append("progress_total=?")
            values.append(int(progress_total))
        if state is not None:
            clauses.append("state_json=?")
            values.append(json.dumps(state, allow_nan=True))
        if resumable is not None:
            clauses.append("resumable=?")
            values.append(int(bool(resumable)))
        values.append(task_id)
        with self._lock, self.connect() as con:
            con.execute(f"UPDATE resumable_tasks SET {','.join(clauses)} WHERE id=?", values)

    def list_resumable_tasks(self, unfinished_only: bool = False) -> list[dict]:
        query = "SELECT * FROM resumable_tasks"
        if unfinished_only:
            query += " WHERE resumable=1 AND status IN ('planned','running','pausing','paused','interrupted','failed')"
        query += " ORDER BY updated_at DESC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query).fetchall()]

    def get_resumable_task(self, task_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM resumable_tasks WHERE id=?", (task_id,)).fetchone()
        return None if row is None else dict(row)

    def delete_resumable_task(self, task_id: str) -> None:
        with self._lock, self.connect() as con:
            con.execute("DELETE FROM resumable_tasks WHERE id=?", (task_id,))

    def mark_stale_running_interrupted(self) -> dict:
        """Recover after an unclean shutdown by making in-flight records resumable.

        This is called once during application startup before any new worker is admitted. Completed
        runs remain untouched; only records that were left in a transient running/pausing state are
        changed.
        """
        now = self._utcnow()
        with self._lock, self.connect() as con:
            campaign_count = con.execute(
                "UPDATE campaigns SET status='interrupted',updated_at=?,last_message=? "
                "WHERE status IN ('running','pausing')",
                (now, "Application restart detected; resume from committed jobs"),
            ).rowcount
            task_count = con.execute(
                "UPDATE campaign_tasks SET status='interrupted',last_activity=? "
                "WHERE status IN ('running','pausing')",
                (now,),
            ).rowcount
            resume_count = con.execute(
                "UPDATE resumable_tasks SET status='interrupted',updated_at=? "
                "WHERE resumable=1 AND status IN ('running','pausing')",
                (now,),
            ).rowcount
            con.execute(
                "UPDATE experiments SET campaign_status='interrupted' "
                "WHERE id IN (SELECT experiment_id FROM campaigns WHERE status='interrupted')"
            )
        return {
            "campaigns": int(campaign_count),
            "campaign_tasks": int(task_count),
            "resume_tasks": int(resume_count),
        }

    # ------------------------------------------------------------------
    # Historical-learning classification
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # CALO v4.1 policy library, qualification, immutable bindings, workspace state
    # ------------------------------------------------------------------

    def upsert_policy(
        self,
        *,
        policy_id: str,
        name: str,
        checkpoint_path: str,
        sha256: str,
        architecture_version: str,
        state_schema_version: str,
        action_schema_version: str,
        training_environment_version: str,
        qualification_status: str = "candidate",
        grade: str = "U",
        active: bool = False,
        archived: bool = False,
        metadata: dict | None = None,
    ) -> None:
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO policies(id,name,checkpoint_path,sha256,architecture_version,state_schema_version,
                   action_schema_version,training_environment_version,qualification_status,grade,active,archived,
                   metadata_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name,checkpoint_path=excluded.checkpoint_path,
                   sha256=excluded.sha256,architecture_version=excluded.architecture_version,
                   state_schema_version=excluded.state_schema_version,action_schema_version=excluded.action_schema_version,
                   training_environment_version=excluded.training_environment_version,
                   qualification_status=excluded.qualification_status,grade=excluded.grade,active=excluded.active,
                   archived=excluded.archived,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""",
                (
                    str(policy_id),
                    str(name),
                    str(checkpoint_path),
                    str(sha256),
                    str(architecture_version),
                    str(state_schema_version),
                    str(action_schema_version),
                    str(training_environment_version),
                    str(qualification_status),
                    str(grade),
                    int(bool(active)),
                    int(bool(archived)),
                    json.dumps(metadata or {}, allow_nan=True),
                    now,
                    now,
                ),
            )

    def get_policy(self, policy_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM policies WHERE id=?", (str(policy_id),)).fetchone()
        return None if row is None else dict(row)

    def get_policy_by_sha256(self, sha256: str) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM policies WHERE sha256=?", (str(sha256),)).fetchone()
        return None if row is None else dict(row)

    def list_policies(self, *, include_archived: bool = False) -> list[dict]:
        query = "SELECT * FROM policies"
        if not include_archived:
            query += " WHERE archived=0"
        query += " ORDER BY active DESC, grade ASC, updated_at DESC, name"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query).fetchall()]

    def update_policy(self, policy_id: str, **fields) -> None:
        allowed = {
            "name",
            "checkpoint_path",
            "qualification_status",
            "grade",
            "active",
            "archived",
            "metadata_json",
            "architecture_version",
            "state_schema_version",
            "action_schema_version",
            "training_environment_version",
        }
        clauses = ["updated_at=?"]
        values = [self._utcnow()]
        for key, value in fields.items():
            if key not in allowed:
                continue
            clauses.append(f"{key}=?")
            if key in {"active", "archived"}:
                value = int(bool(value))
            elif key == "metadata_json" and isinstance(value, dict):
                value = json.dumps(value, allow_nan=True)
            values.append(value)
        values.append(str(policy_id))
        with self._lock, self.connect() as con:
            con.execute(f"UPDATE policies SET {','.join(clauses)} WHERE id=?", values)

    def set_active_policy(self, policy_id: str) -> None:
        with self._lock, self.connect() as con:
            row = con.execute(
                "SELECT id FROM policies WHERE id=? AND archived=0", (str(policy_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown or archived policy: {policy_id}")
            con.execute("UPDATE policies SET active=0")
            con.execute(
                "UPDATE policies SET active=1,updated_at=? WHERE id=?",
                (self._utcnow(), str(policy_id)),
            )

    def delete_policy(self, policy_id: str) -> None:
        with self._lock, self.connect() as con:
            con.execute("DELETE FROM policy_qualifications WHERE policy_id=?", (str(policy_id),))
            con.execute("DELETE FROM policies WHERE id=?", (str(policy_id),))

    def add_policy_qualification(
        self,
        *,
        qualification_id: str,
        policy_id: str,
        reference_policy_id: str = "",
        config: dict | None = None,
        metrics: dict | None = None,
        passed: bool = False,
        grade: str = "U",
        score: float = 0.0,
        qualification_status: str | None = None,
    ) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO policy_qualifications VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(qualification_id),
                    str(policy_id),
                    self._utcnow(),
                    str(reference_policy_id),
                    json.dumps(config or {}, allow_nan=True),
                    json.dumps(metrics or {}, allow_nan=True),
                    int(bool(passed)),
                    str(grade),
                    float(score),
                ),
            )
            con.execute(
                "UPDATE policies SET qualification_status=?,grade=?,updated_at=? WHERE id=?",
                (
                    str(qualification_status or ("qualified" if passed else "failed")),
                    str(grade),
                    self._utcnow(),
                    str(policy_id),
                ),
            )

    def list_policy_qualifications(self, policy_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM policy_qualifications"
        args: list = []
        if policy_id:
            query += " WHERE policy_id=?"
            args.append(str(policy_id))
        query += " ORDER BY created_at DESC"
        with self.connect() as con:
            return [dict(row) for row in con.execute(query, args).fetchall()]

    def bind_policy_to_experiment(self, experiment_id: str, binding: dict) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO experiment_policy_bindings(
                   experiment_id,policy_id,policy_name,checkpoint_path,sha256,binding_json,created_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(experiment_id) DO UPDATE SET policy_id=excluded.policy_id,
                   policy_name=excluded.policy_name,checkpoint_path=excluded.checkpoint_path,
                   sha256=excluded.sha256,binding_json=excluded.binding_json,created_at=excluded.created_at""",
                (
                    str(experiment_id),
                    str(binding.get("policy_id", "")),
                    str(binding.get("policy_name", "")),
                    str(binding.get("policy_checkpoint", "")),
                    str(binding.get("policy_sha256", "")),
                    json.dumps(binding, allow_nan=True),
                    self._utcnow(),
                ),
            )

    def get_experiment_policy_binding(self, experiment_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM experiment_policy_bindings WHERE experiment_id=?",
                (str(experiment_id),),
            ).fetchone()
        if row is None:
            return None
        output = dict(row)
        output["binding"] = json.loads(output.get("binding_json") or "{}")
        return output

    def policy_reference_count(self, policy_id: str, sha256: str = "") -> int:
        with self.connect() as con:
            count = con.execute(
                "SELECT COUNT(*) AS n FROM experiment_policy_bindings WHERE policy_id=? OR (?<>'' AND sha256=?)",
                (str(policy_id), str(sha256), str(sha256)),
            ).fetchone()["n"]
        return int(count)

    def save_workspace_state(
        self, experiment_id: str, *, workflow: dict, ui: dict | None = None
    ) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO experiment_workspace_state(experiment_id,workflow_json,ui_json,updated_at)
                   VALUES(?,?,?,?) ON CONFLICT(experiment_id) DO UPDATE SET
                   workflow_json=excluded.workflow_json,ui_json=excluded.ui_json,updated_at=excluded.updated_at""",
                (
                    str(experiment_id),
                    json.dumps(workflow or {}, allow_nan=True),
                    json.dumps(ui or {}, allow_nan=True),
                    self._utcnow(),
                ),
            )

    def get_workspace_state(self, experiment_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM experiment_workspace_state WHERE experiment_id=?",
                (str(experiment_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "experiment_id": str(row["experiment_id"]),
            "workflow": json.loads(row["workflow_json"] or "{}"),
            "ui": json.loads(row["ui_json"] or "{}"),
            "updated_at": str(row["updated_at"]),
        }

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
    # v5 policy lineage and experiment-evolution persistence
    # ------------------------------------------------------------------

    def create_policy_lineage(
        self,
        name: str,
        *,
        lineage_id: str | None = None,
        parent_lineage_id: str = "",
        forked_from_checkpoint_id: str = "",
        notes: str = "",
    ) -> str:
        lineage_id = str(lineage_id or uuid.uuid4())
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO policy_lineages(id,name,created_at,updated_at,parent_lineage_id,forked_from_checkpoint_id,notes,archived) VALUES(?,?,?,?,?,?,?,0)",
                (
                    lineage_id,
                    str(name),
                    now,
                    now,
                    str(parent_lineage_id),
                    str(forked_from_checkpoint_id),
                    str(notes),
                ),
            )
        return lineage_id

    def upsert_policy_lineage(
        self,
        lineage_id: str,
        *,
        name: str,
        parent_lineage_id: str = "",
        forked_from_checkpoint_id: str = "",
        notes: str = "",
        archived: bool = False,
    ) -> None:
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute(
                """INSERT INTO policy_lineages(id,name,created_at,updated_at,parent_lineage_id,forked_from_checkpoint_id,notes,archived)
                   VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,updated_at=excluded.updated_at,
                   parent_lineage_id=excluded.parent_lineage_id,forked_from_checkpoint_id=excluded.forked_from_checkpoint_id,
                   notes=excluded.notes,archived=excluded.archived""",
                (
                    str(lineage_id),
                    str(name),
                    now,
                    now,
                    str(parent_lineage_id),
                    str(forked_from_checkpoint_id),
                    str(notes),
                    int(bool(archived)),
                ),
            )

    def list_policy_lineages(self, *, include_archived: bool = False) -> list[dict]:
        q = "SELECT * FROM policy_lineages"
        if not include_archived:
            q += " WHERE archived=0"
        q += " ORDER BY updated_at DESC"
        with self.connect() as con:
            return [dict(r) for r in con.execute(q).fetchall()]

    def get_policy_lineage(self, lineage_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM policy_lineages WHERE id=?", (str(lineage_id),)
            ).fetchone()
        return dict(row) if row else None

    def add_policy_checkpoint(
        self,
        *,
        checkpoint_id: str,
        lineage_id: str,
        cumulative_epoch: int,
        phase_index: int,
        checkpoint_path: str,
        resume_path: str,
        sha256: str,
        qualification_status: str = "candidate",
        grade: str = "U",
        is_latest: bool = True,
        is_best: bool = False,
        metadata: dict | None = None,
    ) -> None:
        now = self._utcnow()
        with self._lock, self.connect() as con:
            if is_latest:
                con.execute(
                    "UPDATE policy_checkpoints SET is_latest=0 WHERE lineage_id=?",
                    (str(lineage_id),),
                )
            if is_best:
                con.execute(
                    "UPDATE policy_checkpoints SET is_best=0 WHERE lineage_id=?", (str(lineage_id),)
                )
            con.execute(
                """INSERT INTO policy_checkpoints(id,lineage_id,cumulative_epoch,phase_index,checkpoint_path,resume_path,sha256,
                   qualification_status,grade,is_latest,is_best,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET checkpoint_path=excluded.checkpoint_path,resume_path=excluded.resume_path,
                   sha256=excluded.sha256,qualification_status=excluded.qualification_status,grade=excluded.grade,
                   is_latest=excluded.is_latest,is_best=excluded.is_best,metadata_json=excluded.metadata_json""",
                (
                    str(checkpoint_id),
                    str(lineage_id),
                    int(cumulative_epoch),
                    int(phase_index),
                    str(checkpoint_path),
                    str(resume_path),
                    str(sha256),
                    str(qualification_status),
                    str(grade),
                    int(bool(is_latest)),
                    int(bool(is_best)),
                    json.dumps(metadata or {}, allow_nan=True),
                    now,
                ),
            )
            con.execute(
                "UPDATE policy_lineages SET updated_at=? WHERE id=?", (now, str(lineage_id))
            )

    def list_policy_checkpoints(self, lineage_id: str) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM policy_checkpoints WHERE lineage_id=? ORDER BY cumulative_epoch,created_at",
                (str(lineage_id),),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            out.append(item)
        return out

    def get_policy_checkpoint(self, checkpoint_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM policy_checkpoints WHERE id=?", (str(checkpoint_id),)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        return item

    def get_policy_checkpoint_by_sha256(self, sha256: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM policy_checkpoints WHERE lower(sha256)=lower(?) ORDER BY created_at DESC LIMIT 1",
                (str(sha256),),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        return item

    def policy_checkpoint_fork_reference_count(self, checkpoint_id: str) -> int:
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM policy_lineages WHERE forked_from_checkpoint_id=?",
                (str(checkpoint_id),),
            ).fetchone()
        return int(row["n"] if row else 0)

    def delete_policy_checkpoint(self, checkpoint_id: str) -> None:
        row = self.get_policy_checkpoint(checkpoint_id)
        if row is None:
            return
        if bool(row.get("is_latest")) or bool(row.get("is_best")):
            raise ValueError(
                "Latest or best-qualified lineage checkpoints must be retained; archive the policy instead"
            )
        if self.policy_checkpoint_fork_reference_count(checkpoint_id) > 0:
            raise ValueError(
                "This checkpoint is the parent of a forked policy lineage and cannot be deleted"
            )
        with self._lock, self.connect() as con:
            con.execute("DELETE FROM policy_checkpoints WHERE id=?", (str(checkpoint_id),))

    def update_policy_checkpoint_qualification(
        self,
        checkpoint_id: str,
        *,
        qualification_status: str,
        grade: str,
        metadata_updates: dict | None = None,
    ) -> None:
        current = self.get_policy_checkpoint(checkpoint_id)
        if current is None:
            raise KeyError(checkpoint_id)
        metadata = dict(current.get("metadata", {}))
        metadata.update(metadata_updates or {})
        with self._lock, self.connect() as con:
            con.execute(
                "UPDATE policy_checkpoints SET qualification_status=?,grade=?,metadata_json=? WHERE id=?",
                (
                    str(qualification_status),
                    str(grade),
                    json.dumps(metadata, allow_nan=True),
                    str(checkpoint_id),
                ),
            )

    def mark_best_policy_checkpoint(self, lineage_id: str, checkpoint_id: str) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                "UPDATE policy_checkpoints SET is_best=0 WHERE lineage_id=?", (str(lineage_id),)
            )
            con.execute(
                "UPDATE policy_checkpoints SET is_best=1 WHERE id=? AND lineage_id=?",
                (str(checkpoint_id), str(lineage_id)),
            )

    def create_experiment_revision(
        self,
        experiment_id: str,
        *,
        run_target: int,
        evaluation_target: int,
        extension_mode: str = "original",
        publication_eligible: bool = True,
        protocol: dict | None = None,
        parent_revision_id: str = "",
        status: str = "planned",
    ) -> dict:
        with self._lock, self.connect() as con:
            n = int(
                con.execute(
                    "SELECT COALESCE(MAX(revision_number),0)+1 FROM experiment_revisions WHERE experiment_id=?",
                    (str(experiment_id),),
                ).fetchone()[0]
            )
            revision_id = str(uuid.uuid4())
            con.execute(
                "INSERT INTO experiment_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    revision_id,
                    str(experiment_id),
                    n,
                    str(parent_revision_id),
                    self._utcnow(),
                    int(run_target),
                    int(evaluation_target),
                    str(extension_mode),
                    int(bool(publication_eligible)),
                    str(status),
                    json.dumps(protocol or {}, allow_nan=True),
                ),
            )
        return self.get_experiment_revision(revision_id)

    def get_experiment_revision(self, revision_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM experiment_revisions WHERE id=?", (str(revision_id),)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["protocol"] = json.loads(item.pop("protocol_json") or "{}")
        return item

    def list_experiment_revisions(self, experiment_id: str) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM experiment_revisions WHERE experiment_id=? ORDER BY revision_number",
                (str(experiment_id),),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["protocol"] = json.loads(item.pop("protocol_json") or "{}")
            out.append(item)
        return out

    def update_experiment_revision(self, revision_id: str, *, status: str | None = None) -> None:
        if status is None:
            return
        with self._lock, self.connect() as con:
            con.execute(
                "UPDATE experiment_revisions SET status=? WHERE id=?",
                (str(status), str(revision_id)),
            )

    def add_run_segment(
        self,
        *,
        run_id: str,
        segment_index: int,
        start_evaluations: int,
        end_evaluations: int,
        checkpoint_path: str = "",
        checkpoint_sha256: str = "",
        status: str = "completed",
        publication_eligible: bool = True,
        metadata: dict | None = None,
    ) -> str:
        segment_id = str(uuid.uuid4())
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO run_segments VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    segment_id,
                    str(run_id),
                    int(segment_index),
                    int(start_evaluations),
                    int(end_evaluations),
                    str(checkpoint_path),
                    str(checkpoint_sha256),
                    str(status),
                    int(bool(publication_eligible)),
                    json.dumps(metadata or {}, allow_nan=True),
                    self._utcnow(),
                ),
            )
        return segment_id

    def get_run_by_algorithm_index(
        self, experiment_id: str, algorithm: str, run_index: int
    ) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM runs WHERE experiment_id=? AND algorithm=? AND run_index=? ORDER BY rowid LIMIT 1",
                (str(experiment_id), str(algorithm), int(run_index)),
            ).fetchone()
        return dict(row) if row else None

    def snapshot_run_horizon(
        self, run_id: str, *, evaluation_horizon: int, revision_id: str = ""
    ) -> str:
        """Preserve the complete current evidence pointer before a run row moves to another horizon."""
        with self._lock, self.connect() as con:
            row = con.execute(
                "SELECT result_json,arrays_path,validation_status,scientific_fingerprint FROM runs WHERE id=?",
                (str(run_id),),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            validations = [
                dict(v)
                for v in con.execute(
                    "SELECT id,created_at,validation_json,passed,evaluation_horizon,revision_id FROM validations WHERE run_id=? AND (evaluation_horizon=? OR evaluation_horizon=0) ORDER BY created_at",
                    (str(run_id), int(evaluation_horizon)),
                ).fetchall()
            ]
            # Legacy validation rows had no horizon. Once the current horizon is snapshotted, bind
            # those unambiguous legacy rows to it so future validations cannot be confused with them.
            con.execute(
                "UPDATE validations SET evaluation_horizon=? WHERE run_id=? AND evaluation_horizon=0",
                (int(evaluation_horizon), str(run_id)),
            )
            snapshot_id = str(uuid.uuid4())
            con.execute(
                """INSERT OR IGNORE INTO run_horizon_snapshots(
                       id,run_id,evaluation_horizon,revision_id,result_json,arrays_path,validation_status,
                       scientific_fingerprint,validations_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot_id,
                    str(run_id),
                    int(evaluation_horizon),
                    str(revision_id),
                    str(row["result_json"]),
                    str(row["arrays_path"]),
                    str(row["validation_status"]),
                    str(row["scientific_fingerprint"]),
                    json.dumps(validations, allow_nan=True),
                    self._utcnow(),
                ),
            )
        return snapshot_id

    def update_run_result(
        self, run_id: str, completed, arrays_path: str, *, scientific_fingerprint: str = ""
    ) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                "UPDATE runs SET result_json=?,arrays_path=?,scientific_fingerprint=?,validation_status='unverified' WHERE id=?",
                (
                    json.dumps(self._result_dict(completed.result), allow_nan=True),
                    str(arrays_path),
                    str(scientific_fingerprint),
                    str(run_id),
                ),
            )

    def list_run_horizon_snapshots(self, run_id: str) -> list[dict]:
        with self.connect() as con:
            rows = [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM run_horizon_snapshots WHERE run_id=? ORDER BY evaluation_horizon",
                    (str(run_id),),
                ).fetchall()
            ]
        for row in rows:
            row["validations"] = json.loads(row.pop("validations_json", "[]") or "[]")
        return rows

    def available_run_horizons(self, run_id: str) -> set[int]:
        horizons = {
            int(row["evaluation_horizon"]) for row in self.list_run_horizon_snapshots(run_id)
        }
        current = self.get_run(run_id)
        if current is not None:
            try:
                horizons.add(
                    int(
                        json.loads(str(current.get("result_json", "{}") or "{}")).get(
                            "evaluations", 0
                        )
                        or 0
                    )
                )
            except Exception:
                pass
        horizons.discard(0)
        return horizons

    def get_run_evidence_at_horizon(self, run_id: str, evaluation_horizon: int) -> dict | None:
        """Return preserved/current run evidence for exactly one FE horizon without mixing horizons."""
        horizon = int(evaluation_horizon)
        current = self.get_run(run_id)
        if current is not None:
            try:
                current_eval = int(
                    json.loads(str(current.get("result_json", "{}") or "{}")).get("evaluations", 0)
                    or 0
                )
            except Exception:
                current_eval = 0
            if current_eval == horizon:
                out = dict(current)
                out["evaluation_horizon"] = horizon
                out["evidence_source"] = "current"
                return out
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM run_horizon_snapshots WHERE run_id=? AND evaluation_horizon=?",
                (str(run_id), horizon),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["validations"] = json.loads(out.pop("validations_json", "[]") or "[]")
        out["evidence_source"] = "snapshot"
        return out

    def list_experiment_runs_at_horizon(
        self, experiment_id: str, evaluation_horizon: int
    ) -> list[dict]:
        """Return only runs with evidence at the requested horizon; never mix FE horizons silently."""
        with self.connect() as con:
            run_rows = [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM runs WHERE experiment_id=? ORDER BY algorithm,run_index",
                    (str(experiment_id),),
                ).fetchall()
            ]
        out = []
        for run in run_rows:
            evidence = self.get_run_evidence_at_horizon(str(run["id"]), int(evaluation_horizon))
            if evidence is None:
                continue
            # Snapshot rows have their own snapshot primary key. Preserve it separately while
            # exposing the immutable logical run ID consistently to every statistics/GUI caller.
            if str(evidence.get("evidence_source", "")) == "snapshot":
                evidence["snapshot_id"] = str(evidence.get("id", ""))
            evidence["id"] = str(run["id"])
            evidence["run_id"] = str(run["id"])
            evidence["experiment_id"] = str(experiment_id)
            evidence["algorithm"] = str(run["algorithm"])
            evidence["run_index"] = int(run["run_index"])
            evidence["seed_json"] = str(run["seed_json"])
            out.append(evidence)
        return out

    def list_experiment_horizons(self, experiment_id: str) -> list[int]:
        horizons: set[int] = set()
        for row in self.list_runs(experiment_id):
            horizons.update(self.available_run_horizons(str(row["id"])))
        return sorted(horizons)

    def experiment_horizon_status(self, experiment_id: str, evaluation_horizon: int) -> dict:
        """Describe completeness/eligibility for one evidence horizon without mixing revisions."""
        horizon = int(evaluation_horizon)
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        config = json.loads(str(experiment.get("config_json", "{}") or "{}"))
        algorithms = [str(name) for name in config.get("algorithms", [])]
        revisions = [
            row
            for row in self.list_experiment_revisions(experiment_id)
            if int(row.get("evaluation_target", 0)) == horizon
            and str(row.get("status")) == "completed"
        ]
        # Prefer the latest completed publication-eligible revision at this horizon; otherwise the
        # latest completed exploratory revision. Multiple run-count revisions at the same horizon
        # naturally select the largest/latest completed evidence target.
        eligible = [row for row in revisions if bool(row.get("publication_eligible"))]
        revision = (eligible or revisions)[-1] if (eligible or revisions) else None
        protocol = dict(revision.get("protocol", {}) if revision else {})
        run_target = int(
            revision.get("run_target", config.get("runs", 0)) if revision else config.get("runs", 0)
        )
        selected_algorithms = [str(v) for v in protocol.get("algorithms", algorithms)] or algorithms
        selected_runs = [int(v) for v in protocol.get("run_indices", [])]
        if not selected_runs:
            selected_runs = list(range(run_target))
        expected = {
            (algorithm, run_index)
            for algorithm in selected_algorithms
            for run_index in selected_runs
        }
        rows = self.list_experiment_runs_at_horizon(experiment_id, horizon)
        actual = {(str(row["algorithm"]), int(row["run_index"])) for row in rows}
        missing = sorted(expected - actual)
        legacy_original_eligible = revision is None and horizon == int(
            (config.get("budget", {}) or {}).get("max_evaluations", 0) or 0
        )
        return {
            "experiment_id": str(experiment_id),
            "evaluation_horizon": horizon,
            "revision": revision,
            "publication_eligible": bool(
                (revision and revision.get("publication_eligible")) or legacy_original_eligible
            ),
            "expected_count": len(expected),
            "available_count": len(expected & actual),
            "complete": bool(expected) and expected.issubset(actual),
            "missing": missing,
            "rows": rows,
        }

    def list_run_segments(self, run_id: str) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM run_segments WHERE run_id=? ORDER BY segment_index", (str(run_id),)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            out.append(item)
        return out

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
        snapshot_paths: list[str] = []
        if run_ids:
            with self.connect() as con:
                placeholders = ",".join("?" for _ in run_ids)
                snapshot_paths = [
                    str(row["arrays_path"])
                    for row in con.execute(
                        f"SELECT arrays_path FROM run_horizon_snapshots WHERE run_id IN ({placeholders})",
                        run_ids,
                    ).fetchall()
                ]
        existing, missing, trace_bytes = self._trace_file_stats(
            [row["arrays_path"] for row in run_rows] + snapshot_paths
        )
        return {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "created_at": experiment["created_at"],
            "runs": len(run_rows),
            "failures": int(failures),
            "validations": int(validations),
            "verified_runs": sum(1 for row in run_rows if row["validation_status"] == "verified"),
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
            "missing_trace_files": sum(item["missing_trace_files"] for item in summaries),
            "trace_bytes": sum(item["trace_bytes"] for item in summaries),
        }

    def _delete_trace_files(self, array_paths: list[str]) -> dict:
        deleted = 0
        missing = 0
        failed = 0
        shared = 0
        reclaimed_bytes = 0
        for value in dict.fromkeys(array_paths):
            # Exact-reuse portfolios may share one immutable trace file. Delete it only when the
            # final database reference has been removed.
            with self.connect() as con:
                references = int(
                    con.execute(
                        "SELECT (SELECT COUNT(*) FROM runs WHERE arrays_path=?) + "
                        "(SELECT COUNT(*) FROM run_horizon_snapshots WHERE arrays_path=?)",
                        (str(value), str(value)),
                    ).fetchone()[0]
                )
            if references > 0:
                shared += 1
                continue
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
            "trace_files_shared": shared,
            "trace_files_missing": missing,
            "trace_files_failed": failed,
            "trace_bytes_reclaimed": reclaimed_bytes,
        }

    def _delete_checkpoint_files(self, paths: list[str]) -> dict:
        """Delete run-resume checkpoints owned exclusively by deleted logical runs."""
        deleted = 0
        missing = 0
        failed = 0
        for value in dict.fromkeys(str(v) for v in paths if str(v or "").strip()):
            path = self._resolve_array_path(value) or Path(value).expanduser()
            try:
                if path.is_file():
                    path.unlink()
                    deleted += 1
                else:
                    missing += 1
                path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
            except OSError:
                failed += 1
        return {
            "checkpoint_files_deleted": deleted,
            "checkpoint_files_missing": missing,
            "checkpoint_files_failed": failed,
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
            snapshot_paths = [
                str(item["arrays_path"])
                for item in con.execute(
                    "SELECT arrays_path FROM run_horizon_snapshots WHERE run_id=?", (run_id,)
                ).fetchall()
            ]
            checkpoint_paths = [
                str(item["checkpoint_path"])
                for item in con.execute(
                    "SELECT checkpoint_path FROM run_segments WHERE run_id=?", (run_id,)
                ).fetchall()
            ]
            con.execute("DELETE FROM validations WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM run_segments WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM run_horizon_snapshots WHERE run_id=?", (run_id,))
            con.execute(
                "UPDATE campaign_tasks SET run_id=NULL,status='deleted',last_activity=? WHERE run_id=?",
                (self._utcnow(), run_id),
            )
            con.execute("DELETE FROM runs WHERE id=?", (run_id,))
            array_paths = [row["arrays_path"], *snapshot_paths]
        trace_summary = self._delete_trace_files(array_paths)
        checkpoint_summary = self._delete_checkpoint_files(checkpoint_paths)
        if compact:
            self._compact_database()
        return {
            "experiments_deleted": 0,
            "runs_deleted": 1,
            "failures_deleted": 0,
            "validations_deleted": int(validations),
            **trace_summary,
            **checkpoint_summary,
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
            snapshot_paths: list[str] = []
            checkpoint_paths: list[str] = []
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                validations = con.execute(
                    f"SELECT COUNT(*) FROM validations WHERE run_id IN ({placeholders})",
                    run_ids,
                ).fetchone()[0]
                snapshot_paths = [
                    str(item["arrays_path"])
                    for item in con.execute(
                        f"SELECT arrays_path FROM run_horizon_snapshots WHERE run_id IN ({placeholders})",
                        run_ids,
                    ).fetchall()
                ]
                checkpoint_paths = [
                    str(item["checkpoint_path"])
                    for item in con.execute(
                        f"SELECT checkpoint_path FROM run_segments WHERE run_id IN ({placeholders})",
                        run_ids,
                    ).fetchall()
                ]
                con.execute(f"DELETE FROM validations WHERE run_id IN ({placeholders})", run_ids)
                con.execute(f"DELETE FROM run_segments WHERE run_id IN ({placeholders})", run_ids)
                con.execute(
                    f"DELETE FROM run_horizon_snapshots WHERE run_id IN ({placeholders})", run_ids
                )
            failures = con.execute(
                "SELECT COUNT(*) FROM run_failures WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()[0]
            campaign_rows = con.execute(
                "SELECT id FROM campaigns WHERE experiment_id=?", (experiment_id,)
            ).fetchall()
            campaign_ids = [str(row["id"]) for row in campaign_rows]
            if campaign_ids:
                placeholders = ",".join("?" for _ in campaign_ids)
                task_ids = [
                    str(row["id"])
                    for row in con.execute(
                        f"SELECT id FROM campaign_tasks WHERE campaign_id IN ({placeholders})",
                        campaign_ids,
                    ).fetchall()
                ]
                if task_ids:
                    task_placeholders = ",".join("?" for _ in task_ids)
                    con.execute(
                        f"DELETE FROM task_events WHERE task_id IN ({task_placeholders})", task_ids
                    )
                con.execute(
                    f"DELETE FROM campaign_tasks WHERE campaign_id IN ({placeholders})",
                    campaign_ids,
                )
                con.execute(f"DELETE FROM campaigns WHERE id IN ({placeholders})", campaign_ids)
                con.execute(
                    f"DELETE FROM resumable_tasks WHERE id IN ({placeholders})", campaign_ids
                )
            con.execute("DELETE FROM runs WHERE experiment_id=?", (experiment_id,))
            con.execute("DELETE FROM run_failures WHERE experiment_id=?", (experiment_id,))
            con.execute(
                "DELETE FROM experiment_workspace_state WHERE experiment_id=?", (experiment_id,)
            )
            con.execute(
                "DELETE FROM experiment_policy_bindings WHERE experiment_id=?", (experiment_id,)
            )
            con.execute("DELETE FROM experiment_revisions WHERE experiment_id=?", (experiment_id,))
            con.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
            array_paths = [row["arrays_path"] for row in run_rows] + snapshot_paths
        trace_summary = self._delete_trace_files(array_paths)
        checkpoint_summary = self._delete_checkpoint_files(checkpoint_paths)
        if compact:
            self._compact_database()
        return {
            "experiments_deleted": 1,
            "runs_deleted": len(run_rows),
            "failures_deleted": int(failures),
            "validations_deleted": int(validations),
            **trace_summary,
            **checkpoint_summary,
        }

    def clear_history(self) -> dict:
        """Delete all experiment history and all referenced run-array traces."""
        with self._lock, self.connect() as con:
            experiment_count = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            run_rows = con.execute("SELECT id,arrays_path FROM runs").fetchall()
            snapshot_paths = [
                str(row["arrays_path"])
                for row in con.execute("SELECT arrays_path FROM run_horizon_snapshots").fetchall()
            ]
            checkpoint_paths = [
                str(row["checkpoint_path"])
                for row in con.execute("SELECT checkpoint_path FROM run_segments").fetchall()
            ]
            failure_count = con.execute("SELECT COUNT(*) FROM run_failures").fetchone()[0]
            validation_count = con.execute("SELECT COUNT(*) FROM validations").fetchone()[0]
            con.execute("DELETE FROM validations")
            con.execute("DELETE FROM run_segments")
            con.execute("DELETE FROM run_horizon_snapshots")
            con.execute("DELETE FROM experiment_revisions")
            con.execute("DELETE FROM task_events")
            con.execute("DELETE FROM campaign_tasks")
            con.execute("DELETE FROM campaigns")
            con.execute("DELETE FROM resumable_tasks")
            con.execute("DELETE FROM runs")
            con.execute("DELETE FROM run_failures")
            con.execute("DELETE FROM experiment_workspace_state")
            con.execute("DELETE FROM experiment_policy_bindings")
            con.execute("DELETE FROM experiments")
            con.execute("DELETE FROM portfolios")
            array_paths = [row["arrays_path"] for row in run_rows] + snapshot_paths
        trace_summary = self._delete_trace_files(array_paths)
        checkpoint_summary = self._delete_checkpoint_files(checkpoint_paths)
        self._compact_database()
        return {
            "experiments_deleted": int(experiment_count),
            "runs_deleted": len(run_rows),
            "failures_deleted": int(failure_count),
            "validations_deleted": int(validation_count),
            **trace_summary,
            **checkpoint_summary,
        }
