"""SQLite experiment, run, validation, failure, and trace repository."""

from __future__ import annotations

import logging

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid

from calo_rpd_studio.experiments.execution_plans import resume_contract_sha256
from calo_rpd_studio.version import VERSION


_LOG = logging.getLogger(__name__)
DATABASE_SCHEMA_VERSION = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class ResultDatabase:
    """Persist experiment metadata and provide safe history deletion operations.

    Deletion methods remove the selected database records together with the referenced
    compressed ``.npz`` run-array files.  External publication export directories are
    intentionally not touched because they may contain user-managed copies.
    """

    def __init__(self, path="calo_rpd_results.sqlite", *, read_only: bool = False):
        self.path = str(path)
        self.read_only = bool(read_only)
        self.migration_backup_path: str | None = None
        self._lock = threading.RLock()
        if self.read_only:
            source_version, has_user_tables = self._inspect_existing_schema()
            if has_user_tables and source_version < DATABASE_SCHEMA_VERSION:
                raise RuntimeError(
                    "Read-only database access cannot migrate an older schema; create and review "
                    "a migration backup through the ordinary application workflow first"
                )
        else:
            self._initialize()

    def _inspect_existing_schema(self) -> tuple[int, bool]:
        """Return ``(user_version, has_user_tables)`` without mutating the database."""
        if self.path == ":memory:":
            return 0, False
        source = Path(self.path)
        if not source.is_file() or source.stat().st_size == 0:
            return 0, False
        uri = f"{source.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=30) as con:
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            has_tables = (
                con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone()
                is not None
            )
        if version > DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema v{version} is newer than supported v{DATABASE_SCHEMA_VERSION}; "
                "open it with a compatible CALO-RPD Studio release."
            )
        return version, has_tables

    def _backup_for_migration(self, source_version: int) -> tuple[str, str]:
        """Create and verify a consistent SQLite backup before any legacy-schema DDL."""
        source_path = Path(self.path).resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = source_path.with_name(
            f"{source_path.stem}.pre-schema-v{source_version}-to-v{DATABASE_SCHEMA_VERSION}-{stamp}.sqlite"
        )
        source_uri = f"{source_path.as_uri()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True, timeout=30) as source:
            with sqlite3.connect(backup_path, timeout=30) as backup:
                source.backup(backup)
                integrity = str(backup.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            raise RuntimeError(f"Pre-migration backup failed integrity_check: {integrity}")
        self.migration_backup_path = str(backup_path)
        return str(backup_path), _sha256_file(backup_path)

    @contextmanager
    def connect(self):
        if self.read_only:
            if self.path == ":memory:" or not Path(self.path).is_file():
                raise FileNotFoundError(f"Read-only database does not exist: {self.path}")
            uri = f"{Path(self.path).resolve().as_uri()}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=30)
        else:
            con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        if self.read_only:
            con.execute("PRAGMA query_only=ON")
        else:
            con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            if self.read_only:
                con.rollback()
            else:
                con.commit()
        finally:
            con.close()

    def _initialize(self):
        source_version, has_user_tables = self._inspect_existing_schema()
        backup_path = ""
        backup_sha256 = ""
        if has_user_tables and source_version < DATABASE_SCHEMA_VERSION:
            backup_path, backup_sha256 = self._backup_for_migration(source_version)
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
            execution_plan_id TEXT NOT NULL DEFAULT '',
            workspace_plan_cell_id TEXT NOT NULL DEFAULT '',
            job_identity_sha256 TEXT NOT NULL DEFAULT '',
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
        CREATE TABLE IF NOT EXISTS algorithm_stages(
            id TEXT PRIMARY KEY, schema_version TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, content_json TEXT NOT NULL, content_sha256 TEXT NOT NULL,
            record_sha256 TEXT NOT NULL, superseded_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS execution_plans(
            id TEXT PRIMARY KEY, plan_kind TEXT NOT NULL, schema_version TEXT NOT NULL,
            algorithm_stage_id TEXT NOT NULL, created_at TEXT NOT NULL,
            design_json TEXT NOT NULL, design_sha256 TEXT NOT NULL,
            audit_json TEXT NOT NULL DEFAULT '{}', audit_sha256 TEXT NOT NULL DEFAULT '',
            lifecycle_state TEXT NOT NULL, state_revision INTEGER NOT NULL DEFAULT 0,
            state_receipt_sha256 TEXT NOT NULL, campaign_id TEXT NOT NULL DEFAULT '',
            controller_epoch INTEGER NOT NULL DEFAULT 0, active_slot INTEGER NOT NULL DEFAULT 1,
            last_message TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
            FOREIGN KEY(algorithm_stage_id) REFERENCES algorithm_stages(id)
        );
        CREATE TABLE IF NOT EXISTS workspace_plan_cells(
            id TEXT PRIMARY KEY, workspace_plan_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            config_json TEXT NOT NULL, design_sha256 TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL DEFAULT 'planned', campaign_id TEXT NOT NULL DEFAULT '',
            experiment_id TEXT NOT NULL DEFAULT '', last_message TEXT NOT NULL DEFAULT '',
            UNIQUE(workspace_plan_id,ordinal), UNIQUE(workspace_plan_id,design_sha256),
            FOREIGN KEY(workspace_plan_id) REFERENCES execution_plans(id)
        );
        CREATE TABLE IF NOT EXISTS execution_controller(
            singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
            schema_version TEXT NOT NULL, controller TEXT NOT NULL, owner_plan_id TEXT NOT NULL,
            owner_design_sha256 TEXT NOT NULL, campaign_id TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL, epoch INTEGER NOT NULL, owner_instance_id TEXT NOT NULL,
            record_revision INTEGER NOT NULL, acquired_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            state_receipt_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS execution_lifecycle_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT NOT NULL,
            created_at TEXT NOT NULL, controller TEXT NOT NULL, controller_epoch INTEGER NOT NULL,
            from_state TEXT NOT NULL, to_state TEXT NOT NULL,
            state_revision INTEGER NOT NULL, receipt_sha256 TEXT NOT NULL,
            message TEXT NOT NULL DEFAULT '', payload_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(plan_id) REFERENCES execution_plans(id)
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
        CREATE TABLE IF NOT EXISTS suppressed_policies(
            sha256 TEXT PRIMARY KEY, created_at TEXT NOT NULL, reason TEXT NOT NULL DEFAULT ''
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
        CREATE TABLE IF NOT EXISTS schema_migrations(
            id TEXT PRIMARY KEY, applied_at TEXT NOT NULL,
            source_version INTEGER NOT NULL, target_version INTEGER NOT NULL,
            backup_path TEXT NOT NULL DEFAULT '', backup_sha256 TEXT NOT NULL DEFAULT '',
            application_version TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_failures_experiment ON run_failures(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_validations_run ON validations(run_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_status ON campaigns(status);
        CREATE INDEX IF NOT EXISTS idx_campaign_tasks_status ON campaign_tasks(campaign_id,status);
        CREATE INDEX IF NOT EXISTS idx_campaign_tasks_fingerprint ON campaign_tasks(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_resumable_status ON resumable_tasks(status,resumable);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_algorithm_stage_active
            ON algorithm_stages(status) WHERE status='active';
        CREATE UNIQUE INDEX IF NOT EXISTS uq_execution_plan_active_kind
            ON execution_plans(plan_kind) WHERE active_slot=1;
        CREATE INDEX IF NOT EXISTS idx_execution_plan_state
            ON execution_plans(plan_kind,lifecycle_state,updated_at);
        CREATE INDEX IF NOT EXISTS idx_workspace_cells_plan
            ON workspace_plan_cells(workspace_plan_id,ordinal);
        CREATE INDEX IF NOT EXISTS idx_execution_events_plan
            ON execution_lifecycle_events(plan_id,id);
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
            # BEGIN is inside executescript so its implicit pre-script COMMIT cannot split the
            # migration. The context manager commits only after every dynamic ALTER and the
            # user_version receipt succeed; closing after an exception rolls the transaction back.
            con.executescript("BEGIN IMMEDIATE;\n" + schema)
            # Version-0 migrations preserve all historical rows. Existing experiments are
            # deliberately excluded from learning until the user classifies them.
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
            task_columns = {
                row["name"] for row in con.execute("PRAGMA table_info(campaign_tasks)").fetchall()
            }
            for name in (
                "execution_plan_id",
                "workspace_plan_cell_id",
                "job_identity_sha256",
            ):
                if name not in task_columns:
                    con.execute(
                        f"ALTER TABLE campaign_tasks ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                    )
            con.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_campaign_task_plan_job
                   ON campaign_tasks(execution_plan_id,workspace_plan_cell_id,job_identity_sha256)
                   WHERE execution_plan_id<>'' AND job_identity_sha256<>''"""
            )
            controller_columns = {
                row["name"]
                for row in con.execute("PRAGMA table_info(execution_controller)").fetchall()
            }
            if "acquired_at" not in controller_columns:
                con.execute(
                    "ALTER TABLE execution_controller ADD COLUMN acquired_at TEXT NOT NULL DEFAULT ''"
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
            if source_version < DATABASE_SCHEMA_VERSION:
                con.execute(
                    """INSERT INTO schema_migrations(
                        id,applied_at,source_version,target_version,
                        backup_path,backup_sha256,application_version
                    ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        datetime.now(timezone.utc).isoformat(),
                        int(source_version),
                        DATABASE_SCHEMA_VERSION,
                        backup_path,
                        backup_sha256,
                        VERSION,
                    ),
                )
                con.execute(f"PRAGMA user_version={DATABASE_SCHEMA_VERSION}")
            controller_payload = {
                "schema_version": "calo-rpd-execution-controller-v1",
                "controller": "none",
                "owner_plan_id": "",
                "owner_design_sha256": "",
                "campaign_id": "",
                "lifecycle_state": "",
                "epoch": 0,
                "owner_instance_id": "",
                "record_revision": 0,
                "acquired_at": "",
            }
            controller_receipt = hashlib.sha256(
                json.dumps(
                    controller_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            con.execute(
                """INSERT OR IGNORE INTO execution_controller(
                    singleton_id,schema_version,controller,owner_plan_id,owner_design_sha256,
                    campaign_id,lifecycle_state,epoch,owner_instance_id,record_revision,
                    acquired_at,updated_at,state_receipt_sha256
                ) VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    controller_payload["schema_version"],
                    controller_payload["controller"],
                    "",
                    "",
                    "",
                    "",
                    0,
                    "",
                    0,
                    "",
                    datetime.now(timezone.utc).isoformat(),
                    controller_receipt,
                ),
            )

    @property
    def schema_version(self) -> int:
        """Return the durable SQLite application schema version."""
        with self.connect() as con:
            return int(con.execute("PRAGMA user_version").fetchone()[0])

    # ------------------------------------------------------------------
    # Workspace/individual immutable plan and exclusive-controller state
    # ------------------------------------------------------------------

    @staticmethod
    def _execution_state_receipt(
        *,
        plan_id: str,
        design_sha256: str,
        lifecycle_state: str,
        state_revision: int,
        controller_epoch: int,
        campaign_id: str = "",
        prior_receipt_sha256: str = "",
    ) -> str:
        return _canonical_sha256(
            {
                "schema_version": "calo-rpd-execution-state-receipt-v1",
                "plan_id": str(plan_id),
                "design_sha256": str(design_sha256),
                "lifecycle_state": str(lifecycle_state),
                "state_revision": int(state_revision),
                "controller_epoch": int(controller_epoch),
                "campaign_id": str(campaign_id),
                "prior_receipt_sha256": str(prior_receipt_sha256),
            }
        )

    @staticmethod
    def _controller_receipt(payload: dict) -> str:
        return _canonical_sha256(
            {
                "schema_version": str(payload["schema_version"]),
                "controller": str(payload["controller"]),
                "owner_plan_id": str(payload["owner_plan_id"]),
                "owner_design_sha256": str(payload["owner_design_sha256"]),
                "campaign_id": str(payload["campaign_id"]),
                "lifecycle_state": str(payload["lifecycle_state"]),
                "epoch": int(payload["epoch"]),
                "owner_instance_id": str(payload["owner_instance_id"]),
                "record_revision": int(payload["record_revision"]),
                "acquired_at": str(payload.get("acquired_at", "")),
            }
        )

    @classmethod
    def _verified_controller_payload(cls, row) -> dict:
        if row is None:
            raise RuntimeError("The execution-controller singleton record is missing")
        payload = dict(row)
        if str(payload["state_receipt_sha256"]) != cls._controller_receipt(payload):
            raise RuntimeError("The execution-controller integrity receipt does not match")
        return payload

    @staticmethod
    def _verified_plan_design(row) -> dict:
        if row is None:
            raise RuntimeError("The execution-plan record is missing")
        design = json.loads(str(row["design_json"]))
        if _canonical_sha256(design) != str(row["design_sha256"]):
            raise RuntimeError("The immutable execution-plan design checksum does not match")
        return design

    @classmethod
    def _verified_campaign_plan_binding(cls, con, plan, campaign_id: str) -> dict:
        """Verify a campaign config against the exact immutable plan or Workspace cell."""

        design = cls._verified_plan_design(plan)
        campaign = con.execute(
            "SELECT * FROM campaigns WHERE id=?", (str(campaign_id),)
        ).fetchone()
        if campaign is None:
            raise KeyError(f"Unknown campaign: {campaign_id}")
        stored = json.loads(str(campaign["config_json"]))
        if (
            str(stored.get("execution_plan_id", "")) != str(plan["id"])
            or str(stored.get("execution_plan_design_sha256", ""))
            != str(plan["design_sha256"])
            or str(stored.get("algorithm_stage_id", ""))
            != str(plan["algorithm_stage_id"])
        ):
            raise RuntimeError("The retained campaign identity does not match its execution plan")
        cell_id = str(stored.get("workspace_plan_cell_id", "") or "")
        if str(plan["plan_kind"]) == "workspace":
            if not cell_id:
                raise RuntimeError("A Workspace campaign is missing its immutable cell identity")
            cell = con.execute(
                """SELECT config_json FROM workspace_plan_cells
                   WHERE id=? AND workspace_plan_id=?""",
                (cell_id, str(plan["id"])),
            ).fetchone()
            if cell is None:
                raise RuntimeError("The retained campaign refers to a foreign Workspace cell")
            expected = json.loads(str(cell["config_json"]))
        else:
            if cell_id:
                raise RuntimeError("An individual campaign cannot claim a Workspace cell")
            expected = dict(design["config"])
        expected["execution_plan_id"] = str(plan["id"])
        expected["execution_plan_design_sha256"] = str(plan["design_sha256"])
        expected["algorithm_stage_id"] = str(plan["algorithm_stage_id"])
        expected["workspace_plan_cell_id"] = cell_id
        if resume_contract_sha256(stored) != resume_contract_sha256(expected):
            raise RuntimeError("The retained campaign does not match the frozen resume contract")
        return dict(campaign)

    @classmethod
    def _discard_unstarted_plans(
        cls,
        con,
        rows,
        *,
        now: str,
        message: str,
        controller_epoch: int,
    ) -> None:
        """Close selected drafts while preserving the authenticated receipt chain."""

        for plan in rows:
            cls._verified_plan_design(plan)
            prior_state = str(plan["lifecycle_state"])
            if prior_state not in {"draft", "audited"}:
                raise RuntimeError("Only an unstarted execution plan can be superseded")
            revision = int(plan["state_revision"]) + 1
            receipt = cls._execution_state_receipt(
                plan_id=str(plan["id"]),
                design_sha256=str(plan["design_sha256"]),
                lifecycle_state="discarded_unstarted",
                state_revision=revision,
                controller_epoch=int(controller_epoch),
                campaign_id=str(plan["campaign_id"]),
                prior_receipt_sha256=str(plan["state_receipt_sha256"]),
            )
            con.execute(
                """UPDATE execution_plans SET lifecycle_state='discarded_unstarted',
                    active_slot=0,state_revision=?,state_receipt_sha256=?,updated_at=?,
                    last_message=? WHERE id=?""",
                (revision, receipt, now, str(message), str(plan["id"])),
            )
            con.execute(
                """INSERT INTO execution_lifecycle_events(
                    plan_id,created_at,controller,controller_epoch,from_state,to_state,
                    state_revision,receipt_sha256,message
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(plan["id"]),
                    now,
                    "none",
                    int(controller_epoch),
                    prior_state,
                    "discarded_unstarted",
                    revision,
                    receipt,
                    str(message),
                ),
            )

    @staticmethod
    def _decoded_execution_plan(row) -> dict | None:
        if row is None:
            return None
        payload = dict(row)
        payload["design"] = json.loads(str(payload.pop("design_json")))
        payload["audit"] = json.loads(str(payload.pop("audit_json")))
        return payload

    def get_execution_controller(self) -> dict:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
        return self._verified_controller_payload(row)

    def get_active_algorithm_stage(self) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM algorithm_stages WHERE status='active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["content"] = json.loads(str(payload.pop("content_json")))
        if _canonical_sha256(payload["content"]) != str(payload["content_sha256"]):
            raise RuntimeError("The submitted algorithm-stage content checksum does not match")
        record = {
            "schema_version": str(payload["schema_version"]),
            "stage_id": str(payload["id"]),
            "created_at": str(payload["created_at"]),
            "content_sha256": str(payload["content_sha256"]),
        }
        if _canonical_sha256(record) != str(payload["record_sha256"]):
            raise RuntimeError("The submitted algorithm-stage record checksum does not match")
        return payload

    def replace_algorithm_stage(self, stage) -> None:
        """Submit one explicit stage and invalidate only unstarted drafts from the prior stage."""

        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            controller = self._verified_controller_payload(controller)
            if str(controller["controller"]) != "none":
                owner = str(controller["owner_plan_id"])
                raise RuntimeError(
                    f"Algorithm submission is blocked while execution plan {owner!r} owns control"
                )
            resumable = con.execute(
                """SELECT id FROM execution_plans
                   WHERE active_slot=1 AND lifecycle_state IN ('paused','interrupted_resumable')
                   LIMIT 1"""
            ).fetchone()
            if resumable is not None:
                raise RuntimeError(
                    "Algorithm submission is blocked while resumable plan "
                    f"{str(resumable['id'])!r} remains bound to the current stage"
                )
            drafts = con.execute(
                """SELECT * FROM execution_plans
                   WHERE active_slot=1 AND lifecycle_state IN ('draft','audited')"""
            ).fetchall()
            self._discard_unstarted_plans(
                con,
                drafts,
                now=now,
                message="Superseded by an explicitly submitted algorithm stage",
                controller_epoch=int(controller["epoch"]),
            )
            con.execute(
                "UPDATE algorithm_stages SET status='superseded',superseded_at=? WHERE status='active'",
                (now,),
            )
            con.execute(
                """INSERT INTO algorithm_stages(
                    id,schema_version,created_at,status,content_json,content_sha256,record_sha256
                ) VALUES(?,?,?,?,?,?,?)""",
                (
                    stage.stage_id,
                    stage.schema_version,
                    stage.created_at,
                    "active",
                    json.dumps(stage.content_payload(), sort_keys=True, allow_nan=False),
                    stage.content_sha256,
                    stage.record_sha256,
                ),
            )

    def discard_algorithm_stage(self) -> None:
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            controller = self._verified_controller_payload(controller)
            if str(controller["controller"]) != "none":
                owner = str(controller["owner_plan_id"])
                raise RuntimeError(
                    f"Algorithm reset is blocked while execution plan {owner!r} owns control"
                )
            resumable = con.execute(
                """SELECT id FROM execution_plans
                   WHERE active_slot=1 AND lifecycle_state IN ('paused','interrupted_resumable')
                   LIMIT 1"""
            ).fetchone()
            if resumable is not None:
                raise RuntimeError(
                    "Algorithm reset is blocked while resumable plan "
                    f"{str(resumable['id'])!r} remains bound to the stage"
                )
            con.execute(
                "UPDATE algorithm_stages SET status='discarded',superseded_at=? WHERE status='active'",
                (now,),
            )
            drafts = con.execute(
                """SELECT * FROM execution_plans
                   WHERE active_slot=1 AND lifecycle_state IN ('draft','audited')"""
            ).fetchall()
            self._discard_unstarted_plans(
                con,
                drafts,
                now=now,
                message="Invalidated by explicit algorithm-stage reset",
                controller_epoch=int(controller["epoch"]),
            )

    def create_execution_plan(self, plan, *, plan_kind: str) -> str:
        """Persist one immutable design as a draft without acquiring execution authority."""

        kind = str(plan_kind)
        if kind not in {"workspace", "individual_experiment"}:
            raise ValueError(f"Unsupported execution plan kind: {kind}")
        now = self._utcnow()
        design = plan.design_payload()
        if _canonical_sha256(design) != str(plan.design_sha256):
            raise RuntimeError("Execution-plan design checksum does not match its canonical payload")
        receipt = self._execution_state_receipt(
            plan_id=plan.plan_id,
            design_sha256=plan.design_sha256,
            lifecycle_state="draft",
            state_revision=0,
            controller_epoch=0,
        )
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            controller = self._verified_controller_payload(controller)
            if str(controller["controller"]) != "none":
                raise RuntimeError("A new plan cannot be created while another execution controller is active")
            stage = con.execute(
                "SELECT content_sha256 FROM algorithm_stages WHERE id=? AND status='active'",
                (str(plan.algorithm_stage_id),),
            ).fetchone()
            if stage is None or str(stage["content_sha256"]) != str(plan.algorithm_stage_sha256):
                raise RuntimeError("The plan is not bound to the currently submitted algorithm stage")
            existing = con.execute(
                "SELECT * FROM execution_plans WHERE plan_kind=? AND active_slot=1",
                (kind,),
            ).fetchone()
            if existing is not None:
                if str(existing["lifecycle_state"]) not in {"draft", "audited"}:
                    raise RuntimeError(
                        f"Plan {str(existing['id'])!r} must be resumed or closed before another {kind} plan is created"
                    )
                self._discard_unstarted_plans(
                    con,
                    (existing,),
                    now=now,
                    message="Superseded by a new unstarted draft",
                    controller_epoch=int(controller["epoch"]),
                )
            con.execute(
                """INSERT INTO execution_plans(
                    id,plan_kind,schema_version,algorithm_stage_id,created_at,design_json,
                    design_sha256,lifecycle_state,state_revision,state_receipt_sha256,
                    active_slot,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,1,?)""",
                (
                    plan.plan_id,
                    kind,
                    plan.schema_version,
                    plan.algorithm_stage_id,
                    plan.created_at,
                    json.dumps(design, sort_keys=True, allow_nan=False),
                    plan.design_sha256,
                    "draft",
                    0,
                    receipt,
                    now,
                ),
            )
            for cell in tuple(getattr(plan, "cells", ()) or ()):
                con.execute(
                    """INSERT INTO workspace_plan_cells(
                        id,workspace_plan_id,ordinal,config_json,design_sha256
                    ) VALUES(?,?,?,?,?)""",
                    (
                        str(cell["cell_id"]),
                        plan.plan_id,
                        int(cell["ordinal"]),
                        json.dumps(cell["config"], sort_keys=True, allow_nan=False),
                        str(cell["design_sha256"]),
                    ),
                )
        return str(plan.plan_id)

    def get_execution_plan(self, plan_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM execution_plans WHERE id=?", (str(plan_id),)).fetchone()
        payload = self._decoded_execution_plan(row)
        if payload is None:
            return None
        if _canonical_sha256(payload["design"]) != str(payload["design_sha256"]):
            raise RuntimeError("The immutable execution-plan design checksum does not match")
        audit_sha = str(payload.get("audit_sha256", "") or "")
        if audit_sha:
            canonical_audit = dict(payload["audit"])
            canonical_audit.pop("audit_sha256", None)
            if _canonical_sha256(canonical_audit) != audit_sha:
                raise RuntimeError("The execution-plan audit receipt checksum does not match")
        elif str(payload["lifecycle_state"]) not in {"draft", "discarded_unstarted"}:
            raise RuntimeError("A non-draft execution plan is missing its audit receipt")
        return payload

    def get_active_execution_plan(self, plan_kind: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT id FROM execution_plans WHERE plan_kind=? AND active_slot=1",
                (str(plan_kind),),
            ).fetchone()
        return None if row is None else self.get_execution_plan(str(row["id"]))

    def list_workspace_plan_cells(self, plan_id: str) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM workspace_plan_cells WHERE workspace_plan_id=? ORDER BY ordinal",
                (str(plan_id),),
            ).fetchall()
        payloads = []
        for row in rows:
            payload = dict(row)
            payload["config"] = json.loads(str(payload.pop("config_json")))
            expected = _canonical_sha256(
                {
                    "plan_id": str(plan_id),
                    "ordinal": int(payload["ordinal"]),
                    "config": payload["config"],
                }
            )
            if expected != str(payload["design_sha256"]):
                raise RuntimeError(
                    f"Workspace plan cell {str(payload['id'])!r} checksum does not match"
                )
            payloads.append(payload)
        return payloads

    def set_execution_plan_audited(self, plan_id: str, audit_receipt: dict) -> dict:
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = self._verified_controller_payload(
                con.execute(
                    "SELECT * FROM execution_controller WHERE singleton_id=1"
                ).fetchone()
            )
            if str(controller["controller"]) != "none":
                raise RuntimeError("An audit receipt cannot be committed while execution is owned")
            row = con.execute("SELECT * FROM execution_plans WHERE id=?", (str(plan_id),)).fetchone()
            if row is None:
                raise KeyError(f"Unknown execution plan: {plan_id}")
            self._verified_plan_design(row)
            if str(row["lifecycle_state"]) not in {"draft", "audited"}:
                raise RuntimeError("Only an unstarted draft can receive an audit receipt")
            if str(audit_receipt.get("design_sha256", "")) != str(row["design_sha256"]):
                raise RuntimeError("The fairness audit was produced for a different plan design")
            audit_sha = str(audit_receipt.get("audit_sha256", ""))
            canonical = dict(audit_receipt)
            canonical.pop("audit_sha256", None)
            if not audit_sha or _canonical_sha256(canonical) != audit_sha:
                raise RuntimeError("The fairness-audit receipt checksum does not match")
            revision = int(row["state_revision"]) + 1
            receipt = self._execution_state_receipt(
                plan_id=str(plan_id),
                design_sha256=str(row["design_sha256"]),
                lifecycle_state="audited",
                state_revision=revision,
                controller_epoch=0,
                prior_receipt_sha256=str(row["state_receipt_sha256"]),
            )
            con.execute(
                """UPDATE execution_plans SET audit_json=?,audit_sha256=?,lifecycle_state='audited',
                    state_revision=?,state_receipt_sha256=?,updated_at=?,last_message=? WHERE id=?""",
                (
                    json.dumps(audit_receipt, sort_keys=True, allow_nan=False),
                    audit_sha,
                    revision,
                    receipt,
                    now,
                    "Fairness audit bound to immutable plan",
                    str(plan_id),
                ),
            )
        return self.get_execution_plan(str(plan_id)) or {}

    def discard_unstarted_execution_plan(self, plan_id: str, *, message: str) -> dict:
        """Close a draft/audited plan without acquiring or fabricating execution ownership."""

        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            plan = con.execute(
                "SELECT * FROM execution_plans WHERE id=?", (str(plan_id),)
            ).fetchone()
            controller = self._verified_controller_payload(controller)
            if plan is None:
                raise RuntimeError("Execution plan or controller record is missing")
            self._verified_plan_design(plan)
            if str(controller["controller"]) != "none":
                raise RuntimeError("An unstarted draft cannot be discarded while a controller owns work")
            if str(plan["lifecycle_state"]) not in {"draft", "audited"}:
                raise RuntimeError("Only a draft or audited unstarted plan can use draft discard")
            revision = int(plan["state_revision"]) + 1
            receipt = self._execution_state_receipt(
                plan_id=str(plan_id),
                design_sha256=str(plan["design_sha256"]),
                lifecycle_state="discarded_unstarted",
                state_revision=revision,
                controller_epoch=int(controller["epoch"]),
                campaign_id=str(plan["campaign_id"]),
                prior_receipt_sha256=str(plan["state_receipt_sha256"]),
            )
            con.execute(
                """UPDATE execution_plans SET lifecycle_state='discarded_unstarted',
                   state_revision=?,state_receipt_sha256=?,active_slot=0,updated_at=?,
                   last_message=? WHERE id=?""",
                (revision, receipt, now, str(message), str(plan_id)),
            )
            con.execute(
                """INSERT INTO execution_lifecycle_events(
                    plan_id,created_at,controller,controller_epoch,from_state,to_state,
                    state_revision,receipt_sha256,message
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(plan_id),
                    now,
                    "none",
                    int(controller["epoch"]),
                    str(plan["lifecycle_state"]),
                    "discarded_unstarted",
                    revision,
                    receipt,
                    str(message),
                ),
            )
        return self.get_execution_plan(str(plan_id)) or {}

    def acquire_execution_controller(
        self,
        plan_id: str,
        *,
        controller_kind: str,
        owner_instance_id: str,
        resume: bool = False,
    ) -> dict:
        """Atomically acquire the singleton controller and stage or resume the exact plan."""

        kind = str(controller_kind)
        if kind not in {"workspace", "individual_experiment"}:
            raise ValueError(f"Unsupported execution controller: {kind}")
        expected_plan_kind = "workspace" if kind == "workspace" else "individual_experiment"
        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            controller = self._verified_controller_payload(controller)
            if str(controller["controller"]) != "none":
                raise RuntimeError(
                    "Execution control is already owned by "
                    f"{str(controller['controller'])} plan {str(controller['owner_plan_id'])!r}"
                )
            plan = con.execute("SELECT * FROM execution_plans WHERE id=?", (str(plan_id),)).fetchone()
            if plan is None or str(plan["plan_kind"]) != expected_plan_kind:
                raise RuntimeError("The requested controller does not match the execution-plan kind")
            design = self._verified_plan_design(plan)
            expected_states = {"paused", "interrupted_resumable"} if resume else {"audited"}
            current_state = str(plan["lifecycle_state"])
            if current_state not in expected_states:
                raise RuntimeError(
                    f"Plan {plan_id!r} is {current_state!r}; expected one of {sorted(expected_states)}"
                )
            stage = con.execute(
                "SELECT content_sha256 FROM algorithm_stages WHERE id=? AND status='active'",
                (str(plan["algorithm_stage_id"]),),
            ).fetchone()
            if stage is None or str(stage["content_sha256"]) != str(
                design.get("algorithm_stage_sha256", "")
            ):
                raise RuntimeError("The execution plan no longer matches the active algorithm stage")
            if not resume:
                audit = json.loads(str(plan["audit_json"]))
                audit_sha = str(plan["audit_sha256"])
                canonical_audit = dict(audit)
                canonical_audit.pop("audit_sha256", None)
                if (
                    not audit_sha
                    or _canonical_sha256(canonical_audit) != audit_sha
                    or str(audit.get("design_sha256", "")) != str(plan["design_sha256"])
                ):
                    raise RuntimeError("The unchanged plan does not have a valid fairness-audit receipt")
            elif str(plan["campaign_id"] or ""):
                self._verified_campaign_plan_binding(
                    con, plan, str(plan["campaign_id"])
                )
            epoch = int(controller["epoch"]) + 1
            controller_revision = int(controller["record_revision"]) + 1
            plan_revision = int(plan["state_revision"]) + 1
            new_state = "running" if resume else "staged"
            plan_receipt = self._execution_state_receipt(
                plan_id=str(plan_id),
                design_sha256=str(plan["design_sha256"]),
                lifecycle_state=new_state,
                state_revision=plan_revision,
                controller_epoch=epoch,
                campaign_id=str(plan["campaign_id"]),
                prior_receipt_sha256=str(plan["state_receipt_sha256"]),
            )
            controller_payload = {
                "schema_version": "calo-rpd-execution-controller-v1",
                "controller": kind,
                "owner_plan_id": str(plan_id),
                "owner_design_sha256": str(plan["design_sha256"]),
                "campaign_id": str(plan["campaign_id"]),
                "lifecycle_state": new_state,
                "epoch": epoch,
                "owner_instance_id": str(owner_instance_id),
                "record_revision": controller_revision,
                "acquired_at": now,
            }
            controller_receipt = self._controller_receipt(controller_payload)
            con.execute(
                """UPDATE execution_plans SET lifecycle_state=?,state_revision=?,
                    state_receipt_sha256=?,controller_epoch=?,updated_at=?,last_message=? WHERE id=?""",
                (
                    new_state,
                    plan_revision,
                    plan_receipt,
                    epoch,
                    now,
                    "Execution controller reacquired" if resume else "Execution plan staged",
                    str(plan_id),
                ),
            )
            con.execute(
                """UPDATE execution_controller SET schema_version=?,controller=?,owner_plan_id=?,
                    owner_design_sha256=?,campaign_id=?,lifecycle_state=?,epoch=?,owner_instance_id=?,
                    record_revision=?,acquired_at=?,updated_at=?,state_receipt_sha256=? WHERE singleton_id=1""",
                (
                    controller_payload["schema_version"],
                    kind,
                    str(plan_id),
                    str(plan["design_sha256"]),
                    str(plan["campaign_id"]),
                    new_state,
                    epoch,
                    str(owner_instance_id),
                    controller_revision,
                    now,
                    now,
                    controller_receipt,
                ),
            )
            con.execute(
                """INSERT INTO execution_lifecycle_events(
                    plan_id,created_at,controller,controller_epoch,from_state,to_state,
                    state_revision,receipt_sha256,message
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(plan_id),
                    now,
                    kind,
                    epoch,
                    current_state,
                    new_state,
                    plan_revision,
                    plan_receipt,
                    "Resumed exact frozen plan" if resume else "Staged without starting work",
                ),
            )
        return self.get_execution_controller()

    def transition_execution_plan(
        self,
        plan_id: str,
        *,
        controller_epoch: int,
        expected_states: tuple[str, ...],
        new_state: str,
        message: str,
        campaign_id: str = "",
        release_controller: bool = False,
    ) -> dict:
        """Apply one fenced lifecycle transition and optionally release authority atomically."""

        now = self._utcnow()
        terminal = str(new_state) in {
            "completed",
            "completed_with_failures",
            "cancelled",
            "failed_non_resumable",
            "discarded_unstarted",
        }
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            plan = con.execute("SELECT * FROM execution_plans WHERE id=?", (str(plan_id),)).fetchone()
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            controller = self._verified_controller_payload(controller)
            if plan is None:
                raise RuntimeError("Execution plan or controller record is missing")
            self._verified_plan_design(plan)
            if str(plan["lifecycle_state"]) not in {str(value) for value in expected_states}:
                raise RuntimeError(
                    f"Plan transition rejected from lifecycle {str(plan['lifecycle_state'])!r}"
                )
            if (
                str(controller["owner_plan_id"]) != str(plan_id)
                or str(controller["owner_design_sha256"]) != str(plan["design_sha256"])
                or int(controller["epoch"]) != int(controller_epoch)
            ):
                raise RuntimeError("Stale or foreign execution-controller fencing token")
            prior_state = str(plan["lifecycle_state"])
            plan_revision = int(plan["state_revision"]) + 1
            effective_campaign_id = str(campaign_id or plan["campaign_id"] or "")
            if effective_campaign_id:
                self._verified_campaign_plan_binding(con, plan, effective_campaign_id)
            if str(new_state) == "paused" and effective_campaign_id:
                self._commit_campaign_paused_rows(
                    con,
                    effective_campaign_id,
                    now=now,
                    message=str(message),
                )
            next_epoch = int(controller_epoch) + 1 if release_controller else int(controller_epoch)
            plan_receipt = self._execution_state_receipt(
                plan_id=str(plan_id),
                design_sha256=str(plan["design_sha256"]),
                lifecycle_state=str(new_state),
                state_revision=plan_revision,
                controller_epoch=next_epoch,
                campaign_id=effective_campaign_id,
                prior_receipt_sha256=str(plan["state_receipt_sha256"]),
            )
            con.execute(
                """UPDATE execution_plans SET lifecycle_state=?,state_revision=?,
                    state_receipt_sha256=?,campaign_id=?,controller_epoch=?,active_slot=?,
                    updated_at=?,last_message=? WHERE id=?""",
                (
                    str(new_state),
                    plan_revision,
                    plan_receipt,
                    effective_campaign_id,
                    next_epoch,
                    0 if terminal else 1,
                    now,
                    str(message),
                    str(plan_id),
                ),
            )
            controller_revision = int(controller["record_revision"]) + 1
            controller_payload = {
                "schema_version": str(controller["schema_version"]),
                "controller": "none" if release_controller else str(controller["controller"]),
                "owner_plan_id": "" if release_controller else str(plan_id),
                "owner_design_sha256": "" if release_controller else str(plan["design_sha256"]),
                "campaign_id": "" if release_controller else effective_campaign_id,
                "lifecycle_state": "" if release_controller else str(new_state),
                "epoch": next_epoch,
                "owner_instance_id": "" if release_controller else str(controller["owner_instance_id"]),
                "record_revision": controller_revision,
                "acquired_at": "" if release_controller else str(controller["acquired_at"]),
            }
            controller_receipt = self._controller_receipt(controller_payload)
            con.execute(
                """UPDATE execution_controller SET controller=?,owner_plan_id=?,owner_design_sha256=?,
                    campaign_id=?,lifecycle_state=?,epoch=?,owner_instance_id=?,record_revision=?,
                    acquired_at=?,updated_at=?,state_receipt_sha256=? WHERE singleton_id=1""",
                (
                    controller_payload["controller"],
                    controller_payload["owner_plan_id"],
                    controller_payload["owner_design_sha256"],
                    controller_payload["campaign_id"],
                    controller_payload["lifecycle_state"],
                    next_epoch,
                    controller_payload["owner_instance_id"],
                    controller_revision,
                    controller_payload["acquired_at"],
                    now,
                    controller_receipt,
                ),
            )
            con.execute(
                """INSERT INTO execution_lifecycle_events(
                    plan_id,created_at,controller,controller_epoch,from_state,to_state,
                    state_revision,receipt_sha256,message
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(plan_id),
                    now,
                    str(controller["controller"]),
                    next_epoch,
                    prior_state,
                    str(new_state),
                    plan_revision,
                    plan_receipt,
                    str(message),
                ),
            )
        return self.get_execution_plan(str(plan_id)) or {}

    def cancel_retained_execution_plan(
        self,
        plan_id: str,
        *,
        controller_epoch: int,
        owner_instance_id: str,
        message: str,
    ) -> dict:
        """Atomically cancel an idle retained plan, its unfinished ledgers, and ownership."""

        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            plan = con.execute(
                "SELECT * FROM execution_plans WHERE id=?", (str(plan_id),)
            ).fetchone()
            controller = self._verified_controller_payload(
                con.execute(
                    "SELECT * FROM execution_controller WHERE singleton_id=1"
                ).fetchone()
            )
            if plan is None:
                raise KeyError(f"Unknown execution plan: {plan_id}")
            self._verified_plan_design(plan)
            prior_state = str(plan["lifecycle_state"])
            if prior_state not in {
                "staged",
                "running",
                "pausing",
                "paused",
                "interrupted_resumable",
            }:
                raise RuntimeError(f"Execution plan cannot be cancelled from {prior_state!r}")
            if (
                str(controller["owner_plan_id"]) != str(plan_id)
                or str(controller["owner_design_sha256"]) != str(plan["design_sha256"])
                or str(controller["owner_instance_id"]) != str(owner_instance_id)
                or int(controller["epoch"]) != int(controller_epoch)
            ):
                raise RuntimeError("Stale or foreign execution-controller cancellation token")
            campaign_id = str(plan["campaign_id"] or "")
            if campaign_id:
                self._verified_campaign_plan_binding(con, plan, campaign_id)
                self._cancel_campaign_remaining_rows(
                    con, campaign_id, now=now, message=str(message)
                )
            if str(plan["plan_kind"]) == "workspace":
                con.execute(
                    """UPDATE workspace_plan_cells SET lifecycle_state='cancelled',
                       last_message=? WHERE workspace_plan_id=? AND lifecycle_state NOT IN
                       ('completed','completed_with_failures','cancelled')""",
                    (
                        "Cancelled terminally; completed evidence retained",
                        str(plan_id),
                    ),
                )
            next_epoch = int(controller_epoch) + 1
            revision = int(plan["state_revision"]) + 1
            receipt = self._execution_state_receipt(
                plan_id=str(plan_id),
                design_sha256=str(plan["design_sha256"]),
                lifecycle_state="cancelled",
                state_revision=revision,
                controller_epoch=next_epoch,
                campaign_id=campaign_id,
                prior_receipt_sha256=str(plan["state_receipt_sha256"]),
            )
            con.execute(
                """UPDATE execution_plans SET lifecycle_state='cancelled',state_revision=?,
                   state_receipt_sha256=?,controller_epoch=?,active_slot=0,updated_at=?,
                   last_message=? WHERE id=?""",
                (revision, receipt, next_epoch, now, str(message), str(plan_id)),
            )
            controller_revision = int(controller["record_revision"]) + 1
            controller_payload = {
                "schema_version": str(controller["schema_version"]),
                "controller": "none",
                "owner_plan_id": "",
                "owner_design_sha256": "",
                "campaign_id": "",
                "lifecycle_state": "",
                "epoch": next_epoch,
                "owner_instance_id": "",
                "record_revision": controller_revision,
                "acquired_at": "",
            }
            con.execute(
                """UPDATE execution_controller SET controller='none',owner_plan_id='',
                   owner_design_sha256='',campaign_id='',lifecycle_state='',epoch=?,
                   owner_instance_id='',record_revision=?,acquired_at='',updated_at=?,
                   state_receipt_sha256=? WHERE singleton_id=1""",
                (
                    next_epoch,
                    controller_revision,
                    now,
                    self._controller_receipt(controller_payload),
                ),
            )
            con.execute(
                """INSERT INTO execution_lifecycle_events(
                    plan_id,created_at,controller,controller_epoch,from_state,to_state,
                    state_revision,receipt_sha256,message
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(plan_id),
                    now,
                    str(controller["controller"]),
                    next_epoch,
                    prior_state,
                    "cancelled",
                    revision,
                    receipt,
                    str(message),
                ),
            )
        return self.get_execution_plan(str(plan_id)) or {}

    def update_workspace_plan_cell(
        self,
        cell_id: str,
        *,
        lifecycle_state: str,
        campaign_id: str = "",
        experiment_id: str = "",
        message: str = "",
    ) -> None:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = self._verified_controller_payload(
                con.execute(
                    "SELECT * FROM execution_controller WHERE singleton_id=1"
                ).fetchone()
            )
            cell = con.execute(
                "SELECT workspace_plan_id FROM workspace_plan_cells WHERE id=?",
                (str(cell_id),),
            ).fetchone()
            if cell is None:
                raise KeyError(f"Unknown Workspace plan cell: {cell_id}")
            if (
                str(controller["controller"]) != "workspace"
                or str(controller["owner_plan_id"]) != str(cell["workspace_plan_id"])
            ):
                raise RuntimeError("Only the controlling Workspace plan can update its cell ledger")
            con.execute(
                """UPDATE workspace_plan_cells SET lifecycle_state=?,campaign_id=?,
                    experiment_id=?,last_message=? WHERE id=?""",
                (
                    str(lifecycle_state),
                    str(campaign_id),
                    str(experiment_id),
                    str(message),
                    str(cell_id),
                ),
            )

    def recover_execution_controller(self, *, owner_instance_id: str) -> dict:
        """Fence an old process and restore truthful interrupted ownership after restart."""

        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            controller = con.execute(
                "SELECT * FROM execution_controller WHERE singleton_id=1"
            ).fetchone()
            controller_payload = self._verified_controller_payload(controller)
            if str(controller["controller"]) == "none":
                return controller_payload
            plan = con.execute(
                "SELECT * FROM execution_plans WHERE id=?", (str(controller["owner_plan_id"]),)
            ).fetchone()
            if plan is None:
                raise RuntimeError("Execution controller refers to a missing owner plan")
            design = self._verified_plan_design(plan)
            prior_state = str(plan["lifecycle_state"])
            if prior_state in {
                "completed",
                "completed_with_failures",
                "cancelled",
                "failed_non_resumable",
                "discarded_unstarted",
            }:
                raise RuntimeError("A terminal execution plan cannot retain controller ownership")
            if str(plan["plan_kind"]) == "workspace" and prior_state == "paused":
                raise RuntimeError("A durably paused Workspace plan cannot retain controller ownership")
            stage = con.execute(
                "SELECT content_sha256 FROM algorithm_stages WHERE id=? AND status='active'",
                (str(plan["algorithm_stage_id"]),),
            ).fetchone()
            if stage is None or str(stage["content_sha256"]) != str(
                design.get("algorithm_stage_sha256", "")
            ):
                raise RuntimeError("The interrupted plan no longer matches the active algorithm stage")
            if str(plan["plan_kind"]) == "workspace" and prior_state in {
                "running",
                "pausing",
            }:
                active_cells = con.execute(
                    """SELECT id,campaign_id FROM workspace_plan_cells
                       WHERE workspace_plan_id=? AND lifecycle_state IN ('running','pausing')""",
                    (str(plan["id"]),),
                ).fetchall()
                for cell in active_cells:
                    campaign_id = str(cell["campaign_id"] or "")
                    if not campaign_id:
                        task = con.execute(
                            """SELECT campaign_id FROM campaign_tasks
                               WHERE execution_plan_id=? AND workspace_plan_cell_id=?
                               ORDER BY last_activity DESC LIMIT 1""",
                            (str(plan["id"]), str(cell["id"])),
                        ).fetchone()
                        campaign_id = "" if task is None else str(task["campaign_id"])
                    con.execute(
                        """UPDATE workspace_plan_cells SET lifecycle_state='interrupted_resumable',
                           campaign_id=?,last_message=? WHERE id=?""",
                        (
                            campaign_id,
                            "Application restart retained this in-flight Workspace cell",
                            str(cell["id"]),
                        ),
                    )
            new_state = (
                "interrupted_resumable"
                if prior_state in {"running", "pausing"}
                else prior_state
            )
            epoch = int(controller["epoch"]) + 1
            plan_revision = int(plan["state_revision"]) + 1
            plan_receipt = self._execution_state_receipt(
                plan_id=str(plan["id"]),
                design_sha256=str(plan["design_sha256"]),
                lifecycle_state=new_state,
                state_revision=plan_revision,
                controller_epoch=epoch,
                campaign_id=str(plan["campaign_id"]),
                prior_receipt_sha256=str(plan["state_receipt_sha256"]),
            )
            con.execute(
                """UPDATE execution_plans SET lifecycle_state=?,state_revision=?,
                    state_receipt_sha256=?,controller_epoch=?,updated_at=?,last_message=? WHERE id=?""",
                (
                    new_state,
                    plan_revision,
                    plan_receipt,
                    epoch,
                    now,
                    "Application restart restored authenticated execution ownership",
                    str(plan["id"]),
                ),
            )
            controller_revision = int(controller["record_revision"]) + 1
            controller_payload = {
                "schema_version": str(controller["schema_version"]),
                "controller": str(controller["controller"]),
                "owner_plan_id": str(plan["id"]),
                "owner_design_sha256": str(plan["design_sha256"]),
                "campaign_id": str(plan["campaign_id"]),
                "lifecycle_state": new_state,
                "epoch": epoch,
                "owner_instance_id": str(owner_instance_id),
                "record_revision": controller_revision,
                "acquired_at": str(controller["acquired_at"]),
            }
            con.execute(
                """UPDATE execution_controller SET lifecycle_state=?,epoch=?,owner_instance_id=?,
                    record_revision=?,updated_at=?,state_receipt_sha256=? WHERE singleton_id=1""",
                (
                    new_state,
                    epoch,
                    str(owner_instance_id),
                    controller_revision,
                    now,
                    self._controller_receipt(controller_payload),
                ),
            )
            con.execute(
                """INSERT INTO execution_lifecycle_events(
                    plan_id,created_at,controller,controller_epoch,from_state,to_state,
                    state_revision,receipt_sha256,message
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(plan["id"]),
                    now,
                    str(controller["controller"]),
                    epoch,
                    prior_state,
                    new_state,
                    plan_revision,
                    plan_receipt,
                    "Application restart fenced the prior process instance",
                ),
            )
        return self.get_execution_controller()

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
                    json.dumps(failure.numerical_state, allow_nan=False),
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
                    "SELECT * FROM portfolios ORDER BY updated_at DESC, id DESC"
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
        query += " ORDER BY updated_at DESC, id DESC"
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
        values: list[object] = [self._utcnow()]
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

    @staticmethod
    def _commit_campaign_paused_rows(
        con, campaign_id: str, *, now: str, message: str
    ) -> None:
        """Commit the final campaign pause boundary inside the controller transaction."""

        campaign = con.execute(
            "SELECT experiment_id,config_json FROM campaigns WHERE id=?",
            (str(campaign_id),),
        ).fetchone()
        if campaign is None:
            raise KeyError(f"Unknown campaign: {campaign_id}")
        con.execute(
            """UPDATE campaign_tasks SET status='paused',last_activity=? WHERE campaign_id=?
               AND status IN ('planned','queued','running','pausing','interrupted')""",
            (now, str(campaign_id)),
        )
        con.execute(
            """UPDATE campaigns SET status='paused',updated_at=?,last_message=? WHERE id=?""",
            (now, str(message), str(campaign_id)),
        )
        if campaign["experiment_id"]:
            con.execute(
                "UPDATE experiments SET campaign_status='paused' WHERE id=?",
                (str(campaign["experiment_id"]),),
            )
        config = json.loads(str(campaign["config_json"]))
        revision_id = str(config.get("experiment_revision_id", "") or "")
        if revision_id:
            con.execute(
                "UPDATE experiment_revisions SET status='paused' WHERE id=?",
                (revision_id,),
            )
        con.execute(
            """UPDATE resumable_tasks SET status='paused',resumable=1,updated_at=?
               WHERE id=?""",
            (now, str(campaign_id)),
        )

    @staticmethod
    def _cancel_campaign_remaining_rows(
        con, campaign_id: str, *, now: str, message: str
    ) -> None:
        campaign = con.execute(
            "SELECT * FROM campaigns WHERE id=?", (str(campaign_id),)
        ).fetchone()
        if campaign is None:
            raise KeyError(f"Unknown campaign: {campaign_id}")
        unfinished = con.execute(
            """SELECT id,checkpoint_path FROM campaign_tasks WHERE campaign_id=?
               AND status IN ('planned','queued','running','pausing','paused',
                              'interrupted','failed')""",
            (str(campaign_id),),
        ).fetchall()
        for task in unfinished:
            con.execute(
                "UPDATE campaign_tasks SET status='cancelled',last_activity=? WHERE id=?",
                (now, str(task["id"])),
            )
            con.execute(
                """INSERT INTO task_events(task_id,created_at,event_type,payload_json)
                   VALUES(?,?,?,?)""",
                (
                    str(task["id"]),
                    now,
                    "cancelled",
                    json.dumps(
                        {
                            "checkpoint_retained_for_audit": bool(task["checkpoint_path"]),
                            "resume_eligible": False,
                        },
                        sort_keys=True,
                    ),
                ),
            )
        con.execute(
            """UPDATE campaigns SET status='cancelled',updated_at=?,last_message=?
               WHERE id=?""",
            (now, str(message), str(campaign_id)),
        )
        if campaign["experiment_id"]:
            con.execute(
                "UPDATE experiments SET campaign_status='cancelled' WHERE id=?",
                (str(campaign["experiment_id"]),),
            )
        config = json.loads(str(campaign["config_json"]))
        revision_id = str(config.get("experiment_revision_id", "") or "")
        if revision_id:
            con.execute(
                "UPDATE experiment_revisions SET status='cancelled' WHERE id=?",
                (revision_id,),
            )
        con.execute(
            """UPDATE resumable_tasks SET status='cancelled',resumable=0,updated_at=?
               WHERE id=?""",
            (now, str(campaign_id)),
        )

    def cancel_campaign_remaining(self, campaign_id: str, *, message: str) -> None:
        """Close one retained campaign ledger without altering its committed run evidence."""

        now = self._utcnow()
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._cancel_campaign_remaining_rows(
                con, str(campaign_id), now=now, message=str(message)
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
        *,
        execution_plan_id: str = "",
        workspace_plan_cell_id: str = "",
        job_identity_sha256: str = "",
    ) -> str:
        task_id = str(uuid.uuid4())
        with self._lock, self.connect() as con:
            existing = con.execute(
                """SELECT id,execution_plan_id,workspace_plan_cell_id,job_identity_sha256,
                          algorithm,run_index,fingerprint
                   FROM campaign_tasks WHERE campaign_id=? AND job_index=?""",
                (campaign_id, int(job_index)),
            ).fetchone()
            if existing:
                expected_existing = (
                    str(execution_plan_id),
                    str(workspace_plan_cell_id),
                    str(job_identity_sha256),
                    str(algorithm),
                    int(run_index),
                    str(fingerprint),
                )
                observed_existing = (
                    str(existing["execution_plan_id"]),
                    str(existing["workspace_plan_cell_id"]),
                    str(existing["job_identity_sha256"]),
                    str(existing["algorithm"]),
                    int(existing["run_index"]),
                    str(existing["fingerprint"]),
                )
                if expected_existing != observed_existing:
                    raise RuntimeError(
                        "Campaign job index collision has a different plan, cell, scientific "
                        "identity, algorithm, run index, or fingerprint"
                    )
                return str(existing["id"])
            if execution_plan_id and job_identity_sha256:
                duplicate = con.execute(
                    """SELECT id,campaign_id FROM campaign_tasks
                       WHERE execution_plan_id=? AND workspace_plan_cell_id=?
                         AND job_identity_sha256=?""",
                    (
                        str(execution_plan_id),
                        str(workspace_plan_cell_id),
                        str(job_identity_sha256),
                    ),
                ).fetchone()
                if duplicate is not None:
                    raise RuntimeError(
                        "Duplicate scientific job admission was blocked for execution plan "
                        f"{execution_plan_id!r}; existing campaign {str(duplicate['campaign_id'])!r}"
                    )
            con.execute(
                """INSERT INTO campaign_tasks(
                    id,campaign_id,job_index,algorithm,run_index,seed_json,fingerprint,
                    required_outputs_json,status,attempts,checkpoint_path,checkpoint_sha256,
                    run_id,failure_id,last_activity,execution_plan_id,
                    workspace_plan_cell_id,job_identity_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    str(execution_plan_id),
                    str(workspace_plan_cell_id),
                    str(job_identity_sha256),
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
        values: list[object] = [self._utcnow()]
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
        values: list[object] = [self._utcnow()]
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
        query += " ORDER BY updated_at DESC, id DESC"
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

    def remove_unreferenced_unqualified_policy(
        self,
        policy_id: str,
        *,
        expected_sha256: str,
        reason: str = "user_deleted_completed_campaign",
    ) -> dict:
        """Atomically suppress and remove one exact unused candidate registration."""

        policy_key = str(policy_id)
        expected = str(expected_sha256).strip().lower()
        if not expected:
            raise ValueError("Expected policy SHA-256 cannot be empty")
        with self._lock, self.connect() as con:
            row = con.execute("SELECT * FROM policies WHERE id=?", (policy_key,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown CALO policy: {policy_key}")
            selected = dict(row)
            if str(selected.get("sha256", "")).lower() != expected:
                raise RuntimeError("Selected policy identity changed before removal")
            if bool(selected.get("active", False)):
                raise PermissionError("The active governing policy cannot be removed")
            if str(selected.get("qualification_status", "")).lower() == "qualified":
                raise PermissionError(
                    "A qualified policy requires the reviewed retirement workflow"
                )
            qualification_count = int(
                con.execute(
                    "SELECT COUNT(*) AS n FROM policy_qualifications WHERE policy_id=?",
                    (policy_key,),
                ).fetchone()["n"]
            )
            binding_count = int(
                con.execute(
                    "SELECT COUNT(*) AS n FROM experiment_policy_bindings "
                    "WHERE policy_id=? OR sha256=?",
                    (policy_key, expected),
                ).fetchone()["n"]
            )
            checkpoint_count = int(
                con.execute(
                    "SELECT COUNT(*) AS n FROM policy_checkpoints "
                    "WHERE lower(sha256)=lower(?) OR checkpoint_path=? OR resume_path=?",
                    (
                        expected,
                        str(selected.get("checkpoint_path", "")),
                        str(selected.get("checkpoint_path", "")),
                    ),
                ).fetchone()["n"]
            )
            if qualification_count:
                raise PermissionError(
                    "A policy with qualification evidence requires the reviewed retirement workflow"
                )
            if binding_count:
                raise PermissionError("A policy referenced by an experiment cannot be removed")
            if checkpoint_count:
                raise PermissionError(
                    "A policy referenced by a lineage checkpoint cannot be removed"
                )
            con.execute(
                "INSERT INTO suppressed_policies(sha256,created_at,reason) VALUES(?,?,?) "
                "ON CONFLICT(sha256) DO UPDATE SET reason=excluded.reason",
                (expected, self._utcnow(), str(reason)),
            )
            removed = int(
                con.execute(
                    "DELETE FROM policies WHERE id=? AND lower(sha256)=lower(?)",
                    (policy_key, expected),
                ).rowcount
            )
            if removed != 1:
                raise RuntimeError("The selected policy registration was not removed exactly once")
        return selected

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

    def admit_verified_policy_qualification(
        self,
        *,
        qualification_id: str,
        policy_id: str,
        expected_sha256: str,
        config: dict,
        metrics: dict,
        grade: str,
        score: float,
    ) -> bool:
        """Atomically admit already-verified evidence without activating its policy.

        Returns ``False`` for an exact idempotent re-admission. A reused qualification identity,
        changed policy artifact, active policy, or conflicting evidence fails closed.
        """

        qualification_key = str(qualification_id).strip()
        policy_key = str(policy_id).strip()
        expected = str(expected_sha256).strip().lower()
        config_json = json.dumps(config or {}, sort_keys=True, allow_nan=False)
        metrics_json = json.dumps(metrics or {}, sort_keys=True, allow_nan=False)
        if not qualification_key or not policy_key or len(expected) != 64:
            raise ValueError("Verified qualification admission identities are incomplete")
        with self._lock, self.connect() as con:
            policy_row = con.execute(
                "SELECT sha256,qualification_status,active,archived FROM policies WHERE id=?",
                (policy_key,),
            ).fetchone()
            if policy_row is None:
                raise KeyError(f"Unknown CALO policy: {policy_key}")
            if str(policy_row["sha256"]).lower() != expected:
                raise RuntimeError("Policy artifact identity changed before evidence admission")
            if bool(policy_row["active"]):
                raise PermissionError(
                    "An active policy cannot accept replacement qualification evidence"
                )
            if bool(policy_row["archived"]):
                raise PermissionError(
                    "Restore the archived policy before admitting qualification evidence"
                )
            existing = con.execute(
                "SELECT * FROM policy_qualifications WHERE id=?", (qualification_key,)
            ).fetchone()
            if existing is not None:
                row = dict(existing)
                same = bool(
                    str(row.get("policy_id", "")) == policy_key
                    and str(row.get("config_json", "")) == config_json
                    and str(row.get("metrics_json", "")) == metrics_json
                    and bool(row.get("passed", False))
                    and str(row.get("grade", "")) == str(grade)
                    and float(row.get("score", 0.0)) == float(score)
                )
                if not same:
                    raise RuntimeError(
                        "Qualification identity is already bound to different retained evidence"
                    )
                return False
            con.execute(
                "INSERT INTO policy_qualifications VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    qualification_key,
                    policy_key,
                    self._utcnow(),
                    "",
                    config_json,
                    metrics_json,
                    1,
                    str(grade),
                    float(score),
                ),
            )
            updated = int(
                con.execute(
                    "UPDATE policies SET qualification_status='qualified',grade=?,updated_at=? "
                    "WHERE id=? AND lower(sha256)=lower(?) AND active=0 AND archived=0",
                    (str(grade), self._utcnow(), policy_key, expected),
                ).rowcount
            )
            if updated != 1:
                raise RuntimeError(
                    "Policy qualification admission did not update exactly one policy"
                )
        return True

    def admit_verified_policy_assessment(
        self,
        *,
        assessment_id: str,
        policy_id: str,
        expected_sha256: str,
        config: dict,
        metrics: dict,
        score: float,
    ) -> bool:
        """Atomically retain a verified feasibility dossier without selecting its policy."""

        assessment_key = str(assessment_id).strip()
        policy_key = str(policy_id).strip()
        expected = str(expected_sha256).strip().lower()
        config_json = json.dumps(config or {}, sort_keys=True, allow_nan=False)
        metrics_json = json.dumps(metrics or {}, sort_keys=True, allow_nan=False)
        numeric_score = float(score)
        if (
            not assessment_key
            or not policy_key
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not math.isfinite(numeric_score)
            or numeric_score < 0.0
            or numeric_score > 100.0
        ):
            raise ValueError("Verified feasibility assessment identities are incomplete")
        with self._lock, self.connect() as con:
            policy_row = con.execute(
                "SELECT sha256,qualification_status,active,archived FROM policies WHERE id=?",
                (policy_key,),
            ).fetchone()
            if policy_row is None:
                raise KeyError(f"Unknown CALO policy: {policy_key}")
            if str(policy_row["sha256"]).lower() != expected:
                raise RuntimeError("Policy artifact identity changed before assessment admission")
            if bool(policy_row["active"]):
                raise PermissionError(
                    "An active policy cannot accept replacement assessment evidence"
                )
            if bool(policy_row["archived"]):
                raise PermissionError("Restore the archived policy before admitting an assessment")
            existing = con.execute(
                "SELECT * FROM policy_qualifications WHERE id=?", (assessment_key,)
            ).fetchone()
            if existing is not None:
                row = dict(existing)
                same = bool(
                    str(row.get("policy_id", "")) == policy_key
                    and str(row.get("config_json", "")) == config_json
                    and str(row.get("metrics_json", "")) == metrics_json
                    and str(row.get("grade", "")) == "N/A"
                    and float(row.get("score", 0.0)) == numeric_score
                )
                if not same:
                    raise RuntimeError(
                        "Assessment identity is already bound to different retained evidence"
                    )
                return False
            if str(policy_row["qualification_status"]) not in {"candidate", "assessed"}:
                raise PermissionError(
                    "Feasibility admission cannot replace an existing scientist or legacy decision"
                )
            con.execute(
                "INSERT INTO policy_qualifications VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    assessment_key,
                    policy_key,
                    self._utcnow(),
                    "",
                    config_json,
                    metrics_json,
                    0,
                    "N/A",
                    numeric_score,
                ),
            )
            updated = int(
                con.execute(
                    "UPDATE policies SET qualification_status='assessed',grade='N/A',updated_at=? "
                    "WHERE id=? AND lower(sha256)=lower(?) AND active=0 AND archived=0",
                    (self._utcnow(), policy_key, expected),
                ).rowcount
            )
            if updated != 1:
                raise RuntimeError("Policy assessment admission did not update exactly one policy")
        return True

    def record_scientist_policy_selection(
        self,
        *,
        policy_id: str,
        assessment_id: str,
        expected_sha256: str,
        evidence_sha256: str,
    ) -> bool:
        """Record an explicit candidate-bound scientist selection without activating it."""

        policy_key = str(policy_id).strip()
        assessment_key = str(assessment_id).strip()
        expected = str(expected_sha256).strip().lower()
        evidence = str(evidence_sha256).strip().lower()
        if (
            len(expected) != 64
            or len(evidence) != 64
            or any(character not in "0123456789abcdef" for character in expected + evidence)
        ):
            raise ValueError("Scientist selection requires exact candidate and evidence SHA-256")
        with self._lock, self.connect() as con:
            policy_row = con.execute("SELECT * FROM policies WHERE id=?", (policy_key,)).fetchone()
            if policy_row is None:
                raise KeyError(f"Unknown CALO policy: {policy_key}")
            if str(policy_row["sha256"]).lower() != expected:
                raise RuntimeError("Policy artifact identity changed before scientist selection")
            if bool(policy_row["active"]) or bool(policy_row["archived"]):
                raise PermissionError("Only an inactive, unarchived policy can be selected")
            assessment_row = con.execute(
                "SELECT * FROM policy_qualifications WHERE id=? AND policy_id=?",
                (assessment_key, policy_key),
            ).fetchone()
            if assessment_row is None:
                raise ValueError("Scientist selection requires an admitted feasibility assessment")
            metrics = json.loads(str(assessment_row["metrics_json"] or "{}"))
            if (
                metrics.get("admission_schema_version")
                != "tsh-calo-policy-feasibility-admission-v1"
                or str(metrics.get("candidate_sha256", "")).lower() != expected
                or str(metrics.get("evidence_artifact_sha256", "")).lower() != evidence
            ):
                raise ValueError("Scientist selection assessment evidence is incompatible")
            metadata = json.loads(str(policy_row["metadata_json"] or "{}"))
            existing = dict(metadata.get("scientist_selection", {}) or {})
            if existing:
                if (
                    str(policy_row["qualification_status"]) == "scientist_selected"
                    and existing.get("schema_version") == "tsh-calo-scientist-policy-selection-v1"
                    and existing.get("assessment_id") == assessment_key
                    and str(existing.get("candidate_sha256", "")).lower() == expected
                    and str(existing.get("evidence_sha256", "")).lower() == evidence
                ):
                    return False
                raise RuntimeError("Policy is already bound to another scientist selection")
            if str(policy_row["qualification_status"]) != "assessed":
                raise PermissionError("Scientist selection requires the assessed policy state")
            if bool(assessment_row["passed"]):
                raise ValueError("Feasibility assessment already carries a decision state")
            selected_at = self._utcnow()
            metadata["scientist_selection"] = {
                "schema_version": "tsh-calo-scientist-policy-selection-v1",
                "assessment_id": assessment_key,
                "candidate_sha256": expected,
                "evidence_sha256": evidence,
                "selected_at": selected_at,
                "activation_performed": False,
            }
            con.execute(
                "UPDATE policy_qualifications SET passed=1 WHERE id=? AND policy_id=?",
                (assessment_key, policy_key),
            )
            updated = int(
                con.execute(
                    "UPDATE policies SET qualification_status='scientist_selected',grade='N/A',"
                    "metadata_json=?,updated_at=? WHERE id=? AND lower(sha256)=lower(?) "
                    "AND active=0 AND archived=0",
                    (json.dumps(metadata, allow_nan=False), selected_at, policy_key, expected),
                ).rowcount
            )
            if updated != 1:
                raise RuntimeError("Scientist selection did not update exactly one policy")
        return True

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

    @staticmethod
    def _policy_lifecycle_snapshot_from_connection(con: sqlite3.Connection) -> dict:
        """Return stable policy-lifecycle rows for inventory-bound retirement.

        This intentionally includes immutable experiment bindings and policy-training resume tasks.
        A post-freeze removal receipt can therefore retain the exact evidence that was removed from
        the live database without retaining an executable policy artifact or resumable workflow.
        """

        tables = {
            "policies": ("SELECT * FROM policies ORDER BY id", ()),
            "policy_qualifications": (
                "SELECT * FROM policy_qualifications ORDER BY id",
                (),
            ),
            "experiment_policy_bindings": (
                "SELECT * FROM experiment_policy_bindings ORDER BY experiment_id",
                (),
            ),
            "policy_lineages": ("SELECT * FROM policy_lineages ORDER BY id", ()),
            "policy_checkpoints": ("SELECT * FROM policy_checkpoints ORDER BY id", ()),
            "suppressed_policies": ("SELECT * FROM suppressed_policies ORDER BY sha256", ()),
            "policy_training_resume_tasks": (
                "SELECT * FROM resumable_tasks WHERE task_type='policy_training' ORDER BY id",
                (),
            ),
        }
        return {
            name: [dict(row) for row in con.execute(query, arguments).fetchall()]
            for name, (query, arguments) in tables.items()
        }

    def policy_lifecycle_snapshot(self) -> dict:
        """Return a read-only exact snapshot of policy lifecycle database state."""

        if (
            self.read_only
            and self.path != ":memory:"
            and (not Path(self.path).is_file() or Path(self.path).stat().st_size == 0)
        ):
            return {
                "policies": [],
                "policy_qualifications": [],
                "experiment_policy_bindings": [],
                "policy_lineages": [],
                "policy_checkpoints": [],
                "suppressed_policies": [],
                "policy_training_resume_tasks": [],
            }
        with self.connect() as con:
            return self._policy_lifecycle_snapshot_from_connection(con)

    def clear_policy_lifecycle(self, *, expected_snapshot: dict) -> dict:
        """Remove the exact inventoried lifecycle state in one database transaction.

        This method is deliberately unusable as an unscoped "delete all" operation: the caller must
        provide the complete snapshot observed during the authorized inventory. Any concurrent or
        otherwise unreviewed change aborts before a row is modified. The caller is responsible for
        retaining the snapshot in an immutable external deletion receipt.
        """

        if self.read_only:
            raise PermissionError("A read-only database cannot clear policy lifecycle state")
        expected = json.loads(json.dumps(expected_snapshot, sort_keys=True, allow_nan=True))
        with self._lock, self.connect() as con:
            current = self._policy_lifecycle_snapshot_from_connection(con)
            if current != expected:
                raise RuntimeError(
                    "Policy lifecycle database state changed after inventory; removal is blocked"
                )
            removed = {
                "experiment_policy_bindings": int(
                    con.execute("DELETE FROM experiment_policy_bindings").rowcount
                ),
                "policy_qualifications": int(
                    con.execute("DELETE FROM policy_qualifications").rowcount
                ),
                "policy_checkpoints": int(con.execute("DELETE FROM policy_checkpoints").rowcount),
                "policy_lineages": int(con.execute("DELETE FROM policy_lineages").rowcount),
                "policies": int(con.execute("DELETE FROM policies").rowcount),
                "suppressed_policies": int(con.execute("DELETE FROM suppressed_policies").rowcount),
                "policy_training_resume_tasks": int(
                    con.execute(
                        "DELETE FROM resumable_tasks WHERE task_type='policy_training'"
                    ).rowcount
                ),
            }
            remaining = self._policy_lifecycle_snapshot_from_connection(con)
            if any(remaining.values()):
                raise RuntimeError(
                    "Policy lifecycle cleanup did not produce an empty database state"
                )
        return {"removed": removed, "remaining": remaining}

    def list_suppressed_policy_sha256(self) -> set[str]:
        """Return policy suppressions scoped to this project/results database."""
        with self.connect() as con:
            rows = con.execute("SELECT sha256 FROM suppressed_policies").fetchall()
        return {str(row["sha256"]).lower() for row in rows}

    def suppress_policy_sha256(self, sha256: str, *, reason: str = "user_deleted") -> None:
        value = str(sha256).strip().lower()
        if not value:
            raise ValueError("Policy SHA-256 cannot be empty")
        with self._lock, self.connect() as con:
            con.execute(
                "INSERT INTO suppressed_policies(sha256,created_at,reason) VALUES(?,?,?) "
                "ON CONFLICT(sha256) DO UPDATE SET reason=excluded.reason",
                (value, self._utcnow(), str(reason)),
            )

    def unsuppress_policy_sha256(self, sha256: str) -> None:
        with self._lock, self.connect() as con:
            con.execute(
                "DELETE FROM suppressed_policies WHERE sha256=?", (str(sha256).strip().lower(),)
            )

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
        updated = self.get_experiment(experiment_id)
        if not isinstance(updated, dict):
            raise RuntimeError(
                f"Experiment disappeared after learning-role update: {experiment_id}"
            )
        return updated

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
        q += " ORDER BY updated_at DESC, id DESC"
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
            con.execute("BEGIN IMMEDIATE")
            effective_latest = False
            if is_latest:
                current = con.execute(
                    """SELECT id,cumulative_epoch,phase_index FROM policy_checkpoints
                       WHERE lineage_id=? AND id<>?
                       ORDER BY cumulative_epoch DESC,phase_index DESC,created_at DESC LIMIT 1""",
                    (str(lineage_id), str(checkpoint_id)),
                ).fetchone()
                effective_latest = (
                    current is None
                    or int(cumulative_epoch) > int(current["cumulative_epoch"])
                    or (
                        int(cumulative_epoch) == int(current["cumulative_epoch"])
                        and int(phase_index) >= int(current["phase_index"])
                    )
                )
                if effective_latest:
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
                    int(bool(effective_latest)),
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
        # Read protection state and delete inside one write transaction so no thread/process can
        # change latest/best/fork references between the checks and deletion.
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT is_latest,is_best FROM policy_checkpoints WHERE id=?",
                (str(checkpoint_id),),
            ).fetchone()
            if row is None:
                return
            if bool(row["is_latest"]) or bool(row["is_best"]):
                raise ValueError(
                    "Latest or best-qualified lineage checkpoints must be retained; archive the policy instead"
                )
            forks = con.execute(
                "SELECT COUNT(*) AS n FROM policy_lineages WHERE forked_from_checkpoint_id=?",
                (str(checkpoint_id),),
            ).fetchone()
            if int(forks["n"] if forks else 0) > 0:
                raise ValueError(
                    "This checkpoint is the parent of a forked policy lineage and cannot be deleted"
                )
            con.execute("DELETE FROM policy_checkpoints WHERE id=?", (str(checkpoint_id),))

    def update_policy_checkpoint_qualification(
        self,
        checkpoint_id: str,
        *,
        qualification_status: str,
        grade: str,
        metadata_updates: dict | None = None,
    ) -> None:
        # Merge metadata under the same transaction as the update to prevent lost concurrent writes.
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute(
                "SELECT metadata_json FROM policy_checkpoints WHERE id=?",
                (str(checkpoint_id),),
            ).fetchone()
            if current is None:
                raise KeyError(checkpoint_id)
            try:
                metadata = json.loads(current["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(metadata_updates or {})
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
        created = self.get_experiment_revision(revision_id)
        if created is None:
            raise RuntimeError(
                f"Experiment revision was not retained after creation: {revision_id}"
            )
        return created

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
                _LOG.debug("Suppressed non-fatal fallback/cleanup exception", exc_info=True)
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

    def resolve_array_path(self, value: str) -> Path | None:
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
            path = self.resolve_array_path(value)
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
            path = self.resolve_array_path(value)
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
            path = self.resolve_array_path(value) or Path(value).expanduser()
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
