import hashlib
import sqlite3

import pytest

from calo_rpd_studio.results.database import DATABASE_SCHEMA_VERSION, ResultDatabase
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.provenance import collect_provenance


def test_database_experiment_creation(tmp_path):
    path = tmp_path / "r.sqlite"
    db = ResultDatabase(path)
    eid = db.create_experiment(ExperimentConfig(), collect_provenance())
    assert db.get_experiment(eid)["id"] == eid
    assert db.migration_backup_path is None
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
        receipt = con.execute(
            "SELECT source_version,target_version,backup_path FROM schema_migrations"
        ).fetchone()
    assert receipt == (0, DATABASE_SCHEMA_VERSION, "")


def _create_version_zero_database(path):
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE experiments(
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, name TEXT NOT NULL,
                config_json TEXT NOT NULL, provenance_json TEXT NOT NULL
            );
            CREATE TABLE runs(
                id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, algorithm TEXT NOT NULL,
                run_index INTEGER NOT NULL, seed_json TEXT NOT NULL, result_json TEXT NOT NULL,
                arrays_path TEXT NOT NULL, validation_status TEXT NOT NULL DEFAULT 'unverified'
            );
            CREATE TABLE validations(
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created_at TEXT NOT NULL,
                validation_json TEXT NOT NULL, passed INTEGER NOT NULL
            );
            INSERT INTO experiments VALUES(
                'experiment-1','2026-01-01T00:00:00Z','Legacy study','{"runs": 30}','{"host": "legacy"}'
            );
            INSERT INTO runs VALUES(
                'run-1','experiment-1','CALO',0,'{"algorithm_seed": 7}',
                '{"best_objective": 1.25}','arrays/run-1.npz','verified'
            );
            INSERT INTO validations VALUES(
                'validation-1','run-1','2026-01-01T00:01:00Z','{"passed": true}',1
            );
            PRAGMA user_version=0;
            """
        )


def test_version_zero_database_migrates_with_verified_backup_and_no_data_loss(tmp_path):
    path = tmp_path / "legacy.sqlite"
    _create_version_zero_database(path)

    database = ResultDatabase(path)

    backups = list(tmp_path.glob(f"legacy.pre-schema-v0-to-v{DATABASE_SCHEMA_VERSION}-*.sqlite"))
    assert backups == [tmp_path / database.migration_backup_path]
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 0
        assert backup.execute("SELECT id,name,config_json FROM experiments").fetchone() == (
            "experiment-1",
            "Legacy study",
            '{"runs": 30}',
        )

    with sqlite3.connect(path) as migrated:
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
        assert migrated.execute(
            "SELECT id,name,config_json,data_role,learning_eligible,learning_locked FROM experiments"
        ).fetchone() == ("experiment-1", "Legacy study", '{"runs": 30}', "excluded", 0, 0)
        assert migrated.execute(
            "SELECT id,algorithm,result_json,scientific_fingerprint FROM runs"
        ).fetchone() == ("run-1", "CALO", '{"best_objective": 1.25}', "")
        assert migrated.execute(
            "SELECT id,passed,evaluation_horizon,revision_id FROM validations"
        ).fetchone() == ("validation-1", 1, 0, "")
        receipt = migrated.execute(
            "SELECT source_version,target_version,backup_path,backup_sha256 FROM schema_migrations"
        ).fetchone()
    assert receipt[:3] == (0, DATABASE_SCHEMA_VERSION, str(backups[0]))
    assert receipt[3] == hashlib.sha256(backups[0].read_bytes()).hexdigest()

    reopened = ResultDatabase(path)
    assert reopened.migration_backup_path is None
    assert (
        list(tmp_path.glob(f"legacy.pre-schema-v0-to-v{DATABASE_SCHEMA_VERSION}-*.sqlite"))
        == backups
    )
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1


def test_future_database_schema_is_rejected_without_mutation(tmp_path):
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE preserved(value TEXT NOT NULL)")
        con.execute("INSERT INTO preserved VALUES('unchanged')")
        con.execute(f"PRAGMA user_version={DATABASE_SCHEMA_VERSION + 1}")

    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="newer than supported"):
        ResultDatabase(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("future.pre-schema-*.sqlite"))
