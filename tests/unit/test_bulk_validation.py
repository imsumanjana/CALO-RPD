from __future__ import annotations

from calo_rpd_studio.benchmarking import validation as validation_module


class _FakeDatabase:
    def __init__(self):
        self.rows = [
            {"id": "a", "experiment_id": "e1", "algorithm": "CALO", "run_index": 0, "validation_status": "unverified"},
            {"id": "b", "experiment_id": "e1", "algorithm": "TLBO", "run_index": 0, "validation_status": "verified"},
            {"id": "c", "experiment_id": "e2", "algorithm": "QODE", "run_index": 1, "validation_status": "failed"},
        ]

    def list_runs(self, experiment_id=None):
        rows = self.rows
        if experiment_id is not None:
            rows = [row for row in rows if row["experiment_id"] == experiment_id]
        return [dict(row) for row in rows]

    def get_run(self, run_id):
        return next(dict(row) for row in self.rows if row["id"] == run_id)


def test_bulk_validation_selection_skips_verified_by_default():
    db = _FakeDatabase()
    current = validation_module.select_runs_for_validation(db, experiment_id="e1")
    all_pending = validation_module.select_runs_for_validation(db)
    assert [row["id"] for row in current] == ["a"]
    assert [row["id"] for row in all_pending] == ["a", "c"]


def test_bulk_validate_runs_continues_after_failure_and_reports_progress(monkeypatch):
    db = _FakeDatabase()
    progress = []

    def fake_validate(_database, run_id):
        if run_id == "c":
            raise RuntimeError("synthetic validation failure")
        return {
            "passed": True,
            "maximum_constraint_violation": 0.0,
            "relative_difference": 0.0,
        }

    monkeypatch.setattr(validation_module, "validate_stored_run", fake_validate)
    rows = validation_module.select_runs_for_validation(db)
    summary = validation_module.validate_runs(db, rows, progress_callback=progress.append)

    assert summary["total_selected"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["errors"] == 1
    assert summary["cancelled"] is False
    assert [item["completed"] for item in progress] == [1, 2]
