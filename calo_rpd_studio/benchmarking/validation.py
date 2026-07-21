"""Bulk independent validation for saved experiments and frozen benchmark campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from calo_rpd_studio.results.solution_validator import validate_stored_run


def select_runs_for_validation(
    database,
    *,
    experiment_id: str | None = None,
    only_unverified: bool = True,
) -> list[dict]:
    """Return deterministic run rows for one experiment or the complete repository.

    ``only_unverified`` means all records not currently marked ``verified``. Failed validations
    are therefore eligible for an intentional retry after the underlying issue is corrected.
    """
    runs = database.list_runs(experiment_id) if experiment_id else database.list_runs()
    if only_unverified:
        runs = [row for row in runs if row.get("validation_status") != "verified"]
    return list(runs)


def validate_runs(
    database,
    runs: Iterable[dict] | Iterable[str],
    *,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict:
    """Independently validate many saved runs with progress, cancellation, and error capture."""
    selected = list(runs)
    total = len(selected)
    passed = 0
    failed = 0
    errors = 0
    cancelled = False
    records: list[dict] = []

    for index, item in enumerate(selected, start=1):
        if cancel_callback and cancel_callback():
            cancelled = True
            break
        if isinstance(item, str):
            run_id = item
            row = database.get_run(run_id) or {
                "id": run_id,
                "algorithm": "Unknown",
                "run_index": -1,
            }
        else:
            row = dict(item)
            run_id = row["id"]

        status = "failed"
        error_message = ""
        result = None
        try:
            result = validate_stored_run(database, run_id)
            status = "verified" if result.get("passed") else "failed"
            if result.get("passed"):
                passed += 1
            else:
                failed += 1
        except Exception as exc:  # bulk mode must continue to the remaining independent runs
            errors += 1
            error_message = f"{type(exc).__name__}: {exc}"

        record = {
            "run_id": run_id,
            "algorithm": row.get("algorithm", "Unknown"),
            "run_index": int(row.get("run_index", -1)),
            "status": status if not error_message else "error",
            "error": error_message,
        }
        if result is not None:
            record["maximum_constraint_violation"] = result.get("maximum_constraint_violation")
            record["relative_difference"] = result.get("relative_difference")
        records.append(record)

        if progress_callback:
            progress_callback(
                {
                    "completed": index,
                    "total": total,
                    "percent": int(100 * index / max(total, 1)),
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    **record,
                }
            )

    return {
        "total_selected": total,
        "validated": passed + failed,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "cancelled": cancelled,
        "remaining": max(0, total - len(records)),
        "records": records,
    }


def validate_campaign(
    database,
    campaign_manifest: str | Path,
    *,
    only_unverified: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict:
    manifest = Path(campaign_manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    experiment_ids = [
        task.get("experiment_id")
        for task in payload.get("tasks", [])
        if task.get("experiment_id") and task.get("status") == "completed"
    ]
    runs = []
    for experiment_id in experiment_ids:
        runs.extend(database.list_runs(experiment_id))
    if only_unverified:
        runs = [row for row in runs if row.get("validation_status") != "verified"]
    return validate_runs(
        database,
        runs,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
