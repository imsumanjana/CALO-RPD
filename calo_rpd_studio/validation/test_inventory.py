"""Review test omissions without converting missing execution into acceptance evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


def review_inventory(inventory: dict, dispositions: dict, *, source_root: Path) -> dict:
    """Classify exact historical skips; retain failures and every live omission explicitly."""
    known = {item["nodeid"]: item for item in dispositions.get("historical_cases", [])}
    selected = set(inventory.get("selected", []))
    outcomes = dict(inventory.get("outcomes", {}))
    errors, omitted, historical = [], [], []
    if not selected:
        errors.append("No selected tests were recorded")
    if inventory.get("exit_code") != 0:
        errors.append(f"Test process exited with {inventory.get('exit_code')!r}")
    for nodeid in sorted(selected - set(outcomes)):
        errors.append(f"Selected test never executed: {nodeid}")
    for nodeid, result in outcomes.items():
        outcome = result.get("outcome")
        if outcome in {"failed", "incomplete", "xpassed"}:
            errors.append(f"{outcome}: {nodeid}")
        elif outcome == "skipped":
            reasons = [
                str(phase.get("skip_message") or phase.get("reason", "")).removeprefix("Skipped: ")
                for phase in result.get("phases", {}).values()
                if phase.get("outcome") == "skipped"
            ]
            record = known.get(nodeid)
            path = source_root / nodeid.split("::", 1)[0]
            exact = bool(
                record
                and reasons
                and all(record["reason"] == reason for reason in reasons)
                and path.is_file()
                and hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                == record["source_sha256"]
            )
            if exact:
                missing = [
                    target
                    for target in record.get("current_successors", [])
                    if outcomes.get(target, {}).get("outcome") != "passed"
                ]
                historical.append(
                    {
                        "nodeid": nodeid,
                        "disposition": record["disposition"],
                        "unexecuted_successors": missing,
                    }
                )
                if missing:
                    omitted.append(
                        {
                            "nodeid": nodeid,
                            "kind": "unexecuted_current_successor",
                            "targets": missing,
                        }
                    )
            else:
                omitted.append(
                    {"nodeid": nodeid, "kind": "current_or_unrecognized_skip", "reasons": reasons}
                )
        elif outcome != "passed":
            errors.append(f"Unknown outcome for {nodeid}: {outcome!r}")
    for row in inventory.get("deselected", []):
        nodeid = str(row.get("nodeid", ""))
        protected = (
            nodeid.startswith("tests/scientific/")
            and any(case in nodeid for case in ("case118", "case300"))
            and row.get("reason") == "protected-case scientific execution not authorized"
        )
        omitted.append(
            {
                **row,
                "kind": "separately_authorized_protected_scope"
                if protected
                else "deselected_current_test",
            }
        )
    for row in inventory.get("collection", []):
        omitted.append({**row, "kind": "collection_failure_or_skip"})
    return {
        "schema": "calo-test-omission-review-v1",
        "errors": errors,
        "historical_omissions": historical,
        "unclosed_omissions": omitted,
        "selected_count": len(selected),
        "executed_count": len(outcomes),
        "complete_for_selected_scope": not errors and not omitted,
        "release_ready": False,
    }


def review_required_lanes(required: list[str], evidence: dict[str, dict]) -> dict:
    """No absent, failed, source-unstable or omission-bearing lane counts as qualified."""
    pending = {}
    for lane in required:
        record = evidence.get(lane)
        if record is None:
            pending[lane] = "No evidence supplied"
        elif record.get("status") != "passed" or record.get("source_stable") is not True:
            pending[lane] = "A complete source-stable passing run is required"
        elif record.get("harness_stable") is not True:
            pending[lane] = "Unchanged validator source must be established"
        elif "unclosed_omissions" not in record:
            pending[lane] = "A complete omission inventory is required"
        elif record.get("unclosed_omissions"):
            pending[lane] = "Live omissions remain"
        elif lane == "physical_cuda" and record.get("physical_cuda_available") is not True:
            pending[lane] = "Physical CUDA execution was not established"
    return {
        "required": list(required),
        "pending": pending,
        "engineering_lanes_complete": not pending,
        "release_ready": False,
    }
