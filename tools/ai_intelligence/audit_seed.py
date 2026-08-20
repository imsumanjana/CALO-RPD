from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .common import AI_DIR, INDEX_FILES, canonical_json, read_json, write_if_changed

_REVIEWED_RE = re.compile(r"(?m)^- Audited units retained: \d+$")
_REAUDIT_RE = re.compile(r"(?m)^- Units requiring re-audit: \d+$")


def apply_audit_seed(root: Path) -> int:
    """Apply only hash-bound historical review evidence to the current audit ledger.

    A seed is evidence about an exact file content hash, not a blanket approval for a path.
    If source changes, the normal incremental indexer invalidates the review and this function
    cannot reapply it unless the content returns to the exact reviewed hash.
    """

    ai_dir = root / AI_DIR
    file_index = read_json(ai_dir / INDEX_FILES["file"], {"files": {}})
    audit = read_json(ai_dir / INDEX_FILES["audit"], {"units": {}})
    seed = read_json(ai_dir / "audit-seed.json", {"units": {}})
    files: dict[str, Any] = file_index.get("files", {})
    units: dict[str, Any] = audit.get("units", {})
    applied = 0

    for path, unit in units.items():
        previous = unit.get("previous_review") or {}
        if unit.get("re_audit_required") and not previous.get("reviewed"):
            unit["re_audit_required"] = False
            unit["changed_at_index_commit"] = None

    for path, evidence in sorted(seed.get("units", {}).items()):
        current = files.get(path)
        unit = units.get(path)
        if not current or not unit:
            continue
        if current.get("sha256") != evidence.get("content_hash"):
            continue
        categories = sorted(set(evidence.get("audit_categories", [])))
        if not categories or not evidence.get("last_reviewed_commit"):
            continue
        unit.update(
            reviewed=True,
            audit_categories=categories,
            last_reviewed_commit=evidence["last_reviewed_commit"],
            unresolved_findings=list(unit.get("unresolved_findings", [])),
            re_audit_required=False,
            changed_at_index_commit=None,
            review_scope=evidence.get("scope_note"),
        )
        applied += 1

    write_if_changed(ai_dir / INDEX_FILES["audit"], canonical_json(audit))

    status_path = ai_dir / INDEX_FILES["status"]
    try:
        status = status_path.read_text(encoding="utf-8")
    except OSError:
        return applied
    reviewed = sum(bool(row.get("reviewed")) for row in units.values())
    re_audit = sum(bool(row.get("re_audit_required")) for row in units.values())
    status = _REVIEWED_RE.sub(f"- Audited units retained: {reviewed}", status)
    status = _REAUDIT_RE.sub(f"- Units requiring re-audit: {re_audit}", status)
    write_if_changed(status_path, status)
    return applied
