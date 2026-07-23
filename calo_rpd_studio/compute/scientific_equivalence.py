"""Scientific-equivalence checks for protected scheduling.

Scheduling is allowed to change wall-clock order and device placement, but it must not change branch
scientific identity: seeds, target epochs/budgets, formulation fingerprints, or committed terminal results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class BranchScientificIdentity:
    branch_id: str
    seed: int
    scientific_config_fingerprint: str
    target: str

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def canonical_branch_set(identities: Iterable[BranchScientificIdentity]) -> dict[str, str]:
    return {item.branch_id: item.fingerprint() for item in sorted(identities, key=lambda x: x.branch_id)}


def scheduling_equivalent(
    sequential: Iterable[BranchScientificIdentity],
    concurrent: Iterable[BranchScientificIdentity],
) -> tuple[bool, dict[str, Any]]:
    left = canonical_branch_set(sequential)
    right = canonical_branch_set(concurrent)
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    changed = sorted(key for key in set(left) & set(right) if left[key] != right[key])
    return not (missing or extra or changed), {
        "missing": missing,
        "extra": extra,
        "changed": changed,
        "sequential": left,
        "concurrent": right,
    }


def compare_terminal_records(
    sequential: dict[str, dict[str, Any]], concurrent: dict[str, dict[str, Any]], *, ignore_keys: set[str] | None = None
) -> tuple[bool, dict[str, Any]]:
    """Compare branch terminal records while ignoring declared wall-clock/resource-placement metadata."""
    ignore = set(ignore_keys or {"wall_clock_seconds", "device", "started_at", "completed_at", "queue_wait_seconds"})

    def canonical(record: dict[str, Any]) -> str:
        payload = {k: v for k, v in dict(record).items() if k not in ignore}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

    keys = sorted(set(sequential) | set(concurrent))
    changed = []
    for key in keys:
        if key not in sequential or key not in concurrent or canonical(sequential.get(key, {})) != canonical(concurrent.get(key, {})):
            changed.append(key)
    return not changed, {"changed_branches": changed}


__all__ = [
    "BranchScientificIdentity",
    "canonical_branch_set",
    "scheduling_equivalent",
    "compare_terminal_records",
]
