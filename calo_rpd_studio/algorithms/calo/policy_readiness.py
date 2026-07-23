"""Fail-closed governing-policy readiness used by the v6.0 policy-first workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GoverningPolicyStatus:
    ready: bool
    state: str
    reason: str
    policy_id: str = ""
    policy_name: str = ""
    policy_sha256: str = ""
    qualification_status: str = ""
    grade: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_governing_policy(registry) -> GoverningPolicyStatus:
    """Return the one active policy only when every governing-intelligence gate passes."""
    records = registry.list(include_archived=True)
    active = next((record for record in records if record.active), None)
    if active is None:
        if not records:
            return GoverningPolicyStatus(
                False,
                "missing",
                "No CALO policy exists. Train or import a candidate, qualify it, then activate it.",
            )
        return GoverningPolicyStatus(
            False,
            "inactive",
            "Policy records exist, but no qualified compatible policy is active.",
        )
    if active.archived:
        return GoverningPolicyStatus(False, "archived", "The active policy record is archived.", active.id, active.name)
    if not active.usable:
        return GoverningPolicyStatus(
            False,
            "artifact_unavailable",
            "The active policy artifact is missing or unavailable.",
            active.id,
            active.name,
            active.sha256,
            active.qualification_status,
            active.grade,
        )
    if not active.runtime_compatible:
        return GoverningPolicyStatus(
            False,
            "incompatible",
            "The active policy is not compatible with the current CALO runtime ABI.",
            active.id,
            active.name,
            active.sha256,
            active.qualification_status,
            active.grade,
        )
    if active.qualification_status not in {"qualified", "legacy_qualified"}:
        return GoverningPolicyStatus(
            False,
            "unqualified",
            f"The active policy is {active.qualification_status!r}; a qualified governing policy is required.",
            active.id,
            active.name,
            active.sha256,
            active.qualification_status,
            active.grade,
        )
    try:
        inspected = registry.inspect_checkpoint(active.checkpoint_path)
    except Exception as exc:
        _LOG.warning("Governing-policy integrity inspection failed", exc_info=True)
        return GoverningPolicyStatus(
            False,
            "inspection_failed",
            f"Active policy integrity inspection failed: {type(exc).__name__}: {exc}",
            active.id,
            active.name,
            active.sha256,
            active.qualification_status,
            active.grade,
        )
    if str(inspected.get("sha256", "")).lower() != active.sha256.lower():
        return GoverningPolicyStatus(
            False,
            "checksum_mismatch",
            "The active policy SHA-256 no longer matches the registered immutable artifact.",
            active.id,
            active.name,
            active.sha256,
            active.qualification_status,
            active.grade,
        )
    return GoverningPolicyStatus(
        True,
        "ready",
        "Qualified, runtime-compatible, integrity-verified CALO governing policy is active.",
        active.id,
        active.name,
        active.sha256,
        active.qualification_status,
        active.grade,
    )
