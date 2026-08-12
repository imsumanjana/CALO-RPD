"""Fail-closed governing-policy readiness for the v12 policy-gated TSH-CALO path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from types import SimpleNamespace

from .tsh_calo_schema import TSH_CALO_ALGORITHM_ID

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
    algorithm_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _record_status(
    record,
    *,
    ready: bool,
    state: str,
    reason: str,
) -> GoverningPolicyStatus:
    """Build a fully labelled status without relying on positional record-field ordering."""

    return GoverningPolicyStatus(
        ready=ready,
        state=state,
        reason=reason,
        policy_id=record.id,
        policy_name=record.name,
        policy_sha256=record.sha256,
        qualification_status=record.qualification_status,
        grade=record.grade,
        algorithm_id=record.algorithm_id,
    )


def evaluate_governing_policy(registry) -> GoverningPolicyStatus:
    """Return the one active policy only when every governing-intelligence gate passes."""
    records = registry.list(include_archived=True)
    active = next((record for record in records if record.active), None)
    if active is None:
        if not records:
            return GoverningPolicyStatus(
                False,
                "missing",
                "No release policy exists. CALO remains in the policy-free safe path until a "
                "completely new post-freeze TSH-CALO candidate is independently qualified and "
                "activated.",
            )
        return GoverningPolicyStatus(
            False,
            "inactive",
            "Only development policy records exist, or no new qualified compatible TSH-CALO "
            "policy is active. CALO remains policy-free.",
        )
    if active.archived:
        return _record_status(
            active,
            ready=False,
            state="archived",
            reason="The active policy record is archived.",
        )
    if not active.post_development_eligible:
        return _record_status(
            active,
            ready=False,
            state="development_only",
            reason=(
                "The active record is a pre-freeze development policy and cannot govern v12. "
                "A completely new A-E/F-off policy must be trained after the development freeze."
            ),
        )
    if not active.usable:
        return _record_status(
            active,
            ready=False,
            state="artifact_unavailable",
            reason="The active policy artifact is missing or unavailable.",
        )
    if not active.compatible_with(TSH_CALO_ALGORITHM_ID):
        return _record_status(
            active,
            ready=False,
            state="incompatible",
            reason="The active policy is not compatible with the TSH-CALO A-E/F-off runtime ABI.",
        )
    if active.qualification_status != "qualified":
        return _record_status(
            active,
            ready=False,
            state="unqualified",
            reason=(
                f"The active policy is {active.qualification_status!r}; a qualified governing "
                "policy is required."
            ),
        )
    try:
        inspected = registry.inspect_checkpoint(active.checkpoint_path)
    except Exception as exc:
        _LOG.warning("Governing-policy integrity inspection failed", exc_info=True)
        return _record_status(
            active,
            ready=False,
            state="inspection_failed",
            reason=f"Active policy integrity inspection failed: {type(exc).__name__}: {exc}",
        )
    if str(inspected.get("sha256", "")).lower() != active.sha256.lower():
        return _record_status(
            active,
            ready=False,
            state="checksum_mismatch",
            reason="The active policy SHA-256 no longer matches the registered immutable artifact.",
        )
    try:
        registry.bind_to_experiment_config(
            active.id,
            SimpleNamespace(algorithm_parameters={}),
            deterministic=True,
            allow_unqualified=False,
            algorithm_id=TSH_CALO_ALGORITHM_ID,
        )
    except Exception as exc:
        _LOG.warning("Governing-policy binding validation failed", exc_info=True)
        return _record_status(
            active,
            ready=False,
            state="binding_invalid",
            reason=f"Active TSH-CALO binding validation failed: {type(exc).__name__}: {exc}",
        )
    return _record_status(
        active,
        ready=True,
        state="ready",
        reason=(
            "A new qualified, runtime-compatible, integrity-verified TSH-CALO governing policy "
            "is active."
        ),
    )
