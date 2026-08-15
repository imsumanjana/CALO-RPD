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


def governing_policy_user_message(status: GoverningPolicyStatus) -> str:
    """Describe policy availability without exposing engineering lifecycle terminology."""

    if status.ready:
        identity = status.policy_name or "The selected TSH-CALO policy"
        grade = f" (grade {status.grade})" if status.grade else ""
        return f"{identity}{grade} is verified, compatible, and selected for experiments."
    messages = {
        "archived": "The selected policy is archived. Restore it or select another verified policy.",
        "artifact_unavailable": (
            "The selected policy file is unavailable. Select another verified policy."
        ),
        "incompatible": (
            "The selected policy is not compatible with this application. Select a compatible policy."
        ),
        "inspection_failed": (
            "The selected policy could not be verified. Rule-based CALO remains available."
        ),
        "checksum_mismatch": (
            "The selected policy did not pass integrity verification. Rule-based CALO remains available."
        ),
        "binding_invalid": (
            "The selected policy cannot be applied to experiments. Rule-based CALO remains available."
        ),
    }
    return messages.get(
        getattr(status, "state", ""),
        "Select a verified, compatible TSH-CALO policy for policy-guided experiments. "
        "Rule-based CALO remains available.",
    )


def policy_record_user_status(record) -> str:
    """Return a concise scientific availability label for a policy-library row."""

    if record.archived:
        return "Archived"
    if not record.usable:
        return "File unavailable"
    if not record.compatible_with(TSH_CALO_ALGORITHM_ID):
        return "Not compatible"
    ready = record.qualification_status in {"qualified", "scientist_selected"}
    if not ready:
        return (
            "Assessment ready · scientist decision required"
            if record.qualification_status == "assessed"
            else "Feasibility assessment required"
        )
    if record.active:
        return "Verification required"
    return (
        "Scientist selected · activation available"
        if record.qualification_status == "scientist_selected"
        else "Legacy qualified · activation available"
    )


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
                "No policy exists. CALO remains in the policy-free safe path until a compatible "
                "TSH-CALO candidate is independently qualified and explicitly activated.",
            )
        return GoverningPolicyStatus(
            False,
            "inactive",
            "No qualified compatible TSH-CALO policy is active. CALO remains policy-free.",
        )
    if active.archived:
        return _record_status(
            active,
            ready=False,
            state="archived",
            reason="The active policy record is archived.",
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
    if active.qualification_status not in {"qualified", "scientist_selected"}:
        return _record_status(
            active,
            ready=False,
            state="unqualified",
            reason=(
                f"The active policy is {active.qualification_status!r}; an explicit scientist "
                "selection or retained legacy qualification is required."
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
            "A scientist-selected or legacy-qualified, runtime-compatible, integrity-verified "
            "TSH-CALO governing policy is explicitly active."
        ),
    )
