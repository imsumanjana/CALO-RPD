"""Scientist-authorized permanent policy-artifact deletion receipts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .policy_registry import PolicyRecord, PolicyRegistry


ARTIFACT_DELETION_SCHEMA = "calo-policy-artifact-deletion-v1"


def permanent_artifact_deletion_blocker(policy: PolicyRecord) -> str:
    """Return only exact-target blockers; scientific lifecycle state is not a deletion veto."""

    if policy.archived:
        return "This policy is already archived and is not an executable library model."
    source = Path(policy.checkpoint_path).expanduser()
    if source.is_symlink():
        return "Symbolic-link model targets cannot be permanently deleted from the library."
    if not source.is_file():
        return (
            "The registered model file is unavailable; there is no verified model file to delete."
        )
    return ""


def record_permanent_artifact_deletion(
    registry: PolicyRegistry,
    policy_id: str,
    *,
    expected_sha256: str,
    reason: str = "scientist_permanent_model_deletion",
    deleted_scope: str = "model_artifact",
) -> PolicyRecord:
    """Deactivate/archive a deleted artifact while retaining historical scientific provenance."""

    policy = registry.get(policy_id)
    expected = str(expected_sha256).strip().lower()
    if policy.sha256.lower() != expected:
        raise RuntimeError("Selected policy identity changed before permanent deletion")

    metadata = dict(policy.metadata)
    metadata["artifact_deletion"] = {
        "schema_version": ARTIFACT_DELETION_SCHEMA,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason),
        "deleted_scope": str(deleted_scope),
        "checkpoint_path": policy.checkpoint_path,
        "sha256": expected,
        "was_active": bool(policy.active),
        "qualification_status": policy.qualification_status,
        "qualification_record_count": len(
            registry.database.list_policy_qualifications(policy.id)
        ),
        "experiment_binding_count": registry.database.policy_reference_count(
            policy.id, policy.sha256
        ),
        "lineage_checkpoint_referenced": bool(
            registry.database.get_policy_checkpoint_by_sha256(policy.sha256) is not None
        ),
        "historical_records_retained": True,
    }
    registry.database.update_policy(
        policy.id,
        active=False,
        archived=True,
        metadata_json=metadata,
    )
    registry.suppress(expected, reason=reason)
    updated = registry.get(policy.id)
    if updated.active or not updated.archived:
        raise RuntimeError("Deleted model policy was not safely deactivated and archived")
    return updated
