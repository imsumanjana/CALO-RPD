"""Database-backed CALO policy library and immutable experiment bindings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import uuid

from calo_rpd_studio.ai.model_io import checkpoint_sha256, load_checkpoint
from .policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
    infer_checkpoint_schema,
)
from .policy_lineage import PolicyLineageManager

_LOG = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class PolicyRecord:
    id: str
    name: str
    checkpoint_path: str
    sha256: str
    architecture_version: str
    state_schema_version: str
    action_schema_version: str
    training_environment_version: str
    qualification_status: str
    grade: str
    active: bool
    archived: bool
    metadata: dict

    @property
    def usable(self) -> bool:
        return not self.archived and Path(self.checkpoint_path).is_file()

    @property
    def runtime_compatible(self) -> bool:
        return (
            self.architecture_version == CALO_RUNTIME_ARCHITECTURE
            and self.state_schema_version == POLICY_STATE_SCHEMA
            and self.action_schema_version == POLICY_ACTION_SCHEMA
            and self.training_environment_version == TRAINING_ENVIRONMENT_VERSION
        )


class PolicyRegistry:
    """Manage policy artifacts without silently changing experiment provenance."""

    def __init__(self, database) -> None:
        self.database = database
        self.lineages = PolicyLineageManager(database)

    def suppress(self, sha256: str, *, reason: str = "user_deleted") -> None:
        self.database.suppress_policy_sha256(str(sha256), reason=reason)

    def is_suppressed(self, sha256: str) -> bool:
        return str(sha256).lower() in self.database.list_suppressed_policy_sha256()

    @staticmethod
    def inspect_checkpoint(path: str | Path) -> dict:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"CALO policy checkpoint not found: {source}")
        payload = load_checkpoint(source, map_location="cpu")
        schema = infer_checkpoint_schema(payload)
        metadata = dict(payload.get("metadata", {}) or {})
        checksum = checkpoint_sha256(source)
        return {
            "checkpoint_path": str(source),
            "sha256": checksum,
            "schema": schema,
            "metadata": metadata,
        }

    def register(
        self, path: str | Path, *, name: str | None = None, status: str | None = None
    ) -> PolicyRecord:
        inspected = self.inspect_checkpoint(path)
        source = Path(inspected["checkpoint_path"])
        schema = inspected["schema"]
        metadata = inspected["metadata"]
        existing = self.database.get_policy_by_sha256(inspected["sha256"])
        if existing is not None:
            return self._from_row(existing)
        native = bool(schema.get("native_v59", False))
        policy_id = str(uuid.uuid4())
        qualification_status = status or ("candidate" if native else "legacy_unqualified")
        grade = "U" if native else "C"
        self.database.upsert_policy(
            policy_id=policy_id,
            name=name or source.stem,
            checkpoint_path=str(source),
            sha256=inspected["sha256"],
            architecture_version=str(schema["runtime_architecture_version"]),
            state_schema_version=str(schema["state_schema_version"]),
            action_schema_version=str(schema["action_schema_version"]),
            training_environment_version=str(schema["training_environment_version"]),
            qualification_status=qualification_status,
            grade=grade,
            active=False,
            archived=False,
            metadata=metadata,
        )
        return self.get(policy_id)

    def discover_bundled(self, directory: str | Path) -> list[PolicyRecord]:
        output: list[PolicyRecord] = []
        for path in sorted(Path(directory).glob("*.pt")):
            stem = str(path.stem)
            if stem.endswith(".resume"):
                continue
            if "_lineage" in str(path):
                continue
            try:
                inspected = self.inspect_checkpoint(path)
                if self.is_suppressed(inspected["sha256"]):
                    continue
                output.append(self.register(path, name=path.stem))
            except Exception as exc:
                _LOG.warning("Skipping malformed/incompatible bundled policy %s: %s: %s", path, type(exc).__name__, exc)
                continue
        return output

    def list(self, *, include_archived: bool = False) -> list[PolicyRecord]:
        records = [
            self._from_row(row)
            for row in self.database.list_policies(include_archived=include_archived)
        ]
        grade_rank = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C": 6, "U": 7, "F": 8}
        records.sort(
            key=lambda item: (
                not item.active,
                item.archived,
                grade_rank.get(item.grade.upper(), 9),
                item.name.lower(),
            )
        )
        return records

    def get(self, policy_id: str) -> PolicyRecord:
        row = self.database.get_policy(policy_id)
        if row is None:
            raise KeyError(f"Unknown CALO policy: {policy_id}")
        return self._from_row(row)

    def activate(self, policy_id: str, *, allow_unqualified: bool = False) -> PolicyRecord:
        policy = self.get(policy_id)
        if not policy.usable:
            raise ValueError(f"Policy {policy.name!r} is archived or its checkpoint file is unavailable")
        if not policy.runtime_compatible:
            raise ValueError(
                f"Policy {policy.name!r} is not compatible with the current CALO runtime schema. "
                "Import/train a native compatible policy before activation."
            )
        if (
            policy.qualification_status not in {"qualified", "legacy_qualified"}
            and not allow_unqualified
        ):
            raise ValueError(
                f"Policy {policy.name!r} is {policy.qualification_status!r}. "
                "Only qualified policies can become the default active policy; use an explicit "
                "research-only experiment binding for an unqualified candidate."
            )
        inspected = self.inspect_checkpoint(policy.checkpoint_path)
        if inspected["sha256"] != policy.sha256:
            raise RuntimeError(
                "Policy checkpoint checksum changed since registration; activation is blocked"
            )
        self.database.set_active_policy(policy_id)
        return self.get(policy_id)

    def archive(self, policy_id: str) -> None:
        policy = self.get(policy_id)
        if policy.active:
            raise ValueError("The active policy cannot be archived. Activate another policy first.")
        self.database.update_policy(policy_id, archived=True)

    def unarchive(self, policy_id: str) -> None:
        policy = self.get(policy_id)
        if not policy.archived:
            return
        self.database.update_policy(policy_id, archived=False)

    def delete(self, policy_id: str, *, delete_artifact: bool = False) -> None:
        policy = self.get(policy_id)
        if policy.active:
            raise ValueError("The active policy cannot be deleted")
        references = self.database.policy_reference_count(policy_id, policy.sha256)
        if references > 0:
            raise ValueError(
                f"Policy is referenced by {references} experiment binding(s); archive it instead to preserve reproducibility"
            )
        checkpoint = self.database.get_policy_checkpoint_by_sha256(policy.sha256)
        if delete_artifact and checkpoint is not None:
            self.database.delete_policy_checkpoint(str(checkpoint["id"]))
        self.database.delete_policy(policy_id)
        if not delete_artifact:
            return
        source = Path(policy.checkpoint_path)
        existed = source.is_file()
        try:
            source.unlink(missing_ok=True)
            sidecar = source.with_suffix(source.suffix + ".sha256")
            sidecar.unlink(missing_ok=True)
        except OSError as exc:
            _LOG.error("Failed to delete policy artifact %s: %s", source, exc)
            # Suppress the exact SHA so rediscovery cannot silently resurrect a deliberately deleted record.
            self.suppress(policy.sha256, reason="delete_failed")
            raise
        # A deliberate delete is a project-scoped suppression even when the file was already absent.
        self.suppress(policy.sha256, reason="deleted_artifact" if existed else "artifact_already_absent")

    def bind_to_experiment_config(
        self, policy_id: str, config, *, deterministic: bool, allow_unqualified: bool = False
    ) -> dict:
        policy = self.get(policy_id)
        if not policy.usable:
            raise ValueError(f"Policy {policy.name!r} is archived or its checkpoint file is unavailable")
        if not policy.runtime_compatible:
            raise ValueError(
                f"Policy {policy.name!r} is incompatible with the current CALO runtime; experiment binding refused"
            )
        if (
            policy.qualification_status not in {"qualified", "legacy_qualified"}
            and not allow_unqualified
        ):
            raise ValueError(
                f"Policy {policy.name!r} is {policy.qualification_status!r}, not qualified. "
                "Run Policy Qualification or explicitly enable research-only unqualified use."
            )
        inspected = self.inspect_checkpoint(policy.checkpoint_path)
        if inspected["sha256"] != policy.sha256:
            raise RuntimeError("Policy artifact checksum mismatch; experiment binding refused")
        parameters = dict(config.algorithm_parameters.get("CALO", {}))
        binding = {
            "policy_id": policy.id,
            "policy_name": policy.name,
            "policy_checkpoint": policy.checkpoint_path,
            "policy_sha256": policy.sha256,
            "policy_architecture_version": policy.architecture_version,
            "policy_state_schema_version": policy.state_schema_version,
            "policy_action_schema_version": policy.action_schema_version,
            "policy_training_environment_version": policy.training_environment_version,
            "policy_qualification_status": policy.qualification_status,
            "policy_grade": policy.grade,
            "deterministic_policy": bool(deterministic),
            "strict_policy_binding": True,
            "allow_unqualified_policy": bool(allow_unqualified),
        }
        parameters.update(binding)
        config.algorithm_parameters["CALO"] = parameters
        return binding

    def register_lineage_snapshot(
        self,
        path: str | Path,
        *,
        lineage_id: str,
        cumulative_epoch: int,
        phase_index: int = 1,
        resume_path: str | Path = "",
        name: str | None = None,
    ) -> tuple[PolicyRecord, object]:
        """Register one immutable usable checkpoint in both the policy library and lineage history."""
        policy = self.register(path, name=name or Path(path).stem)
        checkpoint = self.lineages.register_checkpoint(
            lineage_id,
            path,
            cumulative_epoch=int(cumulative_epoch),
            phase_index=int(phase_index),
            resume_path=resume_path,
            metadata={"policy_id": policy.id, "policy_name": policy.name},
        )
        return policy, checkpoint

    def create_lineage(
        self,
        name: str,
        *,
        parent_lineage_id: str = "",
        forked_from_checkpoint_id: str = "",
        notes: str = "",
    ) -> str:
        return self.lineages.create(
            name,
            parent_lineage_id=parent_lineage_id,
            forked_from_checkpoint_id=forked_from_checkpoint_id,
            notes=notes,
        )

    @staticmethod
    def _from_row(row: dict) -> PolicyRecord:
        return PolicyRecord(
            id=str(row["id"]),
            name=str(row["name"]),
            checkpoint_path=str(row["checkpoint_path"]),
            sha256=str(row["sha256"]),
            architecture_version=str(row["architecture_version"]),
            state_schema_version=str(row["state_schema_version"]),
            action_schema_version=str(row["action_schema_version"]),
            training_environment_version=str(row["training_environment_version"]),
            qualification_status=str(row["qualification_status"]),
            grade=str(row["grade"]),
            active=bool(row["active"]),
            archived=bool(row["archived"]),
            metadata=json.loads(row.get("metadata_json") or "{}"),
        )
