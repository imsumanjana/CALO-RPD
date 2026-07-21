"""Database-backed CALO policy library and immutable experiment bindings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from calo_rpd_studio.ai.model_io import load_checkpoint
from .policy_schema import infer_checkpoint_schema


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


class PolicyRegistry:
    """Manage policy artifacts without silently changing experiment provenance."""

    def __init__(self, database) -> None:
        self.database = database

    @staticmethod
    def inspect_checkpoint(path: str | Path) -> dict:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"CALO policy checkpoint not found: {source}")
        payload = load_checkpoint(source, map_location="cpu")
        schema = infer_checkpoint_schema(payload)
        metadata = dict(payload.get("metadata", {}) or {})
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        return {
            "checkpoint_path": str(source),
            "sha256": checksum,
            "schema": schema,
            "metadata": metadata,
        }

    def register(self, path: str | Path, *, name: str | None = None, status: str | None = None) -> PolicyRecord:
        inspected = self.inspect_checkpoint(path)
        source = Path(inspected["checkpoint_path"])
        schema = inspected["schema"]
        metadata = inspected["metadata"]
        existing = self.database.get_policy_by_sha256(inspected["sha256"])
        if existing is not None:
            return self._from_row(existing)
        native = bool(schema["native_v41"])
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
            try:
                output.append(self.register(path, name=path.stem))
            except Exception:
                continue
        return output

    def list(self, *, include_archived: bool = False) -> list[PolicyRecord]:
        records = [self._from_row(row) for row in self.database.list_policies(include_archived=include_archived)]
        grade_rank = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C": 6, "U": 7, "F": 8}
        records.sort(key=lambda item: (not item.active, item.archived, grade_rank.get(item.grade.upper(), 9), item.name.lower()))
        return records

    def get(self, policy_id: str) -> PolicyRecord:
        row = self.database.get_policy(policy_id)
        if row is None:
            raise KeyError(f"Unknown CALO policy: {policy_id}")
        return self._from_row(row)

    def activate(self, policy_id: str, *, allow_unqualified: bool = False) -> PolicyRecord:
        policy = self.get(policy_id)
        if policy.qualification_status not in {"qualified", "legacy_qualified"} and not allow_unqualified:
            raise ValueError(
                f"Policy {policy.name!r} is {policy.qualification_status!r}. "
                "Only qualified policies can become the default active policy; use an explicit "
                "research-only experiment binding for an unqualified candidate."
            )
        inspected = self.inspect_checkpoint(policy.checkpoint_path)
        if inspected["sha256"] != policy.sha256:
            raise RuntimeError("Policy checkpoint checksum changed since registration; activation is blocked")
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
        self.database.delete_policy(policy_id)
        if delete_artifact:
            try:
                Path(policy.checkpoint_path).unlink(missing_ok=True)
            except OSError:
                pass

    def bind_to_experiment_config(self, policy_id: str, config, *, deterministic: bool, allow_unqualified: bool = False) -> dict:
        policy = self.get(policy_id)
        if policy.qualification_status not in {"qualified", "legacy_qualified"} and not allow_unqualified:
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
