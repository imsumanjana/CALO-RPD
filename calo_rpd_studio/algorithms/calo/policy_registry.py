"""Database-backed CALO policy library and immutable experiment bindings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import uuid

from calo_rpd_studio.ai.model_io import checkpoint_sha256, load_checkpoint
from .policy_schema import (
    CALO_ALGORITHM_ID,
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
    infer_checkpoint_schema,
)
from .tsh_calo_schema import (
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_POLICY_ARCHITECTURE,
    TSH_CALO_STATE_SCHEMA,
    TSH_CALO_TRAINING_ENVIRONMENT,
)
from .tsh_calo_qualification import load_tsh_calo_qualification_receipt
from .policy_lineage import PolicyLineageManager

_LOG = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_development_training_provenance(metadata: dict) -> bool:
    """Return whether provenance proves completely new post-freeze training."""

    provenance = dict(metadata.get("training_provenance", {}) or {})
    if str(metadata.get("artifact_kind", "")) == "ensemble_policy":
        members = list(metadata.get("ensemble_members", []) or [])
        provenance_rows = [dict(item.get("training_provenance", {}) or {}) for item in members]
        if len(provenance_rows) < 2:
            return False
    elif provenance.get("source_kind") == "independent_policy_training_ensemble":
        members = list(provenance.get("members", []) or [])
        provenance_rows = [dict(item.get("training_provenance", {}) or {}) for item in members]
        if len(provenance_rows) < 2:
            return False
    else:
        provenance_rows = [provenance]
    identities: set[tuple[str, str, str]] = set()
    for row in provenance_rows:
        source_commit = str(row.get("source_commit", "")).strip().lower()
        freeze_commit = str(row.get("development_freeze_commit", "")).strip().lower()
        freeze_sha256 = str(row.get("development_freeze_sha256", "")).strip().lower()
        acceptance_sha256 = str(row.get("phase4_acceptance_sha256", "")).strip().lower()
        if (
            not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or freeze_commit != source_commit
            or not re.fullmatch(r"[0-9a-f]{64}", freeze_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", acceptance_sha256)
            or str(row.get("initialization_policy_sha256", "")).strip()
        ):
            return False
        identities.add((source_commit, freeze_sha256, acceptance_sha256))
    return len(identities) == 1


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
    algorithm_id: str
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
        """Backward-compatible alias for the frozen CALO v5.9 runtime."""
        return self.compatible_with(CALO_ALGORITHM_ID)

    @property
    def post_development_eligible(self) -> bool:
        """Whether this artifact proves new training against the development freeze."""

        return _post_development_training_provenance(self.metadata)

    def compatible_with(self, algorithm_id: str) -> bool:
        if str(algorithm_id) == TSH_CALO_ALGORITHM_ID:
            return bool(
                self.algorithm_id == TSH_CALO_ALGORITHM_ID
                and self.architecture_version == TSH_CALO_ALGORITHM_VERSION
                and self.state_schema_version == TSH_CALO_STATE_SCHEMA
                and self.action_schema_version == TSH_CALO_ACTION_SCHEMA
                and self.training_environment_version == TSH_CALO_TRAINING_ENVIRONMENT
                and str(self.metadata.get("policy_architecture_version", ""))
                == TSH_CALO_POLICY_ARCHITECTURE
            )
        return bool(
            str(algorithm_id) == CALO_ALGORITHM_ID
            and self.algorithm_id == CALO_ALGORITHM_ID
            and self.architecture_version == CALO_RUNTIME_ARCHITECTURE
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
        metadata.setdefault("algorithm_id", str(schema["algorithm_id"]))
        metadata.setdefault(
            "policy_architecture_version", str(schema["policy_architecture_version"])
        )
        native = bool(schema.get("native_supported", False))
        if bool(schema.get("native_tsh_calo", False)) and status not in {None, "candidate"}:
            raise ValueError(
                "TSH-CALO registration creates candidates only; qualification is a separate lifecycle action"
            )
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
                _LOG.warning(
                    "Skipping malformed/incompatible bundled policy %s: %s: %s",
                    path,
                    type(exc).__name__,
                    exc,
                )
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

    def activate(
        self,
        policy_id: str,
        *,
        allow_unqualified: bool = False,
        algorithm_id: str = CALO_ALGORITHM_ID,
    ) -> PolicyRecord:
        policy = self.get(policy_id)
        if not policy.post_development_eligible:
            raise ValueError(
                "Existing/pre-freeze policies are development-only and cannot be activated in the "
                "v12 release lifecycle. Train and qualify a completely new A-E/F-off policy after "
                "the development freeze."
            )
        if algorithm_id != TSH_CALO_ALGORITHM_ID:
            raise ValueError(
                "v12 activation accepts only a completely new TSH-CALO A-E/F-off ensemble"
            )
        if algorithm_id == TSH_CALO_ALGORITHM_ID and allow_unqualified:
            raise ValueError("TSH-CALO policies cannot be activated before qualification")
        if (
            algorithm_id == TSH_CALO_ALGORITHM_ID
            and int(policy.metadata.get("ensemble_size", 1)) < 2
        ):
            raise ValueError("TSH-CALO activation requires a qualified epistemic ensemble artifact")
        if not policy.usable:
            raise ValueError(
                f"Policy {policy.name!r} is archived or its checkpoint file is unavailable"
            )
        if not policy.compatible_with(algorithm_id):
            raise ValueError(
                f"Policy {policy.name!r} is not compatible with the {algorithm_id} runtime schema. "
                "Import/train a native compatible policy before activation."
            )
        if policy.qualification_status != "qualified":
            raise ValueError(
                f"Policy {policy.name!r} is {policy.qualification_status!r}. "
                "Only an independently qualified new TSH-CALO policy can become active."
            )
        inspected = self.inspect_checkpoint(policy.checkpoint_path)
        if inspected["sha256"] != policy.sha256:
            raise RuntimeError(
                "Policy checkpoint checksum changed since registration; activation is blocked"
            )
        if algorithm_id == TSH_CALO_ALGORITHM_ID:
            rows = [
                row
                for row in self.database.list_policy_qualifications(policy.id)
                if bool(row.get("passed", False))
            ]
            if not rows:
                raise ValueError("TSH-CALO activation requires a passed qualification record")
            qualification = rows[0]
            receipt = load_tsh_calo_qualification_receipt(
                json.loads(str(qualification.get("config_json", "{}")) or "{}"),
                expected_policy_sha256=policy.sha256,
            )
            metadata = dict(policy.metadata)
            metadata["activated_qualification"] = {
                "qualification_id": str(qualification.get("id", "")),
                "created_at": str(qualification.get("created_at", "")),
                "grade": str(qualification.get("grade", "")),
                "receipt": receipt.as_dict(),
            }
            self.database.update_policy(policy.id, metadata_json=metadata)
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
        del policy_id, delete_artifact
        raise PermissionError(
            "Direct policy deletion is disabled. Export an exact policy-retirement inventory and "
            "dry-run plan; post-freeze removal requires separate authorization and an immutable receipt."
        )

    def bind_to_experiment_config(
        self,
        policy_id: str,
        config,
        *,
        deterministic: bool,
        allow_unqualified: bool = False,
        algorithm_id: str = CALO_ALGORITHM_ID,
    ) -> dict:
        policy = self.get(policy_id)
        if not policy.post_development_eligible:
            raise ValueError(
                "Existing/pre-freeze policies cannot be bound to v12 experiments. Only a completely "
                "new post-development A-E/F-off policy may enter the release lifecycle."
            )
        if algorithm_id != TSH_CALO_ALGORITHM_ID:
            raise ValueError("v12 experiments may bind only a new qualified TSH-CALO ensemble")
        if algorithm_id == TSH_CALO_ALGORITHM_ID and allow_unqualified:
            raise ValueError("TSH-CALO experiments cannot consume an unqualified policy")
        if (
            algorithm_id == TSH_CALO_ALGORITHM_ID
            and int(policy.metadata.get("ensemble_size", 1)) < 2
        ):
            raise ValueError("TSH-CALO experiments require an activated epistemic ensemble")
        if algorithm_id == TSH_CALO_ALGORITHM_ID and not policy.active:
            raise ValueError("TSH-CALO experiments require the explicitly activated ensemble")
        if not policy.usable:
            raise ValueError(
                f"Policy {policy.name!r} is archived or its checkpoint file is unavailable"
            )
        if not policy.compatible_with(algorithm_id):
            raise ValueError(
                f"Policy {policy.name!r} is incompatible with the {algorithm_id} runtime; experiment binding refused"
            )
        if policy.qualification_status != "qualified":
            raise ValueError(
                f"Policy {policy.name!r} is {policy.qualification_status!r}, not qualified. "
                "Only a new independently qualified TSH-CALO policy may be bound."
            )
        inspected = self.inspect_checkpoint(policy.checkpoint_path)
        if inspected["sha256"] != policy.sha256:
            raise RuntimeError("Policy artifact checksum mismatch; experiment binding refused")
        parameters = dict(config.algorithm_parameters.get(algorithm_id, {}))
        binding = {
            "policy_algorithm_id": policy.algorithm_id,
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
            "policy_active_at_binding": bool(policy.active),
            "deterministic_policy": bool(deterministic),
            "strict_policy_binding": True,
            "allow_unqualified_policy": bool(allow_unqualified),
            "allow_cpu_fallback": False,
            "baseline_fallback_permitted": False,
        }
        if policy.algorithm_id == TSH_CALO_ALGORITHM_ID:
            binding["policy_feature_flags"] = dict(policy.metadata.get("feature_flags", {}))
            binding["policy_artifact_kind"] = str(policy.metadata.get("artifact_kind", ""))
            binding["policy_ensemble_size"] = int(policy.metadata.get("ensemble_size", 1))
            binding["policy_ensemble_members"] = list(policy.metadata.get("ensemble_members", []))
            binding["policy_training_provenance"] = (
                {
                    "source_kind": "independent_policy_training_ensemble",
                    "members": list(policy.metadata.get("ensemble_members", [])),
                }
                if binding["policy_artifact_kind"] == "ensemble_policy"
                else dict(policy.metadata.get("training_provenance", {}))
            )
            activated = dict(policy.metadata.get("activated_qualification", {}) or {})
            receipt = load_tsh_calo_qualification_receipt(
                {"tsh_calo_qualification_receipt": activated.get("receipt", {})},
                expected_policy_sha256=policy.sha256,
            )
            binding["policy_qualification_id"] = str(activated.get("qualification_id", ""))
            binding["policy_qualification_receipt_sha256"] = receipt.receipt_sha256
            binding["policy_qualification_receipt"] = receipt.as_dict()
            binding["policy_ood_calibration_sha256"] = receipt.ood_calibration_sha256
            binding["ood_calibration"] = dict(receipt.ood_calibration)
        parameters.update(binding)
        config.algorithm_parameters[algorithm_id] = parameters
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
        return str(
            self.lineages.create(
                name,
                parent_lineage_id=parent_lineage_id,
                forked_from_checkpoint_id=forked_from_checkpoint_id,
                notes=notes,
            )
        )

    @staticmethod
    def _from_row(row: dict) -> PolicyRecord:
        metadata = json.loads(row.get("metadata_json") or "{}")
        return PolicyRecord(
            id=str(row["id"]),
            name=str(row["name"]),
            checkpoint_path=str(row["checkpoint_path"]),
            sha256=str(row["sha256"]),
            architecture_version=str(row["architecture_version"]),
            state_schema_version=str(row["state_schema_version"]),
            action_schema_version=str(row["action_schema_version"]),
            training_environment_version=str(row["training_environment_version"]),
            algorithm_id=str(metadata.get("algorithm_id", CALO_ALGORITHM_ID)),
            qualification_status=str(row["qualification_status"]),
            grade=str(row["grade"]),
            active=bool(row["active"]),
            archived=bool(row["archived"]),
            metadata=metadata,
        )
