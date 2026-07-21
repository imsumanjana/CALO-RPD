"""CALO v5 policy lineage and checkpoint-family management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import uuid

from calo_rpd_studio.ai.model_io import load_checkpoint


@dataclass(frozen=True, slots=True)
class PolicyCheckpointRecord:
    id: str
    lineage_id: str
    cumulative_epoch: int
    phase_index: int
    checkpoint_path: str
    resume_path: str
    sha256: str
    qualification_status: str
    grade: str
    is_latest: bool
    is_best: bool
    metadata: dict


class PolicyLineageManager:
    """Track latest and best-qualified checkpoints without rewriting policy history."""

    def __init__(self, database) -> None:
        self.database = database

    def create(
        self,
        name: str,
        *,
        parent_lineage_id: str = "",
        forked_from_checkpoint_id: str = "",
        notes: str = "",
    ) -> str:
        return self.database.create_policy_lineage(
            name,
            parent_lineage_id=parent_lineage_id,
            forked_from_checkpoint_id=forked_from_checkpoint_id,
            notes=notes,
        )

    def fork(self, checkpoint_id: str, name: str, *, notes: str = "") -> str:
        source = self.database.get_policy_checkpoint(checkpoint_id)
        if source is None:
            raise KeyError(f"Unknown policy checkpoint: {checkpoint_id}")
        return self.create(
            name,
            parent_lineage_id=str(source["lineage_id"]),
            forked_from_checkpoint_id=str(checkpoint_id),
            notes=notes,
        )

    def register_checkpoint(
        self,
        lineage_id: str,
        checkpoint_path: str | Path,
        *,
        cumulative_epoch: int,
        phase_index: int = 1,
        resume_path: str | Path = "",
        checkpoint_id: str | None = None,
        metadata: dict | None = None,
        is_latest: bool = True,
    ) -> PolicyCheckpointRecord:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Ensure the artifact is actually a loadable policy before recording it.
        load_checkpoint(path, map_location="cpu")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint_id = str(checkpoint_id or uuid.uuid4())
        self.database.add_policy_checkpoint(
            checkpoint_id=checkpoint_id,
            lineage_id=lineage_id,
            cumulative_epoch=int(cumulative_epoch),
            phase_index=int(phase_index),
            checkpoint_path=str(path),
            resume_path=str(Path(resume_path).expanduser().resolve()) if str(resume_path) else "",
            sha256=sha,
            metadata=metadata or {},
            is_latest=is_latest,
        )
        return self.get_checkpoint(checkpoint_id)

    def get_checkpoint(self, checkpoint_id: str) -> PolicyCheckpointRecord:
        row = self.database.get_policy_checkpoint(checkpoint_id)
        if row is None:
            raise KeyError(checkpoint_id)
        return self._record(row)

    def checkpoints(self, lineage_id: str) -> list[PolicyCheckpointRecord]:
        return [self._record(row) for row in self.database.list_policy_checkpoints(lineage_id)]

    def latest(self, lineage_id: str) -> PolicyCheckpointRecord | None:
        rows = self.checkpoints(lineage_id)
        marked = [row for row in rows if row.is_latest]
        return marked[-1] if marked else (rows[-1] if rows else None)

    def best(self, lineage_id: str) -> PolicyCheckpointRecord | None:
        rows = [row for row in self.checkpoints(lineage_id) if row.is_best]
        return rows[-1] if rows else None

    def mark_best(self, lineage_id: str, checkpoint_id: str) -> None:
        self.database.mark_best_policy_checkpoint(lineage_id, checkpoint_id)

    def export_manifest(self, lineage_id: str, path: str | Path) -> Path:
        lineage = self.database.get_policy_lineage(lineage_id)
        if lineage is None:
            raise KeyError(lineage_id)
        payload = {
            "schema_version": 1,
            "lineage": lineage,
            "checkpoints": [asdict(row) for row in self.checkpoints(lineage_id)],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return target

    @staticmethod
    def _record(row: dict) -> PolicyCheckpointRecord:
        return PolicyCheckpointRecord(
            id=str(row["id"]),
            lineage_id=str(row["lineage_id"]),
            cumulative_epoch=int(row["cumulative_epoch"]),
            phase_index=int(row["phase_index"]),
            checkpoint_path=str(row["checkpoint_path"]),
            resume_path=str(row.get("resume_path", "")),
            sha256=str(row["sha256"]),
            qualification_status=str(row["qualification_status"]),
            grade=str(row["grade"]),
            is_latest=bool(row["is_latest"]),
            is_best=bool(row["is_best"]),
            metadata=dict(row.get("metadata", {})),
        )
