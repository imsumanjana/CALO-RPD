"""Database-backed universal resume registry and atomic checkpoint helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from .models import ResumeItem, ResumeStatus, ResumeTaskType


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResumeService:
    def __init__(self, database, checkpoint_root: str | Path = "results_data/checkpoints") -> None:
        self.database = database
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        task_type: ResumeTaskType | str,
        title: str,
        state: dict,
        *,
        total: int = 0,
        task_id: str | None = None,
        status: ResumeStatus | str = ResumeStatus.PLANNED,
    ) -> str:
        task_id = task_id or str(uuid.uuid4())
        self.database.upsert_resumable_task(
            task_id,
            str(task_type.value if isinstance(task_type, ResumeTaskType) else task_type),
            title,
            str(status.value if isinstance(status, ResumeStatus) else status),
            0,
            int(total),
            state,
            resumable=True,
        )
        return task_id

    def update(
        self,
        task_id: str,
        *,
        status: ResumeStatus | str | None = None,
        current: int | None = None,
        total: int | None = None,
        state: dict | None = None,
        resumable: bool | None = None,
    ) -> None:
        self.database.update_resumable_task(
            task_id,
            status=None
            if status is None
            else str(status.value if isinstance(status, ResumeStatus) else status),
            progress_current=current,
            progress_total=total,
            state=state,
            resumable=resumable,
        )

    def recover_after_restart(self) -> dict:
        return self.database.mark_stale_running_interrupted()

    def unfinished(self) -> list[ResumeItem]:
        rows = self.database.list_resumable_tasks(unfinished_only=True)
        return [
            ResumeItem(
                id=row["id"],
                task_type=row["task_type"],
                title=row["title"],
                status=row["status"],
                progress_current=int(row["progress_current"]),
                progress_total=int(row["progress_total"]),
                updated_at=row["updated_at"],
                state=json.loads(row["state_json"] or "{}"),
                resumable=bool(row["resumable"]),
            )
            for row in rows
        ]

    def list_all(
        self, *, task_type: ResumeTaskType | str | None = None, resumable_only: bool = False
    ) -> list[ResumeItem]:
        rows = self.database.list_resumable_tasks(unfinished_only=False)
        expected = (
            None
            if task_type is None
            else str(task_type.value if isinstance(task_type, ResumeTaskType) else task_type)
        )
        items = []
        for row in rows:
            if expected is not None and str(row["task_type"]) != expected:
                continue
            if resumable_only and not bool(row["resumable"]):
                continue
            items.append(
                ResumeItem(
                    id=row["id"],
                    task_type=row["task_type"],
                    title=row["title"],
                    status=row["status"],
                    progress_current=int(row["progress_current"]),
                    progress_total=int(row["progress_total"]),
                    updated_at=row["updated_at"],
                    state=json.loads(row["state_json"] or "{}"),
                    resumable=bool(row["resumable"]),
                )
            )
        return items

    def checkpoint_path(self, task_id: str, name: str, suffix: str = ".json") -> Path:
        directory = self.checkpoint_root / task_id
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        return directory / f"{safe}{suffix}"

    @staticmethod
    def atomic_write_json(path: str | Path, payload: dict) -> tuple[Path, str]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            delete=False, dir=destination.parent, suffix=".tmp"
        ) as handle:
            handle.write(encoded)
            temp_path = Path(handle.name)
        temp_path.replace(destination)
        return destination, hashlib.sha256(encoded).hexdigest()

    def archive(self, task_id: str) -> None:
        self.update(task_id, status=ResumeStatus.ARCHIVED, resumable=False)

    def delete(self, task_id: str) -> None:
        self.database.delete_resumable_task(task_id)
        directory = self.checkpoint_root / task_id
        if directory.exists():
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
