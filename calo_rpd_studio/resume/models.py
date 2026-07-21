"""Models shared by campaign, training, validation, and export resume flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResumeTaskType(str, Enum):
    EXPERIMENT = "experiment"
    POLICY_TRAINING = "policy_training"
    VALIDATION = "validation"
    PORTFOLIO_EXPORT = "portfolio_export"


class ResumeStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


@dataclass(slots=True)
class ResumeItem:
    id: str
    task_type: str
    title: str
    status: str
    progress_current: int
    progress_total: int
    updated_at: str
    state: dict
    resumable: bool

    @property
    def progress_text(self) -> str:
        return f"{self.progress_current}/{self.progress_total}" if self.progress_total else "—"
