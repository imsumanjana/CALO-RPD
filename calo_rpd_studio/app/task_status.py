"""Application-wide task status used by the persistent bottom status bar."""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class TaskStatus(QObject):
    """Tracks one user-facing foreground scientific task at a time."""

    changed = pyqtSignal(object)
    cancel_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.busy = False
        self.state = "Ready"
        self.title = ""
        self.detail = ""
        self.progress = 0
        self.cancellable = False

    def snapshot(self) -> dict:
        return {
            "busy": self.busy,
            "state": self.state,
            "title": self.title,
            "detail": self.detail,
            "progress": self.progress,
            "cancellable": self.cancellable,
        }

    def _emit(self) -> None:
        self.changed.emit(self.snapshot())

    def begin(
        self,
        title: str,
        *,
        detail: str = "",
        progress: int = -1,
        cancellable: bool = False,
    ) -> bool:
        if self.busy:
            return False
        self.busy = True
        self.state = "Busy"
        self.title = title
        self.detail = detail
        self.progress = int(progress)
        self.cancellable = bool(cancellable)
        self._emit()
        return True

    def update(self, progress: int | None = None, detail: str | None = None) -> None:
        if progress is not None:
            self.progress = max(-1, min(100, int(progress)))
        if detail is not None:
            self.detail = str(detail)
        self._emit()

    def finish(self, detail: str = "Completed") -> None:
        self.busy = False
        self.state = "Completed"
        self.detail = detail
        self.progress = 100
        self.cancellable = False
        self._emit()

    def fail(self, detail: str) -> None:
        self.busy = False
        self.state = "Failed"
        self.detail = detail
        self.progress = 0
        self.cancellable = False
        self._emit()

    def cancelled(self, detail: str = "Cancelled") -> None:
        self.busy = False
        self.state = "Cancelled"
        self.detail = detail
        self.progress = 0
        self.cancellable = False
        self._emit()

    def cancel(self, detail: str = "Cancellation requested") -> None:
        if not self.busy or not self.cancellable:
            return
        self.state = "Busy"
        self.detail = detail
        self.cancellable = False
        self._emit()
        self.cancel_requested.emit()

    def reset_ready(self) -> None:
        if self.busy:
            return
        self.state = "Ready"
        self.title = ""
        self.detail = ""
        self.progress = 0
        self.cancellable = False
        self._emit()
