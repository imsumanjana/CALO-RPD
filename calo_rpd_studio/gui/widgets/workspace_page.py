"""Reusable non-scrollable workspace surface."""

from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .page_header import PageHeader


class WorkspacePage(QWidget):
    """Base widget for workspaces that do not need a page-level scroll area.

    Data tables, plot canvases, and text viewers may still provide their own local
    scrolling. The page itself remains a normal widget so simple workspaces do
    not feel like a form embedded inside a scroll box.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: QWidget | None = None,
        *,
        margins: tuple[int, int, int, int] = (24, 22, 24, 22),
        spacing: int = 16,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("WorkspacePage")
        self.layout_root = QVBoxLayout(self)
        self.layout_root.setContentsMargins(*margins)
        self.layout_root.setSpacing(spacing)
        self.layout_root.addWidget(PageHeader(title, subtitle))
