"""Accessible, consistent tab navigation for multi-section workspaces."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTabWidget, QWidget


class WorkspaceTabs(QTabWidget):
    """Present related workspace sections without a long vertical card stack."""

    def __init__(self, accessible_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkspaceSectionTabs")
        self.setAccessibleName(accessible_name)
        self.setDocumentMode(True)
        self.setUsesScrollButtons(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabBar().setExpanding(False)

    def add_section(self, title: str, page: QWidget, description: str = "") -> int:
        """Add an accessible section and retain its purpose as tab metadata."""
        page.setAccessibleName(title)
        if description:
            page.setAccessibleDescription(description)
        index = self.addTab(page, title)
        if description:
            self.setTabToolTip(index, description)
        return index
