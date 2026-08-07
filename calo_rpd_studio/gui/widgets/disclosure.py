"""Accessible progressive-disclosure widgets for details and advanced controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QToolButton, QVBoxLayout, QWidget


class DisclosurePanel(QFrame):
    """A keyboard-operable details drawer with a concise closed-state summary."""

    def __init__(
        self,
        title: str,
        summary: str,
        content: QWidget,
        parent: QWidget | None = None,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DisclosurePanel")
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self.toggle = QToolButton()
        self.toggle.setObjectName("DisclosureToggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setText(title)
        self.toggle.setAccessibleName(title)
        self.toggle.setAccessibleDescription(summary)
        self.toggle.clicked.connect(self.set_expanded)
        layout.addWidget(self.toggle)

        self.summary = QLabel(summary)
        self.summary.setObjectName("DisclosureSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self.toggle.blockSignals(True)
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.blockSignals(False)
        self.content.setVisible(expanded)
        self.toggle.setAccessibleDescription(
            f"{'Expanded' if expanded else 'Collapsed'}. {self.summary.text()}"
        )
