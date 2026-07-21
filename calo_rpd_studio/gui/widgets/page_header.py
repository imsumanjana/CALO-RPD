"""Consistent modern workspace heading."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PageHeader")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(5)

        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        if subtitle:
            description = QLabel(subtitle)
            description.setObjectName("PageSubtitle")
            description.setWordWrap(True)
            layout.addWidget(description)
