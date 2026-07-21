"""Modern reusable card surfaces used by dashboard and compact workspaces."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class SectionCard(QFrame):
    """A clean card with an optional heading and subtitle."""

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self.layout_root = QVBoxLayout(self)
        self.layout_root.setContentsMargins(18, 16, 18, 18)
        self.layout_root.setSpacing(10)

        if title:
            heading = QLabel(title)
            heading.setObjectName("CardTitle")
            heading.setWordWrap(True)
            self.layout_root.addWidget(heading)
        if subtitle:
            description = QLabel(subtitle)
            description.setObjectName("CardSubtitle")
            description.setWordWrap(True)
            self.layout_root.addWidget(description)


class MetricCard(QFrame):
    """Compact dashboard metric tile with semantic labels."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        detail: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)

        self.label = QLabel(label)
        self.label.setObjectName("MetricLabel")
        self.label.setWordWrap(True)
        self.value = QLabel(value)
        self.value.setObjectName("MetricValue")
        self.value.setWordWrap(True)
        self.detail = QLabel(detail)
        self.detail.setObjectName("MetricDetail")
        self.detail.setWordWrap(True)

        layout.addWidget(self.label)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def set_metric(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)
