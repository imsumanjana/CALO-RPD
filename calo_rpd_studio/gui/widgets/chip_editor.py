"""Compact structured editor for short integer identifier lists."""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget


class IntegerChipEditor(QWidget):
    """Comma-separated entry with parsed chips and a QLineEdit-compatible text API."""

    def __init__(self, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("IntegerChipEditor")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.editor = QLineEdit()
        self.editor.setPlaceholderText(placeholder)
        self.editor.setAccessibleName(placeholder or "Integer list")
        self.editor.editingFinished.connect(self._refresh_chips)
        layout.addWidget(self.editor)
        self.chip_row = QHBoxLayout()
        self.chip_row.setSpacing(4)
        self.chip_row.addStretch(1)
        layout.addLayout(self.chip_row)

    def text(self) -> str:
        return self.editor.text()

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit-compatible API
        self.editor.setText(str(value))
        self._refresh_chips()

    def values(self) -> tuple[int, ...]:
        return tuple(int(item.strip()) for item in self.text().split(",") if item.strip())

    def _refresh_chips(self) -> None:
        while self.chip_row.count() > 1:
            item = self.chip_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        try:
            values = self.values()
        except ValueError:
            values = ()
        for value in values[:12]:
            chip = QLabel(str(value))
            chip.setObjectName("InputChip")
            chip.setAccessibleName(f"Selected identifier {value}")
            self.chip_row.insertWidget(self.chip_row.count() - 1, chip)
