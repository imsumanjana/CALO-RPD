"""Startup notification for unfinished scientific work."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class UnfinishedWorkDialog(QDialog):
    def __init__(self, items, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Unfinished work detected")
        self.resize(820, 430)
        layout = QVBoxLayout(self)
        message = QLabel(
            "CALO-RPD Studio found unfinished work. Nothing is resumed automatically. Use the "
            "relevant workflow controls; policy-training resume and finite extension are available "
            "from Policies > Train policy."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        table = QTableWidget(len(items), 5)
        table.setHorizontalHeaderLabels(["Type", "Task", "Progress", "Status", "Last activity"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        for row, item in enumerate(items):
            for col, value in enumerate(
                (item.task_type, item.title, item.progress_text, item.status, item.updated_at)
            ):
                table.setItem(row, col, QTableWidgetItem(str(value)))
        layout.addWidget(table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
