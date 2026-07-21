"""Startup notification for unfinished scientific work."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class UnfinishedWorkDialog(QDialog):
    def __init__(self, items, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Unfinished work detected")
        self.resize(820, 430)
        self.open_resume_center = False
        layout = QVBoxLayout(self)
        message = QLabel(
            "CALO-RPD Studio found unfinished resumable work. No new campaign is started automatically. Open Resume Center to inspect, resume, archive, or delete the records."
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
        open_button = QPushButton("Open Resume Center")
        buttons.addButton(open_button, QDialogButtonBox.ButtonRole.AcceptRole)
        open_button.clicked.connect(self._open)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open(self) -> None:
        self.open_resume_center = True
        self.accept()
