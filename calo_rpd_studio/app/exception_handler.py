"""User-safe exception dialog with expandable technical details."""

import traceback
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPlainTextEdit, QDialogButtonBox


class ErrorDialog(QDialog):
    def __init__(self, title, message, details="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 420)
        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        box = QPlainTextEdit(details)
        box.setReadOnly(True)
        layout.addWidget(box, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def show_exception(parent, title, exc):
    ErrorDialog(
        title,
        str(exc),
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        parent,
    ).exec()
