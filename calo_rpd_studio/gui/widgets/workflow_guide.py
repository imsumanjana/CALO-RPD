"""Compact guided-workflow banner shown above the current workspace."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class WorkflowGuide(QWidget):
    next_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowGuide")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(14)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        self.step = QLabel("Guided workflow")
        self.step.setObjectName("WorkflowStep")
        self.instruction = QLabel("")
        self.instruction.setObjectName("WorkflowInstruction")
        self.instruction.setWordWrap(True)
        text_layout.addWidget(self.step)
        text_layout.addWidget(self.instruction)
        layout.addLayout(text_layout, 1)

        self.next_button = QPushButton("Go to next step")
        self.next_button.setObjectName("WorkflowNextButton")
        self.next_button.clicked.connect(self.next_clicked)
        layout.addWidget(self.next_button)

    def set_guidance(self, step_text: str, instruction: str, button_text: str, enabled: bool) -> None:
        self.step.setText(step_text)
        self.instruction.setText(instruction)
        self.next_button.setText(button_text)
        self.next_button.setEnabled(enabled)
