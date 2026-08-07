"""Seven-step, single-scroll Study Setup presentation."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class StudySetupWorkflow(QFrame):
    """A compact seven-step shell; scientific state remains owned by existing panels."""

    step_changed = pyqtSignal(int)

    def __init__(
        self,
        steps: Sequence[tuple[str, str, QWidget]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(steps) != 7:
            raise ValueError("Study Setup requires exactly seven steps")
        self.setObjectName("StudySetupWorkflow")
        self.steps = tuple(steps)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        heading = QLabel("Study Setup")
        heading.setObjectName("StudySetupTitle")
        heading.setAccessibleName("Study Setup, seven steps")
        root.addWidget(heading)

        step_row = QHBoxLayout()
        step_row.setSpacing(8)
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons: list[QPushButton] = []
        for index, (title, description, _page) in enumerate(self.steps):
            button = QPushButton(f"{index + 1}. {title}")
            button.setObjectName("StudyStepButton")
            button.setCheckable(True)
            button.setToolTip(description)
            button.setAccessibleName(f"Step {index + 1}: {title}")
            button.setAccessibleDescription(description)
            button.clicked.connect(lambda _checked=False, value=index: self.set_step(value))
            self.step_group.addButton(button)
            self.step_buttons.append(button)
            step_row.addWidget(button, 1)
        root.addLayout(step_row)

        self.stack = QStackedWidget()
        self.stack.setObjectName("StudySetupStack")
        self.stack.setMaximumWidth(900)
        for _title, _description, page in self.steps:
            self.stack.addWidget(page)
        root.addWidget(self.stack)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.setObjectName("StudyPreviousButton")
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("PrimaryButton")
        self.previous_button.clicked.connect(lambda: self.set_step(self.current_step() - 1))
        self.next_button.clicked.connect(lambda: self.set_step(self.current_step() + 1))
        navigation.addWidget(self.previous_button)
        navigation.addStretch(1)
        self.progress = QLabel()
        self.progress.setObjectName("StudyStepProgress")
        navigation.addWidget(self.progress)
        navigation.addWidget(self.next_button)
        root.addLayout(navigation)
        self.set_step(0)

    def current_step(self) -> int:
        return self.stack.currentIndex()

    def set_step(self, index: int) -> None:
        index = max(0, min(int(index), len(self.steps) - 1))
        self.stack.setCurrentIndex(index)
        self.step_buttons[index].setChecked(True)
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.steps) - 1)
        self.next_button.setText("Review + launch" if index == 5 else "Next")
        self.progress.setText(f"Step {index + 1} of {len(self.steps)}")
        self.progress.setAccessibleName(self.progress.text())
        self.step_changed.emit(index)


def linked_step_page(
    title: str,
    summary: str,
    action_text: str,
    action,
) -> QWidget:
    """Build a concise setup step that routes to an authoritative specialist workspace."""
    page = QFrame()
    page.setObjectName("StudyLinkedStep")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("CardTitle")
    detail = QLabel(summary)
    detail.setObjectName("CardSubtitle")
    detail.setWordWrap(True)
    button = QPushButton(action_text)
    button.setObjectName("PrimaryButton")
    button.clicked.connect(action)
    layout.addWidget(heading)
    layout.addWidget(detail)
    layout.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
    layout.addStretch(1)
    return page
