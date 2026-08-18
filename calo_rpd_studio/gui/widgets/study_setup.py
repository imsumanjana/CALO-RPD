"""Inline, single-workspace Study Setup presentation."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class StudySetupWorkflow(QFrame):
    """A compact setup shell whose pages share authoritative application state."""

    step_changed = pyqtSignal(int)

    def __init__(
        self,
        steps: Sequence[tuple[str, str, QWidget]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(steps) < 2:
            raise ValueError("Study Setup requires at least two steps")
        self.setObjectName("StudySetupWorkflow")
        self.steps = tuple(steps)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        self.heading = QLabel("Study Setup")
        self.heading.setObjectName("StudySetupTitle")
        self.heading.setAccessibleName(f"Study Setup, {len(self.steps)} steps")
        root.addWidget(self.heading)

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
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.page_widgets: dict[str, QWidget] = {}
        self.prerequisite_labels: dict[str, QLabel] = {}
        for title, _description, page in self.steps:
            container = QFrame()
            container.setObjectName("StudyStepPage")
            page_layout = QVBoxLayout(container)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(8)
            prerequisite = QLabel()
            prerequisite.setObjectName("BlockedWorkspaceSummary")
            prerequisite.setWordWrap(True)
            prerequisite.setVisible(False)
            page_layout.addWidget(prerequisite)
            page_layout.addWidget(page, 1)
            self.page_widgets[title] = page
            self.prerequisite_labels[title] = prerequisite
            self.stack.addWidget(container)
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

    def set_presentation(self, title: str) -> None:
        """Use mode-correct language without changing the shared setup pages."""

        text = str(title).strip() or "Study Setup"
        self.heading.setText(text)
        self.heading.setAccessibleName(f"{text}, {len(self.steps)} steps")

    def current_step(self) -> int:
        return self.stack.currentIndex()

    def set_step(self, index: int) -> None:
        index = max(0, min(int(index), len(self.steps) - 1))
        self.stack.setCurrentIndex(index)
        self.step_buttons[index].setChecked(True)
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.steps) - 1)
        self.next_button.setText("Review + launch" if index == len(self.steps) - 2 else "Next")
        self.progress.setText(f"Step {index + 1} of {len(self.steps)}")
        self.progress.setAccessibleName(self.progress.text())
        self.step_changed.emit(index)

    def set_step_available(self, title: str, available: bool, reason: str = "") -> None:
        """Keep a step inspectable while preventing edits behind a prerequisite/freeze gate."""

        key = str(title)
        if key not in self.page_widgets:
            raise KeyError(f"Unknown Study Setup step: {key}")
        page = self.page_widgets[key]
        prerequisite = self.prerequisite_labels[key]
        page.setEnabled(bool(available))
        prerequisite.setText("" if available else str(reason or "This step is not available."))
        prerequisite.setVisible(not available)
        index = next(i for i, step in enumerate(self.steps) if step[0] == key)
        description = str(self.steps[index][1])
        self.step_buttons[index].setToolTip(
            description if available else f"{description}\n\n{prerequisite.text()}"
        )
        self.step_buttons[index].setAccessibleDescription(self.step_buttons[index].toolTip())


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
