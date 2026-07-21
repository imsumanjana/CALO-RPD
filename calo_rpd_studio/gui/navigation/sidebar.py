"""Modern workspace navigation sidebar with workflow-aware locking."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class NavigationSidebar(QFrame):
    page_requested = pyqtSignal(int)

    def __init__(self, items: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items = items
        self.setObjectName("Sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(238)
        self.setMaximumWidth(268)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(7)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        mark = QLabel("C")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(38, 38)
        brand_row.addWidget(mark)

        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(0)
        title = QLabel("CALO-RPD")
        title.setObjectName("BrandTitle")
        subtitle = QLabel("Studio")
        subtitle.setObjectName("BrandSubtitle")
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand_row.addLayout(brand_text, 1)
        layout.addLayout(brand_row)
        layout.addSpacing(10)

        section = QLabel("GUIDED RESEARCH WORKFLOW")
        section.setObjectName("NavSectionLabel")
        layout.addWidget(section)

        # The number of research workspaces grew in v2.0.0. Keep the brand/footer fixed and make
        # only the navigation list scrollable so no button is vertically compressed on smaller
        # screens.
        nav_scroll = QScrollArea()
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setFrameShape(QFrame.Shape.NoFrame)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 2, 0)
        nav_layout.setSpacing(7)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        for index, (title_text, _icon) in enumerate(items):
            button = QPushButton(title_text.replace("&", "&&"))
            button.setCheckable(True)
            button.setObjectName("NavButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(37)
            button.setProperty("workflowState", "available")
            button.clicked.connect(
                lambda checked=False, page_index=index: self.page_requested.emit(page_index)
            )
            self.group.addButton(button)
            self.buttons.append(button)
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)
        nav_scroll.setWidget(nav_container)
        layout.addWidget(nav_scroll, 1)

        footer = QFrame()
        footer.setObjectName("SidebarFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(10, 9, 10, 9)
        footer_layout.setSpacing(2)
        edition = QLabel("Scientific workspace")
        edition.setObjectName("SidebarFooterTitle")
        version = QLabel("CALO-RPD Studio 5.0.0")
        version.setObjectName("SidebarFooterText")
        footer_layout.addWidget(edition)
        footer_layout.addWidget(version)
        layout.addWidget(footer)

        if self.buttons:
            self.buttons[0].setChecked(True)

    def set_workflow_state(self, index: int, state: str, reason: str = "") -> None:
        if not 0 <= index < len(self.buttons):
            return
        button = self.buttons[index]
        title = self.items[index][0].replace("&", "&&")
        prefix = {
            "completed": "✓  ",
            "recommended": "→  ",
            "optional": "◇  ",
            "locked": "·  ",
            "available": "   ",
        }.get(state, "   ")
        button.setText(prefix + title)
        button.setEnabled(state != "locked")
        button.setToolTip(reason)
        button.setProperty("workflowState", state)
        button.style().unpolish(button)
        button.style().polish(button)

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)
