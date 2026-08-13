"""Grouped, searchable and persistent scientist-workspace navigation."""

from __future__ import annotations

from collections.abc import Sequence

from calo_rpd_studio.app.workspaces import WorkspaceSpec, grouped_workspace_specs
from calo_rpd_studio.gui.icons.workspace_icons import workspace_icon
from calo_rpd_studio.version import PRODUCT_VERSION

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def _setting_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


class NavigationSidebar(QFrame):
    """Five-group navigation rail that keeps stable workspace indexes authoritative."""

    page_requested = pyqtSignal(int)

    def __init__(
        self,
        items: Sequence[WorkspaceSpec] | Sequence[tuple[str, str]],
        settings=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.items = list(items)
        self._specs = self._normalize_specs(items)
        self._workflow_states = ["available"] * len(self._specs)
        self._workflow_reasons = [""] * len(self._specs)
        self._search_text = ""
        self._compact = _setting_bool(self._value("navigation/compact", False), False)
        self.setObjectName("Sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(8)
        self.brand_mark = QLabel("C")
        self.brand_mark.setObjectName("BrandMark")
        self.brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.brand_mark.setFixedSize(38, 38)
        brand_row.addWidget(self.brand_mark)

        self.brand_text = QWidget()
        brand_text_layout = QVBoxLayout(self.brand_text)
        brand_text_layout.setContentsMargins(0, 0, 0, 0)
        brand_text_layout.setSpacing(0)
        title = QLabel("CALO-RPD")
        title.setObjectName("BrandTitle")
        subtitle = QLabel("Studio")
        subtitle.setObjectName("BrandSubtitle")
        brand_text_layout.addWidget(title)
        brand_text_layout.addWidget(subtitle)
        brand_row.addWidget(self.brand_text, 1)

        self.compact_button = QToolButton()
        self.compact_button.setObjectName("NavigationCompactButton")
        self.compact_button.setCheckable(True)
        self.compact_button.setChecked(self._compact)
        self.compact_button.setText("Collapse")
        self.compact_button.setToolTip("Use compact navigation")
        self.compact_button.setAccessibleName("Toggle compact navigation")
        self.compact_button.clicked.connect(self.set_compact)
        brand_row.addWidget(self.compact_button)
        layout.addLayout(brand_row)

        self.search = QLineEdit()
        self.search.setObjectName("WorkspaceSearch")
        self.search.setPlaceholderText("Find a workspace")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search workspaces")
        self.search.setToolTip("Search available scientific workspaces. Shortcut: Ctrl+K")
        self.search.textChanged.connect(self._filter_navigation)
        layout.addWidget(self.search)

        nav_scroll = QScrollArea()
        nav_scroll.setObjectName("NavigationScroll")
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setFrameShape(QFrame.Shape.NoFrame)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QPushButton] = [QPushButton() for _ in self._specs]
        self.group_headers: dict[str, QToolButton] = {}
        self.group_containers: dict[str, QWidget] = {}
        self.group_indexes: dict[str, tuple[int, ...]] = {}
        self.group_expanded: dict[str, bool] = {}

        specs_by_key = {spec.key: (index, spec) for index, spec in enumerate(self._specs)}
        for group_name, canonical_members in grouped_workspace_specs():
            members = tuple(
                specs_by_key[spec.key]
                for _canonical_index, spec in canonical_members
                if spec.key in specs_by_key
            )
            if not members:
                continue
            expanded = _setting_bool(self._value(f"navigation/group/{group_name}", True), True)
            self.group_expanded[group_name] = expanded
            self.group_indexes[group_name] = tuple(index for index, _spec in members)

            header = QToolButton()
            header.setObjectName("NavigationGroupHeader")
            header.setCheckable(True)
            header.setChecked(expanded)
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
            header.setText(group_name)
            header.setAccessibleName(f"{group_name} workspace group")
            header.clicked.connect(
                lambda checked, name=group_name: self._set_group_expanded(name, checked)
            )
            self.group_headers[group_name] = header
            nav_layout.addWidget(header)

            container = QWidget()
            container.setObjectName("NavigationGroup")
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)
            for index, spec in members:
                button = QPushButton(spec.title.replace("&", "&&"))
                button.setIcon(workspace_icon(spec.icon))
                button.setIconSize(button.iconSize().expandedTo(button.iconSize()))
                button.setCheckable(True)
                button.setObjectName("NavButton")
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.setMinimumHeight(40)
                button.setProperty("workflowState", "available")
                button.setProperty("workspaceKey", spec.key)
                button.setAccessibleName(spec.title)
                button.clicked.connect(
                    lambda checked=False, page_index=index: self.page_requested.emit(page_index)
                )
                self.group.addButton(button)
                self.buttons[index] = button
                container_layout.addWidget(button)
            self.group_containers[group_name] = container
            nav_layout.addWidget(container)
        nav_layout.addStretch(1)
        nav_scroll.setWidget(nav_container)
        layout.addWidget(nav_scroll, 1)

        self.blocked_summary = QLabel()
        self.blocked_summary.setObjectName("BlockedWorkspaceSummary")
        self.blocked_summary.setWordWrap(True)
        self.blocked_summary.setAccessibleName("Hidden blocked workspaces")
        layout.addWidget(self.blocked_summary)

        self.footer = QFrame()
        self.footer.setObjectName("SidebarFooter")
        footer_layout = QVBoxLayout(self.footer)
        footer_layout.setContentsMargins(10, 9, 10, 9)
        footer_layout.setSpacing(2)
        edition = QLabel("Scientific workspace")
        edition.setObjectName("SidebarFooterTitle")
        version = QLabel(f"CALO-RPD Studio {PRODUCT_VERSION}")
        version.setObjectName("SidebarFooterText")
        footer_layout.addWidget(edition)
        footer_layout.addWidget(version)
        layout.addWidget(self.footer)

        if self.buttons:
            self.buttons[0].setChecked(True)
        self.set_compact(self._compact)
        self._refresh_visibility()

    @staticmethod
    def _normalize_specs(
        items: Sequence[WorkspaceSpec] | Sequence[tuple[str, str]],
    ) -> list[WorkspaceSpec]:
        specs: list[WorkspaceSpec] = []
        for index, item in enumerate(items):
            if isinstance(item, WorkspaceSpec):
                specs.append(item)
            else:
                title, description = item
                specs.append(
                    WorkspaceSpec(
                        key=f"workspace_{index}",
                        title=str(title),
                        description=str(description),
                    )
                )
        return specs

    def _value(self, key: str, default: object) -> object:
        return self.settings.value(key, default) if self.settings is not None else default

    def _persist(self, key: str, value: object) -> None:
        if self.settings is not None:
            self.settings.set_value(key, value)

    def focus_search(self) -> None:
        if self._compact:
            self.set_compact(False)
        self.search.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.search.selectAll()

    def set_compact(self, compact: bool) -> None:
        self._compact = bool(compact)
        if self._compact and self._search_text:
            self.search.clear()
        self.compact_button.blockSignals(True)
        self.compact_button.setChecked(self._compact)
        self.compact_button.blockSignals(False)
        self.compact_button.setText("Expand" if self._compact else "Collapse")
        self.brand_text.setVisible(not self._compact)
        self.search.setVisible(not self._compact)
        self.footer.setVisible(not self._compact)
        self.blocked_summary.setVisible(not self._compact and bool(self.blocked_summary.text()))
        for header in self.group_headers.values():
            header.setVisible(not self._compact)
        for index, button in enumerate(self.buttons):
            button.setText(self._button_text(index))
            button.setToolTip(self._button_tooltip(index))
        width = 76 if self._compact else 272
        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self._persist("navigation/compact", self._compact)
        self._refresh_visibility()

    def _set_group_expanded(self, group_name: str, expanded: bool) -> None:
        self.group_expanded[group_name] = bool(expanded)
        header = self.group_headers[group_name]
        header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._persist(f"navigation/group/{group_name}", bool(expanded))
        self._refresh_visibility()

    def _filter_navigation(self, text: str) -> None:
        self._search_text = str(text).strip().casefold()
        self._refresh_visibility()

    def _button_tooltip(self, index: int) -> str:
        spec = self._specs[index]
        reason = self._workflow_reasons[index]
        parts = [spec.title]
        if spec.description:
            parts.append(spec.description)
        if reason:
            parts.append(reason)
        return "\n".join(parts)

    def _button_text(self, index: int) -> str:
        if self._compact:
            return ""
        badge = {
            "completed": "Done",
            "recommended": "Next",
            "optional": "Optional",
        }.get(self._workflow_states[index], "")
        return self._specs[index].title + (f"  ·  {badge}" if badge else "")

    def _matches_search(self, index: int) -> bool:
        if not self._search_text:
            return True
        spec = self._specs[index]
        haystack = " ".join(
            (spec.title, spec.description, spec.group, spec.key, *spec.keywords)
        ).casefold()
        return self._search_text in haystack

    def _refresh_visibility(self) -> None:
        for group_name, indexes in self.group_indexes.items():
            visible_count = 0
            for index in indexes:
                available = self._workflow_states[index] != "locked"
                matches = self._matches_search(index)
                visible = available and matches
                self.buttons[index].setVisible(visible)
                visible_count += int(visible)
            expanded = (
                self._compact
                or self.group_expanded.get(group_name, True)
                or bool(self._search_text)
            )
            self.group_containers[group_name].setVisible(expanded and visible_count > 0)
            self.group_headers[group_name].setVisible(not self._compact and visible_count > 0)
        self._refresh_blocked_summary()

    def _refresh_blocked_summary(self) -> None:
        blocked = [
            (self._specs[index].title, reason)
            for index, reason in enumerate(self._workflow_reasons)
            if self._workflow_states[index] == "locked"
        ]
        if not blocked:
            self.blocked_summary.clear()
            self.blocked_summary.hide()
            return
        self.blocked_summary.setText(f"{len(blocked)} later workspace(s) hidden")
        self.blocked_summary.setToolTip(
            "\n".join(
                f"{title}: {reason or 'Complete the preceding scientific step.'}"
                for title, reason in blocked
            )
        )
        self.blocked_summary.setAccessibleDescription(self.blocked_summary.toolTip())
        self.blocked_summary.setVisible(not self._compact)

    def set_workflow_state(self, index: int, state: str, reason: str = "") -> None:
        if not 0 <= index < len(self.buttons):
            return
        button = self.buttons[index]
        self._workflow_states[index] = str(state)
        self._workflow_reasons[index] = str(reason)
        button.setEnabled(state != "locked")
        button.setText(self._button_text(index))
        button.setToolTip(self._button_tooltip(index))
        button.setAccessibleDescription(reason or f"Workspace status: {state}")
        button.setProperty("workflowState", state)
        button.style().unpolish(button)
        button.style().polish(button)
        self._refresh_visibility()

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)
