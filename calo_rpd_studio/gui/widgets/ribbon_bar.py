"""Registry-generated, keyboard-accessible Phase 6 ribbon."""

from __future__ import annotations

from collections import OrderedDict

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.command_registry import CommandRegistry, RIBBON_CATEGORY_ORDER
from calo_rpd_studio.version import PRODUCT_VERSION


class RibbonCategoryTabs(QFrame):
    """Explicit category buttons and page stack without native tab painting."""

    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonTabs")
        self.setAccessibleName("Ribbon categories")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._category_row = QFrame(self)
        self._category_row.setObjectName("RibbonCategoryRow")
        self._category_row.setAccessibleName("Ribbon category selector")
        self._category_row.setMinimumHeight(31)
        self._category_row.setMaximumHeight(31)
        category_layout = QHBoxLayout(self._category_row)
        category_layout.setContentsMargins(0, 0, 0, 0)
        category_layout.setSpacing(0)
        category_layout.addStretch(1)
        self._category_layout = category_layout
        self._category_group = QButtonGroup(self)
        self._category_group.setExclusive(True)
        self._category_buttons: list[QPushButton] = []
        self._titles: list[str] = []
        self._current_index = -1

        self._page_stack = QStackedWidget(self)
        self._page_stack.setObjectName("RibbonPageStack")
        self._page_stack.setAccessibleName("Current ribbon commands")
        self._page_stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout.addWidget(self._category_row)
        layout.addWidget(self._page_stack, 1)

    def addTab(self, page: QWidget, title: str) -> int:
        page_index = self._page_stack.addWidget(page)
        button = QPushButton(str(title), self._category_row)
        button.setObjectName("RibbonCategoryButton")
        button.setCheckable(True)
        button.setAutoDefault(False)
        button.setAccessibleName(f"{title} ribbon category")
        self._category_layout.insertWidget(len(self._category_buttons), button)
        self._category_group.addButton(button, page_index)
        button.clicked.connect(lambda _checked=False, index=page_index: self._select_page(index))
        self._category_buttons.append(button)
        self._titles.append(str(title))
        if page_index == 0:
            self.setFocusProxy(button)
            self._select_page(0)
        return page_index

    def _select_page(self, index: int) -> None:
        if 0 <= index < self._page_stack.count():
            self._page_stack.setCurrentIndex(index)
            self._category_buttons[index].setChecked(True)
            self._current_index = index
            self.currentChanged.emit(index)

    def count(self) -> int:
        return len(self._category_buttons)

    def currentIndex(self) -> int:
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:
        self._select_page(index)

    def tabText(self, index: int) -> str:
        return self._titles[index]

    def widget(self, index: int) -> QWidget | None:
        return self._page_stack.widget(index)

    def tabBar(self) -> QWidget:
        return self._category_row


class RibbonBar(QFrame):
    def __init__(self, registry: CommandRegistry, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.setObjectName("RibbonBar")
        self.setAccessibleName("Application ribbon")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)

        self.identity_bar = QFrame()
        self.identity_bar.setObjectName("RibbonIdentityBar")
        self.identity_bar.setAccessibleName("CALO-RPD product heading")
        self.identity_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.identity_bar.setMinimumHeight(42)
        self.identity_bar.setMaximumHeight(42)
        identity = QHBoxLayout(self.identity_bar)
        identity.setContentsMargins(12, 6, 12, 6)
        identity.setSpacing(7)
        self.product_label = QLabel("CALO-RPD Studio")
        self.product_label.setObjectName("RibbonProduct")
        self.product_label.setAccessibleName("CALO-RPD Studio")
        self.version_label = QLabel(f"v{PRODUCT_VERSION}")
        self.version_label.setObjectName("RibbonVersion")
        self.version_label.setAccessibleName(f"Version {PRODUCT_VERSION}")
        identity.addWidget(self.product_label)
        identity.addWidget(self.version_label)
        identity.addStretch(1)
        self.state_label = QLabel("Ready")
        self.state_label.setObjectName("RibbonState")
        self.state_label.setAccessibleName("Application state")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        identity.addWidget(self.state_label)
        outer.addWidget(self.identity_bar)

        # Keep every category widget inside a separate clipping boundary. Some
        # Windows Qt styles paint tab/page primitives a few pixels above their
        # nominal rectangle; without this parent those fragments can intrude
        # into the product heading.
        self.navigation_area = QFrame()
        self.navigation_area.setObjectName("RibbonNavigationArea")
        self.navigation_area.setAccessibleName("Ribbon navigation area")
        self.navigation_area.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.navigation_area.setMinimumHeight(118)
        self.navigation_area.setMaximumHeight(126)
        navigation_layout = QVBoxLayout(self.navigation_area)
        navigation_layout.setContentsMargins(10, 0, 10, 0)
        navigation_layout.setSpacing(0)

        self.tabs = RibbonCategoryTabs()
        self.tabs.setMinimumHeight(118)
        self.tabs.setMaximumHeight(126)
        navigation_layout.addWidget(self.tabs)
        outer.addWidget(self.navigation_area)

        categories: OrderedDict[str, OrderedDict[str, list]] = OrderedDict()
        for spec in registry.specs:
            categories.setdefault(spec.category, OrderedDict()).setdefault(spec.group, []).append(
                spec
            )
        ordered_categories = [
            (category, categories.pop(category))
            for category in RIBBON_CATEGORY_ORDER
            if category in categories
        ]
        ordered_categories.extend(categories.items())
        for category, groups in ordered_categories:
            page = QWidget()
            page.setObjectName("RibbonPage")
            page.setProperty("ribbonCategory", category)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            row = QHBoxLayout(page)
            row.setContentsMargins(5, 4, 5, 4)
            row.setSpacing(5)
            for group_name, specs in groups.items():
                # QGroupBox's native title subcontrol is designed for a top
                # legend. Moving it to the bottom lets some platform styles
                # paint the caption outside the frame, where it is clipped by
                # the ribbon page. Keep the frame but render the caption as a
                # normal footer widget contained by its layout.
                group = QGroupBox()
                group.setObjectName("RibbonGroup")
                group.setAccessibleName(f"{group_name} ribbon group")
                group_column = QVBoxLayout(group)
                group_column.setContentsMargins(5, 3, 5, 2)
                group_column.setSpacing(0)

                command_area = QWidget()
                command_area.setObjectName("RibbonGroupCommands")
                group_row = QGridLayout(command_area)
                group_row.setContentsMargins(0, 0, 0, 0)
                group_row.setHorizontalSpacing(3)
                group_row.setVerticalSpacing(1)
                for item_index, spec in enumerate(specs):
                    button = QToolButton()
                    button.setProperty("ribbonCommandId", spec.command_id)
                    button.setProperty("ribbonCategory", category)
                    button.setObjectName(
                        "WorkspaceRibbonButton"
                        if category == "Workspace"
                        else ("RibbonPrimaryButton" if spec.primary else "RibbonButton")
                    )
                    button.setDefaultAction(registry.action(spec.command_id))
                    button.setAccessibleName(spec.label)
                    button.setAccessibleDescription(spec.tooltip)
                    workspace_page = category == "Workspace"
                    button.setToolButtonStyle(
                        Qt.ToolButtonStyle.ToolButtonTextBesideIcon
                        if workspace_page
                        else Qt.ToolButtonStyle.ToolButtonTextUnderIcon
                    )
                    button.setIconSize(
                        QSize(24 if spec.primary else 20, 24 if spec.primary else 20)
                    )
                    button.setAutoRaise(False)
                    if workspace_page:
                        group_row.addWidget(button, item_index % 2, item_index // 2)
                    else:
                        group_row.addWidget(button, 0, item_index)
                group_column.addWidget(command_area, 1)
                caption = QLabel(group_name)
                caption.setObjectName("RibbonGroupCaption")
                caption.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                caption.setAccessibleName(f"{group_name} group")
                group_column.addWidget(caption, 0)
                row.addWidget(group)
            row.addStretch(1)
            self.tabs.addTab(page, category)
        self.tabs.currentChanged.connect(self._guard_inactive_category_interaction)
        self._guard_inactive_category_interaction(self.tabs.currentIndex())
        # The heading is intentionally the uppermost sibling. Its opaque
        # surface seals both painting and pointer input above navigation.
        self.identity_bar.raise_()

    def _guard_inactive_category_interaction(self, current_index: int) -> None:
        """Let Qt own page visibility and make every inactive page noninteractive."""
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            inactive = index != current_index
            page.setEnabled(not inactive)
            page.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, inactive)
            for button in page.findChildren(QToolButton):
                button.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, inactive)
                button.setFocusPolicy(
                    Qt.FocusPolicy.NoFocus if inactive else Qt.FocusPolicy.StrongFocus
                )

    @property
    def compact(self) -> bool:
        return False

    def set_compact(self, compact: bool, *, emit: bool = True) -> None:
        """Compatibility adapter: the application ribbon is permanently expanded."""
        del compact, emit
        self.show()
        self.tabs.show()
        self.tabs.setMinimumHeight(118)
        self.tabs.setMaximumHeight(126)
        self.setProperty("compact", False)

    def toggle_compact(self) -> None:
        self.set_compact(False)

    def set_summary(self, text: str) -> None:
        self.state_label.setText(str(text))
        self.state_label.setToolTip(str(text))

    def select_category(self, title: str) -> None:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == str(title):
                self.tabs.setCurrentIndex(index)
                # The currentChanged signal does not fire for a same-category
                # request, so refresh the interaction boundary explicitly.
                self._guard_inactive_category_interaction(index)
                return
