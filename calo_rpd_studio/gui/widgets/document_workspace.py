"""Central document host that preserves the existing scientific workspace stack."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage


class _PreviewSurface(QWidget):
    """Roomy scroll host for the pinned scientific workspace and preview blocks."""

    def __init__(self, content: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewSurface")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("MainPreviewScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content.setMinimumSize(920, 650)
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.content = content
        self.scroll = scroll


class DocumentWorkspace(QTabWidget):
    def __init__(self, workspace: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("DocumentWorkspace")
        self.setAccessibleName("Results and preview workspace")
        self.setDocumentMode(True)
        self.tabBar().setDrawBase(False)
        self.setTabsClosable(False)
        self.setMovable(True)
        self.setUsesScrollButtons(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabBar().setExpanding(False)
        self._scientific_workspace = workspace
        for page in workspace.findChildren(ScrollablePage):
            page.use_external_vertical_scroll()
        self._preview_surface = _PreviewSurface(workspace, self)
        self._documents: dict[str, QWidget] = {"scientific-workspace": self._preview_surface}
        index = self.addTab(self._preview_surface, "Scientific workspace")
        self.tabBar().setTabButton(index, self.tabBar().ButtonPosition.RightSide, None)
        self.setTabToolTip(index, "Pinned scientific configuration and result workspace")
        self.tabCloseRequested.connect(self._close_requested)
        self._sync_document_header()

    def _sync_document_header(self) -> None:
        """Show document chrome only when it provides real document switching."""
        self.tabBar().setVisible(self.count() > 1)

    def open_document(
        self,
        document_id: str,
        title: str,
        widget: QWidget,
        *,
        tooltip: str = "",
    ) -> QWidget:
        key = str(document_id)
        existing = self._documents.get(key)
        if existing is not None:
            self.setCurrentWidget(existing)
            return existing
        self._documents[key] = widget
        index = self.addTab(widget, str(title))
        close_button = QToolButton(self)
        close_button.setObjectName("DocumentCloseButton")
        close_button.setText("×")
        close_button.setAutoRaise(True)
        close_button.setFixedSize(20, 20)
        close_button.setAccessibleName(f"Close {title}")
        close_button.clicked.connect(lambda _checked=False, item=widget: self._close_widget(item))
        self.tabBar().setTabButton(
            index,
            self.tabBar().ButtonPosition.RightSide,
            close_button,
        )
        if tooltip:
            self.setTabToolTip(index, tooltip)
        self.setCurrentIndex(index)
        self._sync_document_header()
        return widget

    def focus_scientific_workspace(self) -> None:
        self.setCurrentWidget(self._documents["scientific-workspace"])

    @property
    def preview_scroll(self) -> QScrollArea:
        return self._preview_surface.scroll

    @property
    def scientific_workspace(self) -> QWidget:
        return self._scientific_workspace

    def document_ids(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def _close_requested(self, index: int) -> None:
        widget = self.widget(index)
        self._close_widget(widget)

    def _close_widget(self, widget: QWidget) -> None:
        if widget is self._documents["scientific-workspace"]:
            return
        key = next((name for name, item in self._documents.items() if item is widget), "")
        if key:
            self._documents.pop(key, None)
        index = self.indexOf(widget)
        if index < 0:
            return
        self.removeTab(index)
        widget.hide()
        self._sync_document_header()
