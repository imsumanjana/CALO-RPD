"""Width-safe long workspace base with explicit scroll ownership."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QAbstractScrollArea, QScrollArea, QSizePolicy, QWidget


class ScrollablePage(QScrollArea):
    """Long-form page whose vertical navigation belongs to the preview canvas.

    The class remains a QScrollArea for compatibility with existing panels. When
    embedded in the application shell it advertises its full content height and
    delegates vertical navigation to the main preview; standalone panels retain
    their own as-needed scrollbar. Tables, editors, plots, and other purpose-specific
    child scrollers are unaffected.
    """

    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ScrollableWorkspace")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setProperty("verticalScrollOwner", "self")
        self.viewport().setObjectName("ScrollableViewport")
        content.setObjectName("ScrollableContent")
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setWidget(content)

    def use_external_vertical_scroll(self, enabled: bool = True) -> None:
        """Delegate page scrolling to an enclosing main preview when embedded."""
        if enabled:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.setProperty("verticalScrollOwner", "main-preview")
        else:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setProperty("verticalScrollOwner", "self")
        self.updateGeometry()

    def viewportSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        content = self.widget()
        if self.property("verticalScrollOwner") == "main-preview" and content is not None:
            return content.sizeHint()
        return super().viewportSizeHint()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        if self.property("verticalScrollOwner") != "main-preview":
            return super().sizeHint()
        hint = self.viewportSizeHint()
        return QSize(max(hint.width(), 1), max(hint.height(), 1))

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        # With no local page scroll range, allow the containing MainPreviewScroll
        # to handle wheel navigation instead of swallowing the gesture here.
        if self.property("verticalScrollOwner") == "main-preview":
            event.ignore()
            return
        super().wheelEvent(event)
