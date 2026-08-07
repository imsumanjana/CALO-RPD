"""Application-wide compact-input and accessibility policy for scientist workspaces."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QTextEdit,
    QWidget,
)


FORM_CONTENT_MAX_WIDTH = 880
ORDINARY_INPUT_MAX_WIDTH = 480
SELECTOR_MAX_WIDTH = 420
SCALAR_INPUT_MAX_WIDTH = 240


def _accessible_fallback(widget: QWidget) -> str:
    placeholder = getattr(widget, "placeholderText", lambda: "")()
    if placeholder:
        return str(placeholder)
    name = widget.objectName().replace("_", " ").strip()
    return name or widget.metaObject().className()


def _associate_form_labels(root: QWidget) -> None:
    for form in root.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            label = label_item.widget() if label_item is not None else None
            field = field_item.widget() if field_item is not None else None
            if isinstance(label, QLabel) and field is not None:
                label.setBuddy(field)
                if not field.accessibleName():
                    field.setAccessibleName(label.text().replace("&", ""))


def _attach_expanded_editor(widget: QTextEdit | QPlainTextEdit) -> None:
    if widget.property("hasExpandDialog"):
        return
    expand = QToolButton(widget)
    expand.setText("Expand")
    expand.setToolTip("Open this text in a larger focused editor")
    expand.setAccessibleName("Expand text editor")

    def open_dialog() -> None:
        dialog = QDialog(widget)
        dialog.setWindowTitle(widget.accessibleName() or "Expanded text editor")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        editor = QPlainTextEdit()
        editor.setPlainText(widget.toPlainText())
        editor.setAccessibleName(dialog.windowTitle())
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(editor, 1)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            widget.setPlainText(editor.toPlainText())

    expand.clicked.connect(open_dialog)
    widget.setCornerWidget(expand)
    widget.setProperty("hasExpandDialog", True)


def apply_compact_input_policy(root: QWidget, density: str = "comfortable") -> str:
    """Constrain ordinary controls while leaving tables, plots, and log viewers resizable."""
    normalized = "compact" if str(density).strip().lower() == "compact" else "comfortable"
    control_height = 40 if normalized == "compact" else 44

    for widget in root.findChildren(QLineEdit):
        if not bool(widget.property("fullWidthInput")):
            widget.setMaximumWidth(ORDINARY_INPUT_MAX_WIDTH)
        widget.setMinimumHeight(control_height)
        if not widget.accessibleName():
            widget.setAccessibleName(_accessible_fallback(widget))

    for widget in root.findChildren(QComboBox):
        if not bool(widget.property("fullWidthInput")):
            widget.setMaximumWidth(SELECTOR_MAX_WIDTH)
        widget.setMinimumHeight(control_height)
        if not widget.accessibleName():
            widget.setAccessibleName(_accessible_fallback(widget))

    for widget in root.findChildren(QAbstractSpinBox):
        widget.setMaximumWidth(SCALAR_INPUT_MAX_WIDTH)
        widget.setMinimumHeight(control_height)
        if not widget.accessibleName():
            widget.setAccessibleName(_accessible_fallback(widget))

    for widget in (*root.findChildren(QTextEdit), *root.findChildren(QPlainTextEdit)):
        if not widget.isReadOnly() and not bool(widget.property("expandedLongText")):
            line_height = max(18, widget.fontMetrics().lineSpacing())
            widget.setMaximumHeight(line_height * 6 + 24)
            widget.setMinimumHeight(line_height * 3 + 20)
            widget.setProperty("compactLongText", True)
            _attach_expanded_editor(widget)
        if not widget.accessibleName():
            widget.setAccessibleName(_accessible_fallback(widget))

    for widget in root.findChildren(QPushButton):
        widget.setMinimumHeight(control_height)
        if not widget.accessibleName():
            widget.setAccessibleName(widget.text().replace("&", ""))

    for widget in root.findChildren(QAbstractButton):
        if not widget.accessibleName():
            widget.setAccessibleName(widget.text().replace("&", ""))

    _associate_form_labels(root)
    root.setProperty("interfaceDensity", normalized)
    root.style().unpolish(root)
    root.style().polish(root)
    return normalized
