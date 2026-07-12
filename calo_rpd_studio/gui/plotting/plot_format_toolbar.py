"""Compact, popup-based plot editing tools for scientific figures."""
from __future__ import annotations

from copy import deepcopy
import weakref

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.visualization.plot_style import PlotStyle


TARGETS = (
    "All plot text",
    "Title",
    "X-axis label",
    "Y-axis label",
    "X-axis ticks",
    "Y-axis ticks",
    "Legend",
    "Annotations",
)
FONT_FIELDS = {
    "Title": "title_font",
    "X-axis label": "x_label_font",
    "Y-axis label": "y_label_font",
    "X-axis ticks": "x_tick_font",
    "Y-axis ticks": "y_tick_font",
    "Legend": "legend_font",
    "Annotations": "annotation_font",
}
SIZE_FIELDS = {
    "Title": "title_size",
    "X-axis label": "x_label_size",
    "Y-axis label": "y_label_size",
    "X-axis ticks": "x_tick_size",
    "Y-axis ticks": "y_tick_size",
    "Legend": "legend_size",
    "Annotations": "annotation_size",
}


class _PlotToolPopup(QFrame):
    """Transient popup panel positioned beneath a plot-tool button."""

    closed = pyqtSignal()

    def __init__(self, title: str, description: str, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("PlotToolPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(390)
        self.setMaximumWidth(620)

        self.layout_root = QVBoxLayout(self)
        self.layout_root.setContentsMargins(16, 14, 16, 16)
        self.layout_root.setSpacing(12)

        heading = QLabel(title)
        heading.setObjectName("PlotToolPopupTitle")
        self.layout_root.addWidget(heading)

        subtitle = QLabel(description)
        subtitle.setObjectName("PlotToolPopupDescription")
        subtitle.setWordWrap(True)
        self.layout_root.addWidget(subtitle)

        divider = QFrame()
        divider.setObjectName("PlotToolPopupDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        self.layout_root.addWidget(divider)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.closed.emit()
        super().hideEvent(event)


class _ToolSection(QWidget):
    """Small titled section used inside plot popups."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        label = QLabel(title)
        label.setObjectName("PlotToolSectionTitle")
        layout.addWidget(label)
        self.content = QVBoxLayout()
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(7)
        layout.addLayout(self.content)


def _make_icon(kind: str, size: int = 22) -> QIcon:
    """Create sharp dependency-free vector-like icons with Qt painting."""

    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(scale, scale)
    color = QColor("#64748b")
    pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if kind == "text":
        font = QFont("Segoe UI", 15)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(0, 0, size, size, Qt.AlignmentFlag.AlignCenter, "T")
    elif kind == "plot":
        painter.drawLine(3, 18, 3, 4)
        painter.drawLine(3, 18, 19, 18)
        painter.drawPolyline(QPolygon([QPoint(5, 15), QPoint(9, 11), QPoint(13, 13), QPoint(18, 6)]))
        for point in (QPoint(5, 15), QPoint(9, 11), QPoint(13, 13), QPoint(18, 6)):
            painter.drawEllipse(point, 1, 1)
    elif kind == "export":
        painter.drawLine(11, 3, 11, 14)
        painter.drawLine(7, 10, 11, 14)
        painter.drawLine(15, 10, 11, 14)
        painter.drawRoundedRect(4, 16, 14, 3, 1, 1)
    elif kind == "style":
        for y, knob_x in ((6, 8), (11, 15), (16, 11)):
            painter.drawLine(4, y, 18, y)
            painter.setBrush(color)
            painter.drawEllipse(QPoint(knob_x, y), 2, 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


class PlotFormattingToolbar(QWidget):
    """Organized icon toolbar with focused popup editors.

    Only four compact tools remain visible above the plot.  Typography, plot
    appearance, export, and style-profile controls are separated into their own
    popups so the canvas is not surrounded by a dense wall of controls.
    """

    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.setObjectName("PlotToolStrip")
        self.plot_widget_ref = weakref.ref(plot_widget)
        self.style = deepcopy(plot_widget.style)
        self._loading = False
        self._syncing_square_size = False
        self.export_series_checks: dict[str, QCheckBox] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)

        caption = QLabel("Plot tools")
        caption.setObjectName("PlotToolStripLabel")
        layout.addWidget(caption)

        separator = QFrame()
        separator.setObjectName("PlotToolStripSeparator")
        separator.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(separator)

        self.text_tool_button = self._make_tool_button(
            "text", "Text & labels", "Edit fonts, font sizes, labels, tick text, and legend text."
        )
        self.plot_tool_button = self._make_tool_button(
            "plot", "Plot appearance", "Edit axes, grid, line, and marker appearance."
        )
        self.export_tool_button = self._make_tool_button(
            "export", "Export figure", "Save the current figure as PNG, SVG, or PDF."
        )
        self.style_tool_button = self._make_tool_button(
            "style", "Style profiles", "Save, load, reset, or apply a plot style."
        )
        for button in (
            self.text_tool_button,
            self.plot_tool_button,
            self.export_tool_button,
            self.style_tool_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)

        self._build_text_popup()
        self._build_plot_popup()
        self._build_export_popup()
        self._build_style_popup()
        self._connect()
        self.load_from_style()

    def _make_tool_button(self, icon_kind: str, name: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("PlotToolButton")
        button.setIcon(_make_icon(icon_kind))
        button.setIconSize(QSize(21, 21))
        button.setFixedSize(38, 36)
        button.setToolTip(f"{name}\n{tooltip}")
        button.setAccessibleName(name)
        return button

    @staticmethod
    def _form() -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _build_text_popup(self) -> None:
        self.text_popup = _PlotToolPopup(
            "Text & labels",
            "Edit displayed text and control typography independently for titles, axes, ticks, legends, and annotations.",
            self,
        )

        typography = _ToolSection("Typography")
        form = self._form()
        self.target = QComboBox()
        self.target.addItems(TARGETS)
        self.font = QFontComboBox()
        self.size = QSpinBox()
        self.size.setRange(6, 72)
        weight_row = QWidget()
        weight_layout = QHBoxLayout(weight_row)
        weight_layout.setContentsMargins(0, 0, 0, 0)
        self.bold = QCheckBox("Bold")
        self.italic = QCheckBox("Italic")
        weight_layout.addWidget(self.bold)
        weight_layout.addWidget(self.italic)
        weight_layout.addStretch(1)
        self.apply_font_all = QPushButton("Apply selected font to all plot text")
        form.addRow("Text target", self.target)
        form.addRow("Font", self.font)
        form.addRow("Font size", self.size)
        form.addRow("Emphasis", weight_row)
        typography.content.addLayout(form)
        typography.content.addWidget(self.apply_font_all)
        self.text_popup.layout_root.addWidget(typography)

        labels = _ToolSection("Displayed labels")
        label_form = self._form()
        self.title_text = QLineEdit()
        self.x_text = QLineEdit()
        self.y_text = QLineEdit()
        label_form.addRow("Plot title", self.title_text)
        label_form.addRow("X-axis label", self.x_text)
        label_form.addRow("Y-axis label", self.y_text)
        labels.content.addLayout(label_form)
        self.text_popup.layout_root.addWidget(labels)

        legend = _ToolSection("Legend")
        self.legend_show = QCheckBox("Show legend")
        self.legend_location = QComboBox()
        self.legend_location.addItems(
            [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "center",
                "center left",
                "center right",
                "upper center",
                "lower center",
            ]
        )
        self.legend_columns = QSpinBox()
        self.legend_columns.setRange(1, 12)
        self.legend_frame = QCheckBox("Show legend frame")
        self.legend_labels = QLineEdit()
        self.legend_labels.setPlaceholderText("old=new; old2=new2")
        legend_form = self._form()
        legend_form.addRow("Visibility", self.legend_show)
        legend_form.addRow("Location", self.legend_location)
        legend_form.addRow("Columns", self.legend_columns)
        legend_form.addRow("Frame", self.legend_frame)
        legend_form.addRow("Rename entries", self.legend_labels)
        legend.content.addLayout(legend_form)
        self.text_popup.layout_root.addWidget(legend)

    def _build_plot_popup(self) -> None:
        self.plot_popup = _PlotToolPopup(
            "Plot appearance",
            "Adjust axes and data styling without changing the underlying numerical results.",
            self,
        )

        axes = _ToolSection("Axes & grid")
        self.x_scale = QComboBox()
        self.x_scale.addItems(["linear", "log"])
        self.y_scale = QComboBox()
        self.y_scale.addItems(["linear", "log"])
        self.x_min = QLineEdit()
        self.x_max = QLineEdit()
        self.y_min = QLineEdit()
        self.y_max = QLineEdit()
        self.x_min.setPlaceholderText("Auto")
        self.x_max.setPlaceholderText("Auto")
        self.y_min.setPlaceholderText("Auto")
        self.y_max.setPlaceholderText("Auto")
        self.major_grid = QCheckBox("Major grid")
        self.minor_grid = QCheckBox("Minor grid")
        self.axis_width = QDoubleSpinBox()
        self.axis_width.setRange(0.1, 10)
        self.axis_width.setSingleStep(0.1)

        axis_grid = QGridLayout()
        axis_grid.setHorizontalSpacing(10)
        axis_grid.setVerticalSpacing(8)
        axis_grid.addWidget(QLabel("X scale"), 0, 0)
        axis_grid.addWidget(self.x_scale, 0, 1)
        axis_grid.addWidget(QLabel("Y scale"), 0, 2)
        axis_grid.addWidget(self.y_scale, 0, 3)
        axis_grid.addWidget(QLabel("X minimum"), 1, 0)
        axis_grid.addWidget(self.x_min, 1, 1)
        axis_grid.addWidget(QLabel("X maximum"), 1, 2)
        axis_grid.addWidget(self.x_max, 1, 3)
        axis_grid.addWidget(QLabel("Y minimum"), 2, 0)
        axis_grid.addWidget(self.y_min, 2, 1)
        axis_grid.addWidget(QLabel("Y maximum"), 2, 2)
        axis_grid.addWidget(self.y_max, 2, 3)
        grid_row = QWidget()
        grid_row_layout = QHBoxLayout(grid_row)
        grid_row_layout.setContentsMargins(0, 0, 0, 0)
        grid_row_layout.addWidget(self.major_grid)
        grid_row_layout.addWidget(self.minor_grid)
        grid_row_layout.addStretch(1)
        axis_grid.addWidget(QLabel("Grid"), 3, 0)
        axis_grid.addWidget(grid_row, 3, 1, 1, 2)
        axis_grid.addWidget(QLabel("Axis line width"), 4, 0)
        axis_grid.addWidget(self.axis_width, 4, 1)
        for column in (1, 3):
            axis_grid.setColumnStretch(column, 1)
        axes.content.addLayout(axis_grid)
        self.plot_popup.layout_root.addWidget(axes)

        series = _ToolSection("Series")
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.1, 12)
        self.line_width.setSingleStep(0.1)
        self.line_style = QComboBox()
        self.line_style.addItems(["-", "--", "-.", ":"])
        self.marker = QComboBox()
        self.marker.addItems(["", "o", "s", "^", "v", "D", "x", "+", "."])
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0, 30)
        self.series_visible = QCheckBox("Series visible")
        series_form = self._form()
        series_form.addRow("Line width", self.line_width)
        series_form.addRow("Line style", self.line_style)
        series_form.addRow("Marker", self.marker)
        series_form.addRow("Marker size", self.marker_size)
        series_form.addRow("Visibility", self.series_visible)
        series.content.addLayout(series_form)
        self.plot_popup.layout_root.addWidget(series)

    def _build_export_popup(self) -> None:
        self.export_popup = _PlotToolPopup(
            "Export figure",
            "Choose the output format and publication resolution. Square plots retain an exact 1:1 export canvas.",
            self,
        )
        self.export_series_section = _ToolSection("Series to export")
        series_note = QLabel(
            "Choose which previewed series are included in the saved figure. The exported legend is rebuilt from the checked series only."
        )
        series_note.setObjectName("PlotToolPopupNote")
        series_note.setWordWrap(True)
        self.export_series_section.content.addWidget(series_note)

        selection_actions = QWidget()
        selection_actions_layout = QHBoxLayout(selection_actions)
        selection_actions_layout.setContentsMargins(0, 0, 0, 0)
        selection_actions_layout.setSpacing(8)
        self.select_all_series = QPushButton("Select all")
        self.clear_all_series = QPushButton("Clear all")
        selection_actions_layout.addWidget(self.select_all_series)
        selection_actions_layout.addWidget(self.clear_all_series)
        selection_actions_layout.addStretch(1)
        self.export_series_section.content.addWidget(selection_actions)

        self.export_series_scroll = QScrollArea()
        self.export_series_scroll.setObjectName("ExportSeriesScroll")
        self.export_series_scroll.setWidgetResizable(True)
        self.export_series_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.export_series_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.export_series_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.export_series_scroll.setMaximumHeight(210)
        self.export_series_body = QWidget()
        self.export_series_grid = QGridLayout(self.export_series_body)
        self.export_series_grid.setContentsMargins(0, 0, 0, 0)
        self.export_series_grid.setHorizontalSpacing(12)
        self.export_series_grid.setVerticalSpacing(6)
        self.export_series_scroll.setWidget(self.export_series_body)
        self.export_series_section.content.addWidget(self.export_series_scroll)
        self.export_popup.layout_root.addWidget(self.export_series_section)

        export = _ToolSection("Output")
        self.export_format = QComboBox()
        self.export_format.addItems(["PNG", "SVG", "PDF"])
        self.dpi = QSpinBox()
        self.dpi.setRange(600, 2400)
        self.dpi.setSingleStep(100)
        self.dpi.setSuffix(" dpi")
        self.dpi.setToolTip("PNG resolution. Select any value from 600 to 2400 DPI.")
        self.width = QDoubleSpinBox()
        self.width.setRange(1, 30)
        self.width.setSuffix(" in")
        self.height = QDoubleSpinBox()
        self.height.setRange(1, 30)
        self.height.setSuffix(" in")
        self.transparent = QCheckBox("Transparent background")
        self.tight = QCheckBox("Tight bounding box")
        self.width_label = QLabel("Width")
        self.height_label = QLabel("Height")
        export_form = self._form()
        export_form.addRow("Format", self.export_format)
        export_form.addRow("PNG resolution", self.dpi)
        export_form.addRow(self.width_label, self.width)
        export_form.addRow(self.height_label, self.height)
        export_form.addRow("Background", self.transparent)
        export_form.addRow("Cropping", self.tight)
        export.content.addLayout(export_form)
        self.export_popup.layout_root.addWidget(export)

        self.export_note = QLabel("")
        self.export_note.setObjectName("PlotToolPopupNote")
        self.export_note.setWordWrap(True)
        self.export_popup.layout_root.addWidget(self.export_note)

        self.export_button = QPushButton("Save figure…")
        self.export_button.setObjectName("PrimaryButton")
        self.export_popup.layout_root.addWidget(self.export_button)

    def _build_style_popup(self) -> None:
        self.style_popup = _PlotToolPopup(
            "Style profiles",
            "Reuse a publication style or apply the current style consistently across compatible figures.",
            self,
        )
        self.save_style = QPushButton("Save current style…")
        self.load_style = QPushButton("Load style profile…")
        self.reset_style = QPushButton("Reset to application default")
        self.apply_all = QPushButton("Apply to all compatible plots")
        self.apply_all.setObjectName("PrimaryButton")
        for button in (self.save_style, self.load_style, self.reset_style, self.apply_all):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.style_popup.layout_root.addWidget(button)

    def _connect(self) -> None:
        self.text_tool_button.clicked.connect(
            lambda: self._toggle_popup(self.text_popup, self.text_tool_button)
        )
        self.plot_tool_button.clicked.connect(
            lambda: self._toggle_popup(self.plot_popup, self.plot_tool_button)
        )
        self.export_tool_button.clicked.connect(
            lambda: self._toggle_popup(self.export_popup, self.export_tool_button)
        )
        self.style_tool_button.clicked.connect(
            lambda: self._toggle_popup(self.style_popup, self.style_tool_button)
        )

        self.target.currentTextChanged.connect(self._sync_target_controls)
        self.font.currentFontChanged.connect(self._text_changed)
        self.size.valueChanged.connect(self._text_changed)
        self.bold.toggled.connect(self._text_changed)
        self.italic.toggled.connect(self._text_changed)
        self.apply_font_all.clicked.connect(self._apply_font_all)

        for widget in (
            self.title_text,
            self.x_text,
            self.y_text,
            self.legend_labels,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
        ):
            widget.editingFinished.connect(self._general_changed)
        for widget in (
            self.legend_show,
            self.legend_frame,
            self.major_grid,
            self.minor_grid,
            self.series_visible,
            self.transparent,
            self.tight,
        ):
            widget.toggled.connect(self._general_changed)
        for widget in (
            self.legend_location,
            self.x_scale,
            self.y_scale,
            self.line_style,
            self.marker,
        ):
            widget.currentTextChanged.connect(self._general_changed)
        for widget in (
            self.legend_columns,
            self.axis_width,
            self.line_width,
            self.marker_size,
        ):
            widget.valueChanged.connect(self._general_changed)

        self.select_all_series.clicked.connect(lambda: self._set_all_export_series(True))
        self.clear_all_series.clicked.connect(lambda: self._set_all_export_series(False))
        self.export_format.currentTextChanged.connect(self._sync_export_controls)
        self.width.valueChanged.connect(self._square_width_changed)
        self.height.valueChanged.connect(self._square_height_changed)
        self.export_button.clicked.connect(self.export_figure)
        self.save_style.clicked.connect(self.save_style_profile)
        self.load_style.clicked.connect(self.load_style_profile)
        self.reset_style.clicked.connect(self.reset)
        self.apply_all.clicked.connect(self._apply_all)

    def _hide_popups_except(self, selected: _PlotToolPopup) -> None:
        for popup in (self.text_popup, self.plot_popup, self.export_popup, self.style_popup):
            if popup is not selected and popup.isVisible():
                popup.hide()

    def _toggle_popup(self, popup: _PlotToolPopup, anchor: QToolButton) -> None:
        if popup.isVisible():
            popup.hide()
            return
        self._hide_popups_except(popup)
        if popup is self.export_popup:
            self._refresh_export_series_options()
            self._sync_export_controls()
        popup.adjustSize()
        popup.show()
        popup.adjustSize()

        point = anchor.mapToGlobal(QPoint(0, anchor.height() + 7))
        screen = QApplication.screenAt(point) or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            x = min(max(point.x(), available.left() + 8), available.right() - popup.width() - 8)
            y = point.y()
            if y + popup.height() > available.bottom() - 8:
                y = anchor.mapToGlobal(QPoint(0, -popup.height() - 7)).y()
            point = QPoint(x, max(available.top() + 8, y))
        popup.move(point)
        popup.raise_()
        popup.activateWindow()

    def _plot_widget(self):
        return self.plot_widget_ref()

    @staticmethod
    def _set_combo(combo, text) -> None:
        index = combo.findText(str(text))
        combo.setCurrentIndex(max(index, 0))

    def load_from_style(self) -> None:
        self._loading = True
        s = self.style
        self.title_text.setText(s.title_override)
        self.x_text.setText(s.x_label_override)
        self.y_text.setText(s.y_label_override)
        self.legend_show.setChecked(s.show_legend)
        self._set_combo(self.legend_location, s.legend_location)
        self.legend_columns.setValue(s.legend_columns)
        self.legend_frame.setChecked(s.legend_frame)
        self.legend_labels.setText(
            "; ".join(f"{key}={value}" for key, value in s.legend_label_overrides.items())
        )
        self._set_combo(self.x_scale, s.x_scale)
        self._set_combo(self.y_scale, s.y_scale)
        for edit, value in (
            (self.x_min, s.x_min),
            (self.x_max, s.x_max),
            (self.y_min, s.y_min),
            (self.y_max, s.y_max),
        ):
            edit.setText("" if value is None else str(value))
        self.major_grid.setChecked(s.major_grid)
        self.minor_grid.setChecked(s.minor_grid)
        self.axis_width.setValue(s.axis_line_width)
        self.line_width.setValue(s.line_width)
        self._set_combo(self.line_style, s.line_style)
        self._set_combo(self.marker, s.marker)
        self.marker_size.setValue(s.marker_size)
        self.series_visible.setChecked(s.series_visible)
        self.dpi.setValue(600)

        widget = self._plot_widget()
        square = bool(widget and widget.square_export)
        self.width.setValue(7.2)
        self.height.setValue(7.2 if square else 4.6)
        self.tight.setChecked(not square)
        self._loading = False
        self._sync_target_controls()
        self._sync_export_controls()

    def _sync_target_controls(self) -> None:
        if self._loading:
            return
        self._loading = True
        target = self.target.currentText()
        s = self.style
        if target == "All plot text":
            font = s.title_font
            size = s.title_size
            bold = s.title_bold
            italic = s.title_italic
        else:
            font = getattr(s, FONT_FIELDS[target])
            size = getattr(s, SIZE_FIELDS[target])
            bold = (
                s.title_bold
                if target == "Title"
                else s.axis_labels_bold
                if target in ("X-axis label", "Y-axis label")
                else s.tick_labels_bold
                if "ticks" in target
                else s.legend_bold
                if target == "Legend"
                else s.annotations_bold
            )
            italic = (
                s.title_italic
                if target == "Title"
                else s.axis_labels_italic
                if target in ("X-axis label", "Y-axis label")
                else s.tick_labels_italic
                if "ticks" in target
                else s.legend_italic
                if target == "Legend"
                else s.annotations_italic
            )
        self.font.setCurrentFont(QFont(font))
        self.size.setValue(size)
        self.bold.setChecked(bold)
        self.italic.setChecked(italic)
        self._loading = False

    def _clear_export_series_grid(self) -> None:
        while self.export_series_grid.count():
            item = self.export_series_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_export_series_options(self) -> None:
        """Build export checkboxes from the legend-capable series in the current preview."""
        widget = self._plot_widget()
        previous = {key: check.isChecked() for key, check in self.export_series_checks.items()}
        self._clear_export_series_grid()
        self.export_series_checks = {}
        options = widget.export_series_options() if widget else []
        self.export_series_section.setVisible(bool(options))
        if not options:
            self.export_button.setEnabled(True)
            return

        columns = 2 if len(options) > 5 else 1
        for index, (original_label, display_label) in enumerate(options):
            check = QCheckBox(display_label)
            check.setToolTip(f"Export series: {display_label}")
            check.setProperty("seriesKey", original_label)
            check.setChecked(previous.get(original_label, True))
            check.toggled.connect(self._update_export_button_state)
            row = index // columns
            column = index % columns
            self.export_series_grid.addWidget(check, row, column)
            self.export_series_checks[original_label] = check
        for column in range(columns):
            self.export_series_grid.setColumnStretch(column, 1)
        self._update_export_button_state()

    def _set_all_export_series(self, checked: bool) -> None:
        for checkbox in self.export_series_checks.values():
            checkbox.setChecked(checked)
        self._update_export_button_state()

    def _selected_export_series(self) -> set[str] | None:
        if not self.export_series_checks:
            return None
        return {
            key
            for key, checkbox in self.export_series_checks.items()
            if checkbox.isChecked()
        }

    def _update_export_button_state(self, *_args) -> None:
        selected = self._selected_export_series()
        self.export_button.setEnabled(selected is None or bool(selected))
        if selected is not None and not selected:
            self.export_button.setToolTip("Select at least one series to export.")
        else:
            self.export_button.setToolTip("")

    def _sync_export_controls(self, *_args) -> None:
        widget = self._plot_widget()
        square = bool(widget and widget.square_export)
        is_png = self.export_format.currentText() == "PNG"
        self.dpi.setEnabled(is_png)
        self.dpi.setToolTip(
            "PNG resolution. Select any value from 600 to 2400 DPI."
            if is_png
            else "DPI selection applies to PNG export; SVG and PDF are vector formats."
        )
        if square:
            self.width_label.setText("Square size")
            self.height_label.setText("Square size")
            self.height_label.setVisible(False)
            self.height.setVisible(False)
            self.tight.setChecked(False)
            self.tight.setEnabled(False)
            self.tight.setToolTip(
                "Disabled for square export because tight cropping can change the final 1:1 dimensions."
            )
            self.export_note.setText(
                "Square export is active. The selected side length is used for both width and height."
            )
            if abs(self.width.value() - self.height.value()) > 1e-9:
                self._syncing_square_size = True
                self.height.setValue(self.width.value())
                self._syncing_square_size = False
        else:
            self.width_label.setText("Width")
            self.height_label.setText("Height")
            self.height_label.setVisible(True)
            self.height.setVisible(True)
            self.tight.setEnabled(True)
            self.tight.setToolTip("")
            self.export_note.setText(
                "PNG uses the selected DPI. SVG and PDF remain vector formats."
            )

    def _square_width_changed(self, value: float) -> None:
        if self._syncing_square_size:
            return
        widget = self._plot_widget()
        if widget and widget.square_export:
            self._syncing_square_size = True
            self.height.setValue(value)
            self._syncing_square_size = False

    def _square_height_changed(self, value: float) -> None:
        if self._syncing_square_size:
            return
        widget = self._plot_widget()
        if widget and widget.square_export:
            self._syncing_square_size = True
            self.width.setValue(value)
            self._syncing_square_size = False

    def _text_changed(self, *_args) -> None:
        if self._loading:
            return
        target = self.target.currentText()
        font = self.font.currentFont().family()
        size = self.size.value()
        bold = self.bold.isChecked()
        italic = self.italic.isChecked()
        s = self.style
        targets = list(FONT_FIELDS) if target == "All plot text" else [target]
        for item in targets:
            setattr(s, FONT_FIELDS[item], font)
            setattr(s, SIZE_FIELDS[item], size)
        if target in ("All plot text", "Title"):
            s.title_bold = bold
            s.title_italic = italic
        if target == "All plot text" or target in ("X-axis label", "Y-axis label"):
            s.axis_labels_bold = bold
            s.axis_labels_italic = italic
        if target == "All plot text" or "ticks" in target:
            s.tick_labels_bold = bold
            s.tick_labels_italic = italic
        if target in ("All plot text", "Legend"):
            s.legend_bold = bold
            s.legend_italic = italic
        if target in ("All plot text", "Annotations"):
            s.annotations_bold = bold
            s.annotations_italic = italic
        self._redraw()

    def _apply_font_all(self) -> None:
        font = self.font.currentFont().family()
        for field in FONT_FIELDS.values():
            setattr(self.style, field, font)
        self._redraw()

    @staticmethod
    def _float_or_none(text: str):
        try:
            return float(text) if text.strip() else None
        except ValueError:
            return None

    def _parse_legend_labels(self) -> dict[str, str]:
        output = {}
        for item in self.legend_labels.text().split(";"):
            if "=" in item:
                old, new = item.split("=", 1)
                output[old.strip()] = new.strip()
        return output

    def _general_changed(self, *_args) -> None:
        if self._loading:
            return
        s = self.style
        s.title_override = self.title_text.text()
        s.x_label_override = self.x_text.text()
        s.y_label_override = self.y_text.text()
        s.show_legend = self.legend_show.isChecked()
        s.legend_location = self.legend_location.currentText()
        s.legend_columns = self.legend_columns.value()
        s.legend_frame = self.legend_frame.isChecked()
        s.legend_label_overrides = self._parse_legend_labels()
        s.x_scale = self.x_scale.currentText()
        s.y_scale = self.y_scale.currentText()
        s.x_min = self._float_or_none(self.x_min.text())
        s.x_max = self._float_or_none(self.x_max.text())
        s.y_min = self._float_or_none(self.y_min.text())
        s.y_max = self._float_or_none(self.y_max.text())
        s.major_grid = self.major_grid.isChecked()
        s.minor_grid = self.minor_grid.isChecked()
        s.axis_line_width = self.axis_width.value()
        s.line_width = self.line_width.value()
        s.line_style = self.line_style.currentText()
        s.marker = self.marker.currentText()
        s.marker_size = self.marker_size.value()
        s.series_visible = self.series_visible.isChecked()
        self._redraw()

    def _redraw(self) -> None:
        widget = self._plot_widget()
        if widget:
            widget.apply_style(self.style)

    def _apply_all(self) -> None:
        widget = self._plot_widget()
        if widget:
            widget.manager.apply_to_all(self.style)

    def save_style_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot style",
            "plot_style.json",
            "JSON (*.json)",
        )
        if path:
            self.style.save(path)

    def load_style_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load plot style",
            "",
            "JSON (*.json)",
        )
        if path:
            self.style = PlotStyle.load(path)
            self.load_from_style()
            self._redraw()

    def reset(self) -> None:
        self.style = PlotStyle()
        self.load_from_style()
        self._redraw()

    def export_figure(self) -> None:
        fmt = self.export_format.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            f"figure.{fmt}",
            f"{fmt.upper()} (*.{fmt})",
        )
        if not path:
            return
        widget = self._plot_widget()
        if not widget:
            return
        square = bool(widget.square_export)
        side = self.width.value()
        width = side if square else self.width.value()
        height = side if square else self.height.value()
        dpi = self.dpi.value() if fmt == "png" else 600
        widget.manager.export(
            widget.plot_id,
            path,
            dpi,
            self.transparent.isChecked(),
            self.tight.isChecked(),
            width,
            height,
            square=square,
            selected_series=self._selected_export_series(),
        )
