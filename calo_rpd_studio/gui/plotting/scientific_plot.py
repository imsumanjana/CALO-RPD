"""Embedded Matplotlib canvas with live publication-format controls."""
from __future__ import annotations

from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from calo_rpd_studio.visualization.plot_manager import PlotManager
from .plot_format_toolbar import PlotFormattingToolbar


class ScientificPlotWidget(QWidget):
    """Reusable scientific plot surface.

    ``square_preview`` gives the Matplotlib canvas an exact 1:1 display surface.
    ``square_export`` guarantees that exported figures use a square physical page/canvas.
    The two options are separate so other workspaces can retain their preferred layout.
    """

    def __init__(
        self,
        manager=None,
        parent=None,
        title: str = "Scientific plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        *,
        square_preview: bool = False,
        square_export: bool | None = None,
        square_preview_size: int = 720,
        auto_fit_visible_data: bool = False,
        auto_include_zero: bool = False,
        auto_scale_padding: float = 0.08,
    ) -> None:
        super().__init__(parent)
        self.manager = manager or PlotManager()
        self.style = deepcopy(self.manager.default_style)
        self.style.auto_fit_visible_data = bool(auto_fit_visible_data)
        self.style.auto_include_zero = bool(auto_include_zero)
        self.style.auto_scale_padding = max(0.0, min(float(auto_scale_padding), 0.5))
        self.square_preview = bool(square_preview)
        self.square_export = self.square_preview if square_export is None else bool(square_export)
        self.square_preview_size = max(420, int(square_preview_size))

        figure_size = (7.2, 7.2) if self.square_preview else (7.2, 4.6)
        self.figure = Figure(figsize=figure_size)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title(title)
        self.axis.set_xlabel(xlabel)
        self.axis.set_ylabel(ylabel)
        self.plot_id = self.manager.register(
            self.figure,
            self.axis,
            metadata={"title": title, "xlabel": xlabel, "ylabel": ylabel},
            style=self.style,
        )
        self.format_toolbar = PlotFormattingToolbar(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.format_toolbar)

        if self.square_preview:
            self.canvas.setFixedSize(self.square_preview_size, self.square_preview_size)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            canvas_host = QWidget()
            canvas_host.setObjectName("SquarePlotPreviewHost")
            host_layout = QHBoxLayout(canvas_host)
            host_layout.setContentsMargins(0, 0, 0, 0)
            host_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            host_layout.addWidget(self.canvas)
            layout.addWidget(canvas_host, 0)
        else:
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self.canvas, 1)



    def configure_preview_series(self, options_provider, selection_provider, selection_callback) -> None:
        """Expose host-controlled selective preview through the compact Plot Tools strip."""
        self.format_toolbar.configure_preview_series(
            options_provider, selection_provider, selection_callback
        )

    def export_series_options(self) -> list[tuple[str, str]]:
        """Return ``(original_label, displayed_legend_label)`` pairs for export selection."""
        output: list[tuple[str, str]] = []
        for label in self.manager.series_labels(self.plot_id):
            display = self.style.legend_label_overrides.get(label, label)
            output.append((label, display))
        return output

    def apply_style(self, style) -> None:
        self.style = deepcopy(style)
        self.manager.apply(self.plot_id, self.style)

    def set_auto_scale_context(self, *, include_zero: bool | None = None) -> None:
        """Update metric-specific automatic scaling without altering numerical data."""
        if include_zero is not None:
            self.style.auto_include_zero = bool(include_zero)
            self.format_toolbar.style.auto_include_zero = bool(include_zero)
            self.format_toolbar.sync_auto_scale_from_style()

    def clear(self) -> None:
        self.axis.clear()


    def show_message(self, message: str, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
        """Render an informative empty-state message instead of a visually blank chart."""
        self.axis.clear()
        self.axis.text(
            0.5,
            0.5,
            str(message),
            transform=self.axis.transAxes,
            ha="center",
            va="center",
            wrap=True,
        )
        meta = self.manager.records[self.plot_id].metadata
        if title is not None:
            meta["title"] = title
        if xlabel is not None:
            meta["xlabel"] = xlabel
        if ylabel is not None:
            meta["ylabel"] = ylabel
        self.manager.apply(self.plot_id, self.style)

    def plot_series(self, series, title=None, xlabel=None, ylabel=None) -> None:
        self.axis.clear()
        for label, values in series.items():
            self.axis.plot(range(1, len(values) + 1), values, label=label)
        meta = self.manager.records[self.plot_id].metadata
        if title is not None:
            meta["title"] = title
        if xlabel is not None:
            meta["xlabel"] = xlabel
        if ylabel is not None:
            meta["ylabel"] = ylabel
        self.manager.apply(self.plot_id, self.style)
    def plot_xy_series(self, series, title=None, xlabel=None, ylabel=None) -> None:
        """Plot mapping of label -> (x_values, y_values) without inventing an x-axis.

        This is used for convergence plots where objective-function evaluation count is the
        scientifically fair comparison axis and different optimizers may record at different
        evaluation intervals.
        """
        self.axis.clear()
        for label, pair in series.items():
            if pair is None or len(pair) != 2:
                continue
            x_values, y_values = pair
            if not x_values or not y_values:
                continue
            self.axis.plot(x_values, y_values, label=label)
        meta = self.manager.records[self.plot_id].metadata
        if title is not None:
            meta["title"] = title
        if xlabel is not None:
            meta["xlabel"] = xlabel
        if ylabel is not None:
            meta["ylabel"] = ylabel
        self.manager.apply(self.plot_id, self.style)

