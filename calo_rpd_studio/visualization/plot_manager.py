"""Reusable Matplotlib plot registry with independent raw data and style."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import uuid

from .plot_style import PlotStyle


@dataclass
class PlotRecord:
    plot_id: str
    figure: object
    axis: object
    raw_data: object
    metadata: dict
    style: PlotStyle


class PlotManager:
    def __init__(self):
        self.records = {}
        self.default_style = PlotStyle()

    def register(self, figure, axis, raw_data=None, metadata=None, plot_id=None, style=None):
        pid = plot_id or str(uuid.uuid4())
        self.records[pid] = PlotRecord(
            pid,
            figure,
            axis,
            raw_data,
            metadata or {},
            deepcopy(style or self.default_style),
        )
        self.apply(pid)
        return pid

    @staticmethod
    def _valid_series_label(label: str) -> bool:
        return bool(label and not str(label).startswith("_"))

    def series_labels(self, plot_id: str) -> list[str]:
        """Return unique plotted line labels in preview/legend order."""
        labels: list[str] = []
        seen: set[str] = set()
        for line in self.records[plot_id].axis.lines:
            label = str(line.get_label())
            if self._valid_series_label(label) and label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    @staticmethod
    def _finite_line_values(ax, axis_name: str) -> list[float]:
        values: list[float] = []
        scale = ax.get_xscale() if axis_name == "x" else ax.get_yscale()
        for line in ax.lines:
            if hasattr(line, "get_visible") and not line.get_visible():
                continue
            raw = line.get_xdata(orig=False) if axis_name == "x" else line.get_ydata(orig=False)
            for item in raw:
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                if scale == "log" and value <= 0:
                    continue
                values.append(value)
        return values

    @staticmethod
    def _padded_limits(values: list[float], padding: float, *, include_zero: bool = False) -> tuple[float, float] | None:
        if not values:
            return None
        lower = min(values)
        upper = max(values)
        padding = max(0.0, min(float(padding), 0.5))
        if include_zero and lower >= 0.0:
            lower = 0.0
            if upper == 0.0:
                upper = 1e-6
            else:
                upper += max(upper, 1e-12) * padding
            return lower, upper
        span = upper - lower
        if span <= 0.0:
            scale = max(abs(lower), abs(upper), 1e-9)
            span = max(scale * 0.1, 1e-9)
        pad = span * padding
        return lower - pad, upper + pad

    def _apply_axis_limits(self, rec: PlotRecord) -> None:
        """Apply either intelligent visible-data fitting or explicit/manual axis bounds.

        ``Axes.clear`` does not guarantee that a prior manually-fixed limit is forgotten.  This
        routine explicitly re-enables autoscaling before recomputing limits, preventing a stale
        0--0.7 range from hiding convergence differences around 0.02--0.06.
        """
        s = rec.style
        ax = rec.axis

        if s.auto_fit_visible_data:
            ax.set_autoscalex_on(True)
            ax.set_autoscaley_on(True)
            ax.relim(visible_only=True)
            ax.autoscale_view(scalex=True, scaley=True)

            x_values = self._finite_line_values(ax, "x")
            y_values = self._finite_line_values(ax, "y")
            x_limits = self._padded_limits(x_values, s.auto_scale_padding, include_zero=False)
            y_limits = self._padded_limits(
                y_values,
                s.auto_scale_padding,
                include_zero=bool(s.auto_include_zero and ax.get_yscale() == "linear"),
            )
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            return

        # Manual mode still autosizes every unspecified bound from the current visible data.
        # Explicit values are then applied only to the sides the user actually fixed.
        ax.set_autoscalex_on(s.x_min is None or s.x_max is None)
        ax.set_autoscaley_on(s.y_min is None or s.y_max is None)
        ax.relim(visible_only=True)
        ax.autoscale_view(scalex=True, scaley=True)
        current_x = ax.get_xlim()
        current_y = ax.get_ylim()
        ax.set_xlim(
            left=current_x[0] if s.x_min is None else s.x_min,
            right=current_x[1] if s.x_max is None else s.x_max,
        )
        ax.set_ylim(
            bottom=current_y[0] if s.y_min is None else s.y_min,
            top=current_y[1] if s.y_max is None else s.y_max,
        )

    def _update_legend(self, rec: PlotRecord, *, visible_only: bool = False) -> None:
        s = rec.style
        ax = rec.axis
        handles, labels = ax.get_legend_handles_labels()
        valid = []
        for handle, label in zip(handles, labels):
            label = str(label)
            if not self._valid_series_label(label):
                continue
            if visible_only and hasattr(handle, "get_visible") and not handle.get_visible():
                continue
            display_label = s.legend_label_overrides.get(label, label)
            valid.append((handle, display_label))

        old = ax.get_legend()
        if old:
            old.remove()
        if s.show_legend and valid:
            legend_handles, legend_labels = zip(*valid)
            ax.legend(
                legend_handles,
                legend_labels,
                loc=s.legend_location,
                ncol=s.legend_columns,
                frameon=s.legend_frame,
                prop={
                    "family": s.legend_font,
                    "size": s.legend_size,
                    "weight": "bold" if s.legend_bold else "normal",
                    "style": "italic" if s.legend_italic else "normal",
                },
            )

    def apply(self, plot_id, style=None):
        rec = self.records[plot_id]
        if style is not None:
            rec.style = deepcopy(style)
        s = rec.style
        ax = rec.axis
        meta = rec.metadata
        title = s.title_override or meta.get("title", ax.get_title())
        xlabel = s.x_label_override or meta.get("xlabel", ax.get_xlabel())
        ylabel = s.y_label_override or meta.get("ylabel", ax.get_ylabel())
        ax.set_title(
            title,
            fontfamily=s.title_font,
            fontsize=s.title_size,
            fontweight="bold" if s.title_bold else "normal",
            fontstyle="italic" if s.title_italic else "normal",
        )
        ax.set_xlabel(
            xlabel,
            fontfamily=s.x_label_font,
            fontsize=s.x_label_size,
            fontweight="bold" if s.axis_labels_bold else "normal",
            fontstyle="italic" if s.axis_labels_italic else "normal",
        )
        ax.set_ylabel(
            ylabel,
            fontfamily=s.y_label_font,
            fontsize=s.y_label_size,
            fontweight="bold" if s.axis_labels_bold else "normal",
            fontstyle="italic" if s.axis_labels_italic else "normal",
        )
        for label in ax.get_xticklabels():
            label.set_fontfamily(s.x_tick_font)
            label.set_fontsize(s.x_tick_size)
            label.set_fontweight("bold" if s.tick_labels_bold else "normal")
            label.set_fontstyle("italic" if s.tick_labels_italic else "normal")
        for label in ax.get_yticklabels():
            label.set_fontfamily(s.y_tick_font)
            label.set_fontsize(s.y_tick_size)
            label.set_fontweight("bold" if s.tick_labels_bold else "normal")
            label.set_fontstyle("italic" if s.tick_labels_italic else "normal")
        ax.set_xscale(s.x_scale)
        ax.set_yscale(s.y_scale)
        ax.grid(s.major_grid, which="major")
        ax.minorticks_on() if s.minor_grid else ax.minorticks_off()
        ax.grid(True, which="minor", alpha=0.3) if s.minor_grid else ax.grid(False, which="minor")
        for spine in ax.spines.values():
            spine.set_linewidth(s.axis_line_width)
        for line in ax.lines:
            line.set_linewidth(s.line_width)
            line.set_linestyle(s.line_style)
            line.set_marker(s.marker)
            line.set_markersize(s.marker_size)
            line.set_visible(s.series_visible)

        self._apply_axis_limits(rec)
        self._update_legend(rec)
        rec.figure.tight_layout()
        canvas = getattr(rec.figure, "canvas", None)
        if canvas:
            canvas.draw_idle()

    def apply_to_all(self, style):
        self.default_style = deepcopy(style)
        for pid in list(self.records):
            self.apply(pid, style)

    def export(
        self,
        plot_id,
        path,
        dpi=600,
        transparent=False,
        tight=True,
        width=None,
        height=None,
        *,
        square=False,
        selected_series: set[str] | list[str] | tuple[str, ...] | None = None,
    ):
        """Export a registered figure, optionally with only selected line series.

        ``selected_series`` contains the original Matplotlib line labels, not display
        overrides. Non-selected labelled lines are hidden only for the export. The live
        preview is restored exactly after saving. The export legend is rebuilt from the
        visible selected series, so no unselected legend entry is written to the file.

        When ``square`` is true, the physical figure page is forced to 1:1. Tight
        cropping is deliberately disabled because cropping can change the final pixel
        dimensions and violate the exact square-output contract.
        """
        rec = self.records[plot_id]
        old_size = rec.figure.get_size_inches().copy()
        line_visibility = [(line, bool(line.get_visible())) for line in rec.axis.lines]
        old_limits = (rec.axis.get_xlim(), rec.axis.get_ylim())
        selectable = set(self.series_labels(plot_id))
        requested = None if selected_series is None else set(selected_series)
        if requested is not None:
            unknown = requested.difference(selectable)
            if unknown:
                raise ValueError(f"Unknown plot series selected for export: {sorted(unknown)}")
            if selectable and not requested:
                raise ValueError("Select at least one plotted series before exporting.")

        try:
            if requested is not None:
                for line in rec.axis.lines:
                    label = str(line.get_label())
                    if self._valid_series_label(label):
                        line.set_visible(label in requested)
                if rec.style.auto_fit_visible_data:
                    self._apply_axis_limits(rec)
                self._update_legend(rec, visible_only=True)
                rec.figure.tight_layout()

            if square:
                side = float(width or height or 7.2)
                rec.figure.set_size_inches(side, side, forward=True)
                bbox = None
            else:
                if width and height:
                    rec.figure.set_size_inches(width, height, forward=True)
                bbox = "tight" if tight else None
            rec.figure.savefig(
                Path(path),
                dpi=int(dpi),
                transparent=transparent,
                bbox_inches=bbox,
            )
        finally:
            rec.figure.set_size_inches(old_size, forward=True)
            for line, was_visible in line_visibility:
                line.set_visible(was_visible)
            rec.axis.set_xlim(*old_limits[0])
            rec.axis.set_ylim(*old_limits[1])
            self._update_legend(rec, visible_only=False)
            rec.figure.tight_layout()
            canvas = getattr(rec.figure, "canvas", None)
            if canvas:
                canvas.draw_idle()
        return Path(path)
