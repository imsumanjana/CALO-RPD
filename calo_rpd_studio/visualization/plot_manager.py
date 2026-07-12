"""Reusable Matplotlib plot registry with independent raw data and style."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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
        ax.set_xlim(left=s.x_min, right=s.x_max)
        ax.set_ylim(bottom=s.y_min, top=s.y_max)
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
        handles, labels = ax.get_legend_handles_labels()
        labels = [s.legend_label_overrides.get(label, label) for label in labels]
        valid = [(h, label) for h, label in zip(handles, labels) if label and not label.startswith("_")]
        old = ax.get_legend()
        if old:
            old.remove()
        if s.show_legend and valid:
            handles, labels = zip(*valid)
            ax.legend(
                handles,
                labels,
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
    ):
        """Export a registered figure.

        When ``square`` is true, the physical figure page is forced to 1:1. Tight
        cropping is deliberately disabled because cropping can change the final pixel
        dimensions and violate the exact square-output contract.
        """
        rec = self.records[plot_id]
        old = rec.figure.get_size_inches().copy()
        try:
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
            rec.figure.set_size_inches(old, forward=True)
        return Path(path)
