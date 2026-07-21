"""Serializable publication plot style."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


@dataclass(slots=True)
class PlotStyle:
    title_font: str = "Times New Roman"
    x_label_font: str = "Times New Roman"
    y_label_font: str = "Times New Roman"
    x_tick_font: str = "Times New Roman"
    y_tick_font: str = "Times New Roman"
    legend_font: str = "Times New Roman"
    annotation_font: str = "Times New Roman"
    title_size: int = 14
    x_label_size: int = 12
    y_label_size: int = 12
    x_tick_size: int = 10
    y_tick_size: int = 10
    legend_size: int = 10
    annotation_size: int = 10
    title_bold: bool = True
    axis_labels_bold: bool = True
    tick_labels_bold: bool = False
    legend_bold: bool = False
    annotations_bold: bool = False
    title_italic: bool = False
    axis_labels_italic: bool = False
    tick_labels_italic: bool = False
    legend_italic: bool = False
    annotations_italic: bool = False
    title_override: str = ""
    x_label_override: str = ""
    y_label_override: str = ""
    legend_label_overrides: dict[str, str] = field(default_factory=dict)
    show_legend: bool = True
    legend_location: str = "best"
    legend_columns: int = 1
    legend_frame: bool = True
    x_scale: str = "linear"
    y_scale: str = "linear"
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    auto_fit_visible_data: bool = False
    auto_include_zero: bool = False
    auto_scale_padding: float = 0.08
    major_grid: bool = True
    minor_grid: bool = False
    axis_line_width: float = 1.0
    line_width: float = 1.8
    line_style: str = "-"
    marker: str = ""
    marker_size: float = 5.0
    series_visible: bool = True

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
