from __future__ import annotations

import matplotlib.image as mpimg
from matplotlib.figure import Figure

from calo_rpd_studio.visualization.plot_manager import PlotManager


def test_square_export_has_exact_one_to_one_pixel_dimensions(tmp_path):
    figure = Figure(figsize=(3, 2))
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    manager = PlotManager()
    plot_id = manager.register(figure, axis)
    path = tmp_path / "square.png"
    manager.export(plot_id, path, dpi=100, width=2, height=1, square=True)
    image = mpimg.imread(path)
    assert image.shape[0] == image.shape[1] == 200


def test_export_selected_series_filters_saved_lines_and_legend_then_restores_preview(tmp_path, monkeypatch):
    figure = Figure(figsize=(3, 3))
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [1, 0], label="CALO")
    axis.plot([0, 1], [0.8, 0.2], label="PSO")
    manager = PlotManager()
    plot_id = manager.register(figure, axis)

    captured = {}

    def fake_savefig(*args, **kwargs):
        captured["visible"] = [
            line.get_label()
            for line in axis.lines
            if line.get_visible() and not line.get_label().startswith("_")
        ]
        legend = axis.get_legend()
        captured["legend"] = [] if legend is None else [text.get_text() for text in legend.get_texts()]

    monkeypatch.setattr(figure, "savefig", fake_savefig)
    manager.export(
        plot_id,
        tmp_path / "selected.png",
        dpi=600,
        square=True,
        selected_series={"CALO"},
    )

    assert captured["visible"] == ["CALO"]
    assert captured["legend"] == ["CALO"]
    assert [line.get_visible() for line in axis.lines] == [True, True]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["CALO", "PSO"]


def test_export_rejects_empty_selection_when_series_exist(tmp_path):
    figure = Figure(figsize=(3, 3))
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [1, 0], label="CALO")
    manager = PlotManager()
    plot_id = manager.register(figure, axis)

    import pytest

    with pytest.raises(ValueError, match="Select at least one"):
        manager.export(plot_id, tmp_path / "none.png", selected_series=set())
