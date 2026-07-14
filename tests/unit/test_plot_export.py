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


def test_auto_fit_visible_data_tightens_nonnegative_convergence_range():
    from calo_rpd_studio.visualization.plot_style import PlotStyle

    figure = Figure(figsize=(3, 3))
    axis = figure.add_subplot(111)
    axis.plot([100, 5000], [0.06, 0.026], label="CALO")
    axis.plot([100, 5000], [0.04, 0.020], label="QODE")
    # Simulate a stale/manual range left from an earlier preview.
    axis.set_ylim(0.0, 0.7)
    manager = PlotManager()
    style = PlotStyle(auto_fit_visible_data=True, auto_include_zero=True, auto_scale_padding=0.08)
    plot_id = manager.register(figure, axis, style=style)

    lower, upper = axis.get_ylim()
    assert lower == 0.0
    assert 0.06 < upper < 0.08
    assert axis.get_xlim()[1] > 5000
    assert manager.records[plot_id].style.auto_fit_visible_data is True


def test_auto_fit_visible_data_reacts_to_current_visible_series_only():
    from calo_rpd_studio.visualization.plot_style import PlotStyle

    figure = Figure(figsize=(3, 3))
    axis = figure.add_subplot(111)
    low = axis.plot([0, 1], [0.02, 0.03], label="Low")[0]
    high = axis.plot([0, 1], [0.5, 0.6], label="High")[0]
    manager = PlotManager()
    style = PlotStyle(auto_fit_visible_data=True, auto_include_zero=False, auto_scale_padding=0.05)
    plot_id = manager.register(figure, axis, style=style)
    assert axis.get_ylim()[1] > 0.6

    high.set_visible(False)
    manager._apply_axis_limits(manager.records[plot_id])
    assert axis.get_ylim()[1] < 0.04
    assert low.get_visible() is True
