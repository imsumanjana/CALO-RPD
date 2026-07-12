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
