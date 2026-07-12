"""Figure export wrapper."""
from __future__ import annotations


def export_figure(
    figure,
    path,
    dpi: int = 600,
    transparent: bool = False,
    tight: bool = True,
    *,
    square: bool = False,
    size_inches: float = 7.2,
):
    """Export a Matplotlib figure, optionally forcing an exact square page."""
    old = figure.get_size_inches().copy()
    try:
        if square:
            figure.set_size_inches(size_inches, size_inches, forward=True)
            bbox = None
        else:
            bbox = "tight" if tight else None
        figure.savefig(path, dpi=dpi, transparent=transparent, bbox_inches=bbox)
    finally:
        figure.set_size_inches(old, forward=True)
    return path
