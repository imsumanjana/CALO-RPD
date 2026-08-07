"""Small, consistent SVG icon set for scientist workspaces."""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSize
from PyQt6.QtGui import QIcon, QPixmap


_PATHS = {
    "home": '<path d="M3 10.5 12 3l9 7.5V21h-6v-6H9v6H3z"/>',
    "resume": '<path d="M4 5v6h6M5.5 10A7 7 0 1 1 7 18.5"/>',
    "policy": '<path d="M12 3a4 4 0 0 0-4 4v2H6v4h2v2a4 4 0 0 0 8 0v-2h2V9h-2V7a4 4 0 0 0-4-4zM8 11h8"/>',
    "network": '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="m7 7 4 9m6-9-4 9M7 6h10"/>',
    "formulation": '<path d="M5 4h14v16H5zM8 8h8M8 12h3m2 0h3M8 16h8"/>',
    "algorithm": '<path d="M5 5h5v5H5zm9 0h5v5h-5zM9 14h6v5H9zM10 7h4m-2 3v4"/>',
    "scenario": '<path d="M4 18 9 7l4 8 3-5 4 8z"/><path d="M4 21h16"/>',
    "portfolio": '<path d="M4 7h16v13H4zM9 7V4h6v3M4 12h16"/>',
    "study": '<path d="M5 4h14v16H5zM8 8h8M8 12h8M8 16h5"/>',
    "activity": '<path d="M3 13h4l2-6 4 11 2-7 2 2h4"/>',
    "results": '<path d="M5 19V9m5 10V5m5 14v-7m5 7V3"/>',
    "statistics": '<path d="M4 19V5m0 14h16M7 15l4-5 3 2 5-7"/>',
    "validation": '<path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6z"/><path d="m8 12 3 3 5-6"/>',
    "benchmark": '<path d="M5 20h14M7 17l3-7 3 4 4-9"/><circle cx="17" cy="5" r="2"/>',
    "publication": '<path d="M6 3h9l3 3v15H6zM14 3v4h4M9 12h6m-6 4h6"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1m0-12.8-2.1 2.1m-8.6 8.6-2.1 2.1"/>',
    "workspace": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9h16"/>',
}


def workspace_icon(name: str, *, color: str = "#64748b", size: int = 20) -> QIcon:
    """Create a deterministic stroke SVG icon without external font dependencies."""
    body = _PATHS.get(str(name), _PATHS["workspace"])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray(svg.encode("utf-8")), "SVG")
    return QIcon(pixmap.scaled(QSize(size, size)))
