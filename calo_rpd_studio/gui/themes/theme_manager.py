"""Application-wide Qt palette and stylesheet management."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from .dark import DARK_STYLESHEET
from .light import LIGHT_STYLESHEET


def _set_palette_color(palette: QPalette, role: QPalette.ColorRole, value: str) -> None:
    color = QColor(value)
    palette.setColor(QPalette.ColorGroup.Active, role, color)
    palette.setColor(QPalette.ColorGroup.Inactive, role, color)


def _light_palette() -> QPalette:
    p = QPalette()
    values = {
        QPalette.ColorRole.Window: "#f5f7fb",
        QPalette.ColorRole.WindowText: "#172033",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#f1f4f9",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#172033",
        QPalette.ColorRole.Text: "#172033",
        QPalette.ColorRole.Button: "#eef2f8",
        QPalette.ColorRole.ButtonText: "#172033",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: "#3157d5",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#3157d5",
        QPalette.ColorRole.LinkVisited: "#6b4fd3",
        QPalette.ColorRole.PlaceholderText: "#7a8699",
    }
    for role, value in values.items():
        _set_palette_color(p, role, value)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#98a2b3"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#98a2b3"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#98a2b3"))
    return p


def _dark_palette() -> QPalette:
    p = QPalette()
    values = {
        QPalette.ColorRole.Window: "#111722",
        QPalette.ColorRole.WindowText: "#e6eaf2",
        QPalette.ColorRole.Base: "#121925",
        QPalette.ColorRole.AlternateBase: "#18202d",
        QPalette.ColorRole.ToolTipBase: "#202a39",
        QPalette.ColorRole.ToolTipText: "#f2f4f7",
        QPalette.ColorRole.Text: "#e6eaf2",
        QPalette.ColorRole.Button: "#222c3c",
        QPalette.ColorRole.ButtonText: "#e6eaf2",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: "#4d6ee8",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#8aa2ff",
        QPalette.ColorRole.LinkVisited: "#b49cff",
        QPalette.ColorRole.PlaceholderText: "#7f8ba0",
    }
    for role, value in values.items():
        _set_palette_color(p, role, value)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6f7b8e"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6f7b8e"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6f7b8e"))
    return p


def apply_theme(app: QApplication, name: str) -> str:
    """Apply a deterministic application palette and matching stylesheet.

    Returning the normalized theme name is convenient for callers that persist
    settings from older installations.
    """
    normalized = "dark" if str(name).strip().lower() == "dark" else "light"
    app.setStyle("Fusion")
    if normalized == "dark":
        app.setPalette(_dark_palette())
        app.setStyleSheet(DARK_STYLESHEET)
    else:
        app.setPalette(_light_palette())
        app.setStyleSheet(LIGHT_STYLESHEET)
    return normalized
