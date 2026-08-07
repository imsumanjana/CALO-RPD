"""Named Phase 3 semantic colors shared by palettes and programmatic visuals."""

from __future__ import annotations

from types import MappingProxyType


SPACING_UNIT = 8


LIGHT_TOKENS = MappingProxyType(
    {
        "background": "#f4f7fb",
        "surface": "#ffffff",
        "border": "#e2e8f0",
        "text": "#0f172a",
        "muted_text": "#64748b",
        "accent": "#2563eb",
        "ready": "#087f5b",
        "attention": "#7c5c16",
        "blocked": "#94a3b8",
        "failed": "#b42318",
        "historical": "#6b4fd3",
        "focus": "#2563eb",
    }
)

DARK_TOKENS = MappingProxyType(
    {
        "background": "#0f1520",
        "surface": "#151e2c",
        "border": "#2a3548",
        "text": "#f8fafc",
        "muted_text": "#8d9aaf",
        "accent": "#4f7cff",
        "ready": "#73d7b4",
        "attention": "#d6b978",
        "blocked": "#66758c",
        "failed": "#ff9b91",
        "historical": "#b49cff",
        "focus": "#6b92ff",
    }
)
