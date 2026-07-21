"""Portable Matplotlib font preflight without distributing proprietary font files."""

from __future__ import annotations

from functools import lru_cache

from matplotlib import font_manager

DEFAULT_FALLBACK = "DejaVu Serif"


@lru_cache(maxsize=128)
def font_available(family: str) -> bool:
    try:
        font_manager.findfont(family, fallback_to_default=False)
        return True
    except (ValueError, RuntimeError):
        return False


@lru_cache(maxsize=128)
def resolve_font(family: str, fallback: str = DEFAULT_FALLBACK) -> str:
    requested = str(family or "").strip() or fallback
    if font_available(requested):
        return requested
    return fallback if font_available(fallback) else "DejaVu Sans"


def font_resolution_manifest(requested: str = "Times New Roman") -> dict[str, object]:
    resolved = resolve_font(requested)
    return {
        "requested_font": requested,
        "resolved_font": resolved,
        "requested_font_available": bool(font_available(requested)),
        "fallback_disclosed": bool(resolved != requested),
        "font_files_bundled": False,
    }
