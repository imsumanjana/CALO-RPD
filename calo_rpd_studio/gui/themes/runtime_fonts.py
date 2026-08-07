"""Deterministic, license-safe Qt font selection for desktop and offscreen rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys

from PyQt6.QtGui import QFont, QFontDatabase, QFontMetrics
from PyQt6.QtWidgets import QApplication


FONT_VALIDATION_SAMPLE = (
    "CALO-RPD 0123456789 +/- % (.,:;) - \u2014 \u00b7 \u03bc \u03c3 \u0394 \u03a9"
)
_FONT_SOURCE_PROPERTY = "caloRpdFontSource"
_FONT_FAMILY_PROPERTY = "caloRpdFontFamily"
_FONT_REGISTERED_PROPERTY = "caloRpdFontRegistered"


@dataclass(frozen=True, slots=True)
class ApplicationFontRecord:
    """Auditable result of resolving a usable application font."""

    family: str
    source: str
    registered: bool
    supports_validation_sample: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def _font_supports_sample(font: QFont) -> bool:
    metrics = QFontMetrics(font)
    return all(
        character.isspace() or metrics.inFontUcs4(ord(character))
        for character in FONT_VALIDATION_SAMPLE
    )


def _candidate_paths() -> tuple[Path, ...]:
    configured = str(os.environ.get("CALO_RPD_GUI_FONT_PATH", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    if sys.platform == "win32":
        windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates.extend(
            (
                windows_root / "Fonts" / "segoeui.ttf",
                windows_root / "Fonts" / "arial.ttf",
            )
        )
    elif sys.platform == "darwin":
        candidates.extend(
            (
                Path("/System/Library/Fonts/SFNS.ttf"),
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            )
        )
    else:
        candidates.extend(
            (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
                Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
            )
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        identity = str(path).casefold()
        if identity not in seen:
            seen.add(identity)
            unique.append(path)
    return tuple(unique)


def _remember(app: QApplication, record: ApplicationFontRecord) -> ApplicationFontRecord:
    app.setProperty(_FONT_SOURCE_PROPERTY, record.source)
    app.setProperty(_FONT_FAMILY_PROPERTY, record.family)
    app.setProperty(_FONT_REGISTERED_PROPERTY, record.registered)
    return record


def ensure_application_font(app: QApplication) -> ApplicationFontRecord:
    """Ensure Qt can render the ordinary scientific UI text deterministically.

    Native desktop platforms normally provide a complete default font. Qt's
    offscreen plugin may not discover that font, so this function registers an
    existing OS font file explicitly. The file remains system-provided and is
    never copied into or redistributed with CALO-RPD.
    """
    remembered_source = app.property(_FONT_SOURCE_PROPERTY)
    if remembered_source:
        remembered_family = str(app.property(_FONT_FAMILY_PROPERTY) or app.font().family())
        registered = bool(app.property(_FONT_REGISTERED_PROPERTY))
        if not _font_supports_sample(app.font()) and remembered_family:
            point_size = app.font().pointSize() if app.font().pointSize() > 0 else 10
            app.setFont(QFont(remembered_family, point_size))
        remembered = ApplicationFontRecord(
            family=remembered_family,
            source=str(remembered_source),
            registered=registered,
            supports_validation_sample=_font_supports_sample(app.font()),
        )
        if remembered.supports_validation_sample:
            return remembered

    current = app.font()
    if _font_supports_sample(current):
        return _remember(
            app,
            ApplicationFontRecord(
                family=current.family(),
                source="qt-platform-default",
                registered=False,
                supports_validation_sample=True,
            ),
        )

    point_size = current.pointSize() if current.pointSize() > 0 else 10
    for path in _candidate_paths():
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            continue
        selected = QFont(families[0], point_size)
        app.setFont(selected)
        supported = _font_supports_sample(app.font())
        record = ApplicationFontRecord(
            family=app.font().family(),
            source=str(path.resolve()),
            registered=True,
            supports_validation_sample=supported,
        )
        if supported:
            return _remember(app, record)

    return _remember(
        app,
        ApplicationFontRecord(
            family=app.font().family(),
            source="unresolved",
            registered=False,
            supports_validation_sample=False,
        ),
    )
