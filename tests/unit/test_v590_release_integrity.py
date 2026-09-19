from __future__ import annotations

from pathlib import Path
import json


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_historical_v590_freeze_and_archive_routing_are_retained():
    """The old freeze remains readable; retired reports are archived, not active v12 evidence."""
    root = _root()
    freeze = root / "calo_rpd_studio/data/frozen/calo_v590_freeze.json"
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    assert payload["software_version"] == "5.9.0"
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "ba597eb" in readme
    assert "recoverable" in readme
    assert "5.9.0" in (root / "CHANGELOG.md").read_text(encoding="utf-8")
