from __future__ import annotations

from pathlib import Path
import json


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_historical_v590_freeze_and_evidence_are_retained_as_history():
    """v6 keeps v5.9 release evidence, but it is no longer the current software freeze."""
    root = _root()
    freeze = root / "calo_rpd_studio" / "data" / "frozen" / "calo_v590_freeze.json"
    assert freeze.is_file()
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    assert payload["software_version"] == "5.9.0"
    for name in (
        "FINDINGS_CLOSURE_v5.9.0.csv",
        "CALO-RPD-v5.9.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v5.9.0_DEEP_POST_GENERATION_AUDIT.txt",
    ):
        assert (root / name).is_file(), name
