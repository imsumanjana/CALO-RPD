from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from calo_rpd_studio.scripts.validate_packaged_gui import validate_packaged_gui


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None,
    reason="PyQt6 is not installed",
)


def test_packaged_gui_validator_rejects_source_checkout_import(tmp_path):
    import calo_rpd_studio

    checkout = Path(calo_rpd_studio.__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="forbidden checkout root"):
        validate_packaged_gui(tmp_path, forbidden_import_root=checkout)


def test_packaged_gui_validator_renders_and_retains_self_describing_evidence(tmp_path):
    report = validate_packaged_gui(tmp_path)

    screenshot = tmp_path / report["screenshot"]
    report_path = tmp_path / "packaged-gui-evidence.json"
    assert report["schema_version"] == "calo-rpd-packaged-gui-evidence-v1"
    assert report["distribution_name"] == "calo-rpd-studio"
    assert report["source_checkout_imported"] is None
    assert report["initial_workspace"] == "dashboard"
    assert report["workspace_count"] == 16
    assert report["visible_forbidden_hits"] == []
    assert report["screenshot_width"] >= 1120
    assert report["screenshot_height"] >= 720
    assert screenshot.stat().st_size > 10_000
    assert len(report["screenshot_sha256"]) == 64
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validate_packaged_gui(tmp_path)
