from __future__ import annotations

from pathlib import Path
import json

import pytest

from calo_rpd_studio.algorithms.calo.ai_controller import AIController
from calo_rpd_studio.algorithms.registry import SPECS


def test_historical_v580_freeze_artifact_is_preserved():
    """v5.8 freeze is historical evidence, not the current-source release gate."""
    root = Path(__file__).resolve().parents[2]
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / "calo_v580_freeze.json"
    assert freeze_path.is_file()
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert payload["software_version"] == "5.8.0"
    assert payload.get("manifest_sha256")


def test_release_does_not_bundle_or_assume_a_default_neural_policy():
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "calo_rpd_studio" / "data" / "trained_models"
    deployable = [path for path in model_dir.glob("*.pt") if not path.name.endswith(".resume.pt")]
    assert deployable == []
    assert SPECS["CALO"].default_parameters["use_ai"] is True
    assert SPECS["CALO"].default_parameters["strict_policy_binding"] is True


def test_missing_policy_never_constructs_an_untrained_fallback(tmp_path):
    with pytest.raises(RuntimeError, match="fail-closed"):
        AIController(None, seed=7, device="cpu")
    with pytest.raises(FileNotFoundError, match="explicitly imported/trained"):
        AIController(tmp_path / "missing.pt", seed=7, device="cpu")


def test_historical_v580_freeze_did_not_bundle_a_default_policy():
    root = Path(__file__).resolve().parents[2]
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / "calo_v580_freeze.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert not any("data/trained_models/" in item for item in payload["files"])
    assert payload["frozen_scope"]["policy_gated_no_default_neural_policy"] is True
    assert payload["frozen_scope"]["untrained_policy_fallback_forbidden"] is True
