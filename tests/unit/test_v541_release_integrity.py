from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from calo_rpd_studio.algorithms.calo.ai_controller import AIController
from calo_rpd_studio.algorithms.registry import SPECS
from calo_rpd_studio.version import FREEZE_MANIFEST, VERSION

pytestmark = pytest.mark.skipif(VERSION != "5.4.1", reason="historical v5.4.1 release gate")


def test_release_identity_matches_pyproject_and_current_freeze_name():
    root = Path(__file__).resolve().parents[2]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["version"] == VERSION == "5.4.1"
    assert FREEZE_MANIFEST == "calo_v541_freeze.json"


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


def test_current_release_freeze_verifies_and_does_not_freeze_a_default_policy():
    from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest

    root = Path(__file__).resolve().parents[2]
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed is True
    payload = __import__("json").loads(freeze_path.read_text(encoding="utf-8"))
    assert not any("data/trained_models/" in item for item in payload["files"])
    assert payload["frozen_scope"]["policy_gated_no_default_neural_policy"] is True
    assert payload["frozen_scope"]["untrained_policy_fallback_forbidden"] is True
