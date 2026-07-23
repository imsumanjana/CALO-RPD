from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib

import pytest

from calo_rpd_studio.algorithms.calo.ai_controller import AIController
from calo_rpd_studio.algorithms.registry import SPECS
from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION

pytestmark = pytest.mark.skipif(VERSION != "6.0.0a4", reason="historical v6.0 alpha release gate")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v600a4_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.0.0a4"
    assert RELEASE_NAME == metadata["release_name"] == "Policy-First Workflow and Safe-80 Compute Foundation"
    assert FREEZE_ID == "calo_v600a4_software_release"
    assert FREEZE_MANIFEST == "calo_v600a4_freeze.json"


def test_v600a4_freeze_verifies_and_covers_alpha_architecture():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed is True
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert payload["software_version"] == "6.0.0a4"
    required = {
        "calo_rpd_studio/app/workspaces.py",
        "calo_rpd_studio/app/workflow_manager.py",
        "calo_rpd_studio/app/main_window.py",
        "calo_rpd_studio/compute/topology.py",
        "calo_rpd_studio/algorithms/calo/policy_readiness.py",
        "calo_rpd_studio/gui/panels/dashboard_panel.py",
        "calo_rpd_studio/gui/panels/calo_intelligence_panel.py",
        "calo_rpd_studio/algorithms/calo/competitive_training.py",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    assert scope["key_based_workspace_navigation"] is True
    assert scope["legacy_v59_workspace_index_migration"] is True
    assert scope["policy_first_governing_intelligence_gate"] is True
    assert scope["dashboard_compute_topology_map"] is True
    assert scope["safe80_resource_budget_engine"] is True
    assert scope["safe80_global_cpu_budget"] is True
    assert scope["safe80_hard_parallel_branch_ceiling"] is True
    assert scope["automatic_accelerator_to_cpu_branch_spillover"] is False
    # Deliberate alpha boundaries: these are later beta/RC items, not silently claimed complete.
    assert scope["xpu_sidecar_full_branch_certified"] is False
    assert scope["dynamic_thermal_power_governor"] is False
    assert scope["queued_total_vs_concurrent_branch_scheduler"] is False
    assert scope["global_training_exclusive_ui_lock"] is False


def test_v600a4_keeps_native_v59_policy_abi_and_no_default_neural_policy():
    root = _root()
    model_dir = root / "calo_rpd_studio" / "data" / "trained_models"
    deployable = [path for path in model_dir.glob("*.pt") if not path.name.endswith(".resume.pt")]
    assert deployable == []
    assert SPECS["CALO"].default_parameters["use_ai"] is True
    assert SPECS["CALO"].default_parameters["strict_policy_binding"] is True
    with pytest.raises(RuntimeError, match="fail-closed"):
        AIController(None, seed=11, device="cpu")


def test_v600a4_release_evidence_files_are_current():
    root = _root()
    for name in (
        "CALO-RPD-v6.0.0a4_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.0.0a4_ALPHA_AUDIT.txt",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "CALO-RPD Studio 6.0.0a4" in readme
    assert "v6.0-alpha1" in readme
    assert "v6.0-alpha4" in readme


def test_root_manifest_matches_every_packaged_file_and_current_freeze():
    root = _root()
    manifest = root / "MANIFEST.sha256"
    assert manifest.is_file()
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    for relative, expected in rows.items():
        path = root / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    assert "calo_rpd_studio/data/frozen/calo_v600a4_freeze.json" in rows
    assert "tests/unit/test_v600_alpha_architecture.py" in rows
    assert "tests/unit/test_v600_release_integrity.py" in rows
