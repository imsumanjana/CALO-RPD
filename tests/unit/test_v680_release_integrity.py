from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tomllib

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v680_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.8.0"
    assert RELEASE_NAME == metadata["release_name"] == "Independent CALO Intelligence & XPU Recovery"
    assert FREEZE_ID == "calo_v680_software_release"
    assert FREEZE_MANIFEST == "calo_v680_freeze.json"


def test_v680_freeze_verifies_new_scope():
    root = _root()
    freeze = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze, project_root=root)
    assert result.passed
    assert result.checked_files == 149
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    for key in (
        "v680_calo_intelligence_independent_validation",
        "v680_no_cross_tab_auto_rehydration",
        "v680_independent_policy_qualification_template",
        "v680_per_accelerator_backend_repair",
        "v680_xpu_sidecar_live_rediscovery",
        "v680_intel_pnp_hardware_tag_detection",
        "v680_detected_only_xpu_readiness_visibility",
    ):
        assert payload["frozen_scope"][key] is True, key


def test_v680_release_evidence_records_seven_closures():
    root = _root()
    for name in (
        "CALO-RPD-v6.8.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.8.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.8.0.csv",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    with (root / "FINDINGS_CLOSURE_v6.8.0.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    assert set(rows) == {f"V68-R{i:02d}" for i in range(1, 8)}
    assert all(row["status"] == "RESOLVED" for row in rows.values())


def test_v680_root_manifest_matches_every_packaged_file():
    root = _root()
    rows: dict[str, str] = {}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and not any(part in excluded_parts for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
    }
    assert set(rows) == actual
    for relative, expected in rows.items():
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert digest == expected, relative
