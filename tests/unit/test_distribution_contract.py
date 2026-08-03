from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_generated_policy_artifacts_are_excluded_from_wheel_and_sdist():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]["calo_rpd_studio"]
    assert not any("trained_models" in pattern for pattern in package_data)
    exclusions = project["tool"]["setuptools"]["exclude-package-data"][
        "calo_rpd_studio.data.trained_models"
    ]
    assert {"*.pt", "*.pt.sha256", "*.json", "**/*"} <= set(exclusions)
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-exclude calo_rpd_studio/data/trained_models *" in manifest
    assert "include calo_rpd_studio/data/trained_models/__init__.py" in manifest
