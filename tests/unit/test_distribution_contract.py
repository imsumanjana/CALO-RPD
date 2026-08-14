from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from calo_rpd_studio.scripts.verify_distribution_stage import _validate_member


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
    assert "recursive-exclude validation *" in manifest
    assert "recursive-exclude validation_logs *" in manifest
    for suffix in ("*.pt", "*.pt.sha256", "*.pth", "*.ckpt", "*.onnx", "*.safetensors"):
        assert suffix in manifest
    assert "*.branches.json" in manifest


def test_distribution_member_gate_rejects_validation_and_generated_training_formats():
    for member in (
        "validation/Validate-Phase4.ps1",
        "calo_rpd_studio-12.0.0.dev1/validation/logs/phase4.txt",
        "validation_logs/phase4/log.txt",
        "calo_rpd_studio/data/model.pth",
        "calo_rpd_studio/data/model.ckpt",
        "calo_rpd_studio/data/model.onnx",
        "calo_rpd_studio/data/model.safetensors",
        "calo_rpd_studio/data/candidate.branches.json",
        "calo_rpd_studio/data/candidate_branches/receipt.json",
    ):
        with pytest.raises(ValueError):
            _validate_member(member)


def test_distribution_member_gate_allows_application_validation_package():
    for member in (
        "calo_rpd_studio/validation/__init__.py",
        "calo_rpd_studio/validation/gui_contract.py",
        "calo_rpd_studio-12.0.0.dev1/calo_rpd_studio/validation/gui_contract.py",
    ):
        assert _validate_member(member).as_posix() == member


def test_distribution_verifier_requires_release_critical_reference_and_gui_commands():
    verifier = (ROOT / "calo_rpd_studio" / "scripts" / "verify_distribution_stage.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "calo_rpd_studio/orpd/mathematical_reference.py",
        "calo_rpd_studio/scripts/run_mathematical_reference.py",
        "calo_rpd_studio/scripts/validate_packaged_gui.py",
        "calo_rpd_studio/scripts/create_development_freeze_candidate.py",
        "calo_rpd_studio/scripts/accept_development_freeze.py",
        "calo_rpd_studio/scripts/train_tsh_calo.py",
        "calo_rpd_studio/algorithms/calo/tsh_calo_training_extension.py",
        "calo_rpd_studio/validation/gui_contract.py",
    ):
        assert required in verifier
