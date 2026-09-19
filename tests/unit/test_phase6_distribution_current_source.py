from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from calo_rpd_studio.scripts.verify_phase6_distribution import GUI_MEMBERS, SDIST_MEMBERS, verify


ROOT = Path(__file__).resolve().parents[2]


def _stage(tmp_path, *, omit=None, extra=None, native=True):
    stage = tmp_path / "distribution"
    stage.mkdir()
    prefix = "calo_rpd_studio-12.0.0.dev1"
    members = GUI_MEMBERS | SDIST_MEMBERS
    if omit:
        members = members - {omit}
    if extra:
        members = members | {extra}
    with zipfile.ZipFile(stage / f"{prefix}-py3-none-any.whl", "w") as wheel:
        for member in sorted(members - SDIST_MEMBERS):
            wheel.writestr(member, b"# synthetic package member\n")
        entry = "[console_scripts]\n"
        if native:
            entry += "calo-rpd-native = calo_rpd_studio.app.application:main\n"
        wheel.writestr(f"{prefix}.dist-info/entry_points.txt", entry)
    with tarfile.open(stage / f"{prefix}.tar.gz", "w:gz") as sdist:
        for member in sorted(members):
            payload = b"# synthetic source member\n"
            info = tarfile.TarInfo(f"{prefix}/{member}")
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))
    return stage


def test_required_distribution_sources_exist_and_current_operating_docs_are_included():
    assert all((ROOT / member).is_file() for member in GUI_MEMBERS | SDIST_MEMBERS)
    assert SDIST_MEMBERS == {"Launch-CALO-RPD.ps1", "docs/NATIVE_WINDOWS_GUIDE.md"}
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for member in SDIST_MEMBERS:
        assert f"include {member}" in manifest


def test_current_distribution_contract_accepts_complete_synthetic_stage(tmp_path):
    assert verify(_stage(tmp_path))["passed"] is True


@pytest.mark.parametrize(
    "member",
    [
        "calo_rpd_studio/gui/panels/_independent_training_panel_core.py",
        "calo_rpd_studio/gui/panels/_calo_intelligence_panel_core.py",
        "calo_rpd_studio/algorithms/calo/_tsh_calo_training_extension_core.py",
        "docs/NATIVE_WINDOWS_GUIDE.md",
    ],
)
def test_current_distribution_contract_rejects_missing_runtime_or_operating_docs(tmp_path, member):
    with pytest.raises(ValueError, match="missing Phase 6 members"):
        verify(_stage(tmp_path, omit=member))


@pytest.mark.parametrize(
    "member",
    ["validation/Validate-CALO-Closure.ps1", "calo_rpd_studio/data/model.pt"],
)
def test_distribution_still_rejects_local_harness_and_policy_payloads(tmp_path, member):
    with pytest.raises(ValueError, match="local validation evidence|generated policy"):
        verify(_stage(tmp_path, extra=member))


def test_distribution_still_requires_direct_native_entry_point(tmp_path):
    with pytest.raises(ValueError, match="direct native GUI entry point"):
        verify(_stage(tmp_path, native=False))
