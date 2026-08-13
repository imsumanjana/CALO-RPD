"""Verify Phase 6 native entry points and GUI modules in one staged wheel/sdist pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


GUI_MEMBERS = {
    "calo_rpd_studio/gui/command_registry.py",
    "calo_rpd_studio/gui/widgets/ribbon_bar.py",
    "calo_rpd_studio/gui/widgets/context_pane.py",
    "calo_rpd_studio/gui/widgets/document_workspace.py",
    "calo_rpd_studio/gui/widgets/activity_center.py",
    "calo_rpd_studio/gui/panels/independent_training_panel.py",
    "calo_rpd_studio/scripts/validate_phase6_gui_contracts.py",
    "calo_rpd_studio/scripts/verify_phase6_distribution.py",
    "calo_rpd_studio/scripts/train_tsh_calo.py",
}
SDIST_MEMBERS = {
    "Launch-CALO-RPD.ps1",
    "docs/NATIVE_WINDOWS_GUIDE.md",
    "docs/implementation/PHASE_6_NEW_CHAT_PROMPT.md",
    "docs/implementation/PHASE_6_EXACT_CONTINUATION_PROMPT.md",
}


def _reject_local_evidence_or_policy_artifacts(names: set[str], *, label: str) -> None:
    artifact_suffixes = (
        ".pt",
        ".pt.sha256",
        ".pth",
        ".ckpt",
        ".onnx",
        ".safetensors",
        ".branches.json",
    )
    for name in names:
        normalized = PurePosixPath(name).as_posix().lower()
        parts = PurePosixPath(normalized).parts
        if parts and parts[0] in {"validation", "validation_logs"}:
            raise ValueError(f"{label} contains local validation evidence: {name}")
        marker = "calo_rpd_studio/data/trained_models/"
        if marker in normalized and not normalized.endswith(f"{marker}__init__.py"):
            raise ValueError(f"{label} contains generated policy/training data: {name}")
        if normalized.endswith(artifact_suffixes):
            raise ValueError(f"{label} contains a generated policy/runtime artifact: {name}")


def _source_name(name: str, *, sdist: bool) -> str:
    parts = PurePosixPath(name).parts
    if sdist:
        if len(parts) < 2:
            return ""
        parts = parts[1:]
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe distribution member: {name!r}")
    return PurePosixPath(*parts).as_posix()


def verify(stage: Path) -> dict:
    wheels = sorted(stage.glob("*.whl"))
    sdists = sorted(stage.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Phase 6 stage must contain exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = {_source_name(name, sdist=False) for name in archive.namelist()}
        entry_points_name = next(
            (name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")),
            "",
        )
        if not entry_points_name:
            raise ValueError("Wheel has no entry_points.txt")
        entry_points = archive.read(entry_points_name).decode("utf-8")
    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = {_source_name(name, sdist=True) for name in archive.getnames()}
    _reject_local_evidence_or_policy_artifacts(wheel_names, label="Wheel")
    _reject_local_evidence_or_policy_artifacts(sdist_names, label="Sdist")
    missing_wheel = sorted(GUI_MEMBERS - wheel_names)
    missing_sdist = sorted((GUI_MEMBERS | SDIST_MEMBERS) - sdist_names)
    if missing_wheel:
        raise ValueError(f"Wheel is missing Phase 6 members: {missing_wheel}")
    if missing_sdist:
        raise ValueError(f"Sdist is missing Phase 6 members: {missing_sdist}")
    expected_entry = "calo-rpd-native = calo_rpd_studio.app.application:main"
    if expected_entry not in entry_points:
        raise ValueError("Wheel does not expose the direct native GUI entry point")
    return {
        "schema": "calo-rpd-phase6-distribution-verification-v1",
        "passed": True,
        "wheel": wheels[0].name,
        "sdist": sdists[0].name,
        "native_entry_point": expected_entry,
        "wheel_phase6_members": sorted(GUI_MEMBERS),
        "sdist_phase6_members": sorted(GUI_MEMBERS | SDIST_MEMBERS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = verify(arguments.stage.resolve())
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        report = {
            "schema": "calo-rpd-phase6-distribution-verification-v1",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
