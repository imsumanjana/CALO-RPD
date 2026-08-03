"""Fail closed if a staged wheel/sdist contains generated or unsafe files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile


_REQUIRED_SOURCE_FILES = {
    "calo_rpd_studio/algorithms/cma_es.py",
    "calo_rpd_studio/algorithms/lshade.py",
    "calo_rpd_studio/scripts/container_smoke.py",
    "calo_rpd_studio/scripts/generate_artifact_manifest.py",
    "calo_rpd_studio/scripts/verify_distribution_stage.py",
    "calo_rpd_studio/scripts/verify_requirements_lock.py",
}


def _validate_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe distribution member path: {name!r}")
    lowered = normalized.lower()
    if "__pycache__" in path.parts or lowered.endswith((".pyc", ".pyo", ".pt", ".pt.sha256")):
        raise ValueError(f"Generated runtime artifact is packaged: {name!r}")
    marker = "calo_rpd_studio/data/trained_models/"
    if marker in lowered and not lowered.endswith(f"{marker}__init__.py"):
        raise ValueError(f"Generated policy/training data is packaged: {name!r}")
    return path


def _source_relative(name: str, *, sdist: bool) -> str:
    parts = _validate_member(name).parts
    if sdist:
        if len(parts) < 2:
            return ""
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def verify_stage(stage: Path, *, project: Path) -> dict:
    root = stage.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Distribution stage is not a directory: {root}")
    wheel_files = sorted(root.glob("*.whl"))
    source_files = sorted(root.glob("*.tar.gz"))
    selected = {*wheel_files, *source_files}
    other_files = sorted(
        item.name for item in root.iterdir() if item.is_file() and item not in selected
    )
    if len(wheel_files) != 1 or len(source_files) != 1 or other_files:
        raise ValueError(
            "Fresh distribution stage must contain exactly one wheel and one sdist; "
            f"wheels={len(wheel_files)}, sdists={len(source_files)}, other={other_files}"
        )

    version = str(tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"])
    expected_prefix = f"calo_rpd_studio-{version}"
    if not wheel_files[0].name.startswith(expected_prefix) or source_files[0].name != (
        f"{expected_prefix}.tar.gz"
    ):
        raise ValueError("Distribution filenames do not match the project name/version")

    with zipfile.ZipFile(wheel_files[0]) as archive:
        wheel_names = archive.namelist()
        wheel_sources = {_source_relative(name, sdist=False) for name in wheel_names}
    with tarfile.open(source_files[0], mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Distribution contains a symbolic/hard link: {member.name!r}")
        source_names = archive.getnames()
        source_sources = {_source_relative(name, sdist=True) for name in source_names}

    for label, members in (("wheel", wheel_sources), ("sdist", source_sources)):
        missing = sorted(_REQUIRED_SOURCE_FILES - members)
        if missing:
            raise ValueError(f"{label} is missing required source files: {missing}")
    return {
        "schema": "calo_rpd_distribution_stage_verification_v1",
        "project_version": version,
        "wheel": wheel_files[0].name,
        "wheel_members": len(wheel_names),
        "sdist": source_files[0].name,
        "sdist_members": len(source_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    arguments = parser.parse_args()
    report = verify_stage(arguments.stage, project=arguments.project)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
