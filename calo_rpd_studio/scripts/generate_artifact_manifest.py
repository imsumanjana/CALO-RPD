"""Generate a deterministic SHA-256 manifest from a staged artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "calo_rpd_staged_artifact_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(stage: Path, *, output: Path | None = None) -> dict:
    """Describe regular files below *stage* without following symbolic links."""

    root = stage.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Artifact stage is not a directory: {root}")
    excluded = output.resolve() if output is not None else None
    artifacts: list[dict[str, int | str]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"Symbolic links are not permitted in an artifact stage: {candidate}")
        if not candidate.is_file() or (excluded is not None and candidate.resolve() == excluded):
            continue
        artifacts.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
        )
    if not artifacts:
        raise ValueError(f"Artifact stage contains no files: {root}")
    return {"schema": SCHEMA, "artifacts": artifacts}


def write_manifest(stage: Path, output: Path) -> Path:
    destination = output.resolve()
    payload = build_manifest(stage, output=destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    destination = write_manifest(arguments.stage, arguments.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
