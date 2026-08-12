"""Generate distinct immutable wheel and sdist member manifests from a fresh stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import zipfile


SCHEMA = "calo-v12-distribution-member-manifest-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError(f"Unsafe distribution member: {name!r}")
    return normalized


def wheel_manifest(path: Path) -> dict:
    artifact = path.resolve(strict=True)
    members = []
    with zipfile.ZipFile(artifact) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            name = _safe_name(info.filename)
            if info.is_dir():
                continue
            payload = archive.read(info)
            members.append(
                {"path": name, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
            )
    if not members:
        raise ValueError("Wheel contains no regular-file members")
    return {
        "schema": SCHEMA,
        "artifact_kind": "wheel",
        "artifact_name": artifact.name,
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256_file(artifact),
        "member_count": len(members),
        "members": members,
    }


def sdist_manifest(path: Path) -> dict:
    artifact = path.resolve(strict=True)
    members = []
    with tarfile.open(artifact, mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            name = _safe_name(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Distribution contains a symbolic/hard link: {name!r}")
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read sdist member: {name!r}")
            payload = stream.read()
            members.append(
                {"path": name, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
            )
    if not members:
        raise ValueError("Source distribution contains no regular-file members")
    return {
        "schema": SCHEMA,
        "artifact_kind": "sdist",
        "artifact_name": artifact.name,
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256_file(artifact),
        "member_count": len(members),
        "members": members,
    }


def write_exclusive(path: Path, payload: dict) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite distribution manifest: {destination}"
        ) from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    stage = arguments.stage.resolve(strict=True)
    wheels = list(stage.glob("*.whl"))
    sdists = list(stage.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Distribution stage must contain exactly one wheel and one sdist")
    output = arguments.output_directory.resolve()
    wheel_output = write_exclusive(output / "wheel-manifest.json", wheel_manifest(wheels[0]))
    sdist_output = write_exclusive(output / "sdist-manifest.json", sdist_manifest(sdists[0]))
    print(wheel_output)
    print(sdist_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
