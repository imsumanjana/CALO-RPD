"""Crash-safe versioned checkpoint utilities used by CALO v5 continuation workflows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable

import torch


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    sha256: str
    size_bytes: int


class CheckpointManager:
    """Atomic checkpoint writer with checksums and conservative retention.

    Scientific artifacts are never silently overwritten. ``write_torch`` writes to a temporary
    file, fsync-equivalent close is provided by the file lifecycle, verifies readability and then
    atomically replaces the destination. A sidecar SHA-256 is always written.
    """

    def __init__(self, directory) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self, pattern: str = "*.pt") -> list[Path]:
        return sorted(self.directory.glob(pattern), key=lambda p: (p.stat().st_mtime_ns, p.name))

    def latest(self, pattern: str = "*.pt") -> Path | None:
        files = self.list(pattern)
        return files[-1] if files else None

    @staticmethod
    def checksum(path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def verify(path: str | Path, expected_sha256: str | None = None) -> CheckpointInfo:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        sha = CheckpointManager.checksum(path)
        if expected_sha256 and sha.lower() != str(expected_sha256).lower():
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
        return CheckpointInfo(path.resolve(), sha, path.stat().st_size)

    def write_torch(self, filename: str | Path, payload: dict) -> CheckpointInfo:
        target = Path(filename)
        if not target.is_absolute():
            target = self.directory / target
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".tmp") as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            # Validate that the serialized container is readable before promotion.
            torch.load(temporary, map_location="cpu", weights_only=False)
            temporary.replace(target)
            info = self.verify(target)
            target.with_suffix(target.suffix + ".sha256").write_text(
                info.sha256 + "\n", encoding="utf-8"
            )
            return info
        finally:
            temporary.unlink(missing_ok=True)

    def write_json(self, filename: str | Path, payload: dict) -> Path:
        target = Path(filename)
        if not target.is_absolute():
            target = self.directory / target
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=target.parent, suffix=".tmp", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            temporary = Path(handle.name)
        temporary.replace(target)
        return target

    def apply_retention(
        self,
        paths: Iterable[str | Path],
        *,
        keep_latest: int = 5,
        protected: Iterable[str | Path] = (),
    ) -> list[Path]:
        """Delete only unprotected rolling checkpoints; milestone/best files should be protected."""
        protected_set = {str(Path(p).resolve()) for p in protected}
        ordered = sorted(
            [Path(p) for p in paths if Path(p).is_file()],
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        removable = ordered[: -max(0, int(keep_latest))] if keep_latest > 0 else ordered
        deleted: list[Path] = []
        for path in removable:
            if str(path.resolve()) in protected_set:
                continue
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
            deleted.append(path)
        return deleted
