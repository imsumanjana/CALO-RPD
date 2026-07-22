"""Durability-hardened versioned checkpoint utilities used by CALO continuation workflows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from calo_rpd_studio.ai.model_io import durable_torch_save, durable_write_bytes


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    sha256: str
    size_bytes: int


class CheckpointManager:
    """Portable/deployable model checkpoint manager.

    This API is intentionally *not* the exact-resume API. Payloads must remain readable by
    ``torch.load(..., weights_only=True)``. Exact optimizer/RNG resume states use the separate
    authenticated-local resume helpers in :mod:`calo_rpd_studio.ai.model_io`.
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

    def write_deployable_model(self, filename: str | Path, payload: dict) -> CheckpointInfo:
        """Durably write a portable weights-only-safe checkpoint and checksum sidecar."""
        target = Path(filename)
        if not target.is_absolute():
            target = self.directory / target
        durable_torch_save(payload, target)
        # Portable/deployable artifacts must never require pickle-capable loading.
        torch.load(target, map_location="cpu", weights_only=True)
        info = self.verify(target)
        durable_write_bytes(
            target.with_suffix(target.suffix + ".sha256"),
            (info.sha256 + "\n").encode("ascii"),
        )
        return info

    def write_torch(self, filename: str | Path, payload: dict) -> CheckpointInfo:
        """Backward-compatible alias for :meth:`write_deployable_model`; not an exact-resume API."""
        return self.write_deployable_model(filename, payload)

    def write_json(self, filename: str | Path, payload: dict) -> Path:
        target = Path(filename)
        if not target.is_absolute():
            target = self.directory / target
        encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
        durable_write_bytes(target, encoded)
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
