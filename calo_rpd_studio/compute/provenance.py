"""Durable, hash-chained compute-protection provenance for CALO-RPD v6.2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import threading
from typing import Any


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ProvenanceEvent:
    sequence: int
    timestamp_utc: str
    event_type: str
    payload: dict
    previous_hash: str
    event_hash: str

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }


class ComputeProvenanceRecorder:
    """Append-only JSONL event chain with fsync durability.

    The recorder is intentionally simple: each event hashes the canonical event body plus the previous
    event hash. It provides tamper-evident ordering, not cryptographic signer authenticity.
    """

    def __init__(self, path: str | Path, *, session_id: str, metadata: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self._lock = threading.Lock()
        self._sequence = 0
        self._previous_hash = ""
        if self.path.is_file():
            self._restore_tail()
        if metadata is not None and self._sequence == 0:
            self.append("SESSION_STARTED", dict(metadata))

    def _restore_tail(self) -> None:
        try:
            lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                return
            last = json.loads(lines[-1])
            self._sequence = int(last.get("sequence", 0))
            self._previous_hash = str(last.get("event_hash", "") or "")
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            # Fail closed: do not append to a provenance chain whose tail cannot be verified.
            raise RuntimeError(f"Cannot restore compute provenance chain: {self.path}")

    def append(self, event_type: str, payload: dict[str, Any]) -> ProvenanceEvent:
        with self._lock:
            sequence = self._sequence + 1
            timestamp = datetime.now(timezone.utc).isoformat()
            body = {
                "sequence": sequence,
                "timestamp_utc": timestamp,
                "session_id": self.session_id,
                "event_type": str(event_type),
                "payload": dict(payload),
                "previous_hash": self._previous_hash,
            }
            event_hash = hashlib.sha256(_canonical(body)).hexdigest()
            row = {**body, "event_hash": event_hash}
            encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            self._sequence = sequence
            self._previous_hash = event_hash
            return ProvenanceEvent(
                sequence=sequence,
                timestamp_utc=timestamp,
                event_type=str(event_type),
                payload=dict(payload),
                previous_hash=body["previous_hash"],
                event_hash=event_hash,
            )

    @staticmethod
    def verify(path: str | Path) -> dict:
        source = Path(path)
        previous = ""
        checked = 0
        if not source.is_file():
            return {"ok": False, "checked": 0, "error": "missing"}
        try:
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                event_hash = str(row.pop("event_hash", "") or "")
                if str(row.get("previous_hash", "") or "") != previous:
                    return {"ok": False, "checked": checked, "error": "previous_hash_mismatch"}
                actual = hashlib.sha256(_canonical(row)).hexdigest()
                if actual != event_hash:
                    return {"ok": False, "checked": checked, "error": "event_hash_mismatch"}
                previous = event_hash
                checked += 1
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return {"ok": False, "checked": checked, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "checked": checked, "tail_hash": previous}


__all__ = ["ProvenanceEvent", "ComputeProvenanceRecorder"]
