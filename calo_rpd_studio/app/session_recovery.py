"""Durable application-session recovery metadata.

The session journal is intentionally conservative: it restores navigation intent and identifiers only.
It never bypasses WorkflowManager gates, never mutates scientific artifacts, and never treats a stale
session snapshot as authoritative optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid


SESSION_RECOVERY_SCHEMA = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class SessionRecoverySnapshot:
    session_id: str
    started_at: str
    updated_at: str
    clean_shutdown: bool
    workspace_ui: dict[str, Any]
    experiment_id: str
    policy_training_active: bool
    governing_policy_sha256: str
    compute_profile_fingerprint: str
    payload_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SESSION_RECOVERY_SCHEMA,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "clean_shutdown": self.clean_shutdown,
            "workspace_ui": dict(self.workspace_ui),
            "experiment_id": self.experiment_id,
            "policy_training_active": self.policy_training_active,
            "governing_policy_sha256": self.governing_policy_sha256,
            "compute_profile_fingerprint": self.compute_profile_fingerprint,
            "payload_sha256": self.payload_sha256,
        }


class SessionRecoveryJournal:
    """Maintain one crash-detection snapshot for the GUI/application session."""

    def __init__(self, root: str | Path = "results_data/session_recovery") -> None:
        self.root = Path(root)
        self.path = self.root / "latest_session.json"
        self.session_id = uuid.uuid4().hex
        self.started_at = _utc_now()

    @staticmethod
    def _seal(payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.pop("payload_sha256", None)
        body["payload_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        return body

    @staticmethod
    def verify_payload(payload: dict[str, Any]) -> bool:
        expected = str(payload.get("payload_sha256", "") or "")
        if not expected:
            return False
        body = dict(payload)
        body.pop("payload_sha256", None)
        actual = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        return actual == expected

    def previous_unclean(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or not self.verify_payload(payload):
            return None
        if bool(payload.get("clean_shutdown", False)):
            return None
        return payload

    def begin(self, *, workspace_ui: dict[str, Any] | None = None) -> None:
        self.update(workspace_ui=workspace_ui or {}, clean_shutdown=False)

    def update(
        self,
        *,
        workspace_ui: dict[str, Any] | None = None,
        experiment_id: str = "",
        policy_training_active: bool = False,
        governing_policy_sha256: str = "",
        compute_profile_fingerprint: str = "",
        clean_shutdown: bool = False,
    ) -> dict[str, Any]:
        payload = self._seal(
            {
                "schema": SESSION_RECOVERY_SCHEMA,
                "session_id": self.session_id,
                "started_at": self.started_at,
                "updated_at": _utc_now(),
                "clean_shutdown": bool(clean_shutdown),
                "workspace_ui": dict(workspace_ui or {}),
                "experiment_id": str(experiment_id or ""),
                "policy_training_active": bool(policy_training_active),
                "governing_policy_sha256": str(governing_policy_sha256 or ""),
                "compute_profile_fingerprint": str(compute_profile_fingerprint or ""),
            }
        )
        _atomic_json(self.path, payload)
        return payload

    def mark_clean(self, **kwargs: Any) -> dict[str, Any]:
        return self.update(clean_shutdown=True, **kwargs)
