"""CALO serialization helpers with explicit safe-model vs trusted-local-resume boundaries."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
from typing import Any

import torch

_TRUST_SCHEMA = "calo-local-resume-trust-v1"
_TRUST_DIR = Path.home() / ".calo_rpd_studio"
_TRUST_KEY = _TRUST_DIR / "resume_trust.key"


def checkpoint_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint_hash(path, expected_sha256: str | None = None) -> str:
    source = Path(path)
    actual = checkpoint_sha256(source)
    if expected_sha256 and actual.lower() != str(expected_sha256).strip().lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {source.name}: expected {expected_sha256}, got {actual}"
        )
    return actual


def _validate_model_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("CALO model checkpoint must contain a dictionary payload")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError("CALO model checkpoint has no model_state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("CALO model_state_dict must contain only named tensors")
    return payload


def load_checkpoint(path, *, expected_sha256: str | None = None, map_location="cpu") -> dict:
    """Load a portable/deployable model with PyTorch's restricted weights-only loader."""
    source = Path(path)
    verify_checkpoint_hash(source, expected_sha256)
    payload = torch.load(source, map_location=map_location, weights_only=True)
    return _validate_model_payload(payload)


def trusted_resume_sha_path(path) -> Path:
    # Retain the historic .sha256 suffix for compatibility with file discovery, but contents are
    # authenticated JSON rather than a bare self-asserted hash.
    return Path(path).with_suffix(Path(path).suffix + ".sha256")


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability after atomic replace (supported on POSIX)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # The file itself remains fsynced; some filesystems/platforms do not permit dir fsync.
        return


def durable_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes by fsync + atomic replace + best-effort parent-directory fsync."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass



def durable_torch_save(payload: Any, path: str | Path) -> None:
    """Durably save a trusted-local torch payload before any trust sidecar is emitted."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

def _load_or_create_local_trust_key() -> bytes:
    _TRUST_DIR.mkdir(parents=True, exist_ok=True)
    if _TRUST_KEY.is_file():
        key = _TRUST_KEY.read_bytes()
        if len(key) < 32:
            raise RuntimeError("CALO local resume trust key is invalid")
        return key
    key = secrets.token_bytes(32)
    durable_write_bytes(_TRUST_KEY, key)
    try:
        os.chmod(_TRUST_KEY, 0o600)
    except OSError:
        pass
    return key


def _resume_signature(digest: str) -> str:
    return hmac.new(_load_or_create_local_trust_key(), digest.encode("ascii"), hashlib.sha256).hexdigest()


def write_trusted_resume_hash(path) -> str:
    """Authenticate an application-created exact-resume artifact with a machine-local HMAC sidecar."""
    source = Path(path)
    digest = checkpoint_sha256(source)
    payload = {
        "schema": _TRUST_SCHEMA,
        "sha256": digest,
        "hmac_sha256": _resume_signature(digest),
        "trust_boundary": "trusted_local_exact_resume_only",
    }
    durable_write_bytes(
        trusted_resume_sha_path(source),
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return digest


def load_trusted_resume(path, *, map_location="cpu"):
    """Load an application-created exact-resume checkpoint only after local authenticity verification.

    Exact resumes contain optimizer/RNG Python objects and therefore require pickle-capable deserialization.
    A bare SHA sidecar proves integrity but not origin. v5.7+ requires an HMAC signed by a machine-local secret,
    preventing arbitrary downloaded pickle files from becoming trusted merely by shipping their own hash.
    Portable/imported artifacts must use :func:`load_checkpoint` (weights_only=True); users must create a new
    Base-Guided Fork rather than importing an untrusted exact-resume pickle.
    """
    source = Path(path).expanduser().resolve()
    sidecar = trusted_resume_sha_path(source)
    if not sidecar.is_file():
        raise ValueError("Exact-resume checkpoint lacks an authenticated local trust sidecar")
    try:
        trust = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            "Legacy/bare-hash resume sidecars are not trusted in v5.7+; use a deployable policy import and Base-Guided Fork"
        ) from exc
    if str(trust.get("schema", "")) != _TRUST_SCHEMA:
        raise ValueError("Unsupported or unauthenticated exact-resume trust schema")
    expected = str(trust.get("sha256", "")).strip().lower()
    signature = str(trust.get("hmac_sha256", "")).strip().lower()
    if not expected or not signature:
        raise ValueError("Incomplete exact-resume trust sidecar")
    actual = verify_checkpoint_hash(source, expected)
    expected_signature = _resume_signature(actual)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError(
            "Exact-resume artifact is not authenticated as locally created; unsafe deserialization refused"
        )
    return torch.load(source, map_location=map_location, weights_only=False)  # nosec B614 -- HMAC-authenticated local state


def migrate_legacy_local_resume(
    path: str | Path,
    *,
    target: str | Path | None = None,
    explicit_trust: bool = False,
    map_location: str = "cpu",
) -> Path:
    """One-time migration of a *locally trusted* pre-v5.7 exact-resume checkpoint.

    Legacy exact resumes contain pickle-capable optimizer/RNG state. A bare SHA-256 sidecar proves
    integrity, not authorship. Therefore migration is deliberately opt-in: callers must explicitly
    assert that the file is a trusted local artifact. The legacy digest is verified before unsafe
    deserialization, and the migrated copy is re-saved with the v5.8 machine-local HMAC trust
    boundary. The original file is never modified.
    """
    if not explicit_trust:
        raise PermissionError(
            "Legacy exact-resume migration requires explicit_trust=True after the user confirms the checkpoint is a trusted local artifact"
        )
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    sidecar = trusted_resume_sha_path(source)
    if not sidecar.is_file():
        raise ValueError("Legacy resume has no integrity sidecar; migration refused")
    text = sidecar.read_text(encoding="utf-8").strip()
    expected = ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Already authenticated/current files do not need migration.
            if str(parsed.get("schema", "")) == _TRUST_SCHEMA:
                load_trusted_resume(source, map_location=map_location)
                return source
            expected = str(parsed.get("sha256", "") or "").strip()
    except json.JSONDecodeError:
        expected = text.split()[0] if text else ""
    if len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
        raise ValueError("Legacy resume integrity sidecar does not contain a valid SHA-256 digest")
    verify_checkpoint_hash(source, expected)
    # nosec B614 -- explicit user trust + verified legacy digest; migration is the sole compatibility boundary.
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError("Legacy file is not a recognized CALO exact-training resume payload")
    destination = (
        Path(target).expanduser().resolve()
        if target is not None
        else source.with_name(source.stem + ".v58.resume.pt")
    )
    migrated = dict(payload)
    extra = dict(migrated.get("extra", {}) or {})
    extra["legacy_resume_migration"] = {
        "source_path": str(source),
        "source_sha256": expected.lower(),
        "trust_assertion": "explicit_user_confirmed_trusted_local_artifact",
        "migrated_to_trust_schema": _TRUST_SCHEMA,
    }
    migrated["extra"] = extra
    durable_torch_save(migrated, destination)
    write_trusted_resume_hash(destination)
    return destination
