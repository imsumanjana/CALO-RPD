"""Resolve an immutable source identity without weakening dirty-checkout safeguards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess


SOURCE_IDENTITY_SCHEMA = "calo-rpd-built-source-identity-v1"
DEFAULT_DECLARATION_PATH = Path("/opt/calo/.calo-source-identity.json")
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Source identity selected from Git or an immutable image-build declaration."""

    source_commit: str
    tracked_source_clean: bool
    source_identity_kind: str
    declaration_path: str = ""

    @property
    def durable_evidence_eligible(self) -> bool:
        return bool(_FULL_COMMIT.fullmatch(self.source_commit) and self.tracked_source_clean)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["durable_evidence_eligible"] = self.durable_evidence_eligible
        return payload


def _run_git(*arguments: str, cwd: str | Path | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _git_identity(cwd: str | Path | None = None) -> SourceIdentity | None:
    """Return Git identity when inside a worktree; never fall back around Git errors."""

    try:
        probe = _run_git("rev-parse", "--is-inside-work-tree", cwd=cwd)
    except FileNotFoundError:
        return None
    if probe.returncode != 0:
        combined = f"{probe.stdout}\n{probe.stderr}".lower()
        if "not a git repository" in combined:
            return None
        raise RuntimeError(f"Unable to determine Git worktree status: {probe.stderr.strip()}")
    if probe.stdout.strip().lower() != "true":
        return None
    commit_result = _run_git("rev-parse", "HEAD", cwd=cwd)
    # Non-ignored untracked files can participate in imports, packaging, or container contexts and
    # therefore must prevent durable source claims just like modifications to tracked files.
    status_result = _run_git("status", "--porcelain", "--untracked-files=all", cwd=cwd)
    if commit_result.returncode != 0 or status_result.returncode != 0:
        raise RuntimeError("Unable to resolve the complete Git source identity")
    commit = commit_result.stdout.strip().lower()
    if not _FULL_COMMIT.fullmatch(commit):
        raise RuntimeError("Git did not return a full 40-character source commit")
    return SourceIdentity(
        source_commit=commit,
        tracked_source_clean=not bool(status_result.stdout.strip()),
        source_identity_kind="git",
    )


def _declaration_path(path: str | Path | None = None) -> Path:
    configured = os.environ.get("CALO_SOURCE_IDENTITY_PATH", "").strip()
    return (
        Path(path)
        if path is not None
        else Path(configured)
        if configured
        else DEFAULT_DECLARATION_PATH
    )


def _validate_declared_values(commit: str, tracked_source_clean: bool) -> tuple[str, bool]:
    normalized = str(commit).strip().lower()
    clean = bool(tracked_source_clean)
    if normalized == "unavailable":
        if clean:
            raise ValueError("An unavailable source commit cannot be declared clean")
        return normalized, clean
    if not _FULL_COMMIT.fullmatch(normalized):
        raise ValueError("Declared source commit must be 40 lowercase hexadecimal characters")
    return normalized, clean


def write_source_declaration(
    path: str | Path, *, source_commit: str, tracked_source_clean: bool
) -> Path:
    """Write one deterministic build declaration and refuse replacement."""

    commit, clean = _validate_declared_values(source_commit, tracked_source_clean)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "source_commit": commit,
        "tracked_source_clean": clean,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with destination.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite source declaration: {destination}") from exc
    return destination


def _read_source_declaration(path: str | Path | None = None) -> SourceIdentity | None:
    declaration = _declaration_path(path)
    if not declaration.is_file():
        return None
    try:
        payload = json.loads(declaration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid source declaration: {declaration}") from exc
    if payload.get("schema") != SOURCE_IDENTITY_SCHEMA:
        raise RuntimeError(f"Unsupported source-declaration schema: {payload.get('schema')!r}")
    if not isinstance(payload.get("tracked_source_clean"), bool):
        raise RuntimeError("Source declaration tracked_source_clean must be a boolean")
    try:
        commit, clean = _validate_declared_values(
            str(payload.get("source_commit", "")), payload["tracked_source_clean"]
        )
    except ValueError as exc:
        raise RuntimeError(f"Invalid source declaration: {declaration}") from exc
    return SourceIdentity(
        source_commit=commit,
        tracked_source_clean=clean,
        source_identity_kind=(
            "build-declared" if commit != "unavailable" else "build-declared-unavailable"
        ),
        declaration_path=str(declaration),
    )


def resolve_source_identity(
    *,
    cwd: str | Path | None = None,
    declaration_path: str | Path | None = None,
    require_durable: bool = False,
) -> SourceIdentity:
    """Prefer live Git truth, then use an immutable build declaration outside a worktree."""

    identity = _git_identity(cwd)
    if identity is None:
        identity = _read_source_declaration(declaration_path)
    if identity is None:
        identity = SourceIdentity("unavailable", False, "unavailable")
    if require_durable and not identity.durable_evidence_eligible:
        raise RuntimeError(
            "Durable evidence requires a full source commit and clean tracked source identity"
        )
    return identity


def resolve_evidence_source_identity(
    *,
    cwd: str | Path | None = None,
    declaration_path: str | Path | None = None,
    allow_dirty_development_evidence: bool = False,
) -> SourceIdentity:
    """Resolve a durable identity or an explicitly scoped development-only identity.

    Development evidence still requires a full commit identity. The opt-in permits a dirty source
    state to be measured and retained, but callers must label it non-durable and must not convert a
    successful engineering observation into qualification or release evidence.
    """

    identity = resolve_source_identity(cwd=cwd, declaration_path=declaration_path)
    if identity.durable_evidence_eligible:
        return identity
    if allow_dirty_development_evidence and _FULL_COMMIT.fullmatch(identity.source_commit):
        return identity
    raise RuntimeError(
        "Durable evidence requires a full source commit and clean tracked source identity; "
        "dirty source is allowed only with the explicit development-evidence option"
    )


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tracked-source-clean", required=True, type=_parse_bool)
    args = parser.parse_args()
    destination = write_source_declaration(
        args.output,
        source_commit=args.source_commit,
        tracked_source_clean=args.tracked_source_clean,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
