from __future__ import annotations

import json
from pathlib import Path

import pytest

from calo_rpd_studio.compute import source_identity as identity_module
from calo_rpd_studio.compute.source_identity import (
    SOURCE_IDENTITY_SCHEMA,
    SourceIdentity,
    resolve_source_identity,
    write_source_declaration,
)


COMMIT = "a" * 40


def test_build_declaration_is_deterministic_validated_and_never_overwritten(tmp_path: Path):
    path = tmp_path / "source.json"
    write_source_declaration(path, source_commit=COMMIT, tracked_source_clean=True)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "source_commit": COMMIT,
        "tracked_source_clean": True,
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_source_declaration(path, source_commit=COMMIT, tracked_source_clean=True)
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        write_source_declaration(
            tmp_path / "bad.json", source_commit="short", tracked_source_clean=False
        )
    with pytest.raises(ValueError, match="cannot be declared clean"):
        write_source_declaration(
            tmp_path / "unavailable.json",
            source_commit="unavailable",
            tracked_source_clean=True,
        )


def test_declaration_is_used_only_when_git_identity_is_unavailable(monkeypatch, tmp_path: Path):
    path = tmp_path / "source.json"
    write_source_declaration(path, source_commit=COMMIT, tracked_source_clean=True)
    monkeypatch.setattr(identity_module, "_git_identity", lambda _cwd=None: None)

    identity = resolve_source_identity(declaration_path=path, require_durable=True)

    assert identity.source_commit == COMMIT
    assert identity.tracked_source_clean
    assert identity.source_identity_kind == "build-declared"
    assert identity.durable_evidence_eligible


def test_dirty_git_identity_cannot_be_bypassed_by_clean_declaration(monkeypatch, tmp_path: Path):
    path = tmp_path / "source.json"
    write_source_declaration(path, source_commit=COMMIT, tracked_source_clean=True)
    monkeypatch.setattr(
        identity_module,
        "_git_identity",
        lambda _cwd=None: SourceIdentity("b" * 40, False, "git"),
    )

    with pytest.raises(RuntimeError, match="Durable evidence requires"):
        resolve_source_identity(declaration_path=path, require_durable=True)


def test_unavailable_development_declaration_is_never_durable(monkeypatch, tmp_path: Path):
    path = tmp_path / "source.json"
    write_source_declaration(path, source_commit="unavailable", tracked_source_clean=False)
    monkeypatch.setattr(identity_module, "_git_identity", lambda _cwd=None: None)

    identity = resolve_source_identity(declaration_path=path)

    assert identity.source_identity_kind == "build-declared-unavailable"
    assert not identity.durable_evidence_eligible
    with pytest.raises(RuntimeError, match="Durable evidence requires"):
        resolve_source_identity(declaration_path=path, require_durable=True)


def test_malformed_or_unknown_declaration_fails_closed(monkeypatch, tmp_path: Path):
    path = tmp_path / "source.json"
    path.write_text('{"schema":"unknown"}\n', encoding="utf-8")
    monkeypatch.setattr(identity_module, "_git_identity", lambda _cwd=None: None)

    with pytest.raises(RuntimeError, match="Unsupported source-declaration schema"):
        resolve_source_identity(declaration_path=path)
