"""Regression coverage for verified snapshots using temporary, tensor-only fixtures."""

from pathlib import Path
import os
import pytest
import torch
from calo_rpd_studio.ai import model_io as io


@pytest.fixture(autouse=True)
def isolated_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_TRUST_DIR", tmp_path / "trust")
    monkeypatch.setattr(io, "_TRUST_KEY", tmp_path / "trust/key")


@pytest.mark.parametrize("kind", ["portable", "legacy", "migration", "envelope"])
def test_replacement_at_deserialization_cannot_change_verified_bytes(tmp_path, monkeypatch, kind):
    source = tmp_path / "model.pt"
    original = {"model_state_dict": {"w": torch.tensor([1.0])}, "optimizer_state_dict": {}}
    replacement = tmp_path / "replacement.pt"
    torch.save({"model_state_dict": {"w": torch.tensor([9.0])}}, replacement)
    if kind == "envelope":
        io.durable_trusted_torch_save(original, source)
    else:
        torch.save(original, source)
    expected = io.checkpoint_sha256(source)
    if kind == "legacy":
        io.write_trusted_resume_hash(source)
    if kind == "migration":
        io.trusted_resume_sha_path(source).write_text(expected, encoding="utf-8")
    actual_load = torch.load
    calls = []

    def replace_then_load(stream, *args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        os.replace(replacement, source)
        assert not isinstance(stream, (str, Path)), "Loader reopened the verified path"
        return actual_load(stream, *args, **kwargs)

    monkeypatch.setattr(io.torch, "load", replace_then_load)
    if kind == "portable":
        result = io.load_checkpoint(source, expected_sha256=expected)
    elif kind in {"legacy", "envelope"}:
        result = io.load_trusted_resume(source)
    else:
        destination = io.migrate_legacy_local_resume(source, explicit_trust=True)
        result = actual_load(destination, weights_only=True)
    assert torch.equal(result["model_state_dict"]["w"], original["model_state_dict"]["w"])
    assert calls == [kind == "portable"]


def test_wrong_digest_rejects_before_deserialization(tmp_path, monkeypatch):
    source = tmp_path / "wrong.pt"
    torch.save({"w": torch.tensor([1.0])}, source)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid bytes reached the deserializer")

    monkeypatch.setattr(io.torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        io.load_checkpoint(source, expected_sha256="0" * 64)


def test_snapshot_spills_and_releases_source_handle(tmp_path):
    source = tmp_path / "large.pt"
    source.write_bytes(b"x" * (9 * 1024 * 1024))
    expected = io.checkpoint_sha256(source)
    with io._verified_checkpoint_snapshot(source, expected) as (snapshot, digest):
        assert snapshot._rolled
        source.unlink()
        assert digest == expected
        assert snapshot.read(1) == b"x"
    assert snapshot.closed
