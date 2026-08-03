from __future__ import annotations

import hashlib
import json

import pytest

from calo_rpd_studio.scripts.generate_artifact_manifest import (
    SCHEMA,
    build_manifest,
    write_manifest,
)


def test_staged_manifest_is_sorted_hashed_and_self_excluding(tmp_path):
    stage = tmp_path / "stage"
    (stage / "nested").mkdir(parents=True)
    (stage / "z.whl").write_bytes(b"wheel")
    (stage / "nested" / "a.tar.gz").write_bytes(b"source")
    output = stage / "artifact-manifest.json"

    write_manifest(stage, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert [item["path"] for item in payload["artifacts"]] == [
        "nested/a.tar.gz",
        "z.whl",
    ]
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(b"source").hexdigest()
    assert payload["artifacts"][1]["size_bytes"] == len(b"wheel")


def test_staged_manifest_rejects_empty_stage(tmp_path):
    with pytest.raises(ValueError, match="contains no files"):
        build_manifest(tmp_path)


def test_staged_manifest_rejects_symbolic_links_when_supported(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    target = stage / "target"
    target.write_text("data", encoding="utf-8")
    link = stage / "link"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("Symbolic links are unavailable on this platform")
    with pytest.raises(ValueError, match="Symbolic links"):
        build_manifest(stage)
