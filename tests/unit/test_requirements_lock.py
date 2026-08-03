from __future__ import annotations

import pytest

from calo_rpd_studio.scripts.verify_requirements_lock import (
    verify_lock,
    verify_lock_contains_exact_graph,
)


def test_requirements_lock_verification_accepts_exact_hashed_pins(tmp_path):
    lock = tmp_path / "requirements.txt"
    lock.write_text(
        "--extra-index-url https://example.invalid/simple\n\n"
        "Alpha_Pkg==1.2.3 \\\n"
        "    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    result = verify_lock(
        lock,
        expected_index="https://example.invalid/simple",
        expected_pins=("alpha-pkg==1.2.3",),
    )
    assert result.package_count == 1
    assert result.hash_count == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("package==1.0\n", "no SHA-256"),
        ("package @ https://example.invalid/package.whl\n", "mutable requirement source"),
        ("-e .\n", "mutable requirement source"),
    ],
)
def test_requirements_lock_verification_rejects_unreproducible_entries(tmp_path, content, message):
    lock = tmp_path / "requirements.txt"
    lock.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_lock(lock)


def test_candidate_lock_must_preserve_reference_runtime_versions(tmp_path):
    reference = tmp_path / "runtime.txt"
    candidate = tmp_path / "ci.txt"
    hash_text = "    --hash=sha256:" + "a" * 64 + "\n"
    reference.write_text("numpy==2.3.5 \\\n" + hash_text, encoding="utf-8")
    candidate.write_text(
        "numpy==2.3.4 \\\n" + hash_text + "pytest==9.1.1 \\\n" + hash_text,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not preserve"):
        verify_lock_contains_exact_graph(reference, candidate)
