from __future__ import annotations

import json

import pytest

from calo_rpd_studio.scripts.validate_accelerator import (
    _json_safe,
    _runtime_snapshot,
    _write_new_evidence,
)


def test_accelerator_evidence_is_strict_json_and_never_overwritten(tmp_path):
    path = tmp_path / "parity.json"
    _write_new_evidence(
        path,
        {
            "finite": 1.25,
            "positive": float("inf"),
            "negative": float("-inf"),
            "nan": float("nan"),
        },
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "finite": 1.25,
        "nan": "NaN",
        "negative": "-Infinity",
        "positive": "Infinity",
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_new_evidence(path, {"replacement": True})


def test_cpu_runtime_snapshot_records_eighty_percent_of_currently_available_ram():
    snapshot = _runtime_snapshot("cpu")
    admission = snapshot["cpu_memory_admission"]

    assert snapshot["python"]
    assert snapshot["platform"]
    assert snapshot["torch"]
    assert admission["requested_available_fraction"] == 0.80
    assert admission["additional_allowance_bytes"] <= int(
        0.80 * admission["available_bytes_at_admission"]
    )
    assert "cuda" not in snapshot


def test_json_safe_recurses_without_changing_finite_values():
    assert _json_safe({"rows": (1, 2.5, float("inf"))}) == {"rows": [1, 2.5, "Infinity"]}
