from __future__ import annotations

import json

import pytest

from calo_rpd_studio.scripts.validate_accelerator import (
    _cuda_peak_residency_evidence,
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


def test_cuda_peak_residency_proof_requires_workload_allocation_inside_allowance():
    before = {
        "process_allocated_bytes": 1_000,
        "process_reserved_bytes": 1_500,
        "admission": {"additional_allowance_bytes": 8_000},
    }

    accepted = _cuda_peak_residency_evidence(
        before,
        peak_allocated_bytes=5_000,
        peak_reserved_bytes=7_000,
    )
    no_workload_allocation = _cuda_peak_residency_evidence(
        before,
        peak_allocated_bytes=1_000,
        peak_reserved_bytes=1_500,
    )
    over_allowance = _cuda_peak_residency_evidence(
        before,
        peak_allocated_bytes=10_000,
        peak_reserved_bytes=10_000,
    )

    assert accepted["dedicated_vram_execution_verified"] is True
    assert accepted["host_ram_zero_use_claimed"] is False
    assert no_workload_allocation["dedicated_vram_execution_verified"] is False
    assert over_allowance["dedicated_vram_execution_verified"] is False
