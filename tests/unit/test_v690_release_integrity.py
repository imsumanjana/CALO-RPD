from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tomllib

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v690_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.9.0"
    assert RELEASE_NAME == metadata["release_name"] == "VRAM-Resident CUDA Data Plane"
    assert FREEZE_ID == "calo_v690_software_release"
    assert FREEZE_MANIFEST == "calo_v690_freeze.json"


def test_v690_freeze_verifies_vram_residency_scope():
    root = _root()
    freeze = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze, project_root=root)
    assert result.passed
    assert result.checked_files == 150
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    assert "calo_rpd_studio/accelerated/vram_residency.py" in payload["files"]
    for key in (
        "v690_cuda_80_percent_process_vram_ceiling",
        "v690_cuda_eligible_data_plane_device_resident",
        "v690_cuda_oom_microbatch_retry_no_cpu_fallback",
        "v690_fixed_shape_masked_newton_hot_loop",
        "v690_single_completed_population_host_materialization",
        "v690_ppo_epoch_loss_single_host_materialization",
        "v690_ppo_cuda_minibatch_oom_backoff",
        "v690_vram_residency_attestation",
    ):
        assert payload["frozen_scope"][key] is True, key


def test_v690_release_evidence_records_all_eight_closures():
    root = _root()
    for name in (
        "CALO-RPD-v6.9.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.9.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.9.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "STAGE_B_ACCELERATOR_QUALIFICATION_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    with (root / "FINDINGS_CLOSURE_v6.9.0.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    assert set(rows) == {f"V69-R{i:02d}" for i in range(1, 9)}
    assert all(row["status"] == "RESOLVED" for row in rows.values())


def test_v690_metadata_and_qualification_boundaries_are_truthful():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    hardware = json.loads((root / "HARDWARE_QUALIFICATION_STATUS.json").read_text(encoding="utf-8"))
    scientific = json.loads(
        (root / "SCIENTIFIC_EQUIVALENCE_STATUS.json").read_text(encoding="utf-8")
    )
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["freeze"]["checked_files"] == 150
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["focused_v690_vram_residency"] == "7 passed"
    assert metadata["validation"]["physical_cuda"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert (
        hardware["v690_cuda_process_vram_ceiling"] == "CONFIGURABLE_DEFAULT_0.80_RANGE_0.10_TO_0.95"
    )
    assert hardware["v690_cpu_fallback_on_cuda_oom"] is False
    assert scientific["mathematical_formulation_changed"] is False
    assert (
        scientific["physical_cpu_cuda_xpu_equivalence"]
        == "NOT_EXECUTED_BUILD_RUNTIME_NO_PHYSICAL_ACCELERATORS"
    )


def test_v690_root_manifest_matches_every_packaged_file():
    root = _root()
    rows: dict[str, str] = {}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            assert relative not in rows, f"duplicate manifest path: {relative}"
            path = PurePosixPath(relative)
            assert not path.is_absolute(), relative
            assert ".." not in path.parts, relative
            assert "\\" not in relative, relative
            rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    for relative, expected in rows.items():
        artifact_path = root.joinpath(*PurePosixPath(relative).parts)
        assert artifact_path.is_file(), relative
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert digest == expected, relative
