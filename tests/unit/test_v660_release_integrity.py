from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tomllib

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v660_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.6.0"
    assert RELEASE_NAME == metadata["release_name"] == "Remaining Audit Closure"
    assert FREEZE_ID == "calo_v660_software_release"
    assert FREEZE_MANIFEST == "calo_v660_freeze.json"


def test_v660_freeze_verifies_and_covers_remaining_audit_scope():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "calo_rpd_studio/orpd/variable_decoder.py",
        "calo_rpd_studio/orpd/feasibility_rules.py",
        "calo_rpd_studio/orpd/constraints.py",
        "calo_rpd_studio/power_system/newton_raphson.py",
        "calo_rpd_studio/power_system/voltage_stability.py",
        "calo_rpd_studio/accelerated/torch_power_flow.py",
        "calo_rpd_studio/accelerated/torch_orpd.py",
        "calo_rpd_studio/algorithms/calo/training.py",
        "calo_rpd_studio/algorithms/calo/ai_controller.py",
        "calo_rpd_studio/algorithms/calo/device_resident_synthetic.py",
        "calo_rpd_studio/app/experiment_workspace_restorer.py",
        "calo_rpd_studio/experiments/experiment_config.py",
        "calo_rpd_studio/gui/panels/resume_center_panel.py",
        "calo_rpd_studio/gui/panels/validation_audit_panel.py",
        "calo_rpd_studio/gui/panels/publication_export_panel.py",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    for key in (
        "v660_reusable_orpd_decode_workspace",
        "v660_unified_feasibility_tolerance_and_ordering",
        "v660_bounded_dense_large_case_fallbacks",
        "v660_lindex_dimension_identity_validation",
        "v660_vectorized_branch_angle_constraints",
        "v660_separated_training_rng_streams",
        "v660_nan_safe_degenerate_friedman_evidence",
        "v660_bounded_policy_and_static_tensor_caches",
        "v660_narrowed_resource_and_accelerator_error_paths",
        "v660_deterministic_campaign_secondary_order",
        "v660_semantic_key_workspace_restore",
        "v660_read_only_config_validation_and_strict_unknown_fields",
        "v660_structured_workspace_restore_failures",
        "v660_sparse_jacobian_failure_fallback",
        "v660_inactive_candidates_removed_from_batch_solve",
        "v660_cross_scenario_torch_batching",
        "v660_explicit_policy_training_completion_state",
        "v660_universal_resume_dispatch",
        "v660_stale_results_selection_safe",
        "v660_corrupt_portfolio_manifest_diagnostic",
        "v660_verified_result_count_preserved_on_stop",
        "v660_safe80_lazy_governor_config_parity",
        "v660_real_development_config_case_cache",
        "v660_oversized_synthetic_request_chunking",
        "v660_static_curriculum_tensor_lru_cache",
    ):
        assert scope[key] is True, key
    assert scope["stage_b_full_stochastic_calo_controller_gpu_resident"] is False


def test_v660_release_evidence_records_every_remaining_priority_id():
    root = _root()
    for name in (
        "CALO-RPD-v6.6.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.6.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.6.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "STAGE_B_ACCELERATOR_QUALIFICATION_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    with (root / "FINDINGS_CLOSURE_v6.6.0.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    expected = {
        "C02", "C03", "C05", "C06", "C07", "H03", "H10", "H14", "H16", "H17",
        "H18", "H19", "H20", "H23", "H25", "H26", "H27", "M04", "M05", "M16",
        "M34", "M36", "M37", "M48", "M52", "M54", "M57", "L19", "L20", "L23",
        "V64-N04", "V64-N05", "V64-N06",
    }
    assert set(rows) == expected
    assert rows["H19"]["status"] == "VERIFIED_NOT_DEFECT"
    for issue_id in expected - {"H19"}:
        assert rows[issue_id]["status"] == "RESOLVED", issue_id


def test_v660_root_manifest_matches_every_packaged_file():
    root = _root()
    manifest = root / "MANIFEST.sha256"
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    for relative, expected in rows.items():
        path = root / relative
        assert path.is_file(), relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, relative
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and not any(part in excluded_parts for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
    }
    assert set(rows) == actual


def test_v660_metadata_records_truthful_environment_boundaries_and_validation():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["focused_remaining_audit"] == "22 passed"
    assert metadata["validation"]["foundational_regression_selection"] == "82 passed"
    assert metadata["validation"]["accelerator_regression_selection"] == "91 passed"
    assert metadata["validation"]["physical_cuda_xpu"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pyqt6"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pypower"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["guaranteed_high_gpu_utilization"] is False
    assert metadata["remaining_audit_closure"]["priority_ids_total"] == 33
    assert metadata["remaining_audit_closure"]["engineering_resolutions"] == 32
    assert metadata["remaining_audit_closure"]["verified_not_defect"] == ["H19"]
