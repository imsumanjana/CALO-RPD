from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v650_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.5.0"
    assert RELEASE_NAME == metadata["release_name"] == "Must-Resolve Audit Closure"
    assert FREEZE_ID == "calo_v650_software_release"
    assert FREEZE_MANIFEST == "calo_v650_freeze.json"


def test_v650_freeze_verifies_and_covers_must_resolve_science():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "calo_rpd_studio/accelerated/torch_power_flow.py",
        "calo_rpd_studio/orpd/mixed_variable_handler.py",
        "calo_rpd_studio/orpd/constraints.py",
        "calo_rpd_studio/algorithms/calo/policy_qualification.py",
        "calo_rpd_studio/results/database.py",
        "calo_rpd_studio/ai/model_io.py",
        "calo_rpd_studio/algorithms/calo/device_resident_synthetic.py",
        "calo_rpd_studio/algorithms/calo/ai_controller.py",
        "calo_rpd_studio/power_system/case_identity.py",
        "calo_rpd_studio/gui/panels/experiment_manager_panel.py",
        "calo_rpd_studio/gui/panels/results_explorer_panel.py",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    for key in (
        "v650_torch_newton_backtracking_cpu_parity",
        "v650_consistent_zero_impedance_threshold",
        "v650_bounded_discrete_step_generation",
        "v650_safe_zero_span_constraint_normalization",
        "v650_stable_near_zero_policy_qualification",
        "v650_transactional_policy_checkpoint_mutations",
        "v650_monotonic_checkpoint_latest_lineage",
        "v650_self_authenticating_atomic_exact_resume_envelope",
        "v650_streaming_checkpoint_hashing",
        "v650_broker_shutdown_failure_propagation",
        "v650_comparison_applies_current_gui_configuration",
        "v650_results_explorer_corrupt_json_safe",
        "v650_stage_b_equal_length_parity_gate",
        "v650_canonical_holdout_identity_protection",
    ):
        assert scope[key] is True, key
    assert scope["crash_safe_atomic_policy_checkpoints"] is True
    assert scope["stage_b_full_stochastic_calo_controller_gpu_resident"] is False


def test_v650_release_evidence_exists_and_records_all_must_resolve_ids():
    root = _root()
    for name in (
        "CALO-RPD-v6.5.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.5.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.5.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    closure = (root / "FINDINGS_CLOSURE_v6.5.0.csv").read_text(encoding="utf-8")
    for issue_id in (
        "C04", "C10", "H01", "H05", "H15", "H21", "H22", "H28", "H30",
        "M18", "V64-N01", "M30", "M32", "M45", "V64-N02", "V64-N03",
    ):
        assert f"{issue_id}," in closure


def test_v650_root_manifest_matches_every_packaged_file():
    root = _root()
    manifest = root / "MANIFEST.sha256"
    rows = {}
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


def test_v650_metadata_records_truthful_environment_boundaries():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["focused_must_resolve"] == "16 passed"
    assert metadata["validation"]["combined_regression_selection"] == "57 passed"
    assert metadata["validation"]["physical_cuda_xpu"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pyqt6"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pypower"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["guaranteed_high_gpu_utilization"] is False
