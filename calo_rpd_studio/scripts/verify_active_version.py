"""Fail-closed consistency check for the active v12 development identity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tomllib

from calo_rpd_studio.version import (
    DISPLAY_VERSION,
    RELEASE_LINE,
    VERSION,
    VERSION_STAGE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def verify_active_version(root: Path = PROJECT_ROOT) -> dict:
    """Return a machine-readable v12 identity report without changing repository state."""

    expected = {
        "version": VERSION,
        "display_version": DISPLAY_VERSION,
        "release_line": RELEASE_LINE,
        "stage": VERSION_STAGE,
    }
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = str(pyproject["project"]["version"])
    checks["pyproject_matches_runtime"] = package_version == VERSION
    details["pyproject_version"] = package_version

    status = _load_json(root / "ACTIVE_DEVELOPMENT_STATUS.json")
    checks["active_status_identity"] = all(
        status.get(key) == value for key, value in expected.items()
    )
    checks["active_status_is_not_release"] = all(
        status.get(key) is False
        for key in (
            "release_candidate",
            "final_release",
            "final_freeze_available",
            "release_qualification_complete",
            "protected_case_evidence_open",
        )
    )
    checks["active_status_does_not_authorize_policy_work"] = (
        status.get("policy_training_authorized_by_status") is False
        and status.get("policy_evaluation_authorized_by_status") is False
    )

    index = _load_json(root / "STATUS_RECORD_INDEX.json")
    checks["status_index_points_to_active_record"] = (
        index.get("active_status") == "ACTIVE_DEVELOPMENT_STATUS.json"
        and index.get("active_version") == VERSION
        and "RELEASE_METADATA.json" in index.get("historical_records", [])
    )

    readme = (root / "README.md").read_text(encoding="utf-8")
    checks["readme_development_label"] = (
        readme.startswith(f"# CALO-RPD Studio v{DISPLAY_VERSION}\n")
        and "Active status: development only" in readme
    )

    gui_sources = (
        root / "calo_rpd_studio/app/main_window.py",
        root / "calo_rpd_studio/gui/navigation/sidebar.py",
        root / "calo_rpd_studio/gui/panels/application_settings_panel.py",
    )
    checks["gui_uses_display_version"] = all(
        "DISPLAY_VERSION" in path.read_text(encoding="utf-8") for path in gui_sources
    )

    launcher = (root / "calo_bootstrap/launcher.py").read_text(encoding="utf-8")
    checks["cli_version_is_explicit"] = all(
        marker in launcher for marker in ('"--version"', '"-V"', "DISPLAY_VERSION", "VERSION")
    )

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    checks["container_version_label"] = (
        f"ARG APP_VERSION={VERSION}" in dockerfile
        and 'org.opencontainers.image.version="${APP_VERSION}"' in dockerfile
    )

    qualification_source = (
        root / "calo_rpd_studio/algorithms/calo/tsh_calo_qualification_campaign.py"
    ).read_text(encoding="utf-8")
    generic_qualification_source = (
        root / "calo_rpd_studio/algorithms/calo/policy_qualification.py"
    ).read_text(encoding="utf-8")
    checks["qualification_evidence_is_versioned"] = (
        "qualification-plan-v2-exact-pairs" in qualification_source
        and "PAIRED_ANALYSIS_SCHEMA_VERSION" in qualification_source
        and "RELATIVE_IMPROVEMENT_VERSION" in qualification_source
        and "source_tracked_clean" in qualification_source
        and '"source_identity": source_identity.to_dict()' in generic_qualification_source
    )

    checks["phase_allows_development_identity"] = (
        VERSION_STAGE == "development"
        and ".dev" in VERSION
        and "-dev." in DISPLAY_VERSION
        and "rc" not in VERSION
        and VERSION != "12.0.0"
    )
    passed = all(checks.values())
    return {
        "schema_version": "calo-active-version-verification-v1",
        "passed": passed,
        "expected": expected,
        "checks": checks,
        "details": details,
    }


def main() -> int:
    try:
        report = verify_active_version()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        report = {
            "schema_version": "calo-active-version-verification-v1",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
