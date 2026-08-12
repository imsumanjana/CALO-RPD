"""Create a fail-closed v12 release-preparation record from retained manual evidence.

This is a development/RC-preparation record, never final release metadata.  It does not build,
scan, publish, tag, release, or choose a policy scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.scripts.accept_development_freeze import acceptance_matches_freeze
from calo_rpd_studio.scripts.create_development_freeze_candidate import (
    validate_development_freeze_candidate,
)
from calo_rpd_studio.scripts.release_policy_scope import validate_scope_decision
from calo_rpd_studio.version import DISPLAY_VERSION, VERSION, VERSION_STAGE


SCHEMA = "calo-v12-release-preparation-candidate-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_EVIDENCE = {
    "python_distribution_manifest",
    "wheel_manifest",
    "sdist_manifest",
    "cpu_image_inspect",
    "cpu_image_digest",
    "cpu_sbom",
    "cpu_vulnerability_report",
    "cpu_filesystem_manifest",
    "cuda_image_inspect",
    "cuda_image_digest",
    "cuda_sbom",
    "cuda_vulnerability_report",
    "cuda_filesystem_manifest",
    "clean_machine_report",
    "wheel_clean_install_report",
    "sdist_clean_install_report",
    "ci_scope_report",
    "scanner_identity_report",
    "cpu_build_metadata",
    "cuda_build_metadata",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_record(root: Path, relative: str) -> dict:
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"Release evidence cannot be a symlink: {relative}")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Release evidence escaped its root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"Release evidence is not a regular file: {relative}")
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def validate_release_preparation(report: dict) -> None:
    if report.get("schema") != SCHEMA or report.get("status") != "release_preparation_candidate":
        raise ValueError("Unsupported release-preparation candidate")
    if report.get("version") != VERSION or report.get("version_stage") != VERSION_STAGE:
        raise ValueError("Release preparation does not match the active development identity")
    if any(
        report.get(field) is not False
        for field in (
            "release_candidate",
            "final_release",
            "release_ready",
            "publication_authorized",
        )
    ):
        raise ValueError("Development release preparation cannot claim RC or final status")
    claims = report.get("claims")
    if (
        not isinstance(claims, dict)
        or set(claims)
        != {
            "policy_benefit",
            "scientific_superiority",
            "protected_case",
            "human_accessibility",
        }
        or any(value is not False for value in claims.values())
    ):
        raise ValueError("Development release preparation contains an unauthorized claim")
    records = report.get("evidence")
    if not isinstance(records, dict) or set(records) != REQUIRED_EVIDENCE:
        raise ValueError("Release preparation has an incomplete evidence contract")
    for name, record in records.items():
        path = str(record.get("path", "")) if isinstance(record, dict) else ""
        size = record.get("size_bytes") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or _SHA256.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ValueError(f"Release preparation has an invalid {name} record")
    expected = str(report.get("release_preparation_payload_sha256", ""))
    stable = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "release_preparation_payload_sha256"}
    }
    if _SHA256.fullmatch(expected) is None or _canonical_sha256(stable) != expected:
        raise ValueError("Release-preparation payload SHA-256 mismatch")


def verify_release_preparation_evidence(report: dict, evidence_root: str | Path) -> None:
    """Re-hash every retained evidence file bound by a valid preparation report."""

    validate_release_preparation(report)
    root = Path(evidence_root).expanduser().resolve(strict=True)
    for name, record in report["evidence"].items():
        actual = _evidence_record(root, str(record["path"]))
        if actual != record:
            raise ValueError(f"Release-preparation evidence changed: {name}")


def build_release_preparation(
    source_root: str | Path,
    evidence_root: str | Path,
    evidence_map: dict[str, str],
    *,
    scope_decision: dict | None = None,
    phase4_acceptance: dict | None = None,
    post_transition_freeze: dict | None = None,
    require_clean: bool = False,
) -> dict:
    root = Path(source_root).expanduser().resolve(strict=True)
    evidence = Path(evidence_root).expanduser().resolve(strict=True)
    if set(evidence_map) != REQUIRED_EVIDENCE:
        missing = sorted(REQUIRED_EVIDENCE - set(evidence_map))
        extra = sorted(set(evidence_map) - REQUIRED_EVIDENCE)
        raise ValueError(f"Release evidence map mismatch; missing={missing}, extra={extra}")
    identity = resolve_source_identity(cwd=root)
    if require_clean and not identity.durable_evidence_eligible:
        raise RuntimeError("Release preparation requires a clean durable source identity")
    scope_status: dict[str, Any]
    if scope_decision is None:
        scope_status = {
            "status": "pending_explicit_decision",
            "selected_scope": "",
            "decision_payload_sha256": "",
        }
    else:
        if phase4_acceptance is None or post_transition_freeze is None:
            raise ValueError(
                "Approved release scope requires Phase 4 acceptance and post-transition freeze"
            )
        validate_development_freeze_candidate(
            post_transition_freeze,
            require_training_eligible=True,
        )
        acceptance_matches_freeze(phase4_acceptance, post_transition_freeze)
        if scope_decision.get("post_transition_freeze_sha256") != post_transition_freeze.get(
            "development_freeze_payload_sha256"
        ):
            raise ValueError("Release-policy scope does not match the post-transition freeze")
        validated = validate_scope_decision(
            scope_decision,
            acceptance_receipt=phase4_acceptance,
            evidence_root=evidence,
        )
        scope_status = {"status": "approved", **validated}
    stable = {
        "schema": SCHEMA,
        "status": "release_preparation_candidate",
        "version": VERSION,
        "display_version": DISPLAY_VERSION,
        "version_stage": VERSION_STAGE,
        "source_identity": identity.to_dict(),
        "policy_scope": scope_status,
        "evidence": {
            name: _evidence_record(evidence, relative)
            for name, relative in sorted(evidence_map.items())
        },
        "release_candidate": False,
        "final_release": False,
        "release_ready": False,
        "publication_authorized": False,
        "claims": {
            "policy_benefit": False,
            "scientific_superiority": False,
            "protected_case": False,
            "human_accessibility": False,
        },
        "required_later_decisions": [
            "accepted_combined_phase4_phase5_validation",
            "explicit_release_policy_scope",
            "explicit_release_authorization",
        ],
    }
    report = {
        **stable,
        "created_at": _utcnow(),
        "release_preparation_payload_sha256": _canonical_sha256(stable),
    }
    validate_release_preparation(report)
    return report


def write_exclusive(path: str | Path, payload: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite release preparation: {destination}") from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument("--scope-decision", type=Path)
    parser.add_argument("--phase4-acceptance", type=Path)
    parser.add_argument("--post-transition-freeze", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    mapping = json.loads(arguments.evidence_map.read_text(encoding="utf-8-sig"))
    decision = (
        json.loads(arguments.scope_decision.read_text(encoding="utf-8-sig"))
        if arguments.scope_decision
        else None
    )
    acceptance = (
        json.loads(arguments.phase4_acceptance.read_text(encoding="utf-8-sig"))
        if arguments.phase4_acceptance
        else None
    )
    transition = (
        json.loads(arguments.post_transition_freeze.read_text(encoding="utf-8-sig"))
        if arguments.post_transition_freeze
        else None
    )
    report = build_release_preparation(
        arguments.source_root,
        arguments.evidence_root,
        mapping,
        scope_decision=decision,
        phase4_acceptance=acceptance,
        post_transition_freeze=transition,
        require_clean=arguments.require_clean,
    )
    destination = write_exclusive(arguments.output, report)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
