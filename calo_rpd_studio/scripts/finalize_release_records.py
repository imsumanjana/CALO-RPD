"""Generate v12 final metadata only from explicitly authorized retained evidence.

The command never changes the version, commits, tags, pushes, publishes, uploads, or creates a
hosting-provider release. Its disabled authorization template is safe to create during development.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.scripts.create_release_preparation import (
    validate_release_preparation,
    verify_release_preparation_evidence,
)
from calo_rpd_studio.scripts.accept_development_freeze import (
    acceptance_matches_freeze,
    build_acceptance_receipt,
)
from calo_rpd_studio.scripts.create_development_freeze_candidate import (
    validate_development_freeze_candidate,
)
from calo_rpd_studio.scripts.release_policy_scope import validate_scope_decision
from calo_rpd_studio.version import DISPLAY_VERSION, VERSION, VERSION_STAGE


AUTHORIZATION_SCHEMA = "calo-v12-final-record-generation-authorization-v1"
METADATA_SCHEMA = "calo-v12-final-release-metadata-v1"
SOURCE_MANIFEST_SCHEMA = "calo-v12-final-source-manifest-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DECISION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")
_HASH_ROW = re.compile(r"([0-9a-f]{64})  (.+)")
_PHASE5_RESULT_IDS = {
    "01-version",
    "02-compile",
    "03-schema",
    "04-ruff",
    "05-format",
    "06-types",
    "07-tests",
    "08-ci-contract",
    "09-scope-template",
    "10-scope-boundary",
    "10-final-authorization-template",
    "10-final-authorization-boundary",
    "11-build",
    "12-distribution",
    "13-member-manifests",
    "14-distribution-manifest",
    "15-wheel-venv",
    "16-wheel-install",
    "17-wheel-smoke",
    "18-packaged-gui",
    "19-sdist-venv",
    "20-sdist-install",
    "21-sdist-smoke",
    "22-cpu-build",
    "23-cpu-inspect",
    "24-cpu-digest",
    "25-cpu-smoke",
    "26-cpu-sbom",
    "27-cpu-vulnerabilities",
    "28-cpu-security",
    "29-cuda-build",
    "30-cuda-inspect",
    "31-cuda-digest",
    "32-cuda-smoke",
    "33-cuda-sbom",
    "34-cuda-vulnerabilities",
    "35-cuda-security",
    "35-scanner-identity",
    "36-release-preparation",
    "37-release-boundary",
    "38-source-stability",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[Any, Any]:
    """Load an evidence JSON object while rejecting arrays, scalars, and malformed input."""

    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return loaded


def _verify_directory_hash_manifest(run: Path, manifest_name: str) -> dict[str, str]:
    root = run.resolve(strict=True)
    manifest = root / manifest_name
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"Evidence hash manifest is missing or unsafe: {manifest}")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = _HASH_ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"Evidence hash manifest contains an invalid row: {line!r}")
        expected, relative_text = match.groups()
        relative = Path(relative_text.replace("/", os.sep))
        if relative.is_absolute() or ".." in relative.parts or relative_text in rows:
            raise ValueError("Evidence hash manifest contains an unsafe or duplicate path")
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError(f"Evidence contains a forbidden symlink: {relative_text}")
        path = unresolved.resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Evidence path escaped its run directory") from exc
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Evidence SHA-256 mismatch: {relative_text}")
        rows[relative_text] = expected
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(rows):
        raise ValueError("Evidence directory contains unhashed or missing files")
    return rows


def _validate_phase5_run(run_directory: str | Path) -> dict:
    run = Path(run_directory).expanduser().resolve(strict=True)
    rows = _verify_directory_hash_manifest(run, "validation-log-sha256.txt")
    required = {"validation-summary.json", "phase5-source-manifest.json"}
    if not required <= set(rows):
        raise ValueError("Phase 5 evidence manifest omits required records")
    summary = _load_json_object(
        run / "validation-summary.json",
        label="Phase 5 validation summary",
    )
    results = summary.get("results")
    if (
        summary.get("schema") != "calo-local-phase5-validation-v1"
        or summary.get("passed") is not True
        or summary.get("failed_command_count") != 0
        or summary.get("command_count") != len(_PHASE5_RESULT_IDS)
        or not isinstance(results, list)
        or any(not isinstance(row, dict) or row.get("status") != "PASS" for row in results)
    ):
        raise ValueError("Phase 5 validation summary is not a complete passing run")
    ids = [str(row.get("id", "")) for row in results]
    if len(ids) != len(set(ids)) or set(ids) != _PHASE5_RESULT_IDS:
        raise ValueError("Phase 5 validation summary does not match the required result set")
    return summary


def _validate_combined_run(combined_path: Path, combined: dict, *, decision_id: str) -> None:
    combined_rows = _verify_directory_hash_manifest(
        combined_path.parent,
        "combined-log-sha256.txt",
    )
    if combined_path.name not in combined_rows:
        raise ValueError("Combined evidence manifest omits its summary")
    phase4 = dict(combined.get("phase4") or {})
    phase5 = dict(combined.get("phase5") or {})
    phase4_run = Path(str(phase4.get("run_directory", ""))).resolve(strict=True)
    phase5_run = Path(str(phase5.get("run_directory", ""))).resolve(strict=True)
    phase4_receipt = build_acceptance_receipt(phase4_run, decision_id=decision_id)
    phase5_summary = _validate_phase5_run(phase5_run)
    comparisons = (
        (
            phase4.get("summary_sha256"),
            _sha256_file(phase4_run / "validation-summary.json"),
        ),
        (
            phase4.get("hash_manifest_sha256"),
            _sha256_file(phase4_run / "validation-log-sha256.txt"),
        ),
        (
            phase5.get("summary_sha256"),
            _sha256_file(phase5_run / "validation-summary.json"),
        ),
        (
            phase5.get("hash_manifest_sha256"),
            _sha256_file(phase5_run / "validation-log-sha256.txt"),
        ),
    )
    if any(recorded != actual for recorded, actual in comparisons):
        raise ValueError("Combined summary does not match its child evidence hashes")
    if phase4_receipt["validated_source_commit"] != phase5_summary.get(
        "source_commit"
    ) or phase4_receipt["validated_source_dirty"] != phase5_summary.get("source_dirty"):
        raise ValueError("Combined child evidence has inconsistent source identity")


def disabled_authorization_template() -> dict:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_granted": False,
        "decision_id": "replace-with-explicit-final-record-decision-id",
        "source_commit": "",
        "combined_validation_summary_sha256": "",
        "release_preparation_payload_sha256": "",
        "release_policy_scope_decision_sha256": "",
        "authorize_final_metadata_generation": False,
        "authorize_tag": False,
        "authorize_push": False,
        "authorize_publication": False,
        "acknowledge_all_release_gates_directly_evidenced": False,
        "authorization_payload_sha256": "",
    }


def _authorization_stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "authorization_payload_sha256"}


def validate_authorization(authorization: dict) -> None:
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("Unsupported final-record authorization schema")
    if authorization.get("authorization_granted") is not True:
        raise PermissionError("Final-record generation has not been explicitly authorized")
    if _DECISION_ID.fullmatch(str(authorization.get("decision_id", ""))) is None:
        raise ValueError("Final-record decision ID must contain 3-80 safe characters")
    if authorization.get("authorize_final_metadata_generation") is not True:
        raise PermissionError("Authorization does not permit final metadata generation")
    if authorization.get("acknowledge_all_release_gates_directly_evidenced") is not True:
        raise PermissionError("Authorization lacks the direct-evidence acknowledgement")
    for field in (
        "combined_validation_summary_sha256",
        "release_preparation_payload_sha256",
        "release_policy_scope_decision_sha256",
    ):
        if _SHA256.fullmatch(str(authorization.get(field, "")).lower()) is None:
            raise ValueError(f"Final-record authorization has an invalid {field}")
    if re.fullmatch(r"[0-9a-f]{40}", str(authorization.get("source_commit", "")).lower()) is None:
        raise ValueError("Final-record authorization has an invalid source commit")
    expected = str(authorization.get("authorization_payload_sha256", "")).lower()
    if (
        _SHA256.fullmatch(expected) is None
        or _canonical_sha256(_authorization_stable(authorization)) != expected
    ):
        raise ValueError("Final-record authorization payload SHA-256 mismatch")


def _complete_source_manifest(root: Path) -> dict:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to enumerate final source: {result.stderr.strip()}")
    records = []
    for relative in sorted({line for line in result.stdout.splitlines() if line}):
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Final source escaped its root: {relative}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Final source must contain regular files only: {relative}")
        records.append(
            {
                "path": relative.replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError("Final source manifest cannot be empty")
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "file_count": len(records),
        "files_sha256": _canonical_sha256(records),
        "files": records,
    }


def build_final_records(
    source_root: str | Path,
    *,
    combined_summary_path: str | Path,
    release_preparation_path: str | Path,
    release_evidence_root: str | Path,
    scope_decision_path: str | Path,
    phase4_acceptance_path: str | Path,
    post_transition_freeze_path: str | Path,
    authorization_path: str | Path,
    scope_evidence_root: str | Path | None = None,
) -> tuple[dict, dict]:
    root = Path(source_root).expanduser().resolve(strict=True)
    identity = resolve_source_identity(cwd=root)
    if not identity.durable_evidence_eligible:
        raise RuntimeError("Final records require a clean durable source identity")
    if VERSION != "12.0.0" or DISPLAY_VERSION != "12.0.0" or VERSION_STAGE != "release":
        raise RuntimeError("Final records require the already reviewed 12.0.0 release identity")

    combined_path = Path(combined_summary_path).expanduser().resolve(strict=True)
    preparation_path = Path(release_preparation_path).expanduser().resolve(strict=True)
    decision_path = Path(scope_decision_path).expanduser().resolve(strict=True)
    acceptance_path = Path(phase4_acceptance_path).expanduser().resolve(strict=True)
    transition_path = Path(post_transition_freeze_path).expanduser().resolve(strict=True)
    authority_path = Path(authorization_path).expanduser().resolve(strict=True)
    combined = _load_json_object(combined_path, label="Combined validation summary")
    preparation = _load_json_object(preparation_path, label="Release preparation")
    decision = _load_json_object(decision_path, label="Release-policy scope decision")
    acceptance = _load_json_object(acceptance_path, label="Phase 4 acceptance receipt")
    transition = _load_json_object(transition_path, label="Post-transition freeze")
    authorization = _load_json_object(authority_path, label="Final-record authorization")
    if combined.get("schema") != "calo-v12-combined-phase4-phase5-validation-v1" or (
        combined.get("passed") is not True
    ):
        raise ValueError("Final records require a complete passing combined validation summary")
    _validate_combined_run(
        combined_path,
        combined,
        decision_id=str(authorization.get("decision_id", "")),
    )
    phase4 = dict(combined.get("phase4") or {})
    phase5 = dict(combined.get("phase5") or {})
    if (
        phase4.get("source_commit") != identity.source_commit
        or phase5.get("source_commit") != identity.source_commit
        or phase4.get("source_dirty") is not False
        or phase5.get("source_dirty") is not False
    ):
        raise ValueError("Combined validation does not match the clean final source identity")
    validate_release_preparation(preparation)
    verify_release_preparation_evidence(preparation, release_evidence_root)
    preparation_identity = dict(preparation.get("source_identity") or {})
    if (
        preparation_identity.get("source_commit") != identity.source_commit
        or preparation_identity.get("tracked_source_clean") is not True
    ):
        raise ValueError("Release preparation does not match the clean final source identity")
    validate_development_freeze_candidate(transition, require_training_eligible=True)
    acceptance_matches_freeze(acceptance, transition)
    if decision.get("post_transition_freeze_sha256") != transition.get(
        "development_freeze_payload_sha256"
    ):
        raise ValueError("Final scope does not match the clean post-transition freeze")
    validated_scope = validate_scope_decision(
        decision,
        acceptance_receipt=acceptance,
        evidence_root=scope_evidence_root,
    )
    preparation_scope = dict(preparation.get("policy_scope") or {})
    if (
        preparation_scope.get("status") != "approved"
        or preparation_scope.get("selected_scope") != validated_scope["selected_scope"]
        or preparation_scope.get("decision_payload_sha256")
        != validated_scope["decision_payload_sha256"]
    ):
        raise ValueError(
            "Final records require a release preparation rebuilt for the exact approved scope"
        )
    validate_authorization(authorization)
    comparisons = {
        "source_commit": identity.source_commit,
        "combined_validation_summary_sha256": _sha256_file(combined_path),
        "release_preparation_payload_sha256": preparation["release_preparation_payload_sha256"],
        "release_policy_scope_decision_sha256": decision["decision_payload_sha256"],
    }
    for field, expected in comparisons.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"Final-record authorization {field} does not match evidence")
    source_manifest = _complete_source_manifest(root)
    stable = {
        "schema": METADATA_SCHEMA,
        "product": "CALO-RPD Studio",
        "version": VERSION,
        "display_version": DISPLAY_VERSION,
        "stage": VERSION_STAGE,
        "source_identity": identity.to_dict(),
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "combined_validation_summary_sha256": comparisons["combined_validation_summary_sha256"],
        "release_preparation_payload_sha256": comparisons["release_preparation_payload_sha256"],
        "release_policy_scope": validated_scope,
        "authorization_id": authorization["decision_id"],
        "tag_authorized": bool(authorization.get("authorize_tag")),
        "push_authorized": bool(authorization.get("authorize_push")),
        "publication_authorized": bool(authorization.get("authorize_publication")),
        "commands_executed_by_generator": ["verify_retained_evidence", "hash_final_source"],
        "commands_not_executed": ["git_commit", "git_tag", "git_push", "publish", "release"],
    }
    metadata = {
        **stable,
        "created_at": _utcnow(),
        "metadata_payload_sha256": _canonical_sha256(stable),
    }
    return metadata, source_manifest


def _write_exclusive(path: Path, payload: dict) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite final release record: {destination}") from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("authorization-template")
    template.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--source-root", type=Path, default=Path.cwd())
    generate.add_argument("--combined-summary", type=Path, required=True)
    generate.add_argument("--release-preparation", type=Path, required=True)
    generate.add_argument("--release-evidence-root", type=Path, required=True)
    generate.add_argument("--scope-decision", type=Path, required=True)
    generate.add_argument("--phase4-acceptance", type=Path, required=True)
    generate.add_argument("--post-transition-freeze", type=Path, required=True)
    generate.add_argument("--scope-evidence-root", type=Path)
    generate.add_argument("--authorization", type=Path, required=True)
    generate.add_argument("--metadata-output", type=Path, required=True)
    generate.add_argument("--source-manifest-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "authorization-template":
        print(_write_exclusive(arguments.output, disabled_authorization_template()))
        return 0
    metadata, manifest = build_final_records(
        arguments.source_root,
        combined_summary_path=arguments.combined_summary,
        release_preparation_path=arguments.release_preparation,
        release_evidence_root=arguments.release_evidence_root,
        scope_decision_path=arguments.scope_decision,
        phase4_acceptance_path=arguments.phase4_acceptance,
        post_transition_freeze_path=arguments.post_transition_freeze,
        authorization_path=arguments.authorization,
        scope_evidence_root=arguments.scope_evidence_root,
    )
    print(_write_exclusive(arguments.metadata_output, metadata))
    print(_write_exclusive(arguments.source_manifest_output, manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
