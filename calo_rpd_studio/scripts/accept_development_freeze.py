"""Create an explicit Phase 4 acceptance receipt from a complete passing manual evidence run.

This command never runs validation, changes source, deletes a policy, or creates a release. It
verifies an already returned Phase 4 directory and writes one non-overwriting decision receipt.
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

from calo_rpd_studio.scripts.create_development_freeze_candidate import (
    development_source_contract,
    validate_development_freeze_candidate,
)


ACCEPTANCE_SCHEMA = "calo-v12-phase4-development-freeze-acceptance-v1"
ACCEPTANCE_CLAIM_BOUNDARY = (
    "Phase 4 development source accepted only; not policy, scientific, RC, "
    "release-ready, or final-release evidence"
)
_DECISION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")
_MANIFEST_ROW = re.compile(r"([0-9a-f]{64})  (.+)")
_REQUIRED_EVIDENCE = {
    "validation-summary.json",
    "VALIDATION_SUMMARY.md",
    "phase4-source-manifest.json",
    "evidence/development-freeze-candidate.json",
    "evidence/old-policy-inventory.json",
    "evidence/old-policy-removal-plan.json",
    "evidence/old-policy-authorization-template-disabled.json",
    "evidence/git-status.txt",
    "evidence/git-status-final.txt",
    "commands/29-source-stability.txt",
    "commands/30-freeze-source-recheck.txt",
}
_REQUIRED_COMMAND_IDS = {
    "01-python",
    "02-version",
    "03-compile",
    "04-schema",
    "05-ruff",
    "06-format",
    "07-types",
    "08-engineering",
    "09-gui",
    "10-inventory",
    "11-plan",
    "12-authorization-template",
    "12-boundary",
    "13-build",
    "14-distribution",
    "15-clean-venv",
    "16-clean-install",
    "17-clean-smoke",
    "17-empty-policy-cli",
    "18-cpu-container-build",
    "19-cpu-container-smoke",
    "20-cuda-container-build",
    "21-cuda-container-smoke",
    "22-nvidia",
    "23-cuda-parity-30",
    "24-cuda-parity-57",
    "25-cuda-batching-30",
    "26-cuda-batching-57",
    "27-resource-recovery",
    "28-freeze-candidate",
    "29-source-stability",
    "30-freeze-source-recheck",
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


def _load_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Phase 4 evidence JSON is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 4 evidence JSON must be an object: {path}")
    return payload


def _verify_hash_manifest(run: Path) -> dict[str, str]:
    manifest = run / "validation-log-sha256.txt"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("Phase 4 evidence hash manifest is missing or unsafe")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = _MANIFEST_ROW.fullmatch(line)
        if match is None:
            raise ValueError("Phase 4 evidence hash manifest contains an invalid row")
        expected, relative_text = match.groups()
        relative = Path(relative_text.replace("/", os.sep))
        if relative.is_absolute() or ".." in relative.parts or relative_text in rows:
            raise ValueError("Phase 4 evidence hash manifest contains an unsafe or duplicate path")
        unresolved = run / relative
        if unresolved.is_symlink():
            raise ValueError(f"Phase 4 evidence file is a forbidden symlink: {relative_text}")
        candidate = unresolved.resolve(strict=True)
        if not candidate.is_file():
            raise ValueError(f"Phase 4 evidence file is missing or unsafe: {relative_text}")
        try:
            candidate.relative_to(run)
        except ValueError as exc:
            raise ValueError("Phase 4 evidence path escaped its run directory") from exc
        if _sha256_file(candidate) != expected:
            raise ValueError(f"Phase 4 evidence SHA-256 mismatch: {relative_text}")
        rows[relative_text] = expected
    missing = _REQUIRED_EVIDENCE - set(rows)
    if missing:
        raise ValueError(
            "Phase 4 evidence manifest omits required files: " + ", ".join(sorted(missing))
        )
    all_paths = list(run.rglob("*"))
    symlinks = [path.relative_to(run).as_posix() for path in all_paths if path.is_symlink()]
    if symlinks:
        raise ValueError(
            "Phase 4 evidence directory contains forbidden symlinks: " + ", ".join(sorted(symlinks))
        )
    actual_files = {
        path.relative_to(run).as_posix()
        for path in all_paths
        if path.is_file() and path != manifest
    }
    if actual_files != set(rows):
        raise ValueError("Phase 4 evidence directory contains unhashed or missing files")
    return rows


def validate_acceptance_receipt(receipt: dict) -> None:
    if receipt.get("schema") != ACCEPTANCE_SCHEMA or receipt.get("status") != "accepted":
        raise ValueError("Unsupported or non-accepted Phase 4 receipt")
    decision_id = str(receipt.get("decision_id", ""))
    if _DECISION_ID.fullmatch(decision_id) is None:
        raise ValueError("Phase 4 acceptance decision ID is invalid")
    if not str(receipt.get("validation_run_id", "")).startswith("phase4-"):
        raise ValueError("Phase 4 acceptance validation-run identity is invalid")
    source_commit = str(receipt.get("validated_source_commit", "")).lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Phase 4 acceptance source commit is invalid")
    if not isinstance(receipt.get("validated_source_dirty"), bool):
        raise ValueError("Phase 4 acceptance source-dirty state must be boolean")
    if receipt.get("claim_boundary") != ACCEPTANCE_CLAIM_BOUNDARY:
        raise ValueError("Phase 4 acceptance claim boundary is invalid")
    expected = str(receipt.get("acceptance_receipt_sha256", "")).lower()
    stable = {
        key: value
        for key, value in receipt.items()
        if key not in {"created_at", "acceptance_receipt_sha256"}
    }
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or _canonical_sha256(stable) != expected:
        raise ValueError("Phase 4 acceptance receipt SHA-256 mismatch")
    for field in (
        "validation_summary_sha256",
        "validation_log_manifest_sha256",
        "development_freeze_candidate_sha256",
        "development_source_contract_sha256",
        "validator_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))):
            raise ValueError(f"Phase 4 acceptance receipt has invalid {field}")
    if (
        not isinstance(receipt.get("development_source_file_count"), int)
        or receipt["development_source_file_count"] < 1
    ):
        raise ValueError("Phase 4 acceptance receipt source-file count is invalid")


def acceptance_matches_freeze(receipt: dict, freeze_report: dict) -> None:
    validate_acceptance_receipt(receipt)
    validate_development_freeze_candidate(freeze_report)
    contract = development_source_contract(freeze_report)
    if (
        receipt["development_source_contract_sha256"] != contract["files_sha256"]
        or receipt["development_source_file_count"] != contract["file_count"]
    ):
        raise ValueError("Development freeze does not match the accepted Phase 4 source contract")


def build_acceptance_receipt(validation_run: str | Path, *, decision_id: str) -> dict:
    unresolved_run = Path(validation_run).expanduser()
    if unresolved_run.is_symlink():
        raise ValueError("Phase 4 validation run cannot be a symbolic link")
    run = unresolved_run.resolve(strict=True)
    if not run.is_dir():
        raise ValueError("Phase 4 validation run must be a real directory")
    if _DECISION_ID.fullmatch(str(decision_id)) is None:
        raise ValueError("Phase 4 acceptance decision ID must contain 3-80 safe characters")
    _verify_hash_manifest(run)
    summary_path = run / "validation-summary.json"
    summary = _load_object(summary_path)
    results = summary.get("results")
    if (
        summary.get("phase") != "phase4"
        or summary.get("passed") is not True
        or summary.get("failed_command_count") != 0
        or summary.get("command_count") != len(_REQUIRED_COMMAND_IDS)
        or not isinstance(results, list)
        or not results
        or any(not isinstance(row, dict) or row.get("status") != "PASS" for row in results)
    ):
        raise ValueError("Phase 4 validation summary is not a complete passing run")
    result_ids = [str(row.get("id", "")) for row in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != _REQUIRED_COMMAND_IDS:
        raise ValueError("Phase 4 validation summary does not match the required command set")
    freeze_report = _load_object(run / "evidence" / "development-freeze-candidate.json")
    validate_development_freeze_candidate(freeze_report)
    freeze_identity = dict(freeze_report["source_identity"])
    if (
        str(summary.get("source_commit", "")).lower()
        != str(freeze_identity.get("source_commit", "")).lower()
        or bool(summary.get("source_dirty"))
        != (not bool(freeze_identity.get("tracked_source_clean")))
        or str(summary.get("validator_sha256", "")).lower()
        != str(freeze_report["validator"]["sha256"]).lower()
    ):
        raise ValueError("Phase 4 summary and development-freeze candidate identities differ")
    contract = development_source_contract(freeze_report)
    stable = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "decision_id": str(decision_id),
        "validation_run_id": run.name,
        "validated_source_commit": str(summary.get("source_commit", "")),
        "validated_source_dirty": bool(summary.get("source_dirty")),
        "validation_summary_sha256": _sha256_file(summary_path),
        "validation_log_manifest_sha256": _sha256_file(run / "validation-log-sha256.txt"),
        "development_freeze_candidate_sha256": freeze_report["development_freeze_payload_sha256"],
        "development_source_contract_sha256": contract["files_sha256"],
        "development_source_file_count": contract["file_count"],
        "validator_sha256": str(summary.get("validator_sha256", "")),
        "claim_boundary": ACCEPTANCE_CLAIM_BOUNDARY,
    }
    receipt = {
        **stable,
        "created_at": _utcnow(),
        "acceptance_receipt_sha256": _canonical_sha256(stable),
    }
    validate_acceptance_receipt(receipt)
    return receipt


def write_acceptance_receipt(path: str | Path, receipt: dict) -> Path:
    validate_acceptance_receipt(receipt)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite Phase 4 acceptance: {destination}") from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.validation_run.expanduser().resolve(strict=True)
    output = arguments.output.expanduser().resolve()
    try:
        output.relative_to(run)
    except ValueError:
        pass
    else:
        parser.error("--output must remain outside the immutable Phase 4 validation run")
    freeze_report = _load_object(run / "evidence" / "development-freeze-candidate.json")
    source_root = Path(str(freeze_report.get("source_root", ""))).expanduser().resolve(strict=True)
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        parser.error("--output must remain outside the validated repository source root")
    receipt = build_acceptance_receipt(arguments.validation_run, decision_id=arguments.decision_id)
    destination = write_acceptance_receipt(arguments.output, receipt)
    print(destination)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
