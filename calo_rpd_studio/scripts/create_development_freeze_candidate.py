"""Create a source-bound Phase 4 development-freeze candidate report.

This is not a release manifest, release candidate, qualification receipt, or policy artifact. It
records implementation interfaces and claim boundaries for manual Phase 4 engineering validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from calo_rpd_studio.algorithms.calo.policy_retirement import PolicyRetirementManager
from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.version import DISPLAY_VERSION, VERSION, VERSION_STAGE


DEVELOPMENT_FREEZE_SCHEMA = "calo-v12-phase4-development-freeze-candidate-v1"
COMPLETE_SOURCE_MANIFEST_SCHEMA = "calo-v12-phase4-complete-source-manifest-v1"
INTERFACE_FILES = {
    "algorithm_registry": "calo_rpd_studio/algorithms/registry.py",
    "algorithm_configuration": "calo_rpd_studio/algorithms/calo/tsh_calo_schema.py",
    "application_state": "calo_rpd_studio/app/state_manager.py",
    "canonical_transition": "calo_rpd_studio/algorithms/calo/tsh_calo_transition_kernel.py",
    "container_smoke": "calo_rpd_studio/scripts/container_smoke.py",
    "runtime": "calo_rpd_studio/algorithms/calo/tsh_calo_optimizer.py",
    "experiment_runner": "calo_rpd_studio/experiments/experiment_runner.py",
    "experiment_config": "calo_rpd_studio/experiments/experiment_config.py",
    "counted_evaluator": "calo_rpd_studio/orpd/problem.py",
    "training": "calo_rpd_studio/algorithms/calo/tsh_calo_training.py",
    "training_campaign": "calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py",
    "training_command": "calo_rpd_studio/scripts/train_tsh_calo.py",
    "phase4_acceptance": "calo_rpd_studio/scripts/accept_development_freeze.py",
    "training_session": "calo_rpd_studio/algorithms/calo/tsh_calo_training_session.py",
    "training_environment": "calo_rpd_studio/algorithms/calo/tsh_calo_training_environment.py",
    "policy_abi": "calo_rpd_studio/algorithms/calo/tsh_calo_policy_artifact.py",
    "policy_inference": "calo_rpd_studio/algorithms/calo/tsh_calo_inference.py",
    "policy_readiness": "calo_rpd_studio/algorithms/calo/policy_readiness.py",
    "policy_registry": "calo_rpd_studio/algorithms/calo/policy_registry.py",
    "policy_retirement": "calo_rpd_studio/algorithms/calo/policy_retirement.py",
    "policy_lifecycle_database": "calo_rpd_studio/results/database.py",
    "phase4_validation_instructions": "validation/PHASE4_VALIDATION.md",
    "accounting_receipt": "calo_rpd_studio/algorithms/calo/tsh_calo_training_receipt.py",
    "qualification_authority": "calo_rpd_studio/algorithms/calo/tsh_calo_qualification.py",
    "qualification_campaign": "calo_rpd_studio/algorithms/calo/tsh_calo_qualification_campaign.py",
    "physics_repair": "calo_rpd_studio/algorithms/calo/tsh_calo_physics_repair.py",
    "decoder": "calo_rpd_studio/orpd/variable_decoder.py",
    "experiment_schema": "calo_rpd_studio/data/schemas/experiment_config.schema.json",
}
IGNORED_DECLARED_INTERFACE_FILES = {"validation/PHASE4_VALIDATION.md"}
DEPENDENCY_FILES = (
    "requirements-lock-ci-py311-linux.txt",
    "requirements-lock-cpu-py311-linux.txt",
    "requirements-lock-cuda128-py311-linux.txt",
    "requirements-lock-cpu.txt",
)
CONTAINER_FILES = ("Dockerfile", "compose.yaml", ".dockerignore")
EXCLUSION_FILES = (
    ".gitignore",
    ".dockerignore",
    "MANIFEST.in",
    "pyproject.toml",
    "calo_rpd_studio/scripts/verify_distribution_stage.py",
)


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


def _valid_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_file_record(record: Any, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Development-freeze {label} must be a file record")
    path = str(record.get("path", ""))
    size = record.get("size_bytes")
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError(f"Development-freeze {label} has an invalid relative path")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"Development-freeze {label} has an invalid file size")
    if not _valid_sha256(record.get("sha256")):
        raise ValueError(f"Development-freeze {label} has an invalid SHA-256")


def _validate_record_list(
    value: Any,
    *,
    label: str,
    expected_paths: tuple[str, ...],
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Development-freeze {label} must be a non-empty file-record list")
    for index, record in enumerate(value):
        _validate_file_record(record, label=f"{label}[{index}]")
    paths = [str(record["path"]) for record in value]
    if paths != list(expected_paths):
        raise ValueError(f"Development-freeze {label} does not match the declared source contract")


def _is_development_policy_record(record: dict) -> bool:
    path = str(record.get("path", "")).replace("\\", "/").lower()
    parts = tuple(part for part in path.split("/") if part)
    name = parts[-1] if parts else ""
    in_policy_store = "calo_rpd_studio/data/trained_models/" in path
    protected_marker = name in {"agents.md", "__init__.py"}
    # Only the designated policy store may be removed from the accepted production-source
    # contract.  A policy-looking suffix elsewhere in the repository is still ordinary source
    # content and must not become an untracked bypass around source-bound acceptance.
    return bool(in_policy_store and not protected_marker)


def development_source_contract(report: dict) -> dict:
    """Return the production-code content identity that survives old-policy retirement."""

    validate_development_freeze_candidate(report)
    records = [
        dict(record)
        for record in report["complete_source_manifest"]["files"]
        if not _is_development_policy_record(record)
    ]
    return {
        "schema": "calo-v12-phase4-development-source-contract-v1",
        "file_count": len(records),
        "files_sha256": _canonical_sha256(records),
    }


def validate_development_freeze_candidate(
    report: dict,
    *,
    require_training_eligible: bool = False,
) -> None:
    """Validate an exact retained development-freeze report without trusting its filename."""

    if report.get("schema") != DEVELOPMENT_FREEZE_SCHEMA:
        raise ValueError("Unsupported development-freeze candidate schema")
    expected = str(report.get("development_freeze_payload_sha256", "")).lower()
    stable = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "development_freeze_payload_sha256"}
    }
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or _canonical_sha256(stable) != expected
    ):
        raise ValueError("Development-freeze candidate payload SHA-256 mismatch")
    if report.get("status") != "development_freeze_candidate":
        raise ValueError("Development-freeze report does not represent a freeze candidate")
    identity = dict(report.get("source_identity", {}) or {})
    commit = str(identity.get("source_commit", "")).strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Development-freeze report requires a full source commit")
    if not isinstance(identity.get("tracked_source_clean"), bool):
        raise ValueError("Development-freeze source cleanliness must be boolean")
    if identity.get("durable_evidence_eligible") is not identity.get("tracked_source_clean"):
        raise ValueError("Development-freeze durable source eligibility is inconsistent")

    _validate_file_record(report.get("validator"), label="validator")
    interfaces = report.get("interfaces")
    if not isinstance(interfaces, dict) or not interfaces:
        raise ValueError("Development-freeze interfaces must be a non-empty mapping")
    if set(interfaces) != set(INTERFACE_FILES):
        raise ValueError("Development-freeze interfaces do not match the declared source contract")
    for name, record in interfaces.items():
        if not str(name).strip():
            raise ValueError("Development-freeze interface names must be non-empty")
        _validate_file_record(record, label=f"interfaces.{name}")
        if record["path"] != INTERFACE_FILES[name]:
            raise ValueError(f"Development-freeze interface path is invalid: {name}")
    for key, expected_paths in (
        ("dependencies", DEPENDENCY_FILES),
        ("containers", CONTAINER_FILES),
        ("distribution_exclusions", EXCLUSION_FILES),
    ):
        _validate_record_list(report.get(key), label=key, expected_paths=expected_paths)

    complete_source = report.get("complete_source_manifest")
    if (
        not isinstance(complete_source, dict)
        or complete_source.get("schema") != COMPLETE_SOURCE_MANIFEST_SCHEMA
    ):
        raise ValueError("Development-freeze complete source manifest is missing or incompatible")
    if not _valid_sha256(complete_source.get("source_status_sha256")):
        raise ValueError("Development-freeze complete source-status SHA-256 is invalid")
    if complete_source.get("source_status_clean") is not identity.get("tracked_source_clean"):
        raise ValueError("Development-freeze complete source status conflicts with source identity")
    source_files = complete_source.get("files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("Development-freeze complete source manifest cannot be empty")
    if complete_source.get("file_count") != len(source_files):
        raise ValueError("Development-freeze complete source file count does not match")
    source_paths: list[str] = []
    for index, record in enumerate(source_files):
        _validate_file_record(record, label=f"complete_source_manifest.files[{index}]")
        source_paths.append(str(record["path"]))
    if source_paths != sorted(set(source_paths)):
        raise ValueError("Development-freeze complete source paths must be sorted and unique")
    complete_source_set = set(source_paths)
    required_source_set = (
        (set(INTERFACE_FILES.values()) - IGNORED_DECLARED_INTERFACE_FILES)
        | set(DEPENDENCY_FILES)
        | set(CONTAINER_FILES)
        | set(EXCLUSION_FILES)
    )
    if not required_source_set.issubset(complete_source_set):
        raise ValueError("Development-freeze complete source manifest omits required source")

    policy_inventory = report.get("policy_inventory")
    if not isinstance(policy_inventory, dict) or not _valid_sha256(
        policy_inventory.get("inventory_sha256")
    ):
        raise ValueError("Development-freeze policy inventory identity is invalid")
    counts = policy_inventory.get("database_row_counts")
    if not isinstance(counts, dict) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise ValueError("Development-freeze policy database counts are invalid")
    removable_count = policy_inventory.get("removable_development_file_count")
    external_count = policy_inventory.get("external_existing_artifact_count")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (removable_count, external_count)
    ):
        raise ValueError("Development-freeze policy artifact counts are invalid")
    if (
        policy_inventory.get("release_scope_policy_count") != 0
        or policy_inventory.get("old_policy_removal_executed") is not False
    ):
        raise ValueError("Development-freeze cannot include or delete policy state")
    policy_empty = not removable_count and not external_count and not any(counts.values())
    if policy_inventory.get("old_policy_transition_pending") is not (not policy_empty):
        raise ValueError("Development-freeze policy-transition state is inconsistent")

    policy_scope = report.get("policy_scope")
    if not isinstance(policy_scope, dict) or any(
        policy_scope.get(key) is not False
        for key in (
            "qualified_policy_in_development_freeze",
            "active_policy_in_development_freeze",
            "final_policy_in_development_freeze",
        )
    ):
        raise ValueError("Development-freeze policy authority boundary is invalid")
    if policy_scope.get("future_policy_initialization_policy_sha256") != "":
        raise ValueError("Development-freeze future-policy initialization must be empty")

    prohibited = {
        "policy_training",
        "policy_evaluation",
        "policy_qualification",
        "policy_registration",
        "policy_activation",
        "policy_deletion",
        "protected_case_campaign",
        "release_publication",
    }
    if not prohibited.issubset(set(report.get("commands_not_executed", []))):
        raise ValueError("Development-freeze prohibited-workflow record is incomplete")
    expected_training_eligible = bool(identity.get("durable_evidence_eligible") and policy_empty)
    if report.get("post_transition_training_eligible") is not expected_training_eligible:
        raise ValueError("Development-freeze training-eligibility state is inconsistent")
    if require_training_eligible and (
        identity.get("durable_evidence_eligible") is not True
        or report.get("post_transition_training_eligible") is not True
    ):
        raise ValueError(
            "New-policy training requires a clean, empty-policy, post-transition development freeze"
        )


def _file_record(root: Path, relative: str) -> dict:
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"Development-freeze input cannot be a symbolic link: {relative}")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Development-freeze input is not a regular file: {relative}")
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Development-freeze input escaped the source root: {relative}") from exc
    return {
        "path": normalized,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _records(root: Path, paths: tuple[str, ...]) -> list[dict]:
    return [_file_record(root, path) for path in paths]


def _complete_source_manifest(root: Path) -> dict:
    """Hash every tracked or non-ignored untracked file participating in this source tree."""

    try:
        enumeration = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Unable to enumerate the complete Git source tree") from exc
    relative_paths = sorted(
        {Path(os.fsdecode(raw)).as_posix() for raw in enumeration.stdout.split(b"\0") if raw}
    )
    if not relative_paths:
        raise RuntimeError("The complete Git source tree is empty")
    return {
        "schema": COMPLETE_SOURCE_MANIFEST_SCHEMA,
        "enumeration": "git_ls_files_cached_and_nonignored_untracked",
        "source_status_sha256": hashlib.sha256(status.stdout).hexdigest(),
        "source_status_clean": not bool(status.stdout),
        "file_count": len(relative_paths),
        "files": [_file_record(root, relative) for relative in relative_paths],
    }


def verify_development_freeze_source(report: dict, source_root: str | Path) -> dict:
    """Re-hash the retained report against its live complete source and ignored validator."""

    validate_development_freeze_candidate(report)
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Source root is not a directory: {root}")
    if Path(str(report.get("source_root", ""))).expanduser().resolve() != root:
        raise ValueError("Development-freeze report source root does not match the live source")
    identity = resolve_source_identity(cwd=root)
    if identity.to_dict() != report["source_identity"]:
        raise ValueError("Development-freeze report source identity no longer matches")
    actual_manifest = _complete_source_manifest(root)
    if actual_manifest != report["complete_source_manifest"]:
        raise ValueError("Development-freeze complete source manifest no longer matches")
    validator = report["validator"]
    if _file_record(root, str(validator["path"])) != validator:
        raise ValueError("Development-freeze validator identity no longer matches")
    return {
        "source_commit": identity.source_commit,
        "source_clean": identity.tracked_source_clean,
        "source_status_sha256": actual_manifest["source_status_sha256"],
        "source_file_count": actual_manifest["file_count"],
        "validator_sha256": validator["sha256"],
        "development_freeze_payload_sha256": report["development_freeze_payload_sha256"],
    }


def build_development_freeze_candidate(
    source_root: str | Path,
    *,
    policy_inventory: dict,
    validator_path: str | Path,
    require_clean: bool = False,
) -> dict:
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Source root is not a directory: {root}")
    PolicyRetirementManager.validate_inventory(policy_inventory)
    identity = resolve_source_identity(cwd=root)
    if require_clean and not identity.durable_evidence_eligible:
        raise RuntimeError(
            "A clean full Git source identity is required for the development freeze"
        )
    inventory_identity = dict(policy_inventory.get("source_identity", {}))
    if (
        str(inventory_identity.get("source_commit", "")) != identity.source_commit
        or bool(inventory_identity.get("tracked_source_clean")) != identity.tracked_source_clean
    ):
        raise ValueError("Policy inventory and development freeze have different source identities")

    validator = Path(validator_path).expanduser().resolve(strict=True)
    if not validator.is_file():
        raise ValueError(f"Phase 4 validator is not a file: {validator}")
    try:
        validator_relative = validator.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Phase 4 validator must remain inside the source root") from exc
    policy_database = dict(policy_inventory.get("database", {}))
    policy_row_counts = {name: len(rows) for name, rows in sorted(policy_database.items())}
    removable = list(policy_inventory.get("removable_files", []))
    external = list(policy_inventory.get("external_existing_artifacts", []))
    policy_state_empty = not removable and not external and not any(policy_row_counts.values())
    stable = {
        "schema": DEVELOPMENT_FREEZE_SCHEMA,
        "status": "development_freeze_candidate",
        "version": VERSION,
        "display_version": DISPLAY_VERSION,
        "version_stage": VERSION_STAGE,
        "source_root": str(root),
        "source_identity": identity.to_dict(),
        "complete_source_manifest": _complete_source_manifest(root),
        "validator": {
            "path": validator_relative,
            "size_bytes": int(validator.stat().st_size),
            "sha256": _sha256_file(validator),
        },
        "interfaces": {
            name: _file_record(root, relative) for name, relative in sorted(INTERFACE_FILES.items())
        },
        "dependencies": _records(root, DEPENDENCY_FILES),
        "containers": _records(root, CONTAINER_FILES),
        "distribution_exclusions": _records(root, EXCLUSION_FILES),
        "policy_inventory": {
            "inventory_sha256": str(policy_inventory["inventory_sha256"]),
            "removable_development_file_count": len(removable),
            "external_existing_artifact_count": len(external),
            "database_row_counts": policy_row_counts,
            "release_scope_policy_count": 0,
            "old_policy_removal_executed": False,
            "old_policy_transition_pending": bool(
                removable or external or any(policy_row_counts.values())
            ),
        },
        "post_transition_training_eligible": bool(
            identity.durable_evidence_eligible and policy_state_empty
        ),
        "supported_execution_modes": ["cuda_preferred", "cpu_only"],
        "prohibited_execution_modes": ["intel_xpu"],
        "resource_admission_ceiling": {
            "fraction": 0.80,
            "cuda_basis": "currently_free_vram",
            "cpu_basis": "currently_available_ram",
        },
        "scientific_scope": {
            "approved": ["TSH-CALO-A", "TSH-CALO-B", "TSH-CALO-C", "TSH-CALO-D", "TSH-CALO-E"],
            "experimental_disabled_by_default": ["TSH-CALO-F"],
        },
        "policy_scope": {
            "qualified_policy_in_development_freeze": False,
            "active_policy_in_development_freeze": False,
            "final_policy_in_development_freeze": False,
            "existing_policy_state_is_development_only_and_excluded": True,
            "future_policy_must_be_new_and_trained_against_exact_freeze_commit": True,
            "future_policy_initialization_policy_sha256": "",
        },
        "commands_executed_by_report": [
            "read_source_identity",
            "enumerate_complete_nonignored_source",
            "hash_complete_source_and_declared_inputs",
        ],
        "commands_not_executed": [
            "policy_training",
            "policy_evaluation",
            "policy_qualification",
            "policy_registration",
            "policy_activation",
            "policy_deletion",
            "protected_case_campaign",
            "release_publication",
        ],
        "claim_limitations": [
            "development freeze only; not a release candidate or final release",
            "no policy benefit, superiority, qualification, or protected-case claim",
            "automation does not infer human screen-reader, usability, or scientist acceptance",
            "physical CUDA, package, container, and GUI claims require retained validator evidence",
        ],
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    report = {
        **stable,
        "created_at": _utcnow(),
        "development_freeze_payload_sha256": _canonical_sha256(stable),
    }
    validate_development_freeze_candidate(report)
    return report


def write_report(path: str | Path, report: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite development-freeze evidence: {destination}"
        ) from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--policy-inventory", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args()
    inventory = json.loads(arguments.policy_inventory.read_text(encoding="utf-8"))
    report = build_development_freeze_candidate(
        arguments.source_root,
        policy_inventory=inventory,
        validator_path=arguments.validator,
        require_clean=arguments.require_clean,
    )
    write_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
