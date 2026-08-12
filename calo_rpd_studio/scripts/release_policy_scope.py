"""Create or verify an explicit v12 release-policy scope decision.

The command never selects, trains, qualifies, activates, deletes, or packages a policy.  It writes
only a disabled template, or verifies a separately approved immutable decision and its referenced
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    TSH_CALO_QUALIFICATION_RECEIPT_KEY,
    load_tsh_calo_qualification_receipt,
)
from calo_rpd_studio.scripts.accept_development_freeze import validate_acceptance_receipt


SCOPE_SCHEMA = "calo-v12-release-policy-scope-decision-v1"
POLICY_MANIFEST_SCHEMA = "calo-v12-qualified-policy-manifest-v1"
POLICY_FREE = "policy-free"
NEWLY_QUALIFIED_POLICY = "newly-qualified-policy"
_DECISION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def disabled_template() -> dict:
    """Return a non-authorizing decision template for later owner completion."""

    return {
        "schema": SCOPE_SCHEMA,
        "approval_granted": False,
        "decision_id": "replace-with-explicit-release-scope-decision-id",
        "selected_scope": "",
        "release_line": "12.0",
        "phase4_acceptance_receipt_sha256": "",
        "post_transition_freeze_sha256": "",
        "development_source_contract_sha256": "",
        "policy_id": "",
        "policy_sha256": "",
        "policy_manifest_path": "",
        "policy_manifest_sha256": "",
        "qualification_receipt_path": "",
        "qualification_receipt_file_sha256": "",
        "qualification_receipt_sha256": "",
        "algorithm_id": "",
        "initialization_policy_sha256": "",
        "change_f_enabled": False,
        "policy_benefit_claims_permitted": False,
        "old_policy_reuse_permitted": False,
        "acknowledge_scope_is_exact_and_immutable": False,
        "acknowledge_no_automatic_policy_activation": False,
        "decision_payload_sha256": "",
    }


def _stable(decision: dict) -> dict:
    return {key: value for key, value in decision.items() if key != "decision_payload_sha256"}


def validate_scope_decision(
    decision: dict,
    *,
    acceptance_receipt: dict | None = None,
    evidence_root: str | Path | None = None,
) -> dict:
    """Validate one exact approved policy-free or newly-qualified-policy decision."""

    if decision.get("schema") != SCOPE_SCHEMA:
        raise ValueError("Unsupported release-policy scope schema")
    if decision.get("approval_granted") is not True:
        raise PermissionError("Release-policy scope has not been explicitly approved")
    decision_id = str(decision.get("decision_id", ""))
    if _DECISION_ID.fullmatch(decision_id) is None:
        raise ValueError("Release-policy decision ID must contain 3-80 safe characters")
    if decision.get("release_line") != "12.0":
        raise ValueError("Release-policy decision is not for the v12.0 release line")
    scope = str(decision.get("selected_scope", ""))
    if scope not in {POLICY_FREE, NEWLY_QUALIFIED_POLICY}:
        raise ValueError("Release-policy decision must select exactly one supported scope")
    if decision.get("old_policy_reuse_permitted") is not False:
        raise PermissionError("Old-policy reuse must remain prohibited")
    if any(
        decision.get(field) is not True
        for field in (
            "acknowledge_scope_is_exact_and_immutable",
            "acknowledge_no_automatic_policy_activation",
        )
    ):
        raise PermissionError(
            "Release-policy decision lacks the required explicit acknowledgements"
        )
    for field in (
        "phase4_acceptance_receipt_sha256",
        "post_transition_freeze_sha256",
        "development_source_contract_sha256",
    ):
        if _SHA256.fullmatch(str(decision.get(field, "")).lower()) is None:
            raise ValueError(f"Release-policy decision has an invalid {field}")

    if scope == POLICY_FREE:
        forbidden = (
            "policy_id",
            "policy_sha256",
            "policy_manifest_path",
            "policy_manifest_sha256",
            "qualification_receipt_path",
            "qualification_receipt_file_sha256",
            "qualification_receipt_sha256",
            "algorithm_id",
            "initialization_policy_sha256",
        )
        if any(str(decision.get(field, "")) for field in forbidden):
            raise ValueError("Policy-free release scope cannot identify or include a policy")
        if decision.get("policy_benefit_claims_permitted") is not False:
            raise PermissionError("Policy-free scope cannot permit policy-benefit claims")
        if decision.get("change_f_enabled") is not False:
            raise PermissionError("Change F must remain disabled")
    else:
        for field in (
            "policy_sha256",
            "policy_manifest_sha256",
            "qualification_receipt_file_sha256",
            "qualification_receipt_sha256",
        ):
            if _SHA256.fullmatch(str(decision.get(field, "")).lower()) is None:
                raise ValueError(f"New-policy release scope has an invalid {field}")
        if not str(decision.get("policy_id", "")):
            raise ValueError("New-policy release scope must identify exactly one policy")
        if decision.get("algorithm_id") != TSH_CALO_ALGORITHM_ID:
            raise ValueError("Release policy must use the TSH-CALO A-E/F-off ABI")
        if str(decision.get("initialization_policy_sha256", "")):
            raise PermissionError("Release policy cannot initialize from an old policy")
        if decision.get("change_f_enabled") is not False:
            raise PermissionError("Change F must remain disabled in the release policy")
        if not isinstance(decision.get("policy_benefit_claims_permitted"), bool):
            raise ValueError("Policy-benefit claim permission must be an explicit boolean")
        if evidence_root is None:
            raise ValueError("New-policy release scope requires an evidence root")
        root = Path(evidence_root).expanduser().resolve(strict=True)
        manifest = (root / str(decision.get("policy_manifest_path", ""))).resolve(strict=True)
        try:
            manifest.relative_to(root)
        except ValueError as exc:
            raise ValueError("Policy manifest escaped its evidence root") from exc
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError("Policy manifest must be a regular immutable evidence file")
        if _sha256_file(manifest) != str(decision["policy_manifest_sha256"]).lower():
            raise ValueError("Policy manifest SHA-256 mismatch")
        qualification = (root / str(decision.get("qualification_receipt_path", ""))).resolve(
            strict=True
        )
        try:
            qualification.relative_to(root)
        except ValueError as exc:
            raise ValueError("Qualification receipt escaped its evidence root") from exc
        if not qualification.is_file() or qualification.is_symlink():
            raise ValueError("Qualification receipt must be a regular immutable evidence file")
        if (
            _sha256_file(qualification)
            != str(decision["qualification_receipt_file_sha256"]).lower()
        ):
            raise ValueError("Qualification receipt file SHA-256 mismatch")
        try:
            qualification_payload = json.loads(qualification.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Qualification receipt is not a readable JSON document") from exc
        loaded_qualification = load_tsh_calo_qualification_receipt(
            {TSH_CALO_QUALIFICATION_RECEIPT_KEY: qualification_payload},
            expected_policy_sha256=str(decision["policy_sha256"]),
        )
        if (
            loaded_qualification.receipt_sha256
            != str(decision["qualification_receipt_sha256"]).lower()
        ):
            raise ValueError("Qualification receipt payload SHA-256 mismatch")
        try:
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Policy manifest is not a readable JSON document") from exc
        if not isinstance(manifest_payload, dict) or manifest_payload.get("schema") != (
            POLICY_MANIFEST_SCHEMA
        ):
            raise ValueError("Policy manifest has an unsupported schema")
        comparisons = {
            "policy_id": decision["policy_id"],
            "policy_sha256": decision["policy_sha256"],
            "algorithm_id": TSH_CALO_ALGORITHM_ID,
            "initialization_policy_sha256": "",
            "change_f_enabled": False,
            "qualification_receipt_sha256": decision["qualification_receipt_sha256"],
            "phase4_acceptance_receipt_sha256": decision["phase4_acceptance_receipt_sha256"],
            "post_transition_freeze_sha256": decision["post_transition_freeze_sha256"],
        }
        for field, expected_value in comparisons.items():
            if manifest_payload.get(field) != expected_value:
                raise ValueError(f"Policy manifest {field} does not match the scope decision")
        artifact_relative = str(manifest_payload.get("policy_artifact_path", ""))
        artifact = (root / artifact_relative).resolve(strict=True)
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError("Policy artifact escaped its evidence root") from exc
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError("Release policy artifact must be a regular immutable evidence file")
        if _sha256_file(artifact) != str(decision["policy_sha256"]).lower():
            raise ValueError("Release policy artifact SHA-256 mismatch")

    if acceptance_receipt is not None:
        validate_acceptance_receipt(acceptance_receipt)
        if (
            decision["phase4_acceptance_receipt_sha256"]
            != acceptance_receipt["acceptance_receipt_sha256"]
            or decision["development_source_contract_sha256"]
            != acceptance_receipt["development_source_contract_sha256"]
        ):
            raise ValueError("Release-policy scope does not match the Phase 4 acceptance receipt")

    expected = str(decision.get("decision_payload_sha256", "")).lower()
    if _SHA256.fullmatch(expected) is None or _canonical_sha256(_stable(decision)) != expected:
        raise ValueError("Release-policy decision payload SHA-256 mismatch")
    return {
        "schema": SCOPE_SCHEMA,
        "decision_id": decision_id,
        "selected_scope": scope,
        "decision_payload_sha256": expected,
    }


def write_exclusive(path: str | Path, payload: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite release-policy evidence: {destination}"
        ) from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--decision", type=Path, required=True)
    verify.add_argument("--phase4-acceptance", type=Path)
    verify.add_argument("--evidence-root", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "template":
        print(write_exclusive(arguments.output, disabled_template()))
        return 0
    decision = json.loads(arguments.decision.read_text(encoding="utf-8-sig"))
    acceptance = (
        json.loads(arguments.phase4_acceptance.read_text(encoding="utf-8-sig"))
        if arguments.phase4_acceptance
        else None
    )
    print(
        json.dumps(
            validate_scope_decision(
                decision,
                acceptance_receipt=acceptance,
                evidence_root=arguments.evidence_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
