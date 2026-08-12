"""Inventory-bound retirement of development-only policy lifecycle state.

Phase 4 uses only :meth:`PolicyRetirementManager.inventory` and ``dry_run``. The destructive
``execute`` path exists for the separately authorized post-freeze transition and cannot run without
an exact inventory, a matching authorization document, a clean durable source identity, and an
out-of-store immutable receipt destination.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from calo_rpd_studio.compute.source_identity import SourceIdentity, resolve_source_identity


INVENTORY_SCHEMA = "calo-policy-retirement-inventory-v1"
PLAN_SCHEMA = "calo-policy-retirement-plan-v1"
AUTHORIZATION_SCHEMA = "calo-policy-retirement-authorization-v1"
RECEIPT_SCHEMA = "calo-policy-retirement-receipt-v1"
AUTHORIZED_ACTION = "delete-exact-inventoried-development-policy-state"
_PROTECTED_STORE_FILES = {"AGENTS.md", "__init__.py"}
_AUTHORIZATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _write_json_exclusive(path: Path, payload: dict) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite policy-retirement evidence: {destination}"
        ) from exc
    return destination


def _assert_evidence_outside_store(path: str | Path, policy_store: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    root = Path(policy_store).expanduser().resolve(strict=True)
    try:
        destination.relative_to(root)
    except ValueError:
        return
    raise ValueError("Policy-retirement evidence must be retained outside the policy store")


def load_json_document(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return payload


class PolicyRetirementManager:
    """Prepare and, only after authorization, apply exact policy-store retirement."""

    def __init__(
        self,
        policy_store: str | Path,
        database,
        *,
        source_root: str | Path | None = None,
        source_identity: SourceIdentity | None = None,
    ) -> None:
        self.policy_store = Path(policy_store).expanduser().resolve(strict=True)
        if not self.policy_store.is_dir():
            raise ValueError(f"Policy store is not a directory: {self.policy_store}")
        if self.policy_store.is_symlink():
            raise ValueError("Policy store cannot be a symbolic link")
        self.database = database
        self.source_root = Path(source_root).expanduser().resolve() if source_root else None
        self._source_identity = source_identity

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _resolve_source_identity(self) -> SourceIdentity:
        if self._source_identity is not None:
            return self._source_identity
        return resolve_source_identity(cwd=self.source_root or self.policy_store)

    def _assert_external_evidence_path(self, path: Path, *, label: str) -> None:
        """Require destructive-transition evidence outside both store and source tree."""

        if self._is_within(path, self.policy_store):
            raise ValueError(f"{label} must be retained outside the policy store")
        if self.source_root is None:
            raise RuntimeError(
                "Policy retirement execution requires an explicit source root for evidence confinement"
            )
        source_root = self.source_root.resolve(strict=True)
        if not source_root.is_dir():
            raise ValueError(f"Policy-retirement source root is not a directory: {source_root}")
        if self._is_within(path, source_root):
            raise ValueError(f"{label} must be retained outside the repository source root")

    def _file_inventory(self) -> tuple[list[dict], list[dict]]:
        removable: list[dict] = []
        protected: list[dict] = []
        for candidate in sorted(self.policy_store.rglob("*"), key=lambda item: item.as_posix()):
            if candidate.is_symlink():
                raise ValueError(f"Policy store contains a forbidden symbolic link: {candidate}")
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            if not self._is_within(resolved, self.policy_store):
                raise ValueError(f"Policy-store entry escaped its root: {candidate}")
            relative = candidate.relative_to(self.policy_store).as_posix()
            entry = {
                "path": relative,
                "size_bytes": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
            if candidate.name in _PROTECTED_STORE_FILES:
                protected.append(entry)
            else:
                removable.append(entry)
        return removable, protected

    def _external_artifacts(self, database_snapshot: dict) -> list[dict]:
        output: list[dict] = []
        seen: set[str] = set()
        path_fields = (
            ("policies", "checkpoint_path"),
            ("policy_checkpoints", "checkpoint_path"),
            ("policy_checkpoints", "resume_path"),
        )
        for table, field in path_fields:
            for row in database_snapshot.get(table, []):
                raw = str(row.get(field, "") or "").strip()
                if not raw:
                    continue
                candidate = Path(raw).expanduser().resolve(strict=False)
                if self._is_within(candidate, self.policy_store) or not candidate.exists():
                    continue
                key = os.path.normcase(str(candidate))
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    {
                        "path": str(candidate),
                        "source_table": table,
                        "source_field": field,
                        "exists": True,
                        "is_file": candidate.is_file(),
                    }
                )
        return sorted(output, key=lambda item: str(item["path"]).casefold())

    def inventory(self) -> dict:
        """Return an exact read-only policy-store and database inventory."""

        removable, protected = self._file_inventory()
        database_snapshot = self.database.policy_lifecycle_snapshot()
        identity = self._resolve_source_identity()
        stable = {
            "schema": INVENTORY_SCHEMA,
            "policy_store_root": str(self.policy_store),
            "source_identity": identity.to_dict(),
            "removable_files": removable,
            "protected_files": protected,
            "database": database_snapshot,
            "external_existing_artifacts": self._external_artifacts(database_snapshot),
        }
        return {
            **stable,
            "created_at": _utcnow(),
            "inventory_sha256": _canonical_sha256(stable),
        }

    @staticmethod
    def validate_inventory(inventory: dict) -> None:
        if inventory.get("schema") != INVENTORY_SCHEMA:
            raise ValueError("Unsupported policy-retirement inventory schema")
        expected = str(inventory.get("inventory_sha256", ""))
        stable = {
            key: value
            for key, value in inventory.items()
            if key not in {"created_at", "inventory_sha256"}
        }
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or _canonical_sha256(stable) != expected:
            raise ValueError("Policy-retirement inventory SHA-256 mismatch")

    def dry_run(self, inventory: dict | None = None) -> dict:
        """Describe exact targets and blockers without changing files or database rows."""

        selected = inventory or self.inventory()
        self.validate_inventory(selected)
        if Path(str(selected["policy_store_root"])).resolve() != self.policy_store:
            raise ValueError("Inventory policy-store root does not match this manager")
        external = list(selected.get("external_existing_artifacts", []))
        database = dict(selected.get("database", {}))
        row_counts = {name: len(rows) for name, rows in sorted(database.items())}
        blockers = []
        if external:
            blockers.append(
                "Existing policy artifacts outside the designated policy store require a separate "
                "path-confined inventory and cannot be removed by this plan"
            )
        stable = {
            "schema": PLAN_SCHEMA,
            "inventory_sha256": selected["inventory_sha256"],
            "policy_store_root": str(self.policy_store),
            "source_identity": selected["source_identity"],
            "removable_files": list(selected.get("removable_files", [])),
            "protected_files": list(selected.get("protected_files", [])),
            "database_row_counts": row_counts,
            "external_existing_artifacts": external,
            "blockers": blockers,
            "destructive_action_executed": False,
            "requires_separate_post_freeze_authorization": True,
        }
        return {
            **stable,
            "created_at": _utcnow(),
            "plan_sha256": _canonical_sha256(stable),
        }

    @staticmethod
    def authorization_template(plan: dict) -> dict:
        PolicyRetirementManager.validate_plan(plan)
        return {
            "schema": AUTHORIZATION_SCHEMA,
            "authorization_granted": False,
            "authorization_id": "replace-with-explicit-post-freeze-authorization-id",
            "authorized_action": AUTHORIZED_ACTION,
            "inventory_sha256": str(plan.get("inventory_sha256", "")),
            "plan_sha256": str(plan.get("plan_sha256", "")),
            "policy_store_root": str(plan.get("policy_store_root", "")),
            "source_commit": str(dict(plan.get("source_identity", {})).get("source_commit", "")),
            "development_freeze_payload_sha256": "",
            "phase4_acceptance_receipt_sha256": "",
            "acknowledge_phase4_development_freeze_accepted": False,
            "acknowledge_irreversible_artifact_deletion": False,
            "acknowledge_database_lifecycle_cleanup": False,
        }

    @staticmethod
    def validate_plan(plan: dict) -> None:
        if plan.get("schema") != PLAN_SCHEMA:
            raise ValueError("Unsupported policy-retirement plan schema")
        expected = str(plan.get("plan_sha256", ""))
        stable = {
            key: value for key, value in plan.items() if key not in {"created_at", "plan_sha256"}
        }
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or _canonical_sha256(stable) != expected:
            raise ValueError("Policy-retirement plan SHA-256 mismatch")

    def _validate_authorization(self, authorization: dict, inventory: dict, plan: dict) -> str:
        if authorization.get("schema") != AUTHORIZATION_SCHEMA:
            raise ValueError("Unsupported policy-retirement authorization schema")
        if authorization.get("authorized_action") != AUTHORIZED_ACTION:
            raise ValueError("Authorization does not permit the required retirement action")
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not _AUTHORIZATION_ID.fullmatch(authorization_id):
            raise ValueError("Authorization ID must be a stable 3-80 character identifier")
        required_true = (
            "authorization_granted",
            "acknowledge_phase4_development_freeze_accepted",
            "acknowledge_irreversible_artifact_deletion",
            "acknowledge_database_lifecycle_cleanup",
        )
        if any(authorization.get(field) is not True for field in required_true):
            raise PermissionError("Policy removal lacks explicit post-freeze authorization")
        freeze_sha256 = str(authorization.get("development_freeze_payload_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", freeze_sha256):
            raise PermissionError(
                "Policy removal authorization must identify the accepted Phase 4 freeze payload"
            )
        acceptance_sha256 = str(authorization.get("phase4_acceptance_receipt_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", acceptance_sha256):
            raise PermissionError(
                "Policy removal authorization must identify the Phase 4 acceptance receipt"
            )
        comparisons = {
            "inventory_sha256": inventory["inventory_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "policy_store_root": str(self.policy_store),
            "source_commit": str(inventory["source_identity"]["source_commit"]),
        }
        for field, expected in comparisons.items():
            if str(authorization.get(field, "")) != str(expected):
                raise PermissionError(f"Authorization {field} does not match the reviewed plan")
        if plan.get("blockers"):
            raise RuntimeError("Policy-retirement plan has unresolved blockers")
        identity = self._resolve_source_identity()
        if not identity.durable_evidence_eligible:
            raise RuntimeError("Policy retirement requires a clean durable source identity")
        if identity.source_commit != str(authorization["source_commit"]):
            raise RuntimeError("Source commit changed after policy-removal authorization")
        return authorization_id

    def execute(
        self,
        *,
        inventory: dict,
        plan: dict,
        authorization: dict,
        authorization_path: str | Path,
        phase4_acceptance_path: str | Path,
        receipt_path: str | Path,
    ) -> dict:
        """Apply a separately authorized post-freeze retirement and retain an exact receipt."""

        self.validate_inventory(inventory)
        current = self.inventory()
        if current["inventory_sha256"] != inventory["inventory_sha256"]:
            raise RuntimeError("Policy store or lifecycle database changed after inventory")
        self.validate_plan(plan)
        expected_plan = self.dry_run(inventory)
        if plan["plan_sha256"] != expected_plan["plan_sha256"]:
            raise ValueError("Policy-retirement plan is not the exact dry-run for this inventory")
        if str(plan.get("inventory_sha256", "")) != inventory["inventory_sha256"]:
            raise ValueError("Policy-retirement plan is not bound to the supplied inventory")
        authorization_id = self._validate_authorization(authorization, inventory, plan)
        authorization_source = Path(authorization_path).expanduser().resolve(strict=True)
        self._assert_external_evidence_path(
            authorization_source,
            label="Policy-removal authorization",
        )
        if load_json_document(authorization_source) != authorization:
            raise PermissionError(
                "In-memory authorization differs from the exact retained authorization file"
            )
        acceptance_source = Path(phase4_acceptance_path).expanduser().resolve(strict=True)
        self._assert_external_evidence_path(
            acceptance_source,
            label="Phase 4 acceptance receipt",
        )
        from calo_rpd_studio.scripts.accept_development_freeze import (
            validate_acceptance_receipt,
        )

        acceptance = load_json_document(acceptance_source)
        validate_acceptance_receipt(acceptance)
        if (
            acceptance["acceptance_receipt_sha256"]
            != authorization["phase4_acceptance_receipt_sha256"]
        ):
            raise PermissionError(
                "Policy removal authorization does not match the retained Phase 4 acceptance receipt"
            )
        if (
            acceptance["development_freeze_candidate_sha256"]
            != authorization["development_freeze_payload_sha256"]
        ):
            raise PermissionError(
                "Policy removal authorization does not match the accepted Phase 4 candidate"
            )
        destination = Path(receipt_path).expanduser().resolve()
        self._assert_external_evidence_path(destination, label="Deletion receipt")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite deletion receipt: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Stage beside the externally retained receipt, never inside the governed store or source
        # tree by construction of the caller's evidence location. If database cleanup succeeds but
        # a later filesystem operation fails, surviving artifacts and their hashes are retained in
        # the immutable failure receipt. The name is one confined path segment because authorization
        # IDs cannot contain directory separators.
        staging = destination.parent / f".calo-policy-removal-staging-{authorization_id}"
        self._assert_external_evidence_path(staging, label="Policy-removal recovery staging")
        if staging.exists():
            raise FileExistsError(f"Refusing to reuse policy-removal staging directory: {staging}")
        moved: list[tuple[Path, Path]] = []
        database_cleared = False
        database_result: dict = {}
        status = "failed"
        error = ""
        try:
            for entry in inventory.get("removable_files", []):
                relative = Path(str(entry["path"]))
                source = (self.policy_store / relative).resolve(strict=True)
                if not self._is_within(source, self.policy_store) or source.is_symlink():
                    raise RuntimeError(f"Refusing unsafe policy-removal target: {source}")
                if _sha256_file(source) != str(entry["sha256"]):
                    raise RuntimeError(f"Policy-removal target changed after inventory: {relative}")
                staged = staging / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                source.replace(staged)
                moved.append((source, staged))

            database_result = self.database.clear_policy_lifecycle(
                expected_snapshot=inventory["database"]
            )
            database_cleared = True
            post = self.inventory()
            if post["removable_files"] or any(post["database"].values()):
                raise RuntimeError(
                    "Post-removal verification did not produce an empty policy store"
                )
            for _source, staged in reversed(moved):
                staged.unlink()
            for directory in sorted(
                (item for item in staging.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                directory.rmdir()
            staging.rmdir()
            status = "completed"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not database_cleared:
                for source, staged in reversed(moved):
                    if staged.exists() and not source.exists():
                        source.parent.mkdir(parents=True, exist_ok=True)
                        staged.replace(source)
                if staging.exists():
                    for directory in sorted(
                        (item for item in staging.rglob("*") if item.is_dir()),
                        key=lambda item: len(item.parts),
                        reverse=True,
                    ):
                        directory.rmdir()
                    if not any(staging.iterdir()):
                        staging.rmdir()
            raise
        finally:
            staged_artifacts = []
            for source, staged in moved:
                retained = staged.is_file()
                staged_artifacts.append(
                    {
                        "original_path": str(source),
                        "staging_path": str(staged),
                        "retained_for_recovery": retained,
                        "sha256": _sha256_file(staged) if retained else "",
                    }
                )
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "status": status,
                "created_at": _utcnow(),
                "authorization_id": authorization_id,
                "authorization_file": str(authorization_source),
                "authorization_file_sha256": _sha256_file(authorization_source),
                "authorization": authorization,
                "phase4_acceptance_file": str(acceptance_source),
                "phase4_acceptance_file_sha256": _sha256_file(acceptance_source),
                "phase4_acceptance": acceptance,
                "inventory": inventory,
                "plan": plan,
                "database_result": database_result,
                "database_cleared": database_cleared,
                "moved_target_count": len(moved),
                "recovery_staging_directory": str(staging),
                "staged_artifacts": staged_artifacts,
                "error": error,
            }
            receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
            _write_json_exclusive(destination, receipt)
        return receipt


def write_inventory(path: str | Path, inventory: dict) -> Path:
    PolicyRetirementManager.validate_inventory(inventory)
    _assert_evidence_outside_store(path, str(inventory["policy_store_root"]))
    return _write_json_exclusive(Path(path), inventory)


def write_plan(path: str | Path, plan: dict) -> Path:
    PolicyRetirementManager.validate_plan(plan)
    _assert_evidence_outside_store(path, str(plan["policy_store_root"]))
    return _write_json_exclusive(Path(path), plan)


def write_authorization_template(path: str | Path, plan: dict) -> Path:
    _assert_evidence_outside_store(path, str(plan["policy_store_root"]))
    return _write_json_exclusive(
        Path(path),
        PolicyRetirementManager.authorization_template(plan),
    )
