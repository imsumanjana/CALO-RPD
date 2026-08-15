"""Frozen one-action orchestration contract for independent TSH-CALO feasibility measurement.

The current workflow freezes the candidate's architecture/training-parameter contract and one
deterministic formal quality campaign. It has no registry, activation, or experiment-binding
authority. Legacy component-campaign helpers remain readable only for retained historical plans.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

from calo_rpd_studio.ai.model_io import checkpoint_sha256, durable_write_bytes

from .tsh_calo_component_ablation import (
    TSH_CALO_COMPONENT_ABLATION_CAMPAIGN_SCHEMA,
    TSHCALOComponentAblationPlan,
)
from .tsh_calo_policy_artifact import TSHCALOCandidateArtifact
from .tsh_calo_qualification_campaign import (
    TSHCALOQualificationPlan,
    _verify_component_evidence,
    qualification_candidate_contract,
)


TSH_CALO_AUTOMATIC_QUALIFICATION_PROTOCOL = "tsh-calo-one-action-feasibility-v1-transactional-cells"
AUTOMATIC_QUALIFICATION_CASES = ("case30", "case57")
AUTOMATIC_QUALIFICATION_RUNS = 30
AUTOMATIC_QUALIFICATION_POPULATION_SIZE = 20
AUTOMATIC_QUALIFICATION_MAX_EVALUATIONS = 10_000
AUTOMATIC_ABLATION_MASTER_SEED = 2_026_081_401
AUTOMATIC_QUALIFICATION_MASTER_SEED = 2_026_081_402
AUTOMATIC_ABLATION_LABEL_COUNT = 8
AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST = "automatic_source_snapshot_manifest.json"


def _remove_source_snapshot_staging(staging: Path) -> None:
    """Remove one exact temporary snapshot, including read-only Windows Git objects."""

    def make_writable_and_retry(function, path, _error_info) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(staging, onerror=make_writable_and_retry)


class AutomaticQualificationRejected(ValueError):
    """The frozen candidate workflow failed a technical integrity or compatibility boundary."""


@dataclass(frozen=True, slots=True)
class AutomaticQualificationSourceSnapshot:
    """An internally committed clean snapshot of one exact non-ignored source tree."""

    root: Path
    source_commit: str
    worktree_sha256: str
    manifest_sha256: str
    file_count: int


@dataclass(frozen=True, slots=True)
class AutomaticQualificationWorkspace:
    """Stable, candidate/source-bound paths used for exact restart and resume."""

    root: Path
    workflow_plan: Path
    qualification_plan: Path
    qualification_output: Path

    @classmethod
    def create(
        cls,
        base_directory: str | Path,
        *,
        candidate_sha256: str,
        source_commit: str,
        restart_ordinal: int = 0,
    ) -> "AutomaticQualificationWorkspace":
        base = Path(base_directory).expanduser().resolve()
        if int(restart_ordinal) < 0:
            raise ValueError("Automatic qualification restart ordinal cannot be negative")
        restart_suffix = f"-restart-{int(restart_ordinal):03d}" if restart_ordinal else ""
        identity = (
            f"architecture-v2-{candidate_sha256[:16].lower()}-{source_commit[:12].lower()}"
            f"{restart_suffix}"
        )
        root = base / identity
        return cls(
            root=root,
            workflow_plan=root / "automatic_qualification_workflow.json",
            qualification_plan=root / "formal_qualification_plan.json",
            qualification_output=root / "formal-qualification-evidence",
        )


def build_automatic_component_ablation_plan(
    *, candidate_path: str | Path, candidate_sha256: str, source_commit: str
) -> TSHCALOComponentAblationPlan:
    """Build the versioned, outcome-independent A--E prerequisite plan."""

    identity = f"auto-ae-{candidate_sha256[:16].lower()}-{source_commit[:12].lower()}"
    plan = TSHCALOComponentAblationPlan(
        campaign_id=identity,
        source_commit=source_commit.lower(),
        candidate_path=str(Path(candidate_path).expanduser().resolve()),
        candidate_sha256=candidate_sha256.lower(),
        development_cases=AUTOMATIC_QUALIFICATION_CASES,
        runs=AUTOMATIC_QUALIFICATION_RUNS,
        master_seed=AUTOMATIC_ABLATION_MASTER_SEED,
        population_size=AUTOMATIC_QUALIFICATION_POPULATION_SIZE,
        max_evaluations=AUTOMATIC_QUALIFICATION_MAX_EVALUATIONS,
        source_tracked_clean=True,
    )
    plan.validate()
    return plan


def _isolated_git_environment(extra: dict | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(name, None)
    environment.update(extra or {})
    return environment


def _source_worktree_manifest(repository_root: str | Path) -> tuple[dict, list[tuple[Path, Path]]]:
    root = Path(repository_root).expanduser().resolve()
    git_environment = _isolated_git_environment()
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            env=git_environment,
        )
        base_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                env=git_environment,
            )
            .stdout.strip()
            .lower()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Automatic qualification requires an inspectable Git source tree"
        ) from exc
    entries: list[dict] = []
    files: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        relative_text = os.fsdecode(encoded)
        relative = Path(relative_text)
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("Git source inventory escaped the repository root") from exc
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Automatic source snapshot refuses a non-regular file: {relative}")
        normalized = relative.as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        digest = checkpoint_sha256(source)
        size = source.stat().st_size
        entries.append({"path": normalized, "sha256": digest, "bytes": size})
        files.append((relative, source))
    entries.sort(key=lambda item: item["path"])
    files.sort(key=lambda item: item[0].as_posix())
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (
        {
            "schema_version": "tsh-calo-automatic-source-snapshot-v1",
            "base_commit": base_commit,
            "files": entries,
            "worktree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        },
        files,
    )


def _run_snapshot_git(arguments: list[str], *, root: Path, environment: dict | None = None) -> str:
    process_environment = _isolated_git_environment(environment)
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=process_environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"Automatic qualification source snapshot failed: {detail}") from exc
    return result.stdout.strip()


def prepare_automatic_source_snapshot(
    repository_root: str | Path,
    snapshot_base_directory: str | Path,
) -> AutomaticQualificationSourceSnapshot:
    """Freeze the current non-ignored worktree in a separate deterministic clean Git commit."""

    source_root = Path(repository_root).expanduser().resolve()
    snapshot_base = Path(snapshot_base_directory).expanduser().resolve()
    snapshot_base.mkdir(parents=True, exist_ok=True)
    before, files = _source_worktree_manifest(source_root)
    staging = Path(tempfile.mkdtemp(prefix="source-snapshot-", dir=snapshot_base)).resolve()
    try:
        expected_hashes = {item["path"]: item["sha256"] for item in before["files"]}
        for relative, source in files:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            expected = expected_hashes[relative.as_posix()]
            if checkpoint_sha256(destination) != expected:
                raise RuntimeError(f"Source changed while snapshotting: {relative.as_posix()}")
        manifest_path = staging / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST
        if manifest_path.exists():
            raise ValueError(
                f"Source tree uses the reserved snapshot path: {AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST}"
            )
        encoded = json.dumps(before, indent=2, sort_keys=True, allow_nan=False) + "\n"
        durable_write_bytes(manifest_path, encoded.encode("utf-8"))
        after, _after_files = _source_worktree_manifest(source_root)
        if after["worktree_sha256"] != before["worktree_sha256"]:
            raise RuntimeError(
                "Source changed while the automatic qualification snapshot was built"
            )
        _run_snapshot_git(["init", "-q", "--object-format=sha1"], root=staging)
        _run_snapshot_git(["config", "core.autocrlf", "false"], root=staging)
        _run_snapshot_git(["config", "core.filemode", "false"], root=staging)
        _run_snapshot_git(["add", "--all"], root=staging)
        fixed_environment = {
            "GIT_AUTHOR_NAME": "CALO-RPD Qualification",
            "GIT_AUTHOR_EMAIL": "qualification@calo-rpd.invalid",
            "GIT_COMMITTER_NAME": "CALO-RPD Qualification",
            "GIT_COMMITTER_EMAIL": "qualification@calo-rpd.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
        _run_snapshot_git(
            ["commit", "-q", "-m", "Frozen automatic qualification source"],
            root=staging,
            environment=fixed_environment,
        )
        source_commit = _run_snapshot_git(["rev-parse", "HEAD"], root=staging).lower()
        if len(source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in source_commit
        ):
            raise RuntimeError("Automatic qualification snapshot commit identity is invalid")
        if _run_snapshot_git(["status", "--porcelain", "--untracked-files=all"], root=staging):
            raise RuntimeError("Automatic qualification source snapshot is not clean")
        destination = snapshot_base / source_commit
        if destination.exists():
            existing_commit = _run_snapshot_git(["rev-parse", "HEAD"], root=destination).lower()
            existing_status = _run_snapshot_git(
                ["status", "--porcelain", "--untracked-files=all"], root=destination
            )
            existing_manifest = json.loads(
                (destination / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST).read_text(encoding="utf-8")
            )
            if (
                existing_commit != source_commit
                or existing_status
                or existing_manifest.get("worktree_sha256") != before["worktree_sha256"]
            ):
                raise RuntimeError(
                    "Existing automatic qualification source snapshot is incompatible"
                )
            _remove_source_snapshot_staging(staging)
        else:
            os.replace(staging, destination)
        return AutomaticQualificationSourceSnapshot(
            root=destination,
            source_commit=source_commit,
            worktree_sha256=str(before["worktree_sha256"]),
            manifest_sha256=checkpoint_sha256(destination / AUTOMATIC_SOURCE_SNAPSHOT_MANIFEST),
            file_count=len(before["files"]),
        )
    except BaseException:
        if staging.exists():
            try:
                _remove_source_snapshot_staging(staging)
            except OSError:
                pass
        raise


def accepted_component_references(
    output_directory: str | Path,
    *,
    plan: TSHCALOComponentAblationPlan,
) -> dict[str, dict]:
    """Return exact A--E references only from the completed frozen ablation campaign."""

    output = Path(output_directory).expanduser().resolve()
    evidence_path = output / "campaign_evidence.json"
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The frozen A-E component campaign is not complete") from exc
    if not isinstance(payload, dict):
        raise ValueError("The A-E component campaign evidence must be a JSON object")
    if payload.get("schema_version") != TSH_CALO_COMPONENT_ABLATION_CAMPAIGN_SCHEMA:
        raise ValueError("The A-E component campaign evidence schema is incompatible")
    if str(payload.get("execution_plan_sha256", "")).lower() != (plan.execution_plan_sha256()):
        raise ValueError("The A-E component evidence belongs to another frozen plan")
    if str(payload.get("source_policy_sha256", "")).lower() != plan.candidate_sha256:
        raise ValueError("The A-E component evidence belongs to another policy")
    if str(payload.get("source_commit", "")).lower() != plan.source_commit.lower():
        raise ValueError("The A-E component evidence belongs to another source architecture")
    references = dict(payload.get("component_evidence", {}) or {})
    if set(references) != set("ABCDE"):
        raise ValueError("The A-E component campaign did not retain all five component decisions")
    rejected = sorted(
        component for component, item in references.items() if not bool(item.get("accepted"))
    )
    if rejected or payload.get("all_A_E_accepted") is not True:
        labels = ", ".join(rejected) if rejected else "one or more A-E gates"
        raise AutomaticQualificationRejected(
            f"Policy rejected: frozen component evidence did not accept {labels}"
        )
    resolved: dict[str, dict] = {}
    for component, item in sorted(references.items()):
        path = Path(str(item.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = output / path
        path = path.resolve()
        resolved[component] = {
            "path": str(path),
            "sha256": str(item.get("sha256", "")).lower(),
            "accepted": True,
        }
    return resolved


def build_automatic_formal_qualification_plan(
    *,
    candidate_path: str | Path,
    candidate_sha256: str,
    source_commit: str,
    candidate_artifact: TSHCALOCandidateArtifact | None = None,
    component_evidence: dict[str, dict] | None = None,
    restart_ordinal: int = 0,
) -> TSHCALOQualificationPlan:
    """Build the exact formal plan from a frozen, stage-neutral candidate contract."""

    if candidate_artifact is None and not component_evidence:
        raise ValueError("Automatic qualification requires an inspected candidate artifact")

    if int(restart_ordinal) < 0:
        raise ValueError("Automatic qualification restart ordinal cannot be negative")
    restart_suffix = f"-restart-{int(restart_ordinal):03d}" if restart_ordinal else ""
    identity = (
        f"auto-formal-{candidate_sha256[:16].lower()}-{source_commit[:12].lower()}{restart_suffix}"
    )
    plan = TSHCALOQualificationPlan(
        qualification_run_id=identity,
        source_commit=source_commit.lower(),
        candidate_path=str(Path(candidate_path).expanduser().resolve()),
        candidate_sha256=candidate_sha256.lower(),
        development_cases=AUTOMATIC_QUALIFICATION_CASES,
        runs=AUTOMATIC_QUALIFICATION_RUNS,
        master_seed=AUTOMATIC_QUALIFICATION_MASTER_SEED,
        population_size=AUTOMATIC_QUALIFICATION_POPULATION_SIZE,
        max_evaluations=AUTOMATIC_QUALIFICATION_MAX_EVALUATIONS,
        source_tracked_clean=True,
        mode="formal",
        candidate_contract=(
            qualification_candidate_contract(candidate_artifact)
            if candidate_artifact is not None
            else {}
        ),
        component_evidence=dict(component_evidence or {}),
    )
    plan.validate()
    if component_evidence:
        _verify_component_evidence(plan)
    return plan


def frozen_qualification_restart_design(plan: TSHCALOQualificationPlan) -> dict:
    """Return every frozen operative field, excluding only new-run provenance identities."""

    plan.validate()
    payload = plan.to_dict()
    for field_name in (
        "qualification_run_id",
        "source_commit",
        "source_tracked_clean",
        "candidate_path",
    ):
        payload.pop(field_name, None)
    return payload


def frozen_qualification_restart_design_sha256(plan: TSHCALOQualificationPlan) -> str:
    """Bind a fresh infrastructure restart to the unchanged preregistered design."""

    encoded = json.dumps(
        frozen_qualification_restart_design(plan),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def automatic_qualification_workflow_payload(
    *, qualification_plan: TSHCALOQualificationPlan
) -> dict:
    """Preregister one immutable architecture and non-decisional feasibility campaign."""

    qualification_plan.validate()
    if not qualification_plan.candidate_contract:
        raise ValueError("Current automatic qualification requires a candidate contract")
    return {
        "schema_version": TSH_CALO_AUTOMATIC_QUALIFICATION_PROTOCOL,
        "source_commit": qualification_plan.source_commit,
        "source_tracked_clean": True,
        "candidate_path": qualification_plan.candidate_path,
        "candidate_sha256": qualification_plan.candidate_sha256,
        "candidate_contract": dict(qualification_plan.candidate_contract),
        "formal_qualification_plan": qualification_plan.to_dict(),
        "transition": (
            "run_or_exactly_resume_formal; admit_integrity_valid_measurements; "
            "scientist_selection_separate"
        ),
        "resume_count_limit": None,
        "finite_budgets_immutable": True,
        "automatic_activation": False,
        "automatic_experiment_binding": False,
        "automatic_suitability_decision": False,
    }


def freeze_plan(path: str | Path, payload: dict) -> str:
    """Create an immutable JSON plan or prove an existing plan is byte-equivalent in meaning."""

    target = Path(path).expanduser().resolve()
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Frozen plan is unreadable: {target}") from exc
        if existing != payload:
            raise ValueError(f"Frozen plan already exists with different content: {target}")
        return str(checkpoint_sha256(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    durable_write_bytes(target, encoded.encode("utf-8"))
    return str(checkpoint_sha256(target))


def automatic_qualification_workload() -> dict[str, int]:
    """Expose exact optimizer-cell counts for UI disclosure and contract tests."""

    cases = len(AUTOMATIC_QUALIFICATION_CASES)
    qualification_cells = cases * AUTOMATIC_QUALIFICATION_RUNS * 2
    return {
        "cases": cases,
        "runs_per_case": AUTOMATIC_QUALIFICATION_RUNS,
        "evaluations_per_cell": AUTOMATIC_QUALIFICATION_MAX_EVALUATIONS,
        "qualification_cells": qualification_cells,
        "total_cells": qualification_cells,
    }
