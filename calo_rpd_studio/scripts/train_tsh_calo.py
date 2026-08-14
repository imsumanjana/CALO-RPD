"""Explicitly validate, start, or resume an independent frozen TSH-CALO training campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    IndependentTSHCALOTrainingCampaign,
    TSHCALOTrainingCampaignPlan,
    TSHCALOTrainingPauseRequested,
    TSH_CALO_TRAINING_EVENT_SCHEMA,
    TSH_CALO_TRAINING_PAUSE_EXIT_CODE,
    parse_tsh_calo_extension_plan,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training import (
    preflight_tsh_calo_training_resources,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_extension import (
    IndependentTSHCALOTrainingExtension,
    extension_plan_summary,
)
from calo_rpd_studio.scripts.create_development_freeze_candidate import (
    validate_development_freeze_candidate,
)
from calo_rpd_studio.scripts.accept_development_freeze import (
    acceptance_matches_freeze,
    validate_acceptance_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINING_EVENT_PREFIX = "CALO_TRAINING_EVENT "


def emit_training_event(event: dict) -> None:
    """Emit one machine-readable checkpoint event while keeping stdout human-inspectable."""

    payload = dict(event)
    if payload.get("schema_version") != TSH_CALO_TRAINING_EVENT_SCHEMA:
        raise ValueError("TSH-CALO training progress event schema is incompatible")
    print(TRAINING_EVENT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def load_plan(
    path: str | Path,
    *,
    compatible_extension: bool = False,
) -> TSHCALOTrainingCampaignPlan:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"TSH-CALO training plan is unreadable: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("TSH-CALO training plan must be a JSON object")
    if compatible_extension:
        return parse_tsh_calo_extension_plan(payload)
    return TSHCALOTrainingCampaignPlan.from_dict(payload)


def repository_state(root: str | Path = ROOT) -> tuple[str, str]:
    directory = Path(root).resolve()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("TSH-CALO training requires an inspectable Git source tree") from exc
    return head, tracked_status


def validate_repository_for_plan(
    plan: TSHCALOTrainingCampaignPlan,
    *,
    root: str | Path = ROOT,
    compatible_extension: bool = False,
) -> str:
    head, tracked_status = repository_state(root)
    if not compatible_extension and head.lower() != plan.source_commit.lower():
        raise RuntimeError(
            "TSH-CALO training plan source commit does not match the checked-out repository"
        )
    if tracked_status:
        raise RuntimeError("TSH-CALO training requires a clean non-ignored source tree")
    if (
        not compatible_extension
        and plan.development_freeze_commit
        and plan.source_commit.lower() != plan.development_freeze_commit.lower()
    ):
        raise RuntimeError("TSH-CALO training source does not match the development freeze")
    return head.lower()


def validate_development_freeze_for_plan(
    plan: TSHCALOTrainingCampaignPlan,
    path: str | Path,
    acceptance_path: str | Path,
) -> dict:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Development-freeze report is unreadable: {source}") from exc
    if not isinstance(report, dict):
        raise ValueError("Development-freeze report must be a JSON object")
    validate_development_freeze_candidate(report, require_training_eligible=True)
    acceptance_source = Path(acceptance_path).expanduser().resolve(strict=True)
    try:
        acceptance = json.loads(acceptance_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Phase 4 acceptance receipt is unreadable: {acceptance_source}") from exc
    if not isinstance(acceptance, dict):
        raise ValueError("Phase 4 acceptance receipt must be a JSON object")
    validate_acceptance_receipt(acceptance)
    acceptance_matches_freeze(acceptance, report)
    identity = dict(report.get("source_identity", {}) or {})
    comparisons = {
        "source commit": (
            str(identity.get("source_commit", "")).lower(),
            plan.development_freeze_commit.lower(),
        ),
        "payload SHA-256": (
            str(report.get("development_freeze_payload_sha256", "")).lower(),
            plan.development_freeze_sha256.lower(),
        ),
        "acceptance receipt SHA-256": (
            str(acceptance.get("acceptance_receipt_sha256", "")).lower(),
            plan.phase4_acceptance_sha256.lower(),
        ),
    }
    for label, (actual, expected) in comparisons.items():
        if actual != expected:
            raise ValueError(f"Training plan development-freeze {label} does not match the report")
    return report


def _summary(
    plan: TSHCALOTrainingCampaignPlan,
    *,
    state: str,
    result=None,
    resource_preflight: dict | None = None,
) -> dict:
    payload: dict[str, object] = {
        "state": state,
        "campaign_id": plan.campaign_id,
        "scientific_design_sha256": plan.scientific_design_hash(),
        "execution_plan_sha256": plan.execution_plan_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
        "source_commit": plan.source_commit,
        "development_freeze_commit": plan.development_freeze_commit,
        "development_freeze_sha256": plan.development_freeze_sha256,
        "phase4_acceptance_sha256": plan.phase4_acceptance_sha256,
        "authority_boundary": "independent_training_only",
    }
    if result is not None:
        payload.update(
            {
                "output_directory": result.output_directory,
                "member_candidate_sha256": [item.sha256 for item in result.member_candidates],
                "ensemble_candidate_sha256": result.ensemble_candidate.sha256,
                "manifest_path": result.manifest_path,
                "manifest_sha256": result.manifest_sha256,
                "lifecycle_status": "candidate_unqualified",
            }
        )
    if resource_preflight is not None:
        payload["resource_preflight"] = resource_preflight
    return payload


def validate_training_resources(plan: TSHCALOTrainingCampaignPlan) -> dict:
    """Apply the trainer's current Safe-80 admission without starting training."""

    return preflight_tsh_calo_training_resources(plan.training_config(plan.members[0]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Frozen JSON training plan")
    parser.add_argument(
        "--development-freeze",
        type=Path,
        help="Legacy accepted development-freeze report",
    )
    parser.add_argument(
        "--phase4-acceptance",
        type=Path,
        help="Legacy accepted source-contract receipt",
    )
    parser.add_argument("--output", type=Path, help="New or explicitly resumed output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly resume the exact authenticated campaign state",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help=(
            "Explicitly run or resume one finite extension segment from authenticated completed "
            "trainer state"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate plan and source tree without starting training",
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.resume:
        parser.error("--check and --resume are mutually exclusive")
    if arguments.extend and arguments.resume:
        parser.error("--extend already resumes a pending extension when present")
    if not arguments.check and arguments.output is None:
        parser.error("--output is required unless --check is used")
    if arguments.extend and arguments.output is None:
        parser.error("--extend requires the completed campaign --output directory")
    if bool(arguments.development_freeze) != bool(arguments.phase4_acceptance):
        parser.error("legacy authority paths must be supplied together")

    plan = load_plan(arguments.plan, compatible_extension=arguments.extend)
    execution_source_commit = validate_repository_for_plan(
        plan,
        compatible_extension=arguments.extend,
    )
    if arguments.development_freeze is not None:
        validate_development_freeze_for_plan(
            plan,
            arguments.development_freeze,
            arguments.phase4_acceptance,
        )
    elif not arguments.extend and any(
        (
            plan.development_freeze_commit,
            plan.development_freeze_sha256,
            plan.phase4_acceptance_sha256,
        )
    ):
        parser.error("this legacy plan requires its internal authority records")
    if arguments.check:
        resource_preflight = validate_training_resources(plan)
        if arguments.extend:
            extension = extension_plan_summary(plan, arguments.output)
            print(
                json.dumps(
                    {
                        **_summary(
                            plan,
                            state="extension_validated_not_started",
                            resource_preflight=resource_preflight,
                        ),
                        "extension": extension,
                    },
                    sort_keys=True,
                )
            )
            return 0
        print(
            json.dumps(
                _summary(
                    plan,
                    state="validated_not_started",
                    resource_preflight=resource_preflight,
                ),
                sort_keys=True,
            )
        )
        return 0

    print(
        TRAINING_EVENT_PREFIX
        + json.dumps(
            {
                "schema_version": TSH_CALO_TRAINING_EVENT_SCHEMA,
                "event": "process_started",
                "campaign_id": plan.campaign_id,
                "member_count": len(plan.members),
                "episode_count": sum(len(member.episodes) for member in plan.members),
                "total_candidate_evaluations": sum(
                    len(member.episodes) for member in plan.members
                )
                * plan.max_evaluations,
                "progress_percent": 0,
                "resume": bool(arguments.resume),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    campaign = None
    extension = None
    if arguments.extend:
        extension = IndependentTSHCALOTrainingExtension(
            plan,
            arguments.output,
            event_callback=emit_training_event,
            execution_source_commit=execution_source_commit,
        )
    else:
        campaign = IndependentTSHCALOTrainingCampaign(
            plan,
            arguments.output,
            event_callback=emit_training_event,
        )
    try:
        if extension is not None:
            result = extension.start_or_resume()
        else:
            assert campaign is not None
            result = campaign.resume() if arguments.resume else campaign.start()
    except TSHCALOTrainingPauseRequested as exc:
        paused_output = (
            extension.segment_directory
            if extension is not None and extension.segment_directory is not None
            else arguments.output.expanduser().resolve()
        )
        print(
            json.dumps(
                {
                    **_summary(plan, state="paused_resumable"),
                    "output_directory": str(paused_output),
                    "pause": exc.event,
                    "resumable": True,
                    "extension": extension is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return TSH_CALO_TRAINING_PAUSE_EXIT_CODE
    print(
        json.dumps(
            {
                **_summary(
                    plan,
                    state=(
                        "completed_unqualified_extension"
                        if extension is not None
                        else "completed_unqualified"
                    ),
                    result=result,
                ),
                "extension": extension is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
