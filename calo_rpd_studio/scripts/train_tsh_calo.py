"""Explicitly validate, start, or resume an independent frozen TSH-CALO training campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    IndependentTSHCALOTrainingCampaign,
    TSHCALOTrainingCampaignPlan,
)
from calo_rpd_studio.scripts.create_development_freeze_candidate import (
    validate_development_freeze_candidate,
)
from calo_rpd_studio.scripts.accept_development_freeze import (
    acceptance_matches_freeze,
    validate_acceptance_receipt,
)


ROOT = Path(__file__).resolve().parents[2]


def load_plan(path: str | Path) -> TSHCALOTrainingCampaignPlan:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"TSH-CALO training plan is unreadable: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("TSH-CALO training plan must be a JSON object")
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
) -> None:
    head, tracked_status = repository_state(root)
    if head.lower() != plan.source_commit.lower():
        raise RuntimeError(
            "TSH-CALO training plan source commit does not match the checked-out repository"
        )
    if tracked_status:
        raise RuntimeError("TSH-CALO training requires a clean non-ignored source tree")
    if (
        plan.development_freeze_commit
        and plan.source_commit.lower() != plan.development_freeze_commit.lower()
    ):
        raise RuntimeError("TSH-CALO training source does not match the development freeze")


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


def _summary(plan: TSHCALOTrainingCampaignPlan, *, state: str, result=None) -> dict:
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
    return payload


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
        "--check",
        action="store_true",
        help="Validate plan and source tree without starting training",
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.resume:
        parser.error("--check and --resume are mutually exclusive")
    if not arguments.check and arguments.output is None:
        parser.error("--output is required unless --check is used")
    if bool(arguments.development_freeze) != bool(arguments.phase4_acceptance):
        parser.error("legacy authority paths must be supplied together")

    plan = load_plan(arguments.plan)
    validate_repository_for_plan(plan)
    if arguments.development_freeze is not None:
        validate_development_freeze_for_plan(
            plan,
            arguments.development_freeze,
            arguments.phase4_acceptance,
        )
    elif any(
        (
            plan.development_freeze_commit,
            plan.development_freeze_sha256,
            plan.phase4_acceptance_sha256,
        )
    ):
        parser.error("this legacy plan requires its internal authority records")
    if arguments.check:
        print(json.dumps(_summary(plan, state="validated_not_started"), sort_keys=True))
        return 0

    campaign = IndependentTSHCALOTrainingCampaign(plan, arguments.output)
    result = campaign.resume() if arguments.resume else campaign.start()
    print(json.dumps(_summary(plan, state="completed_unqualified", result=result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
