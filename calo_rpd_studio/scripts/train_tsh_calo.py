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
            ["git", "status", "--porcelain", "--untracked-files=no"],
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
        raise RuntimeError("TSH-CALO training requires a clean tracked working tree")


def _summary(plan: TSHCALOTrainingCampaignPlan, *, state: str, result=None) -> dict:
    payload = {
        "state": state,
        "campaign_id": plan.campaign_id,
        "scientific_design_sha256": plan.scientific_design_hash(),
        "execution_plan_sha256": plan.execution_plan_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
        "source_commit": plan.source_commit,
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

    plan = load_plan(arguments.plan)
    validate_repository_for_plan(plan)
    if arguments.check:
        print(json.dumps(_summary(plan, state="validated_not_started"), sort_keys=True))
        return 0

    campaign = IndependentTSHCALOTrainingCampaign(plan, arguments.output)
    result = campaign.resume() if arguments.resume else campaign.start()
    print(json.dumps(_summary(plan, state="completed_unqualified", result=result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
