"""Validate, start, or resume an independent TSH-CALO qualification campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    TSHCALOQualificationCampaign,
    TSHCALOQualificationPlan,
)


ROOT = Path(__file__).resolve().parents[2]


def load_plan(path: str | Path) -> TSHCALOQualificationPlan:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"TSH-CALO qualification plan is unreadable: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("TSH-CALO qualification plan must be a JSON object")
    return TSHCALOQualificationPlan.from_dict(payload)


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
        raise RuntimeError(
            "TSH-CALO qualification requires an inspectable Git source tree"
        ) from exc
    return head, tracked_status


def validate_repository_for_plan(
    plan: TSHCALOQualificationPlan, *, root: str | Path = ROOT
) -> None:
    head, tracked_status = repository_state(root)
    if head.lower() != plan.source_commit.lower():
        raise RuntimeError("TSH-CALO qualification plan source commit does not match the checkout")
    if tracked_status:
        raise RuntimeError("TSH-CALO qualification requires a clean tracked working tree")


def _summary(plan: TSHCALOQualificationPlan, *, state: str, result=None) -> dict:
    payload = {
        "state": state,
        "qualification_run_id": plan.qualification_run_id,
        "mode": plan.mode,
        "source_policy_sha256": plan.candidate_sha256,
        "scientific_design_sha256": plan.scientific_design_sha256(),
        "execution_plan_sha256": plan.execution_plan_sha256(),
        "seed_manifest_sha256": plan.seed_manifest_sha256(),
        "authority_boundary": "independent_qualification_only",
        "registration_performed": False,
        "activation_performed": False,
    }
    if result is not None:
        payload.update(result)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Frozen JSON qualification plan")
    parser.add_argument("--output", type=Path, help="New or explicitly resumed evidence directory")
    parser.add_argument("--resume", action="store_true", help="Resume retained plan/run cells")
    parser.add_argument("--check", action="store_true", help="Validate without executing")
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
    campaign = TSHCALOQualificationCampaign(plan, arguments.output)
    result = campaign.resume() if arguments.resume else campaign.start()
    state = "completed_qualified" if result["passed"] else "completed_not_qualified"
    print(json.dumps(_summary(plan, state=state, result=result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
