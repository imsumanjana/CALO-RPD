"""Run the current v6.2.1 frozen 20-algorithm benchmark campaign from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calo_rpd_studio.benchmarking.campaign import (
    BenchmarkCampaignConfig,
    build_campaign,
    write_campaign_plan,
)
from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.benchmarking.package import TransactionsPackageBuilder
from calo_rpd_studio.benchmarking.validation import validate_campaign
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import run_sequential_resilient
from calo_rpd_studio.experiments.parallel_runner import run_parallel_resilient
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.results.result_store import ResultStore


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Run the frozen CALO-RPD v6.2.1 benchmark campaign."
    )
    command.add_argument("--database", default="calo_rpd_results.sqlite")
    command.add_argument("--output", default="benchmark_v600a4")
    command.add_argument("--runs", type=int, default=30)
    command.add_argument("--budget", type=int, default=5000)
    command.add_argument("--population", type=int, default=50)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--cases", default="case30,case57,case118,case300")
    command.add_argument(
        "--studies",
        default="deterministic,mixed,load_mean_risk,renewable_cvar,branch_worst_case",
    )
    command.add_argument("--freeze-manifest", default=None)
    command.add_argument("--validate", action="store_true")
    command.add_argument("--package", action="store_true")
    return command


def _update_manifest(path: Path, task_index: int, experiment_id: str, status: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tasks"][task_index]["experiment_id"] = experiment_id
    payload["tasks"][task_index]["status"] = status
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parser().parse_args()
    campaign = BenchmarkCampaignConfig(
        cases=tuple(item.strip() for item in args.cases.split(",") if item.strip()),
        study_keys=tuple(item.strip() for item in args.studies.split(",") if item.strip()),
        runs=args.runs,
        max_evaluations=args.budget,
        population_size=args.population,
        master_seed=args.seed,
        output_directory=args.output,
        parallel_workers=args.workers,
    )
    if args.freeze_manifest:
        campaign.freeze_manifest = args.freeze_manifest
    verification = verify_freeze_manifest(campaign.freeze_manifest)
    if not verification.passed:
        raise SystemExit(verification.message)

    tasks = build_campaign(campaign, base_config=ExperimentConfig())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_campaign_plan(campaign, tasks, output / "campaign_manifest.json")
    database = ResultDatabase(args.database)

    for task in tasks:
        # Re-check before every TEST task so a source/policy change cannot silently enter a later
        # portion of the same final campaign.
        verification = verify_freeze_manifest(campaign.freeze_manifest)
        if not verification.passed:
            raise SystemExit(verification.message)
        experiment_id = database.create_experiment(task.config, collect_provenance())
        database.set_experiment_learning_role(experiment_id, "test", eligible=False, locked=True)
        _update_manifest(manifest, task.task_index, experiment_id, "running")
        print(
            f"[{task.task_index + 1}/{len(tasks)}] {task.task_id} · {task.planned_jobs} optimizer jobs"
        )
        store = ResultStore(task.config.output_directory)
        completed, failed = (
            run_parallel_resilient(task.config)
            if task.config.parallel_workers > 1
            else run_sequential_resilient(task.config)
        )
        for run in completed:
            arrays = store.save_arrays(run.result)
            database.add_run(experiment_id, run, str(arrays))
        for failure in failed:
            database.add_failure(experiment_id, failure)
        status = "completed" if not failed else "completed_with_failures"
        _update_manifest(manifest, task.task_index, experiment_id, status)
        print(f"  completed={len(completed)} failed={len(failed)} experiment={experiment_id}")

    if args.validate:
        summary = validate_campaign(database, manifest, only_unverified=True)
        print("Validation:", summary)
    if args.package:
        archive = TransactionsPackageBuilder(database).build(
            campaign_manifest=manifest,
            output_directory=output / "transactions_research_package",
            freeze_manifest=campaign.freeze_manifest,
        )
        print("Research package:", archive.resolve())
    print("Campaign manifest:", manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
