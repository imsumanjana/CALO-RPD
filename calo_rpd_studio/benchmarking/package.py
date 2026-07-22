"""Transactions-level campaign evidence package builder."""

from __future__ import annotations

from calo_rpd_studio.version import VERSION

import json
from dataclasses import asdict
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from .evidence import build_campaign_evidence
from .freeze import verify_freeze_manifest
from calo_rpd_studio.visualization.publication_evidence import generate_campaign_figures
from calo_rpd_studio.visualization.font_preflight import font_resolution_manifest


class TransactionsPackageBuilder:
    def __init__(self, database):
        self.database = database

    def build(
        self,
        *,
        campaign_manifest: str | Path,
        output_directory: str | Path,
        freeze_manifest: str | Path,
    ) -> Path:
        campaign_manifest = Path(campaign_manifest)
        payload = json.loads(campaign_manifest.read_text(encoding="utf-8"))
        tasks = payload.get("tasks", [])
        incomplete = [
            task.get("task_id", "unknown")
            for task in tasks
            if task.get("status") != "completed" or not task.get("experiment_id")
        ]
        if incomplete:
            raise ValueError(
                "A final Transactions research package requires every planned benchmark task to complete without recorded job failures. Incomplete tasks: "
                + ", ".join(incomplete)
            )
        task_experiments = {task["task_id"]: task["experiment_id"] for task in tasks}
        if not task_experiments:
            raise ValueError("The campaign manifest contains no completed benchmark tasks.")
        verification = verify_freeze_manifest(freeze_manifest)
        if not verification.passed:
            raise RuntimeError(verification.message)

        root = Path(output_directory)
        tables = root / "tables"
        figures = root / "figures"
        raw = root / "raw_results"
        reports = root / "reports"
        configs = root / "experiment_configurations"
        validation = root / "validation"
        frozen = root / "frozen_calo"
        for directory in (tables, figures, raw, reports, configs, validation, frozen):
            directory.mkdir(parents=True, exist_ok=True)

        total_runs = sum(
            len(self.database.list_runs(experiment_id))
            for experiment_id in task_experiments.values()
        )
        total_verified = sum(
            len(self.database.list_runs(experiment_id, verified_only=True))
            for experiment_id in task_experiments.values()
        )
        if total_runs <= 0 or total_verified != total_runs:
            raise ValueError(
                f"Article-ready export requires every stored run to be independently verified; found {total_verified}/{total_runs} verified runs."
            )
        evidence = build_campaign_evidence(self.database, task_experiments, verified_only=True)
        (reports / "campaign_evidence_verified_only.json").write_text(
            json.dumps(evidence.to_dict(), indent=2, allow_nan=True), encoding="utf-8"
        )
        (reports / "automatic_interpretation_verified_only.txt").write_text(
            "\n".join(evidence.interpretations) + "\n", encoding="utf-8"
        )
        (reports / "evidence_basis.json").write_text(
            json.dumps(
                {
                    "total_runs": total_runs,
                    "verified_runs": total_verified,
                    "font_preflight": font_resolution_manifest(),
                    "publication_claim_basis": (
                        "verified_only"
                        if total_runs > 0 and total_verified == total_runs
                        else "incomplete_validation"
                    ),
                    "warning": (
                        "All campaign runs are independently verified."
                        if total_runs > 0 and total_verified == total_runs
                        else "Publication-level claims should not be finalized until all intended benchmark runs are independently verified."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (reports / "freeze_verification.json").write_text(
            json.dumps(asdict(verification), indent=2), encoding="utf-8"
        )
        shutil.copy2(campaign_manifest, root / "campaign_manifest.json")
        shutil.copy2(freeze_manifest, root / "frozen_calo_manifest.json")
        freeze_payload = json.loads(Path(freeze_manifest).read_text(encoding="utf-8"))
        project_root = Path(__file__).resolve().parents[2]
        for relative in freeze_payload.get("files", {}):
            source = project_root / relative
            if source.is_file():
                destination = frozen / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

        article_lines = [
            f"# CALO-RPD v{VERSION} — Article-ready evidence summary",
            "",
            "## Evidence basis",
            f"- Completed campaign tasks: {len(task_experiments)}",
            f"- Total stored runs: {total_runs}",
            f"- Independently verified runs: {total_verified}",
            f"- Frozen CALO verification: {verification.message}",
            "",
            "## Automatic evidence-based interpretation",
        ]
        article_lines.extend(f"- {line}" for line in evidence.interpretations)
        article_lines.extend(
            [
                "",
                "## Claim discipline",
                "Final superiority claims should be based on the independently verified evidence tables and corrected nonparametric tests. No universal superiority statement is generated automatically.",
                "",
            ]
        )
        (reports / "article_ready_evidence_summary.md").write_text(
            "\n".join(article_lines), encoding="utf-8"
        )

        descriptive_rows = []
        pairwise_rows = []
        for task_id, task in evidence.task_summaries.items():
            experiment = self.database.get_experiment(task["experiment_id"])
            if experiment:
                (configs / f"{task_id}.json").write_text(
                    experiment["config_json"], encoding="utf-8"
                )
            for algorithm, summary in task["algorithms"].items():
                objective = summary.get("objective", {})
                runtime = summary.get("runtime_seconds", {})
                first = summary.get("evaluations_to_first_feasible", {})
                descriptive_rows.append(
                    {
                        "task": task_id,
                        "algorithm": algorithm,
                        "runs": summary.get("runs", 0),
                        "feasible_run_rate": summary.get("feasible_run_rate", 0.0),
                        "best": objective.get("best"),
                        "mean": objective.get("mean"),
                        "median": objective.get("median"),
                        "worst": objective.get("worst"),
                        "std": objective.get("std"),
                        "iqr": objective.get("iqr"),
                        "ci_low": objective.get("confidence_low"),
                        "ci_high": objective.get("confidence_high"),
                        "median_runtime_seconds": runtime.get("median"),
                        "median_evaluations_to_first_feasible": first.get("median"),
                        "verified_runs": summary.get("verified_runs", 0),
                    }
                )
        for baseline, row in evidence.global_statistics.get("calo_pairwise_rank_tests", {}).items():
            pairwise_rows.append({"baseline": baseline, **row})

        descriptive = pd.DataFrame(descriptive_rows)
        descriptive.to_csv(tables / "descriptive_statistics_verified_only.csv", index=False)
        (tables / "descriptive_statistics_verified_only.tex").write_text(
            descriptive.to_latex(index=False), encoding="utf-8"
        )
        pd.DataFrame(pairwise_rows).to_csv(tables / "calo_pairwise_holm_wilcoxon.csv", index=False)
        ranks = evidence.global_statistics.get("average_ranks", {})
        pd.DataFrame(
            [{"algorithm": key, "average_rank": value} for key, value in ranks.items()]
        ).sort_values("average_rank").to_csv(tables / "average_ranks.csv", index=False)

        validation_rows = []
        for task_id, experiment_id in task_experiments.items():
            rows = self.database.list_runs(experiment_id)
            raw_records = []
            task_raw_dir = raw / task_id
            task_raw_dir.mkdir(parents=True, exist_ok=True)
            task_array_dir = task_raw_dir / "arrays"
            task_array_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                result = json.loads(row["result_json"])
                seeds = json.loads(row["seed_json"])
                raw_records.append(
                    {
                        "run_id": row["id"],
                        "algorithm": row["algorithm"],
                        "run_index": row["run_index"],
                        **seeds,
                        "objective": result.get("best_objective"),
                        "feasible": result.get("feasible"),
                        "constraint_violation": result.get("total_constraint_violation"),
                        "runtime_seconds": result.get("runtime_seconds"),
                        "evaluations": result.get("evaluations"),
                        "validation_status": row.get("validation_status"),
                        "decoded_controls": json.dumps(result.get("decoded_controls", {})),
                        "best_vector": json.dumps(result.get("best_vector", [])),
                        "solution_state": json.dumps(
                            (result.get("metadata", {}) or {}).get("solution_state", {})
                        ),
                    }
                )
                complete_record = {
                    "database_row": {key: row[key] for key in row if key != "result_json"},
                    "seeds": seeds,
                    "result": result,
                    "validations": [
                        json.loads(item["validation_json"])
                        for item in self.database.list_validations(row["id"])
                    ],
                }
                (task_raw_dir / f"{row['id']}.json").write_text(
                    json.dumps(complete_record, indent=2, allow_nan=True),
                    encoding="utf-8",
                )
                source_arrays = self.database.resolve_array_path(row.get("arrays_path", ""))
                if source_arrays is not None and source_arrays.is_file():
                    shutil.copy2(source_arrays, task_array_dir / f"{row['id']}.npz")
                validation_rows.append(
                    {
                        "task": task_id,
                        "run_id": row["id"],
                        "algorithm": row["algorithm"],
                        "run_index": row["run_index"],
                        "validation_status": row.get("validation_status"),
                    }
                )
            pd.DataFrame(raw_records).to_csv(task_raw_dir / "runs.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(validation / "validation_status.csv", index=False)

        generate_campaign_figures(self.database, task_experiments, figures)

        archive = root / "CALO_RPD_Transactions_Research_Package.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in root.rglob("*"):
                if path.is_file() and path != archive:
                    bundle.write(path, path.relative_to(root))
        return archive
