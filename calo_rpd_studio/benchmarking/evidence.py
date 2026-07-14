"""Campaign-level descriptive, nonparametric, and evidence-based interpretation engine."""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

from calo_rpd_studio.algorithms.registry import primary_algorithm_names
from calo_rpd_studio.statistics.descriptive import descriptive_statistics
from calo_rpd_studio.statistics.effect_sizes import cliffs_delta
from calo_rpd_studio.statistics.posthoc import holm_correction


@dataclass(slots=True)
class CampaignEvidence:
    task_summaries: dict[str, Any]
    global_statistics: dict[str, Any]
    interpretations: list[str]

    def to_dict(self) -> dict:
        return {
            "task_summaries": self.task_summaries,
            "global_statistics": self.global_statistics,
            "interpretations": self.interpretations,
        }


def _run_record(row: dict) -> dict:
    data = json.loads(row["result_json"])
    metadata = data.get("metadata", {}) or {}
    first = metadata.get("first_feasible_evaluation")
    return {
        "algorithm": row["algorithm"],
        "run_index": int(row["run_index"]),
        "objective": float(data.get("best_objective", np.inf)),
        "feasible": bool(data.get("feasible", False)),
        "violation": float(data.get("total_constraint_violation", np.inf)),
        "runtime_seconds": float(data.get("runtime_seconds", np.nan)),
        "evaluations": int(data.get("evaluations", 0)),
        "first_feasible_evaluation": None if first is None else int(first),
        "validation_status": row.get("validation_status", "unverified"),
        "metadata": metadata,
    }


def _algorithm_summary(records: list[dict]) -> dict:
    feasible = [record for record in records if record["feasible"] and isfinite(record["objective"])]
    objectives = [record["objective"] for record in feasible]
    runtimes = [record["runtime_seconds"] for record in records if isfinite(record["runtime_seconds"])]
    first_feasible = [
        record["first_feasible_evaluation"]
        for record in records
        if record["first_feasible_evaluation"] is not None
    ]
    violations = [record["violation"] for record in records if isfinite(record["violation"])]
    return {
        "runs": len(records),
        "feasible_runs": len(feasible),
        "feasible_run_rate": len(feasible) / len(records) if records else 0.0,
        "objective": descriptive_statistics(objectives),
        "runtime_seconds": descriptive_statistics(runtimes),
        "evaluations_to_first_feasible": descriptive_statistics(first_feasible),
        "final_constraint_violation": descriptive_statistics(violations),
        "verified_runs": sum(record["validation_status"] == "verified" for record in records),
    }


def _block_rank_values(by_algorithm: dict[str, dict]) -> dict[str, float]:
    algorithms = list(by_algorithm)
    feasible_values = [r["objective"] for r in by_algorithm.values() if r["feasible"] and isfinite(r["objective"])]
    finite_objectives = [r["objective"] for r in by_algorithm.values() if isfinite(r["objective"])]
    objective_ceiling = max(feasible_values or finite_objectives or [0.0])
    margin = max(abs(objective_ceiling), 1.0)
    values = []
    for algorithm in algorithms:
        record = by_algorithm[algorithm]
        if record["feasible"] and isfinite(record["objective"]):
            merit = record["objective"]
        else:
            violation = record["violation"] if isfinite(record["violation"]) else 1e12
            merit = objective_ceiling + margin + violation
        values.append(merit)
    ranks = rankdata(values, method="average")
    return {algorithm: float(rank) for algorithm, rank in zip(algorithms, ranks)}


def build_campaign_evidence(
    database,
    task_experiments: dict[str, str],
    alpha: float = 0.05,
    *,
    verified_only: bool = False,
) -> CampaignEvidence:
    algorithms = list(primary_algorithm_names())
    task_summaries: dict[str, Any] = {}
    blocks: list[dict[str, float]] = []
    raw_records: dict[str, dict[str, list[dict]]] = {}

    for task_id, experiment_id in task_experiments.items():
        rows = database.list_runs(experiment_id, verified_only=verified_only)
        grouped: dict[str, list[dict]] = {name: [] for name in algorithms}
        by_run: dict[int, dict[str, dict]] = {}
        for row in rows:
            record = _run_record(row)
            if record["algorithm"] not in grouped:
                continue
            grouped[record["algorithm"]].append(record)
            by_run.setdefault(record["run_index"], {})[record["algorithm"]] = record
        raw_records[task_id] = grouped
        summary = {name: _algorithm_summary(grouped[name]) for name in algorithms}
        task_summaries[task_id] = {"experiment_id": experiment_id, "algorithms": summary}
        for run_index in sorted(by_run):
            block = by_run[run_index]
            if all(name in block for name in algorithms):
                blocks.append(_block_rank_values(block))

    global_statistics: dict[str, Any] = {
        "complete_paired_blocks": len(blocks),
        "friedman": None,
        "average_ranks": {},
        "calo_pairwise_rank_tests": {},
    }
    if blocks:
        matrix = np.asarray([[block[name] for name in algorithms] for block in blocks], float)
        average = matrix.mean(axis=0)
        global_statistics["average_ranks"] = {
            name: float(value) for name, value in zip(algorithms, average)
        }
        if len(blocks) >= 2:
            result = friedmanchisquare(*[matrix[:, i] for i in range(len(algorithms))])
            global_statistics["friedman"] = {
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "significant": bool(result.pvalue < alpha),
            }
            k = len(algorithms)
            q_alpha = float(studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))
            critical_difference = q_alpha * np.sqrt(k * (k + 1) / (6.0 * len(blocks)))
            global_statistics["nemenyi_critical_difference"] = {
                "alpha": float(alpha),
                "q_alpha": q_alpha,
                "blocks": len(blocks),
                "algorithms": k,
                "critical_difference": float(critical_difference),
            }
        calo = matrix[:, algorithms.index("CALO")]
        names: list[str] = []
        raw_p: list[float] = []
        test_rows: list[dict] = []
        for name in algorithms:
            if name == "CALO":
                continue
            baseline = matrix[:, algorithms.index(name)]
            try:
                test = wilcoxon(calo, baseline, zero_method="zsplit", alternative="two-sided")
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            except ValueError:
                p_value = 1.0
                statistic = 0.0
            names.append(name)
            raw_p.append(p_value)
            test_rows.append(
                {
                    "algorithm": name,
                    "statistic": statistic,
                    "p_value": p_value,
                    "cliffs_delta_rank": float(cliffs_delta(calo, baseline)),
                    "calo_mean_rank": float(np.mean(calo)),
                    "baseline_mean_rank": float(np.mean(baseline)),
                }
            )
        corrected = holm_correction(raw_p)
        for row, adjusted in zip(test_rows, corrected):
            row["holm_p_value"] = float(adjusted)
            row["significant"] = bool(adjusted < alpha)
            global_statistics["calo_pairwise_rank_tests"][row["algorithm"]] = row

    interpretations: list[str] = []
    for task_id, task in task_summaries.items():
        summaries = task["algorithms"]
        feasible_rates = {name: summaries[name]["feasible_run_rate"] for name in algorithms}
        best_rate = max(feasible_rates.values(), default=0.0)
        feasible_candidates = []
        for name in algorithms:
            median = summaries[name]["objective"].get("median") if summaries[name]["objective"] else None
            if median is not None:
                feasible_candidates.append((float(median), name))
        if not feasible_candidates:
            interpretations.append(f"{task_id}: no algorithm produced a feasible run; objective superiority cannot be claimed.")
            continue
        best_median, best_name = min(feasible_candidates)
        calo_summary = summaries["CALO"]
        calo_median = calo_summary["objective"].get("median") if calo_summary["objective"] else None
        if best_name == "CALO" and calo_summary["feasible_run_rate"] >= best_rate - 1e-12:
            interpretations.append(
                f"{task_id}: CALO achieved the lowest feasible median objective ({best_median:.6g}) with a feasible-run rate of {calo_summary['feasible_run_rate']:.1%}. Statistical significance must be read from the paired corrected tests."
            )
        else:
            interpretations.append(
                f"{task_id}: {best_name} achieved the lowest feasible median objective ({best_median:.6g}); CALO's feasible median was {calo_median if calo_median is not None else 'unavailable'} and its feasible-run rate was {calo_summary['feasible_run_rate']:.1%}."
            )

    pairwise = global_statistics.get("calo_pairwise_rank_tests", {})
    better = [name for name, row in pairwise.items() if row.get("significant") and row["calo_mean_rank"] < row["baseline_mean_rank"]]
    worse = [name for name, row in pairwise.items() if row.get("significant") and row["calo_mean_rank"] > row["baseline_mean_rank"]]
    if better:
        interpretations.append("Across complete paired benchmark blocks, CALO had a significantly better Holm-corrected rank than: " + ", ".join(better) + ".")
    if worse:
        interpretations.append("Across complete paired benchmark blocks, CALO had a significantly worse Holm-corrected rank than: " + ", ".join(worse) + ".")
    if pairwise and not better and not worse:
        interpretations.append("Across complete paired benchmark blocks, no Holm-corrected pairwise rank difference between CALO and the baselines reached the selected significance threshold.")

    return CampaignEvidence(task_summaries, global_statistics, interpretations)
