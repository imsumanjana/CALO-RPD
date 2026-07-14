"""Publication-grade campaign figures for final benchmark evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calo_rpd_studio.algorithms.registry import primary_algorithm_names
from calo_rpd_studio.benchmarking.evidence import build_campaign_evidence


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _load_records(database, experiment_id: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in database.list_runs(experiment_id):
        data = json.loads(row["result_json"])
        groups.setdefault(row["algorithm"], []).append(data)
    return groups


def _step_align(runs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not runs:
        return None
    grids = [x for x, _ in runs if len(x)]
    if not grids:
        return None
    grid = np.unique(np.concatenate(grids))
    matrix = []
    for x, y in runs:
        values = np.full(len(grid), np.nan)
        for index, gx in enumerate(grid):
            pos = np.searchsorted(x, gx, side="right") - 1
            if pos >= 0:
                values[index] = y[pos]
        matrix.append(values)
    matrix = np.asarray(matrix, float)
    with np.errstate(all="ignore"):
        median = np.nanmedian(matrix, axis=0)
        q1 = np.nanpercentile(matrix, 25, axis=0)
        q3 = np.nanpercentile(matrix, 75, axis=0)
    valid = np.isfinite(median)
    if not np.any(valid):
        return None
    return grid[valid], median[valid], q1[valid], q3[valid]


def plot_median_convergence(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for algorithm in primary_algorithm_names():
        runs = []
        for data in groups.get(algorithm, []):
            metadata = data.get("metadata", {}) or {}
            x = np.asarray(metadata.get("convergence_evaluations", []), int)
            y = np.asarray(metadata.get("best_feasible_objective_history", []), float)
            if len(x) == len(y) and len(x):
                runs.append((x, y))
        aligned = _step_align(runs)
        if aligned is None:
            continue
        x, median, q1, q3 = aligned
        line = ax.plot(x, median, label=algorithm)[0]
        ax.fill_between(x, q1, q3, alpha=0.12, color=line.get_color())
    ax.set_title("Median best-feasible convergence with interquartile band")
    ax.set_xlabel("Objective-function evaluations")
    ax.set_ylabel("Best feasible objective")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, destination)


def plot_feasible_probability(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    algorithms = list(primary_algorithm_names())
    rates = []
    for algorithm in algorithms:
        runs = groups.get(algorithm, [])
        rates.append(sum(bool(item.get("feasible")) for item in runs) / len(runs) if runs else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    positions = np.arange(len(algorithms))
    ax.barh(positions, rates)
    ax.set_yticks(positions, algorithms)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Feasible-run probability")
    ax.set_title("Feasible-run probability by algorithm")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, destination)


def plot_evaluations_to_feasibility(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    labels = []
    values = []
    for algorithm in primary_algorithm_names():
        data = [
            (item.get("metadata", {}) or {}).get("first_feasible_evaluation")
            for item in groups.get(algorithm, [])
        ]
        data = [float(value) for value in data if value is not None and np.isfinite(value)]
        if data:
            labels.append(algorithm)
            values.append(data)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    if values:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Evaluations to first feasible solution")
    ax.set_title("Feasibility attainment distribution")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)


def plot_constraint_decomposition(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    component_names = ["bus_voltage", "generator_q", "generator_p", "branch_thermal"]
    algorithms = list(primary_algorithm_names())
    bottoms = np.zeros(len(algorithms))
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for component in component_names:
        values = []
        for algorithm in algorithms:
            samples = []
            for data in groups.get(algorithm, []):
                histories = (data.get("metadata", {}) or {}).get("constraint_component_histories", {})
                series = histories.get(component, [])
                if series:
                    samples.append(float(series[-1]))
            values.append(float(np.median(samples)) if samples else 0.0)
        ax.bar(algorithms, values, bottom=bottoms, label=component)
        bottoms += np.asarray(values)
    ax.set_ylabel("Median normalized constraint violation")
    ax.set_title("Final constraint-violation decomposition")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)


def plot_calo_operator_usage(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    totals: dict[str, float] = {}
    for data in groups.get("CALO", []):
        history = (data.get("metadata", {}) or {}).get("operator_usage_history", [])
        for record in history:
            for name, value in record.items():
                totals[name] = totals.get(name, 0.0) + float(value)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    names = list(totals)
    values = [totals[name] for name in names]
    if values and sum(values) > 0:
        values = [value / sum(values) for value in values]
    ax.bar(names, values)
    ax.set_ylabel("Fraction of CALO operator assignments")
    ax.set_title("CALO operator utilization")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)


def plot_calo_operator_success(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    collected: dict[str, list[float]] = {}
    for data in groups.get("CALO", []):
        history = (data.get("metadata", {}) or {}).get("operator_success_history", [])
        for record in history:
            for name, value in record.items():
                collected.setdefault(name, []).append(float(value))
    names = list(collected)
    medians = [float(np.median(collected[name])) for name in names]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(names, medians)
    ax.set_ylabel("Median operator success rate")
    ax.set_title("CALO operator success")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)


def plot_calo_regime_timeline(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    plotted = False
    for run_index, data in enumerate(groups.get("CALO", [])[:10]):
        regimes = np.asarray((data.get("metadata", {}) or {}).get("regime_history", []), float)
        if len(regimes):
            ax.step(np.arange(len(regimes)), regimes + run_index * 0.05, where="post", alpha=0.75)
            plotted = True
    ax.set_xlabel("CALO iteration")
    ax.set_ylabel("Cognitive regime index")
    ax.set_title("CALO cognitive-regime timeline (up to 10 runs)")
    ax.grid(True, alpha=0.25)
    if not plotted:
        ax.text(0.5, 0.5, "No CALO regime history available", ha="center", va="center", transform=ax.transAxes)
    _save(fig, destination)


def plot_objective_boxplot(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    labels = []
    values = []
    for algorithm in primary_algorithm_names():
        data = [
            float(item["best_objective"])
            for item in groups.get(algorithm, [])
            if item.get("feasible") and np.isfinite(item.get("best_objective", np.inf))
        ]
        if data:
            labels.append(algorithm)
            values.append(data)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    if values:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Best feasible objective")
    ax.set_title("Feasible objective boxplots")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)


def plot_objective_violin(database, experiment_id: str, destination: Path) -> None:
    groups = _load_records(database, experiment_id)
    labels = []
    values = []
    for algorithm in primary_algorithm_names():
        data = [float(item["best_objective"]) for item in groups.get(algorithm, []) if item.get("feasible") and np.isfinite(item.get("best_objective", np.inf))]
        if data:
            labels.append(algorithm)
            values.append(data)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    if values:
        ax.violinplot(values, showmedians=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=60)
    ax.set_ylabel("Best feasible objective")
    ax.set_title("Feasible objective distributions")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, destination)


def plot_average_ranking(database, task_experiments: dict[str, str], destination: Path) -> None:
    evidence = build_campaign_evidence(database, task_experiments)
    ranks = evidence.global_statistics.get("average_ranks", {})
    ordered = sorted(ranks.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    if ordered:
        names = [item[0] for item in ordered]
        values = [item[1] for item in ordered]
        ax.barh(np.arange(len(names)), values)
        ax.set_yticks(np.arange(len(names)), names)
        ax.invert_yaxis()
    ax.set_xlabel("Average feasibility-first rank (lower is better)")
    ax.set_title("Global algorithm ranking across paired benchmark blocks")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, destination)


def plot_critical_difference_style(database, task_experiments: dict[str, str], destination: Path) -> None:
    evidence = build_campaign_evidence(database, task_experiments)
    ranks = evidence.global_statistics.get("average_ranks", {})
    cd_info = evidence.global_statistics.get("nemenyi_critical_difference", {})
    critical_difference = cd_info.get("critical_difference")
    ordered = sorted(ranks.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    if ordered:
        values = [rank for _, rank in ordered]
        names = [name for name, _ in ordered]
        ax.scatter(values, np.zeros(len(values)), s=55)
        for index, (name, rank) in enumerate(zip(names, values)):
            offset = 0.10 if index % 2 == 0 else -0.12
            ax.text(
                rank,
                offset,
                name,
                ha="center",
                va="center",
                rotation=45 if len(names) > 12 else 0,
            )
        if critical_difference is not None and np.isfinite(critical_difference):
            start = min(values)
            y = 0.28
            ax.plot([start, start + critical_difference], [y, y], linewidth=2.2)
            ax.plot([start, start], [y - 0.025, y + 0.025], linewidth=1.2)
            ax.plot(
                [start + critical_difference, start + critical_difference],
                [y - 0.025, y + 0.025],
                linewidth=1.2,
            )
            ax.text(
                start + critical_difference / 2,
                y + 0.055,
                f"Nemenyi CD = {critical_difference:.3f}",
                ha="center",
            )
        ax.set_xlim(max(0.5, min(values) - 0.5), max(values) + 0.75)
    ax.set_ylim(-0.35, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Critical-difference diagram")
    ax.axhline(0, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, destination)


def plot_robustness_summary(database, task_experiments: dict[str, str], destination: Path) -> None:
    evidence = build_campaign_evidence(database, task_experiments)
    task_ids = list(evidence.task_summaries)
    algorithms = list(primary_algorithm_names())
    matrix = np.full((len(task_ids), len(algorithms)), np.nan)
    for row_index, task_id in enumerate(task_ids):
        for col_index, algorithm in enumerate(algorithms):
            matrix[row_index, col_index] = evidence.task_summaries[task_id]["algorithms"][algorithm]["feasible_run_rate"]
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.35 * len(task_ids))))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(algorithms)), algorithms, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(task_ids)), task_ids)
    ax.set_title("Robustness map: feasible-run probability")
    fig.colorbar(image, ax=ax, label="Feasible-run probability")
    _save(fig, destination)


def generate_campaign_figures(database, task_experiments: dict[str, str], directory: str | Path) -> None:
    directory = Path(directory)
    for task_id, experiment_id in task_experiments.items():
        task_dir = directory / task_id
        plot_median_convergence(database, experiment_id, task_dir / "median_convergence_iqr.png")
        plot_feasible_probability(database, experiment_id, task_dir / "feasible_run_probability.png")
        plot_evaluations_to_feasibility(database, experiment_id, task_dir / "evaluations_to_feasibility.png")
        plot_constraint_decomposition(database, experiment_id, task_dir / "constraint_decomposition.png")
        plot_calo_operator_usage(database, experiment_id, task_dir / "calo_operator_utilization.png")
        plot_calo_operator_success(database, experiment_id, task_dir / "calo_operator_success.png")
        plot_calo_regime_timeline(database, experiment_id, task_dir / "calo_cognitive_regime_timeline.png")
        plot_objective_boxplot(database, experiment_id, task_dir / "objective_boxplot.png")
        plot_objective_violin(database, experiment_id, task_dir / "objective_violin.png")
    plot_average_ranking(database, task_experiments, directory / "global_average_ranking.png")
    plot_critical_difference_style(database, task_experiments, directory / "global_critical_difference_diagram.png")
    plot_robustness_summary(database, task_experiments, directory / "global_robustness_map.png")
