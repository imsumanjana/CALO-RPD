"""Selective, resumable portfolio export from stored experiment evidence."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calo_rpd_studio.statistics.descriptive import descriptive_statistics
from calo_rpd_studio.statistics.effect_sizes import cliffs_delta
from calo_rpd_studio.statistics.posthoc import holm_correction
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from .catalog import OUTPUT_REQUIREMENTS


@dataclass(slots=True)
class ArtifactResult:
    key: str
    status: str
    path: str = ""
    reason: str = ""


class PortfolioExportCancelled(RuntimeError):
    """Raised when a safe portfolio pause is requested during a long artifact."""


_ALREADY_COMPRESSED_SUFFIXES = {
    ".7z", ".bz2", ".gif", ".gz", ".jpeg", ".jpg", ".npz", ".pdf",
    ".png", ".rar", ".webp", ".xz", ".zip",
}


class PortfolioExporter:
    """Generate only the artifacts selected in Portfolio Manager.

    Each artifact is committed independently. Existing files listed in the manifest are reused,
    so a stopped export resumes without recreating valid figures or tables.
    """

    def __init__(self, database) -> None:
        self.database = database

    @staticmethod
    def _load_result(row: dict) -> dict:
        return json.loads(row["result_json"])

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _write_captions(out: Path, tasks: list[str], manifest: dict) -> None:
        lines = ["# Figure and table captions", ""]
        artifacts = manifest.get("artifacts", {}) or {}
        for key in tasks:
            if key == "reproducibility_bundle":
                continue
            record = artifacts.get(key, {}) or {}
            status = str(record.get("status", "pending"))
            reason = str(record.get("reason", ""))
            label = OUTPUT_REQUIREMENTS[key].label if key in OUTPUT_REQUIREMENTS else key.replace("_", " ").title()
            lines.append(f"- **{label}:** {status}" + (f" — {reason}" if reason else ""))
        (out / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _bundle_candidates(out: Path, archive: Path, manifest_path: Path) -> list[Path]:
        """Return only files owned by the current portfolio export.

        Older releases recursively archived every file below the user-selected output directory.
        That could accidentally re-compress unrelated prior exports and make the final 17th artifact
        appear stuck at 94%.  The bundle is now deliberately scoped to this portfolio's evidence.
        """
        candidates: set[Path] = set()
        for folder_name in ("figures", "tables", "raw_results", "configurations"):
            folder = out / folder_name
            if folder.is_dir():
                candidates.update(path for path in folder.rglob("*") if path.is_file())
        for filename in ("portfolio_metadata.json", "figure_captions.md"):
            path = out / filename
            if path.is_file():
                candidates.add(path)
        excluded = {archive.resolve(strict=False), manifest_path.resolve(strict=False)}
        return sorted(
            (path for path in candidates if path.resolve(strict=False) not in excluded and not path.name.endswith(".tmp")),
            key=lambda path: path.as_posix().lower(),
        )

    def _write_reproducibility_bundle(
        self,
        out: Path,
        archive: Path,
        manifest_path: Path,
        manifest: dict,
        *,
        completed_before: int,
        total_tasks: int,
        progress_callback=None,
        cancel_callback=None,
    ) -> str:
        candidates = self._bundle_candidates(out, archive, manifest_path)
        temp_archive = archive.with_name(archive.name + ".tmp")
        temp_archive.unlink(missing_ok=True)
        total_files = len(candidates)
        base_percent = int(100 * completed_before / max(total_tasks, 1))
        final_percent = min(99, int(100 * (completed_before + 1) / max(total_tasks, 1)) - 1)
        last_percent = -1

        def emit_progress(done: int, status: str) -> None:
            nonlocal last_percent
            if progress_callback is None:
                return
            if total_files:
                fraction = done / total_files
                percent = base_percent + int(max(0, final_percent - base_percent) * fraction)
            else:
                percent = final_percent
            percent = min(99, max(base_percent, percent))
            if percent != last_percent or done in {0, total_files}:
                last_percent = percent
                progress_callback({
                    "completed": completed_before,
                    "total": total_tasks,
                    "percent": percent,
                    "artifact": "reproducibility_bundle",
                    "status": status,
                })

        emit_progress(0, f"packing 0/{total_files} files")
        try:
            with zipfile.ZipFile(
                temp_archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as zf:
                for index, candidate in enumerate(candidates, start=1):
                    if cancel_callback and cancel_callback():
                        raise PortfolioExportCancelled("Safe pause requested during reproducibility bundle creation")
                    compression = (
                        zipfile.ZIP_STORED
                        if candidate.suffix.lower() in _ALREADY_COMPRESSED_SUFFIXES
                        else zipfile.ZIP_DEFLATED
                    )
                    zf.write(candidate, candidate.relative_to(out), compress_type=compression)
                    emit_progress(index, f"packing {index}/{total_files} files")

                snapshot = json.loads(json.dumps(manifest, allow_nan=True))
                snapshot.setdefault("artifacts", {})["reproducibility_bundle"] = {
                    "status": "completed",
                    "path": str(archive),
                    "reason": "",
                }
                snapshot.update({
                    "completed_artifacts": completed_before + 1,
                    "total_artifacts": total_tasks,
                    "cancelled": False,
                })
                zf.writestr(
                    "portfolio_manifest_snapshot.json",
                    json.dumps(snapshot, indent=2, allow_nan=True),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=1,
                )
            temp_archive.replace(archive)
        except BaseException:
            temp_archive.unlink(missing_ok=True)
            raise
        return str(archive)

    @staticmethod
    def _save_figure(fig, base: Path) -> str:
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return str(base.with_suffix(".png"))

    @staticmethod
    def _best_row(rows: list[dict]) -> dict | None:
        candidates = []
        for row in rows:
            data = json.loads(row["result_json"])
            if data.get("feasible") and np.isfinite(data.get("best_objective", np.inf)):
                candidates.append((float(data["best_objective"]), row))
        return min(candidates, key=lambda item: item[0])[1] if candidates else (rows[0] if rows else None)

    @staticmethod
    def _aligned_histories(rows: list[dict], key: str) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
        by_algorithm: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        max_eval = 0
        for row in rows:
            data = json.loads(row["result_json"])
            md = data.get("metadata", {}) or {}
            x = np.asarray(md.get("convergence_evaluations", []), dtype=float)
            y = np.asarray(md.get(key, []), dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                x, y = x[mask], y[mask]
                max_eval = max(max_eval, int(x[-1]))
                by_algorithm.setdefault(row["algorithm"], []).append((x, y))
        if max_eval <= 0:
            return np.asarray([]), {}
        grid = np.unique(np.linspace(0, max_eval, min(300, max_eval + 1), dtype=int)).astype(float)
        aligned: dict[str, list[np.ndarray]] = {}
        for algorithm, histories in by_algorithm.items():
            series = []
            for x, y in histories:
                indices = np.searchsorted(x, grid, side="right") - 1
                indices = np.clip(indices, 0, len(y) - 1)
                values = y[indices]
                values[grid < x[0]] = np.nan
                series.append(values)
            aligned[algorithm] = series
        return grid, aligned

    def _single_convergence(self, row: dict, key: str, ylabel: str, title: str, path: Path) -> str:
        data = self._load_result(row)
        md = data.get("metadata", {}) or {}
        x = np.asarray(md.get("convergence_evaluations", []), dtype=float)
        y = np.asarray(md.get(key, []), dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            raise ValueError("Stored run does not contain the requested convergence history")
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.plot(x[mask], y[mask], label=row["algorithm"])
        ax.set_xlabel("Objective-function evaluations")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend()
        if np.nanmin(y[mask]) >= 0:
            ymax = float(np.nanmax(y[mask])); ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
        return self._save_figure(fig, path)

    def _constraint_decomposition(self, row: dict, path: Path) -> str:
        data = self._load_result(row); md = data.get("metadata", {}) or {}
        histories = dict(md.get("constraint_component_histories", {}))
        x = np.asarray(md.get("convergence_evaluations", []), dtype=float)
        if not histories or x.size == 0:
            raise ValueError("Constraint-component histories are unavailable")
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        for label, values in histories.items():
            y = np.asarray(values, dtype=float); n = min(len(x), len(y))
            ax.plot(x[:n], y[:n], label=label.replace("_", " ").title())
        ax.set_xlabel("Objective-function evaluations"); ax.set_ylabel("Normalized violation")
        ax.set_title("Constraint-violation decomposition"); ax.grid(True, alpha=0.35); ax.legend()
        return self._save_figure(fig, path)

    def _solution_scenario(self, row: dict) -> dict:
        data = self._load_result(row); state = (data.get("metadata", {}) or {}).get("solution_state", {}) or {}
        scenarios = state.get("scenarios", [])
        if not scenarios:
            raise ValueError("Full power-flow solution state is unavailable")
        return scenarios[0]

    def _voltage_profile(self, row: dict, path: Path) -> str:
        scenario = self._solution_scenario(row)
        buses = np.asarray(scenario.get("bus_numbers", np.arange(len(scenario.get("vm_pu", []))) + 1))
        vm = np.asarray(scenario.get("vm_pu", []), dtype=float)
        if vm.size == 0: raise ValueError("Voltage magnitudes are unavailable")
        fig, ax = plt.subplots(figsize=(8.0, 4.8)); ax.plot(buses, vm, marker="o")
        ax.axhline(1.0, linestyle="--", linewidth=1); ax.set_xlabel("Bus number"); ax.set_ylabel("Voltage magnitude (p.u.)")
        ax.set_title(f"Optimized bus-voltage profile — {row['algorithm']}"); ax.grid(True, alpha=0.35)
        return self._save_figure(fig, path)

    def _heatmap(self, values: np.ndarray, xlabels, title: str, xlabel: str, path: Path) -> str:
        values = np.asarray(values, dtype=float)
        if values.size == 0: raise ValueError("Heatmap data are unavailable")
        fig, ax = plt.subplots(figsize=(max(7.0, min(14.0, values.size / 8)), 3.0))
        image = ax.imshow(values.reshape(1, -1), aspect="auto")
        ax.set_yticks([0]); ax.set_yticklabels(["Value"]); ax.set_xlabel(xlabel); ax.set_title(title)
        step = max(1, len(xlabels) // 20); ticks = np.arange(0, len(xlabels), step)
        ax.set_xticks(ticks); ax.set_xticklabels([str(xlabels[i]) for i in ticks], rotation=45, ha="right")
        fig.colorbar(image, ax=ax, shrink=0.75)
        return self._save_figure(fig, path)

    def _branch_loading(self, row: dict, path: Path, heatmap: bool = False) -> str:
        scenario = self._solution_scenario(row); loading = np.asarray(scenario.get("loading_percent", []), dtype=float)
        if loading.size == 0: raise ValueError("Branch-loading state is unavailable")
        labels = [f"{a}-{b}" for a, b in zip(scenario.get("branch_from_bus", range(len(loading))), scenario.get("branch_to_bus", range(len(loading))))]
        if heatmap:
            return self._heatmap(loading, labels, f"Branch-loading heatmap — {row['algorithm']}", "Branch", path)
        fig, ax = plt.subplots(figsize=(9.0, 5.0)); ax.bar(np.arange(len(loading)), loading)
        ax.axhline(100, linestyle="--", linewidth=1); ax.set_xlabel("Branch index"); ax.set_ylabel("Loading (%)")
        ax.set_title(f"Optimized branch loading — {row['algorithm']}"); ax.grid(True, axis="y", alpha=0.35)
        return self._save_figure(fig, path)

    def _controls(self, row: dict, path: Path) -> str:
        data = self._load_result(row); controls = data.get("decoded_controls", {}) or {}
        labels, values = [], []
        for key, value in controls.items():
            if isinstance(value, (int, float)):
                labels.append(str(key)); values.append(float(value))
        if not values: raise ValueError("Decoded scalar controls are unavailable")
        fig, ax = plt.subplots(figsize=(max(8.0, len(values) * 0.35), 5.0)); ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values))); ax.set_xticklabels(labels, rotation=65, ha="right")
        ax.set_title(f"Optimized ORPD controls — {row['algorithm']}"); ax.set_ylabel("Physical control value")
        return self._save_figure(fig, path)

    def _median_convergence(self, rows: list[dict], path: Path, uncertainty: bool = False) -> str:
        grid, aligned = self._aligned_histories(rows, "best_feasible_objective_history")
        if grid.size == 0 or not aligned: raise ValueError("Repeated feasible convergence histories are unavailable")
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        for algorithm, series in aligned.items():
            matrix = np.asarray(series, dtype=float)
            valid_columns = np.any(np.isfinite(matrix), axis=0)
            if not np.any(valid_columns):
                continue
            local_grid = grid[valid_columns]
            local_matrix = matrix[:, valid_columns]
            median = np.nanmedian(local_matrix, axis=0)
            line = ax.plot(local_grid, median, label=algorithm)[0]
            if uncertainty:
                q1 = np.nanpercentile(local_matrix, 25, axis=0)
                q3 = np.nanpercentile(local_matrix, 75, axis=0)
                ax.fill_between(local_grid, q1, q3, alpha=0.18, color=line.get_color())
        ax.set_xlabel("Objective-function evaluations"); ax.set_ylabel("Best feasible objective")
        ax.set_title("Median feasible convergence" + (" with IQR" if uncertainty else "")); ax.grid(True, alpha=0.35); ax.legend()
        return self._save_figure(fig, path)

    def _distribution(self, frame: pd.DataFrame, path: Path, violin: bool = False) -> str:
        grouped = [group["objective"].dropna().to_numpy() for _, group in frame.groupby("algorithm")]
        labels = [name for name, _ in frame.groupby("algorithm")]
        if not grouped: raise ValueError("Repeated objective records are unavailable")
        fig, ax = plt.subplots(figsize=(max(7.5, len(labels) * 0.9), 5.2))
        if violin:
            ax.violinplot(grouped, showmedians=True); ax.set_xticks(np.arange(1, len(labels) + 1))
        else:
            ax.boxplot(grouped, tick_labels=labels, showfliers=True); ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_ylabel("Final objective")
        ax.set_title("Final feasible objective distribution"); ax.grid(True, axis="y", alpha=0.35)
        return self._save_figure(fig, path)

    def _feasible_probability(self, frame: pd.DataFrame, path: Path) -> str:
        rates = frame.groupby("algorithm")["feasible"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(max(7.2, len(rates) * 0.8), 4.8)); ax.bar(rates.index, rates.values)
        ax.set_ylim(0, 105); ax.set_ylabel("Feasible runs (%)"); ax.set_title("Feasible-run probability")
        ax.tick_params(axis="x", rotation=45); ax.grid(True, axis="y", alpha=0.35)
        return self._save_figure(fig, path)

    def _evaluations_to_feasibility(self, rows: list[dict], path: Path) -> str:
        records = []
        for row in rows:
            data = self._load_result(row); first = (data.get("metadata", {}) or {}).get("first_feasible_evaluation")
            if first is not None: records.append((row["algorithm"], float(first)))
        if not records: raise ValueError("No run reached feasibility or the first-feasible history is unavailable")
        frame = pd.DataFrame(records, columns=["algorithm", "evaluations"])
        groups = [g["evaluations"].to_numpy() for _, g in frame.groupby("algorithm")]; labels = [n for n, _ in frame.groupby("algorithm")]
        fig, ax = plt.subplots(figsize=(max(7.2, len(labels) * 0.8), 4.8)); ax.boxplot(groups, tick_labels=labels)
        ax.set_ylabel("Evaluations to first feasible solution"); ax.set_title("Feasibility attainment distribution")
        ax.tick_params(axis="x", rotation=45); ax.grid(True, axis="y", alpha=0.35)
        return self._save_figure(fig, path)

    def _objective_violation_scatter(self, rows: list[dict], path: Path) -> str:
        points = []
        for row in rows:
            data = self._load_result(row)
            points.append((row["algorithm"], float(data.get("total_constraint_violation", np.nan)), float(data.get("best_objective", np.nan))))
        frame = pd.DataFrame(points, columns=["algorithm", "violation", "objective"]).dropna()
        if frame.empty: raise ValueError("Objective/violation records are unavailable")
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for algorithm, group in frame.groupby("algorithm"):
            ax.scatter(group["violation"], group["objective"], label=algorithm, alpha=0.75)
        ax.set_xlabel("Final normalized constraint violation"); ax.set_ylabel("Final objective"); ax.set_title("Objective–violation relationship")
        ax.grid(True, alpha=0.35); ax.legend()
        return self._save_figure(fig, path)

    @staticmethod
    def _calo_row(rows: list[dict]) -> dict:
        for row in rows:
            if row["algorithm"] == "CALO":
                return row
        raise ValueError("No CALO run is available")

    def _calo_operator_plot(self, rows: list[dict], path: Path, success: bool) -> str:
        key = "operator_success_history" if success else "operator_usage_history"
        collected: dict[str, list[float]] = {}
        for row in rows:
            if row["algorithm"] != "CALO":
                continue
            history = (self._load_result(row).get("metadata", {}) or {}).get(key, []) or []
            for record in history:
                for name, value in record.items():
                    collected.setdefault(str(name), []).append(float(value))
        if not collected:
            raise ValueError("CALO operator history is unavailable")
        names = list(collected)
        if success:
            values = [float(np.median(collected[name])) for name in names]
            ylabel, title = "Median operator success rate", "CALO operator success"
        else:
            totals = np.asarray([sum(collected[name]) for name in names], dtype=float)
            values = (totals / totals.sum()).tolist() if totals.sum() > 0 else totals.tolist()
            ylabel, title = "Fraction of operator assignments", "CALO operator utilization"
        fig, ax = plt.subplots(figsize=(max(7.5, len(names) * 0.9), 5.0))
        ax.bar(names, values); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis="x", rotation=35); ax.grid(True, axis="y", alpha=0.35)
        return self._save_figure(fig, path)

    def _calo_regime_plot(self, rows: list[dict], path: Path) -> str:
        row = self._calo_row(rows)
        metadata = self._load_result(row).get("metadata", {}) or {}
        regimes = list(metadata.get("regime_history", []) or [])
        if not regimes:
            raise ValueError("CALO regime history is unavailable")
        labels = list(dict.fromkeys(str(value) for value in regimes))
        mapping = {name: index for index, name in enumerate(labels)}
        values = [mapping[str(value)] for value in regimes]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.step(np.arange(len(values)), values, where="post")
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("CALO iteration"); ax.set_ylabel("Cognitive regime")
        ax.set_title("CALO cognitive-regime timeline"); ax.grid(True, alpha=0.35)
        return self._save_figure(fig, path)

    @staticmethod
    def _feasibility_first_merit(data: dict, objective_ceiling: float, margin: float) -> float:
        objective = float(data.get("best_objective", np.inf))
        if bool(data.get("feasible")) and np.isfinite(objective):
            return objective
        violation = float(data.get("total_constraint_violation", np.inf))
        if not np.isfinite(violation):
            violation = 1e12
        return objective_ceiling + margin + violation

    def _paired_merits(self, rows: list[dict]) -> tuple[list[str], list[int], np.ndarray]:
        algorithms = sorted({str(row["algorithm"]) for row in rows})
        run_indices = sorted({int(row["run_index"]) for row in rows})
        lookup = {(str(row["algorithm"]), int(row["run_index"])): self._load_result(row) for row in rows}
        complete = [index for index in run_indices if all((name, index) in lookup for name in algorithms)]
        if not complete or len(algorithms) < 2:
            raise ValueError("Complete paired algorithm/run blocks are unavailable")
        finite_feasible = [
            float(data.get("best_objective"))
            for data in lookup.values()
            if bool(data.get("feasible")) and np.isfinite(data.get("best_objective", np.inf))
        ]
        finite_any = [float(data.get("best_objective")) for data in lookup.values() if np.isfinite(data.get("best_objective", np.inf))]
        ceiling = max(finite_feasible or finite_any or [0.0]); margin = max(abs(ceiling), 1.0)
        matrix = np.asarray([
            [self._feasibility_first_merit(lookup[(name, index)], ceiling, margin) for name in algorithms]
            for index in complete
        ], dtype=float)
        return algorithms, complete, matrix

    def _pairwise_statistics(self, rows: list[dict], tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        algorithms, _blocks, matrix = self._paired_merits(rows)
        reference = "CALO" if "CALO" in algorithms else algorithms[0]
        ref_index = algorithms.index(reference)
        raw = []
        for index, name in enumerate(algorithms):
            if name == reference:
                continue
            try:
                test = wilcoxon(matrix[:, ref_index], matrix[:, index], zero_method="zsplit", alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            raw.append({
                "reference": reference,
                "baseline": name,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "cliffs_delta": float(cliffs_delta(matrix[:, ref_index], matrix[:, index])),
                "reference_mean_merit": float(np.mean(matrix[:, ref_index])),
                "baseline_mean_merit": float(np.mean(matrix[:, index])),
            })
        corrected = holm_correction([record["p_value"] for record in raw]) if raw else []
        for record, value in zip(raw, corrected):
            record["holm_p_value"] = float(value); record["significant_0_05"] = bool(value < 0.05)
        pairwise = pd.DataFrame(raw)
        rank_matrix = np.asarray([rankdata(row, method="average") for row in matrix], dtype=float)
        rankings = pd.DataFrame({"algorithm": algorithms, "average_rank": np.mean(rank_matrix, axis=0)}).sort_values("average_rank")
        pairwise.to_csv(tables / "wilcoxon_holm_effect_sizes.csv", index=False)
        rankings.to_csv(tables / "average_rankings.csv", index=False)
        return pairwise, rankings

    def _effect_size_plot(self, pairwise: pd.DataFrame, path: Path) -> str:
        if pairwise.empty: raise ValueError("Pairwise effect sizes are unavailable")
        fig, ax = plt.subplots(figsize=(8.0, max(4.8, len(pairwise) * 0.45)))
        positions = np.arange(len(pairwise)); ax.barh(positions, pairwise["cliffs_delta"])
        ax.set_yticks(positions); ax.set_yticklabels(pairwise["baseline"]); ax.axvline(0, linewidth=1)
        ax.set_xlabel("Cliff's delta (reference minus baseline merit)"); ax.set_title("Paired effect sizes")
        ax.grid(True, axis="x", alpha=0.35)
        return self._save_figure(fig, path)

    def _ranking_plot(self, rankings: pd.DataFrame, path: Path) -> str:
        if rankings.empty: raise ValueError("Rankings are unavailable")
        fig, ax = plt.subplots(figsize=(8.0, max(4.8, len(rankings) * 0.42)))
        positions = np.arange(len(rankings)); ax.barh(positions, rankings["average_rank"])
        ax.set_yticks(positions); ax.set_yticklabels(rankings["algorithm"]); ax.invert_yaxis()
        ax.set_xlabel("Average feasibility-first rank (lower is better)"); ax.set_title("Algorithm ranking across paired runs")
        ax.grid(True, axis="x", alpha=0.35)
        return self._save_figure(fig, path)

    def _scenario_matrix(self, rows: list[dict], field: str) -> tuple[list[str], np.ndarray]:
        labels, matrix = [], []
        for row in rows:
            state = (self._load_result(row).get("metadata", {}) or {}).get("solution_state", {}) or {}
            scenarios = list(state.get("scenarios", []) or [])
            if not scenarios:
                continue
            values = []
            for scenario in scenarios:
                if field == "feasible":
                    value = 1.0 if bool(scenario.get("converged", False)) else 0.0
                elif field == "loss":
                    value = scenario.get("active_loss_mw", scenario.get("loss_mw", np.nan))
                else:
                    value = scenario.get(field, np.nan)
                values.append(float(value))
            if values:
                labels.append(f"{row['algorithm']} r{int(row['run_index']) + 1}"); matrix.append(values)
        if not matrix:
            raise ValueError("Scenario-wise solution records are unavailable")
        width = max(len(values) for values in matrix)
        padded = np.full((len(matrix), width), np.nan)
        for index, values in enumerate(matrix): padded[index, :len(values)] = values
        return labels, padded

    def _scenario_heatmap(self, rows: list[dict], path: Path, field: str) -> str:
        labels, matrix = self._scenario_matrix(rows, field)
        fig, ax = plt.subplots(figsize=(max(8.0, matrix.shape[1] * 0.22), max(4.5, len(labels) * 0.32)))
        image = ax.imshow(matrix, aspect="auto")
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Scenario index"); ax.set_title("Scenario feasibility" if field == "feasible" else "Scenario active-loss map")
        fig.colorbar(image, ax=ax, shrink=0.8)
        return self._save_figure(fig, path)

    def export(self, experiment_id: str, directory: str | Path, *, progress_callback=None, cancel_callback=None) -> Path:
        out = Path(directory)
        figures = out / "figures"; tables = out / "tables"; raw = out / "raw_results"; configs = out / "configurations"
        for folder in (figures, tables, raw, configs): folder.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "portfolio_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"artifacts": {}}

        experiment = self.database.get_experiment(experiment_id)
        if experiment is None: raise KeyError(f"Unknown experiment: {experiment_id}")
        config = json.loads(experiment["config_json"])
        portfolio = dict(config.get("portfolio", {}))
        requested = list(portfolio.get("requested_outputs", []))
        all_rows = self.database.list_runs(experiment_id)
        verified_rows = self.database.list_runs(experiment_id, verified_only=True)
        publication_rows = verified_rows if portfolio.get("require_independent_validation", True) else all_rows
        if not all_rows: raise ValueError("The experiment has no completed runs")
        best = self._best_row(publication_rows or all_rows)

        records = []
        for row in all_rows:
            data = self._load_result(row)
            records.append({
                "run_id": row["id"], "algorithm": row["algorithm"], "run_index": row["run_index"],
                "objective": data.get("best_objective"), "feasible": bool(data.get("feasible")),
                "violation": data.get("total_constraint_violation"), "runtime_seconds": data.get("runtime_seconds"),
                "evaluations": data.get("evaluations"), "validation_status": row.get("validation_status", "unverified"),
            })
        frame = pd.DataFrame(records)
        frame.to_csv(raw / "all_runs.csv", index=False)
        (configs / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        tasks = list(dict.fromkeys(requested + ["portfolio_tables", "portfolio_metadata", "reproducibility_bundle"]))
        completed_count = 0
        results: list[ArtifactResult] = []
        for index, key in enumerate(tasks):
            if cancel_callback and cancel_callback():
                manifest["cancelled"] = True; self._atomic_json(manifest_path, manifest); break
            existing = manifest.get("artifacts", {}).get(key, {})
            if existing.get("status") == "completed" and existing.get("path") and Path(existing["path"]).exists():
                completed_count += 1
                results.append(ArtifactResult(key, "reused", existing["path"]))
                if progress_callback:
                    progress_callback({
                        "completed": completed_count,
                        "total": len(tasks),
                        "percent": int(100 * completed_count / max(len(tasks), 1)),
                        "artifact": key,
                        "status": "reused",
                    })
                continue
            try:
                path = ""
                target = figures / key
                row = best or all_rows[0]
                if key == "objective_convergence": path = self._single_convergence(row, "best_feasible_objective_history", "Best feasible objective", "Objective convergence", target)
                elif key == "constraint_convergence": path = self._single_convergence(row, "best_constraint_violation_history", "Best normalized constraint violation", "Constraint-violation convergence", target)
                elif key == "constraint_decomposition": path = self._constraint_decomposition(row, target)
                elif key in {"voltage_profile", "best_validated_voltage_profile"}: path = self._voltage_profile(row, target)
                elif key == "voltage_heatmap":
                    sc = self._solution_scenario(row); path = self._heatmap(np.asarray(sc.get("vm_pu", [])), sc.get("bus_numbers", []), f"Bus-voltage heatmap — {row['algorithm']}", "Bus", target)
                elif key == "branch_loading": path = self._branch_loading(row, target, False)
                elif key in {"branch_loading_heatmap", "best_validated_branch_heatmap"}: path = self._branch_loading(row, target, True)
                elif key == "generator_reactive_power":
                    sc = self._solution_scenario(row); path = self._heatmap(np.asarray(sc.get("qg_mvar", [])), sc.get("generator_bus", []), f"Generator reactive power — {row['algorithm']}", "Generator bus", target)
                elif key == "control_changes": path = self._controls(row, target)
                elif key == "objective_violation_scatter": path = self._objective_violation_scatter(all_rows, target)
                elif key == "median_convergence": path = self._median_convergence(all_rows, target, False)
                elif key == "convergence_uncertainty_band": path = self._median_convergence(all_rows, target, True)
                elif key == "objective_boxplot": path = self._distribution(frame[frame["feasible"]], target, False)
                elif key == "objective_violin": path = self._distribution(frame[frame["feasible"]], target, True)
                elif key == "feasible_run_probability": path = self._feasible_probability(frame, target)
                elif key == "evaluations_to_feasibility": path = self._evaluations_to_feasibility(all_rows, target)
                elif key == "calo_regime_timeline": path = self._calo_regime_plot(all_rows, target)
                elif key == "calo_operator_usage": path = self._calo_operator_plot(all_rows, target, False)
                elif key == "calo_operator_success": path = self._calo_operator_plot(all_rows, target, True)
                elif key == "scenario_loss_heatmap": path = self._scenario_heatmap(all_rows, target, "loss")
                elif key == "scenario_feasibility_heatmap": path = self._scenario_heatmap(all_rows, target, "feasible")
                elif key in {"cvar_curve", "contingency_matrix"}:
                    raise ValueError("The selected experiment does not contain the specialized risk/contingency array required for this plot")
                elif key in {"throughput_batch_scaling", "device_speedup", "parity_scatter"}:
                    raise ValueError("Accelerator calibration/parity records are unavailable in the selected experiment")
                elif key in {"wilcoxon_holm", "effect_sizes", "friedman_ranking"}:
                    pairwise, rankings = self._pairwise_statistics(publication_rows or all_rows, tables)
                    if key == "wilcoxon_holm": path = str(tables / "wilcoxon_holm_effect_sizes.csv")
                    elif key == "effect_sizes": path = self._effect_size_plot(pairwise, target)
                    else:
                        algorithms, blocks, matrix = self._paired_merits(publication_rows or all_rows)
                        if len(blocks) < 2 or len(algorithms) < 3: raise ValueError("Friedman ranking requires at least two complete paired blocks and three algorithms")
                        result = friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
                        (tables / "friedman_test.json").write_text(json.dumps({"statistic": float(result.statistic), "p_value": float(result.pvalue), "blocks": len(blocks), "algorithms": algorithms}, indent=2), encoding="utf-8")
                        path = self._ranking_plot(rankings, target)
                elif key == "critical_difference":
                    raise ValueError("Critical-difference evidence must be generated from at least two separate benchmark cases/formulations in the campaign portfolio")
                elif key == "descriptive_statistics":
                    stats = {name: descriptive_statistics(group["objective"].dropna().to_numpy()) for name, group in frame.groupby("algorithm")}
                    pd.DataFrame(stats).T.to_csv(tables / "descriptive_statistics.csv")
                    path = str(tables / "descriptive_statistics.csv")
                elif key == "portfolio_tables":
                    frame.to_csv(tables / "run_level_results.csv", index=False)
                    if not frame.empty:
                        stats = {name: descriptive_statistics(group["objective"].dropna().to_numpy()) for name, group in frame.groupby("algorithm")}
                        pd.DataFrame(stats).T.to_csv(tables / "descriptive_statistics.csv")
                        (tables / "run_level_results.tex").write_text(frame.to_latex(index=False), encoding="utf-8")
                    path = str(tables / "run_level_results.csv")
                elif key == "portfolio_metadata":
                    metadata = {"experiment": experiment, "portfolio": portfolio, "requested_outputs": requested, "completed_runs": len(all_rows), "verified_runs": len(verified_rows)}
                    self._atomic_json(out / "portfolio_metadata.json", metadata); path = str(out / "portfolio_metadata.json")
                elif key == "reproducibility_bundle":
                    self._write_captions(out, tasks, manifest)
                    archive = out / "reproducibility_bundle.zip"
                    path = self._write_reproducibility_bundle(
                        out,
                        archive,
                        manifest_path,
                        manifest,
                        completed_before=completed_count,
                        total_tasks=len(tasks),
                        progress_callback=progress_callback,
                        cancel_callback=cancel_callback,
                    )
                else:
                    raise ValueError("No exporter is registered for this selected output")
                result = ArtifactResult(key, "completed", path)
                manifest.setdefault("artifacts", {})[key] = {"status": "completed", "path": path, "reason": ""}
            except PortfolioExportCancelled:
                manifest["cancelled"] = True
                self._atomic_json(manifest_path, manifest)
                break
            except Exception as exc:
                result = ArtifactResult(key, "skipped", "", str(exc))
                manifest.setdefault("artifacts", {})[key] = {"status": "skipped", "path": "", "reason": str(exc)}
            results.append(result); completed_count += 1
            manifest.update({"experiment_id": experiment_id, "requested_outputs": requested, "completed_artifacts": completed_count, "total_artifacts": len(tasks), "cancelled": False})
            self._atomic_json(manifest_path, manifest)
            if progress_callback:
                progress_callback({"completed": completed_count, "total": len(tasks), "percent": int(100 * completed_count / max(len(tasks), 1)), "artifact": key, "status": result.status})

        self._write_captions(out, tasks, manifest)
        return out
