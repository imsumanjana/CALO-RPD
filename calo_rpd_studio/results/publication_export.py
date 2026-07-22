"""Verified-and-feasible publication and reproducibility export."""

from __future__ import annotations

from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd

from calo_rpd_studio.statistics.descriptive import descriptive_statistics


class PublicationExportCancelled(RuntimeError):
    """Raised when a publication export is cancelled between atomic file steps."""


class PublicationExporter:
    def __init__(self, database):
        self.database = database

    @staticmethod
    def _emit(progress_callback, completed: int, total: int, artifact: str, status: str) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "completed": completed,
                "total": total,
                "percent": int(100 * completed / max(total, 1)),
                "artifact": artifact,
                "status": status,
            }
        )

    @staticmethod
    def _check_cancel(cancel_callback) -> None:
        if cancel_callback and cancel_callback():
            raise PublicationExportCancelled("Publication export cancelled safely")

    def export(self, experiment_id, directory, *, progress_callback=None, cancel_callback=None):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        total_steps = 5
        completed = 0

        self._check_cancel(cancel_callback)
        horizons = self.database.list_experiment_horizons(experiment_id)
        if not horizons:
            raise ValueError("Publication export requires completed run evidence at a declared FE horizon")
        revisions = [
            row for row in self.database.list_experiment_revisions(experiment_id)
            if bool(row.get("publication_eligible")) and str(row.get("status")) == "completed"
        ]
        horizon = int(revisions[-1]["evaluation_target"]) if revisions else int(horizons[-1])
        horizon_status = self.database.experiment_horizon_status(experiment_id, horizon)
        if not horizon_status.get("complete"):
            raise ValueError(
                f"Publication export blocked: evidence horizon {horizon} FE is incomplete "
                f"({horizon_status.get('available_count', 0)}/{horizon_status.get('expected_count', 0)})."
            )
        rows = [
            row for row in horizon_status.get("rows", [])
            if str(row.get("validation_status", "unverified")) == "verified"
        ]
        expected_count = int(horizon_status.get("expected_count", 0))
        if len(rows) != expected_count:
            raise ValueError(
                f"Publication export blocked: all {expected_count} expected paired runs must be independently verified; found {len(rows)}."
            )
        records = []
        for row in rows:
            data = json.loads(row["result_json"])
            records.append(
                {
                    "run_id": row["id"],
                    "algorithm": row["algorithm"],
                    "run_index": row["run_index"],
                    "objective": data.get("best_objective"),
                    "feasible": bool(data.get("feasible", False)),
                    "violation": data.get("total_constraint_violation"),
                    "runtime_seconds": data.get("runtime_seconds"),
                    "evaluations": data.get("evaluations"),
                    "validation_status": row.get("validation_status", "unverified"),
                }
            )
        frame = pd.DataFrame(records)
        frame.to_csv(out / "verified_runs.csv", index=False)
        if not frame.empty:
            frame.to_latex(out / "verified_runs.tex", index=False)
        completed += 1
        self._emit(progress_callback, completed, total_steps, "verified_runs", "completed")

        self._check_cancel(cancel_callback)
        # Objective performance claims are based only on independently verified, feasible, finite runs.
        if frame.empty:
            feasible = frame.copy()
        else:
            objective = pd.to_numeric(frame["objective"], errors="coerce")
            feasible = frame[frame["feasible"].astype(bool) & np.isfinite(objective)].copy()
        feasible.to_csv(out / "verified_feasible_runs.csv", index=False)
        completed += 1
        self._emit(progress_callback, completed, total_steps, "verified_feasible_runs", "completed")

        self._check_cancel(cancel_callback)
        statistics_rows = []
        algorithms = sorted(set(frame.get("algorithm", pd.Series(dtype=str)).astype(str)))
        for algorithm in algorithms:
            all_group = frame[frame["algorithm"] == algorithm]
            valid_group = feasible[feasible["algorithm"] == algorithm]
            objective_stats = descriptive_statistics(
                pd.to_numeric(valid_group.get("objective", []), errors="coerce").to_numpy()
            )
            violation_values = pd.to_numeric(
                all_group.get("violation", []), errors="coerce"
            ).to_numpy()
            violation_values = violation_values[np.isfinite(violation_values)]
            statistics_rows.append(
                {
                    "algorithm": algorithm,
                    "verified_runs": len(all_group),
                    "verified_feasible_runs": len(valid_group),
                    "verified_infeasible_runs": len(all_group) - len(valid_group),
                    "feasibility_rate": len(valid_group) / len(all_group)
                    if len(all_group)
                    else 0.0,
                    "objective_claim_status": "available"
                    if len(valid_group)
                    else "no_verified_feasible_run",
                    **{f"objective_{key}": value for key, value in objective_stats.items()},
                    "mean_final_violation": float(np.mean(violation_values))
                    if len(violation_values)
                    else np.nan,
                    "max_final_violation": float(np.max(violation_values))
                    if len(violation_values)
                    else np.nan,
                }
            )
        pd.DataFrame(statistics_rows).to_csv(
            out / "descriptive_statistics_verified_feasible.csv", index=False
        )
        completed += 1
        self._emit(progress_callback, completed, total_steps, "descriptive_statistics", "completed")

        self._check_cancel(cancel_callback)
        experiment = self.database.get_experiment(experiment_id)
        algorithms = sorted(frame["algorithm"].astype(str).unique().tolist()) if not frame.empty else []
        feasible_algorithms = set(feasible["algorithm"].astype(str).tolist()) if not feasible.empty else set()
        publication_ready = bool(
            horizon_status.get("complete")
            and len(frame) == expected_count
            and expected_count > 0
            and set(algorithms).issubset(feasible_algorithms)
        )
        metadata = {
            "experiment": experiment,
            "evaluation_horizon": horizon,
            "revision": horizon_status.get("revision"),
            "expected_paired_run_count": expected_count,
            "verified_run_count": int(len(frame)),
            "verified_feasible_run_count": int(len(feasible)),
            "verified_infeasible_run_count": int(len(frame) - len(feasible)),
            "objective_statistics_basis": "independently verified AND feasible finite runs only",
            "publication_ready": publication_ready,
            "publication_ready_rule": "complete declared paired horizon + every expected run independently verified + at least one verified-feasible run for every compared algorithm",
        }
        (out / "experiment_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
        )
        completed += 1
        self._emit(progress_callback, completed, total_steps, "publication_metadata", "completed")

        self._check_cancel(cancel_callback)
        archive = out / "reproducibility_bundle.zip"
        temp_archive = archive.with_name(archive.name + ".tmp")
        temp_archive.unlink(missing_ok=True)
        candidates = [
            path
            for path in (
                out / "verified_runs.csv",
                out / "verified_runs.tex",
                out / "verified_feasible_runs.csv",
                out / "descriptive_statistics_verified_feasible.csv",
                out / "experiment_metadata.json",
            )
            if path.is_file()
        ]
        total_files = len(candidates)
        try:
            with zipfile.ZipFile(
                temp_archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as zf:
                for index, path in enumerate(candidates, start=1):
                    self._check_cancel(cancel_callback)
                    zf.write(path, path.name)
                    if progress_callback:
                        # Keep the UI visibly moving inside the long final archive step.
                        percent = 80 + int(19 * index / max(total_files, 1))
                        progress_callback(
                            {
                                "completed": completed,
                                "total": total_steps,
                                "percent": min(percent, 98),
                                "artifact": "reproducibility_bundle",
                                "status": f"packing {index}/{total_files} files",
                            }
                        )
            temp_archive.replace(archive)
        except Exception:
            temp_archive.unlink(missing_ok=True)
            raise
        completed += 1
        self._emit(progress_callback, completed, total_steps, "reproducibility_bundle", "completed")
        return out
