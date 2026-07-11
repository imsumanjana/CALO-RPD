"""Verified-only publication and reproducibility export."""
from __future__ import annotations
from pathlib import Path
import json
import zipfile
import pandas as pd
from calo_rpd_studio.statistics.descriptive import descriptive_statistics

class PublicationExporter:
    def __init__(self, database):
        self.database = database

    def export(self, experiment_id, directory):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        rows = self.database.list_runs(experiment_id, verified_only=True)
        records = []
        for row in rows:
            data = json.loads(row["result_json"])
            records.append({
                "run_id": row["id"], "algorithm": row["algorithm"], "run_index": row["run_index"],
                "objective": data["best_objective"], "feasible": data["feasible"],
                "violation": data["total_constraint_violation"], "runtime_seconds": data["runtime_seconds"],
                "evaluations": data["evaluations"],
            })
        frame = pd.DataFrame(records)
        frame.to_csv(out / "verified_runs.csv", index=False)
        if not frame.empty:
            stats = {name: descriptive_statistics(group["objective"].to_numpy()) for name, group in frame.groupby("algorithm")}
            pd.DataFrame(stats).T.to_csv(out / "descriptive_statistics.csv")
            (out / "verified_runs.tex").write_text(frame.to_latex(index=False), encoding="utf-8")
        experiment = self.database.get_experiment(experiment_id)
        (out / "experiment_metadata.json").write_text(
            json.dumps({"experiment": experiment, "verified_run_count": len(rows)}, indent=2), encoding="utf-8"
        )
        archive = out / "reproducibility_bundle.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in out.iterdir():
                if path != archive:
                    zf.write(path, path.name)
        return out
