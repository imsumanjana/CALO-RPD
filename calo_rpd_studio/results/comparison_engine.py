"""Evidence-based algorithm comparison summaries with fail-closed scientific filtering."""

import json
import numpy as np
import pandas as pd


def summarize_runs(rows, *, require_verified: bool = True, feasible_objectives_only: bool = True):
    records = []
    for row in rows:
        data = json.loads(row["result_json"])
        validation = str(row.get("validation_status", "unverified"))
        feasible = bool(data.get("feasible", False))
        objective = float(data.get("best_objective", float("nan")))
        records.append({
            "algorithm": row["algorithm"], "objective": objective, "feasible": feasible,
            "violation": data.get("total_constraint_violation"), "runtime": data.get("runtime_seconds"),
            "evaluations": data.get("evaluations"), "validation": validation,
            "eligible_objective": bool((not require_verified or validation == "verified") and (not feasible_objectives_only or feasible) and np.isfinite(objective)),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    rows_out = []
    for algorithm, group in df.groupby("algorithm"):
        eligible = group[group["eligible_objective"]]
        values = eligible["objective"].astype(float).to_numpy()
        rows_out.append({
            "algorithm": algorithm,
            "best": float(np.min(values)) if values.size else np.nan,
            "mean": float(np.mean(values)) if values.size else np.nan,
            "median": float(np.median(values)) if values.size else np.nan,
            "worst": float(np.max(values)) if values.size else np.nan,
            "std": float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else np.nan),
            "eligible_objective_runs": int(values.size),
            "feasible_rate": float(group["feasible"].mean()),
            "verified_rate": float((group["validation"] == "verified").mean()),
            "runtime_mean": float(pd.to_numeric(group["runtime"], errors="coerce").mean()),
        })
    summary = pd.DataFrame(rows_out).set_index("algorithm")
    return summary.sort_values("median", na_position="last")


def interpret_comparison(summary, calo="CALO"):
    if summary.empty or calo not in summary.index:
        return "No complete eligible CALO comparison is available."
    eligible = summary[np.isfinite(pd.to_numeric(summary["median"], errors="coerce"))]
    if eligible.empty:
        return "No independently verified feasible objective evidence is available for comparison."
    leader = eligible.index[0]
    if leader == calo:
        return "CALO achieved the lowest median objective among the eligible independently verified feasible runs."
    return f"{leader} achieved the lowest median objective among the eligible independently verified feasible runs; CALO ranked lower on this metric."
