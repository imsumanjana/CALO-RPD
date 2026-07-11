"""Evidence-based algorithm comparison summaries."""
import json
import pandas as pd

def summarize_runs(rows):
    records = []
    for row in rows:
        data = json.loads(row["result_json"])
        records.append({
            "algorithm": row["algorithm"],
            "objective": data["best_objective"],
            "feasible": data["feasible"],
            "violation": data["total_constraint_violation"],
            "runtime": data["runtime_seconds"],
            "evaluations": data["evaluations"],
            "validation": row["validation_status"],
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.groupby("algorithm").agg(
        best=("objective", "min"), mean=("objective", "mean"), median=("objective", "median"),
        worst=("objective", "max"), std=("objective", "std"), feasible_rate=("feasible", "mean"),
        runtime_mean=("runtime", "mean")
    ).sort_values("median")

def interpret_comparison(summary, calo="CALO"):
    if summary.empty or calo not in summary.index:
        return "No complete CALO comparison is available."
    leader = summary.index[0]
    if leader == calo:
        return "CALO achieved the lowest median objective in the selected complete result set."
    return f"{leader} achieved the lowest median objective in the selected complete result set; CALO ranked lower on this metric."
