"""Result record integrity utilities."""

import json
import math


def check_run_record(row):
    data = json.loads(row["result_json"])
    issues = []
    if not math.isfinite(float(data["best_objective"])):
        issues.append("Best objective is not finite.")
    if int(data["evaluations"]) <= 0:
        issues.append("Evaluation count is not positive.")
    if len(data.get("best_vector", [])) == 0:
        issues.append("Decision vector is empty.")
    return {"passed": not issues, "issues": issues}
