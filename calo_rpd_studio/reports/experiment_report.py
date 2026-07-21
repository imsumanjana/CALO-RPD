"""Experiment summary report."""

import json
from .report_builder import ReportBuilder


def build_experiment_report(database, experiment_id):
    e = database.get_experiment(experiment_id)
    runs = database.list_runs(experiment_id)
    failures = database.list_failures(experiment_id)
    b = ReportBuilder("CALO-RPD Experiment Report")
    b.add_section("Experiment", json.dumps(e, indent=2))
    b.add_section(
        "Run summary",
        f"Completed runs: {len(runs)}\nFailed runs: {len(failures)}\nVerified runs: {sum(r['validation_status'] == 'verified' for r in runs)}",
    )
    return b
