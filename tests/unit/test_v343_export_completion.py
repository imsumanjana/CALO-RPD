from __future__ import annotations

import json
import warnings
import zipfile
from types import SimpleNamespace

import numpy as np

from calo_rpd_studio.algorithms.result import OptimizerResult
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.portfolio.exporter import PortfolioExporter
from calo_rpd_studio.portfolio.models import EvidenceProfile, PortfolioConfig, PortfolioKind, StorageProfile
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.results.result_store import ResultStore


def _completed(run_index: int = 0):
    result = OptimizerResult(
        algorithm="CALO",
        seed=100 + run_index,
        parameters={},
        best_vector=np.asarray([0.25, 0.75]),
        decoded_controls={"vg_1": 1.01},
        best_objective=2.1,
        objective_components={"active_power_loss": 2.1},
        total_constraint_violation=0.0,
        feasible=True,
        evaluations=100,
        iterations=4,
        convergence_history=[3.0, 2.5, 2.1],
        runtime_seconds=0.2,
        final_population=np.asarray([[0.25, 0.75], [0.3, 0.7]]),
        termination_reason="budget",
        metadata={
            "convergence_evaluations": [20, 60, 100],
            "best_feasible_objective_history": [3.0, 2.5, 2.1],
            "best_constraint_violation_history": [0.02, 0.01, 0.0],
            "constraint_component_histories": {"bus_voltage": [0.02, 0.01, 0.0]},
            "first_feasible_evaluation": 100,
            "solution_state": {
                "scenarios": [
                    {
                        "bus_numbers": [1, 2],
                        "vm_pu": [1.0, 0.99],
                        "branch_from_bus": [1],
                        "branch_to_bus": [2],
                        "loading_percent": [42.0],
                        "qg_mvar": [5.0],
                        "generator_bus": [1],
                        "converged": True,
                        "active_loss_mw": 2.1,
                    }
                ]
            },
        },
    )
    return SimpleNamespace(
        algorithm="CALO",
        run_index=run_index,
        seeds=SimpleNamespace(
            algorithm_seed=100 + run_index,
            scenario_seed=200 + run_index,
            ai_inference_seed=300 + run_index,
        ),
        result=result,
    )


def _database(tmp_path, requested_outputs=None):
    database = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig(algorithms=["CALO"])
    config.portfolio = PortfolioConfig(
        kind=PortfolioKind.SINGLE_RUN,
        evidence_profile=EvidenceProfile.DIAGNOSTIC,
        requested_outputs=requested_outputs or [],
        require_independent_validation=False,
        storage_profile=StorageProfile.FULL_SINGLE_RUN,
    )
    config.validate()
    experiment_id = database.create_experiment(config, {})
    completed = _completed()
    store = ResultStore(tmp_path / "arrays", storage_profile="full_single_run")
    arrays_path = store.save_arrays(completed.result)
    database.add_run(experiment_id, completed, str(arrays_path))
    return database, experiment_id


def test_portfolio_bundle_is_scoped_and_reports_progress_to_100(tmp_path):
    database, experiment_id = _database(tmp_path, ["feasible_run_probability"])
    output = tmp_path / "publication_export"
    unrelated = output / "old_export" / "unrelated-large-file.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"x" * 1024 * 256)

    progress = []
    PortfolioExporter(database).export(experiment_id, output, progress_callback=progress.append)

    archive = output / "reproducibility_bundle.zip"
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "old_export/unrelated-large-file.bin" not in names
    assert "raw_results/all_runs.csv" in names
    assert "configurations/experiment_config.json" in names
    assert "portfolio_manifest_snapshot.json" in names
    assert any(
        item["artifact"] == "reproducibility_bundle" and str(item["status"]).startswith("packing")
        for item in progress
    )
    assert progress[-1]["percent"] == 100
    assert progress[-1]["artifact"] == "reproducibility_bundle"
    assert progress[-1]["status"] == "completed"


def test_portfolio_bundle_can_pause_during_final_artifact_and_resume(tmp_path):
    database, experiment_id = _database(tmp_path, ["feasible_run_probability"])
    output = tmp_path / "portfolio"
    cancel = {"requested": False}

    def on_progress(payload):
        if payload["artifact"] == "reproducibility_bundle" and str(payload["status"]).startswith("packing"):
            cancel["requested"] = True

    PortfolioExporter(database).export(
        experiment_id,
        output,
        progress_callback=on_progress,
        cancel_callback=lambda: cancel["requested"],
    )
    manifest = json.loads((output / "portfolio_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cancelled"] is True
    assert not (output / "reproducibility_bundle.zip.tmp").exists()
    assert manifest.get("artifacts", {}).get("reproducibility_bundle", {}).get("status") != "completed"

    progress = []
    PortfolioExporter(database).export(experiment_id, output, progress_callback=progress.append)
    manifest = json.loads((output / "portfolio_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cancelled"] is False
    assert manifest["artifacts"]["reproducibility_bundle"]["status"] == "completed"
    assert progress[-1]["percent"] == 100


def test_median_convergence_does_not_warn_for_all_nan_leading_grid_columns(tmp_path):
    database, experiment_id = _database(tmp_path, ["median_convergence"])
    rows = database.list_runs(experiment_id)
    destination = tmp_path / "median_convergence"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PortfolioExporter(database)._median_convergence(rows, destination, False)
    messages = [str(item.message) for item in caught]
    assert not any("All-NaN slice encountered" in message for message in messages)


def test_standard_publication_export_reports_bundle_progress_and_reaches_100(tmp_path):
    from calo_rpd_studio.results.publication_export import PublicationExporter

    database = ResultDatabase(tmp_path / "standard.sqlite")
    config = ExperimentConfig(algorithms=["CALO"])
    experiment_id = database.create_experiment(config, {})
    completed = _completed()
    arrays_path = ResultStore(tmp_path / "standard_arrays").save_arrays(completed.result)
    run_id = database.add_run(experiment_id, completed, str(arrays_path))
    with database.connect() as con:
        con.execute("UPDATE runs SET validation_status='verified' WHERE id=?", (run_id,))

    progress = []
    output = PublicationExporter(database).export(
        experiment_id,
        tmp_path / "standard_publication",
        progress_callback=progress.append,
    )
    assert (output / "reproducibility_bundle.zip").is_file()
    assert any(
        item["artifact"] == "reproducibility_bundle" and str(item["status"]).startswith("packing")
        for item in progress
    )
    percentages = [int(item["percent"]) for item in progress]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100
