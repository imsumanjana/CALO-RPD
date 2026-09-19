"""Bounded case30 execution through real Qt audit, manager, optimizer and persistence."""

from __future__ import annotations
import json

import pytest

from calo_rpd_studio.app.experiment_manager import ExperimentManager
from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.experiments.execution_plans import ExecutionPlanKind
from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel


def make_panel(qtbot, tmp_path, monkeypatch, *, workers=1, database=None, workspace=False):
    import calo_rpd_studio.gui.panels.experiment_manager_panel as module

    errors = []
    monkeypatch.setattr(module, "show_error", lambda *a, **k: errors.append((a, k)))
    from PyQt6.QtWidgets import QMessageBox

    for method in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox,
            method,
            lambda *a, **k: (
                errors.append(tuple(str(x) for x in a[1:])),
                QMessageBox.StandardButton.Ok,
            )[1],
        )
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    state = AppState(database or tmp_path / "real-lifecycle.sqlite")
    config = state.config
    config.name = "bounded real lifecycle acceptance"
    config.case_name = "case30"
    config.study_case_plan = ["case30"]
    config.algorithms = ["TLBO"]
    config.algorithm_parameters = {"TLBO": {}}
    config.runs = 2
    config.population_size = 4
    config.max_iterations = 20
    config.budget.max_evaluations = 24
    config.execution_backend = "cpu_only"
    config.requested_compute_device = "cpu"
    config.execution_purpose = "exploratory"
    config.scientific_backend = "numpy"
    config.device_resident_execution = False
    config.require_backend_parity = False
    config.throughput_engine_enabled = False
    config.automatic_batch_calibration = False
    config.reuse_compatible_results = False
    config.parallel_workers = workers
    config.output_directory = str(tmp_path / "results")
    config.run_checkpoint_root = str(tmp_path / "checkpoints")
    config.checkpoint_interval_evaluations = 8
    if state.execution_control.active_stage() is None:
        state.execution_control.submit_algorithm_stage(config)
    manager = ExperimentManager(state)
    manager.failed.connect(lambda message: errors.append(message))
    # Exercise distinct real execution lanes by controlling only the hardware-width probe.
    # The GUI intentionally chooses automatic width and enforces FP64/parity on Apply.
    monkeypatch.setattr(
        ExperimentManagerPanel, "_recommended_worker_count", staticmethod(lambda: workers)
    )
    from calo_rpd_studio.app.workspace_campaign import WorkspaceCampaignCoordinator

    coordinator = WorkspaceCampaignCoordinator(state, manager) if workspace else None
    panel = ExperimentManagerPanel(state, manager, workspace_coordinator=coordinator)
    qtbot.addWidget(panel)
    panel.show_context("individual_experiment")
    panel.refresh()
    return state, manager, panel, errors


def audit_and_stage(qtbot, state, panel):
    retained_runs = state.database.list_runs()
    assert panel.run_fairness_audit(), panel.audit.toPlainText()
    qtbot.waitUntil(
        lambda: panel.audit_worker is not None and not panel.audit_worker.isRunning(), timeout=60000
    )
    qtbot.waitUntil(lambda: not state.task_status.busy, timeout=10000)
    assert panel.fairness_passed, panel.audit.toPlainText()
    plan = state.execution_control.active_plan(ExecutionPlanKind(panel.execution_mode))
    assert plan["lifecycle_state"] == "audited"
    panel.stage_current_plan()
    plan = state.database.get_execution_plan(plan["id"])
    assert plan["lifecycle_state"] == "staged", panel.status.text()
    assert state.database.list_runs() == retained_runs
    return plan


def wait_finished(qtbot, manager):
    try:
        qtbot.waitUntil(lambda: not manager.running, timeout=90000)
    finally:
        if manager.worker is not None and manager.worker.isRunning():
            manager.cancel()
            manager.worker.wait(10000)


@pytest.mark.parametrize("workers", [1, 2])
def test_real_individual_audit_launch_results_and_reopen(qtbot, tmp_path, monkeypatch, workers):
    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch, workers=workers)
    plan = audit_and_stage(qtbot, state, panel)
    started, completed, progress = [], [], []
    manager.started.connect(started.append)
    manager.completed.connect(completed.append)
    manager.progress.connect(progress.append)
    panel.start_comparison()
    assert manager.running, (errors, panel.status.text())
    wait_finished(qtbot, manager)
    assert errors == [], errors
    assert len(started) == len(completed) == 1
    assert started == completed
    experiment_id = completed[0]
    saved = json.loads(state.database.get_experiment(experiment_id)["config_json"])
    assert saved["case_name"] == "case30"
    assert saved["parallel_workers"] == workers
    assert saved["scientific_backend"] == "torch_fp64"
    assert saved["require_backend_parity"] is True
    assert saved["runs"] == 2 and saved["population_size"] == 4
    assert saved["algorithms"] == ["TLBO"] and saved["budget"]["max_evaluations"] == 24
    rows = state.database.list_runs(experiment_id)
    assert len(rows) == 2, rows
    assert {row["algorithm"] for row in rows} == {"TLBO"}
    assert {row["run_index"] for row in rows} == {0, 1}
    for row in rows:
        segments = state.database.list_run_segments(row["id"])
        assert segments[-1]["end_evaluations"] == 24, segments
    assert state.database.get_execution_plan(plan["id"])["lifecycle_state"] == "completed"
    assert state.execution_control.controller()["controller"] == "none"
    assert progress and not state.task_status.busy
    from calo_rpd_studio.results.database import ResultDatabase

    reopened = ResultDatabase(state.database.path)
    assert reopened.list_runs(experiment_id) == rows
    assert list((tmp_path / "results").rglob("*.npz")), "Numerical arrays must be retained"


def test_real_safe_pause_reopen_and_exact_resume(qtbot, tmp_path, monkeypatch):
    import threading
    import calo_rpd_studio.app.experiment_manager as module

    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch)
    plan = audit_and_stage(qtbot, state, panel)
    original = module.run_single
    evaluated, release = threading.Event(), threading.Event()
    calls = []

    def actual_run_with_test_boundary(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args[1], args[2]))
        if len(calls) == 1:
            evaluated.set()
            if not release.wait(10):
                raise TimeoutError("Acceptance test did not release the completed numerical job")
        return result

    monkeypatch.setattr(module, "run_single", actual_run_with_test_boundary)
    panel.start_comparison()
    try:
        qtbot.waitUntil(evaluated.is_set, timeout=60000)
        manager.pause()
    finally:
        release.set()
    wait_finished(qtbot, manager)
    assert errors == [], errors
    paused = state.database.get_execution_plan(plan["id"])
    assert paused["lifecycle_state"] == "paused", paused
    rows = state.database.list_runs()
    assert len(rows) == 1
    original_row = dict(rows[0])
    assert state.execution_control.controller()["controller"] == "individual_experiment"
    campaign_id = paused["campaign_id"]
    assert state.database.get_campaign(campaign_id)["status"] == "paused"

    recovered, resumed_manager, resumed_panel, resumed_errors = make_panel(
        qtbot, tmp_path, monkeypatch, database=state.database.path
    )
    resumed_panel.resume_current_plan()
    assert resumed_manager.running, resumed_errors
    wait_finished(qtbot, resumed_manager)
    assert resumed_errors == [], resumed_errors
    final_rows = recovered.database.list_runs(original_row["experiment_id"])
    assert len(final_rows) == 2
    assert recovered.database.get_run(original_row["id"]) == original_row
    assert calls == [("TLBO", 0), ("TLBO", 1)], "Completed jobs must not be recomputed on resume"
    assert recovered.database.get_execution_plan(plan["id"])["lifecycle_state"] == "completed"
    assert recovered.execution_control.controller()["controller"] == "none"


def test_real_terminal_cancellation_preserves_committed_work(qtbot, tmp_path, monkeypatch):
    import threading
    import calo_rpd_studio.app.experiment_manager as module

    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch)
    plan = audit_and_stage(qtbot, state, panel)
    original = module.run_single
    evaluated, release = threading.Event(), threading.Event()

    def actual_run_with_cancellation_boundary(*args, **kwargs):
        result = original(*args, **kwargs)
        evaluated.set()
        if not release.wait(10):
            raise TimeoutError("Acceptance cancellation boundary was not released")
        return result

    monkeypatch.setattr(module, "run_single", actual_run_with_cancellation_boundary)
    panel.start_comparison()
    try:
        qtbot.waitUntil(evaluated.is_set, timeout=60000)
        manager.cancel()
    finally:
        release.set()
    wait_finished(qtbot, manager)
    assert errors == [], errors
    assert len(state.database.list_runs()) == 1
    assert state.database.get_execution_plan(plan["id"])["lifecycle_state"] == "cancelled"
    assert state.execution_control.controller()["controller"] == "none"
    with pytest.raises(RuntimeError):
        state.execution_control.resume(plan["id"], ExecutionPlanKind.INDIVIDUAL_EXPERIMENT)


def test_real_result_write_failure_is_retained_and_never_shown_as_success(
    qtbot, tmp_path, monkeypatch
):
    from calo_rpd_studio.results.result_store import ResultStore

    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch)
    plan = audit_and_stage(qtbot, state, panel)
    original = ResultStore.save_arrays
    attempts = []

    def fail_first_write(store, result):
        attempts.append(result)
        if len(attempts) == 1:
            raise OSError("Synthetic result-storage interruption")
        return original(store, result)

    monkeypatch.setattr(ResultStore, "save_arrays", fail_first_write)
    panel.start_comparison()
    wait_finished(qtbot, manager)
    assert errors == [], errors
    final = state.database.get_execution_plan(plan["id"])
    assert final["lifecycle_state"] == "completed_with_failures", final
    assert state.task_status.state == "Failed"
    assert manager.last_completion_succeeded is False
    assert panel.failed_runs == panel.completed_runs == 1
    tasks = state.database.list_campaign_tasks(final["campaign_id"])
    assert sorted(row["status"] for row in tasks) == ["completed", "failed"]
    assert len(state.database.list_runs()) == 1
    assert state.execution_control.controller()["controller"] == "none"


def test_parallel_storage_failure_stops_as_resumable_not_success(qtbot, tmp_path, monkeypatch):
    from calo_rpd_studio.results.result_store import ResultStore

    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch, workers=2)
    plan = audit_and_stage(qtbot, state, panel)

    def refuse_array_write(*args, **kwargs):
        raise OSError("Synthetic unavailable result volume")

    monkeypatch.setattr(ResultStore, "save_arrays", refuse_array_write)
    panel.start_comparison()
    wait_finished(qtbot, manager)
    assert errors and "Synthetic unavailable result volume" in str(errors)
    assert state.task_status.state == "Failed"
    assert manager.last_completion_succeeded is not True
    assert (
        state.database.get_execution_plan(plan["id"])["lifecycle_state"] == "interrupted_resumable"
    )
    assert state.database.list_runs() == []


def test_mismatched_campaign_is_rejected_before_any_numerical_job(qtbot, tmp_path, monkeypatch):
    import calo_rpd_studio.app.experiment_manager as module

    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch)
    plan = audit_and_stage(qtbot, state, panel)
    config = state.execution_control.plan_configuration(plan["id"])
    config.budget.max_evaluations += 4
    attempted = []

    def forbidden_numerical_admission(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("Tampered scientific intent reached numerical execution")

    monkeypatch.setattr(module, "run_single", forbidden_numerical_admission)
    state.execution_control.begin_run(plan["id"])
    assert manager.start_comparison(config)
    wait_finished(qtbot, manager)
    assert attempted == []
    assert errors and "frozen" in str(errors).lower()
    assert state.database.list_runs() == []
    assert state.task_status.state == "Failed"
    assert manager.last_completion_succeeded is not True


def test_real_workspace_runs_frozen_case_cells_through_shared_worker(
    qtbot, tmp_path, monkeypatch, apply_workspace_study
):
    state, manager, panel, errors = make_panel(
        qtbot, tmp_path, monkeypatch, workers=2, workspace=True
    )
    panel.apply()
    state.config.study_case_plan = ["case30", "case57"]
    apply_workspace_study(state.execution_control, state.config, ("TLBO",))
    panel.show_context("workspace_study")
    panel.refresh()
    plan = audit_and_stage(qtbot, state, panel)
    cells = state.database.list_workspace_plan_cells(plan["id"])
    assert len(cells) == 2
    expected_runs = sum(int(cell["config"]["runs"]) for cell in cells)
    finished = []
    panel.workspace_coordinator.finished.connect(finished.append)
    panel.start_comparison()
    try:
        qtbot.waitUntil(
            lambda: not manager.running and not panel.workspace_coordinator.active, timeout=180000
        )
    finally:
        if manager.running:
            panel.workspace_coordinator.cancel_remaining()
            manager.worker.wait(15000)
    assert errors == [], errors
    assert finished == [plan["id"]]
    assert state.database.get_execution_plan(plan["id"])["lifecycle_state"] == "completed"
    cells = state.database.list_workspace_plan_cells(plan["id"])
    assert all(cell["lifecycle_state"] == "completed" for cell in cells), cells
    experiments = state.database.list_experiments()
    assert {json.loads(row["config_json"])["case_name"] for row in experiments} == {
        "case30",
        "case57",
    }
    assert len(state.database.list_runs()) == expected_runs
    assert state.execution_control.controller()["controller"] == "none"
    for cell in cells:
        assert cell["campaign_id"]
        state.execution_control.verify_campaign_binding(plan["id"], cell["campaign_id"])


def test_real_cuda_experiment_requires_physical_device(qtbot, tmp_path, monkeypatch):
    import os
    import torch

    if os.environ.get("CALO_REQUIRE_PHYSICAL_CUDA") != "1":
        pytest.skip("physical CUDA experiment requires the separately recorded CUDA lane")
    assert torch.cuda.is_available(), "Required physical CUDA is not visible"
    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch)
    panel.runs.setValue(1)
    panel.execution_backend.setCurrentIndex(panel.execution_backend.findData("cuda_preferred"))
    panel.device_resident_execution.setChecked(True)
    panel.persistent_workers.setChecked(False)
    panel.auto_batch_calibration.setChecked(False)
    plan = audit_and_stage(qtbot, state, panel)
    panel.start_comparison()
    wait_finished(qtbot, manager)
    assert errors == [], errors
    rows = state.database.list_runs()
    assert len(rows) == 1
    result = json.loads(rows[0]["result_json"])
    metadata = result["metadata"]
    attestation = metadata["device_attestation"]
    assert str(attestation["actual_evaluator_device"]).startswith("cuda"), attestation
    assert result["evaluations"] == 24
    assert manager.last_completion_succeeded is True
    final = state.database.get_execution_plan(plan["id"])
    assert final["lifecycle_state"] == "completed" and final["campaign_id"]
    state.execution_control.verify_campaign_binding(plan["id"], final["campaign_id"])
    (tmp_path / "cuda-experiment.json").write_text(
        json.dumps(
            {
                "attestation": attestation,
                "evaluations": result["evaluations"],
                "controller": state.execution_control.controller(),
                "campaign_id": final["campaign_id"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_real_independent_validation_export_and_exact_verified_reuse(qtbot, tmp_path, monkeypatch):
    import csv
    import zipfile
    import calo_rpd_studio.app.experiment_manager as module
    from calo_rpd_studio.benchmarking.validation import validate_runs
    from calo_rpd_studio.results.publication_export import PublicationExporter

    state, manager, panel, errors = make_panel(qtbot, tmp_path, monkeypatch)
    audit_and_stage(qtbot, state, panel)
    panel.start_comparison()
    wait_finished(qtbot, manager)
    assert errors == [], errors
    rows = state.database.list_runs()
    original_experiment = rows[0]["experiment_id"]
    exporter = PublicationExporter(state.database)
    with pytest.raises(ValueError, match="independently verified"):
        exporter.export(original_experiment, tmp_path / "unverified-export")
    validation = validate_runs(state.database, rows)
    assert validation["passed"] == 2 and validation["failed"] == validation["errors"] == 0, (
        validation
    )
    events = []
    exporter.export(
        original_experiment, tmp_path / "verified-export", progress_callback=events.append
    )
    completed_events = [event for event in events if event["status"] == "completed"]
    assert len(completed_events) == 5 and events[-1]["percent"] == 100
    assert len({event["artifact"] for event in completed_events}) == 5
    assert [event["percent"] for event in events] == sorted(event["percent"] for event in events)
    with (tmp_path / "verified-export/verified_runs.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        exported = list(csv.DictReader(stream))
    assert {item["run_id"] for item in exported} == {row["id"] for row in rows}
    assert all(item["validation_status"] == "verified" for item in exported)
    assert all(int(item["evaluations"]) == 24 for item in exported)
    with zipfile.ZipFile(tmp_path / "verified-export/reproducibility_bundle.zip") as archive:
        assert "experiment_metadata.json" in archive.namelist()
        assert "verified_runs.csv" in archive.namelist()

    original_rows = state.database.list_runs(original_experiment)
    # A fresh immutable plan may reuse exact independently verified evidence, not recompute it.
    panel.reuse_results.setChecked(True)
    plan = audit_and_stage(qtbot, state, panel)
    attempted = []

    def disallow_recomputation(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("Exact verified result reuse unexpectedly executed the optimizer")

    monkeypatch.setattr(module, "run_single", disallow_recomputation)
    panel.start_comparison()
    wait_finished(qtbot, manager)
    assert errors == [], errors
    final = state.database.get_execution_plan(plan["id"])
    assert final["lifecycle_state"] == "completed", final
    tasks = state.database.list_campaign_tasks(final["campaign_id"])
    assert len(tasks) == 2 and all(row["status"] == "reused" for row in tasks), tasks
    assert attempted == []
    assert state.database.list_runs(original_experiment) == original_rows
    reused_ids = {row["run_id"] for row in tasks}
    assert all(
        state.database.get_run(run_id)["validation_status"] == "verified" for run_id in reused_ids
    )
