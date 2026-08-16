from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from calo_rpd_studio.app.execution_control import ExecutionControlService
from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.app.workspace_campaign import WorkspaceCampaignCoordinator
from calo_rpd_studio.experiments.execution_plans import ExecutionLifecycle, ExecutionPlanKind
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.results.database import DATABASE_SCHEMA_VERSION, ResultDatabase


def prepared_service(tmp_path):
    database = ResultDatabase(tmp_path / "execution.sqlite")
    service = ExecutionControlService(database)
    config = ExperimentConfig()
    config.algorithms = ["CALO", "TLBO"]
    config.algorithm_parameters = {
        "CALO": {
            "use_ai": False,
            "strict_policy_binding": False,
            "allow_unqualified_policy": False,
        },
        "TLBO": {},
    }
    stage = service.submit_algorithm_stage(config)
    return database, service, config, stage


def audited_workspace(service, config):
    plan = service.create_workspace_draft(config, ("CALO",))
    service.record_audit(plan["id"], {"fair": True, "backend_parity_passed": True})
    return service.active_plan(ExecutionPlanKind.WORKSPACE)


def test_workspace_pause_releases_controller_but_retains_plan(tmp_path) -> None:
    _database, service, config, _stage = prepared_service(tmp_path)
    plan = audited_workspace(service, config)
    service.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    service.begin_run(plan["id"])
    service.request_pause(plan["id"])
    paused = service.commit_paused(plan["id"])

    assert paused["lifecycle_state"] == ExecutionLifecycle.PAUSED.value
    assert paused["active_slot"] == 1
    assert service.controller()["controller"] == "none"


def test_paused_workspace_can_reacquire_only_to_commit_terminal_cancel(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    plan = audited_workspace(service, config)
    service.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    cell = database.list_workspace_plan_cells(plan["id"])[0]
    frozen = service.plan_configuration(plan["id"], cell_id=cell["id"])
    experiment_id = database.create_experiment(frozen, {})
    campaign_id = database.create_campaign(
        experiment_id, "", "comparison", frozen.to_dict(), 1
    )
    database.add_campaign_task(
        campaign_id,
        0,
        "CALO",
        0,
        {"algorithm_seed": 1, "scenario_seed": 2, "ai_inference_seed": 3},
        "f" * 64,
        [],
    )
    database.upsert_resumable_task(
        campaign_id,
        "experiment_campaign",
        "Retained Workspace campaign",
        "paused",
        0,
        1,
        {"campaign_id": campaign_id},
    )
    service.begin_run(plan["id"], campaign_id=campaign_id)
    service.commit_paused(plan["id"], campaign_id=campaign_id)

    assert database.get_campaign(campaign_id)["status"] == "paused"
    assert database.list_campaign_tasks(campaign_id)[0]["status"] == "paused"
    assert database.get_resumable_task(campaign_id)["resumable"] == 1

    service.resume(plan["id"], ExecutionPlanKind.WORKSPACE)
    cancelled = service.cancel_retained(
        plan["id"],
        message="Scientist cancelled the retained remaining Workspace work",
    )

    assert cancelled["lifecycle_state"] == ExecutionLifecycle.CANCELLED.value
    assert cancelled["active_slot"] == 0
    assert service.controller()["controller"] == "none"
    assert database.get_campaign(campaign_id)["status"] == "cancelled"
    assert database.list_campaign_tasks(campaign_id)[0]["status"] == "cancelled"
    assert database.get_resumable_task(campaign_id)["resumable"] == 0
    assert database.list_workspace_plan_cells(plan["id"])[0]["lifecycle_state"] == "cancelled"


def test_individual_pause_retains_exclusive_controller(tmp_path) -> None:
    _database, service, config, _stage = prepared_service(tmp_path)
    plan = service.create_individual_draft(config)
    service.record_audit(plan["id"], {"fair": True})
    service.stage(plan["id"], ExecutionPlanKind.INDIVIDUAL_EXPERIMENT)
    service.begin_run(plan["id"])
    service.request_pause(plan["id"])
    paused = service.commit_paused(plan["id"])

    assert paused["lifecycle_state"] == ExecutionLifecycle.PAUSED.value
    assert service.controller()["controller"] == "individual_experiment"
    assert service.controller()["owner_plan_id"] == plan["id"]


def test_paused_workspace_resume_waits_for_individual_owner(tmp_path) -> None:
    _database, service, config, _stage = prepared_service(tmp_path)
    workspace = audited_workspace(service, config)
    service.stage(workspace["id"], ExecutionPlanKind.WORKSPACE)
    service.begin_run(workspace["id"])
    service.commit_paused(workspace["id"])

    individual = service.create_individual_draft(config)
    service.record_audit(individual["id"], {"fair": True})
    service.stage(individual["id"], ExecutionPlanKind.INDIVIDUAL_EXPERIMENT)

    with pytest.raises(RuntimeError, match="already owned"):
        service.resume(workspace["id"], ExecutionPlanKind.WORKSPACE)

    service.commit_terminal(
        individual["id"],
        lifecycle=ExecutionLifecycle.DISCARDED_UNSTARTED,
        message="Test closes unstarted individual staging",
    )
    resumed = service.resume(workspace["id"], ExecutionPlanKind.WORKSPACE)
    assert resumed["controller"] == "workspace"
    assert resumed["owner_plan_id"] == workspace["id"]


def test_controller_acquisition_is_singleton_and_fenced(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    plan = audited_workspace(service, config)
    service.stage(plan["id"], ExecutionPlanKind.WORKSPACE)

    with pytest.raises(RuntimeError, match="already owned"):
        database.acquire_execution_controller(
            plan["id"],
            controller_kind="workspace",
            owner_instance_id="competing-instance",
            resume=False,
        )

    controller = service.controller()
    with pytest.raises(RuntimeError, match="fencing token"):
        database.transition_execution_plan(
            plan["id"],
            controller_epoch=int(controller["epoch"]) + 1,
            expected_states=(ExecutionLifecycle.STAGED.value,),
            new_state=ExecutionLifecycle.RUNNING.value,
            message="stale writer",
        )


def test_restart_fences_prior_instance_and_restores_staged_owner(tmp_path) -> None:
    database, first, config, _stage = prepared_service(tmp_path)
    plan = audited_workspace(first, config)
    first.stage(plan["id"], ExecutionPlanKind.WORKSPACE)

    recovered = ExecutionControlService(database)

    assert recovered.controller()["controller"] == "workspace"
    assert recovered.controller()["owner_plan_id"] == plan["id"]
    with pytest.raises(RuntimeError, match="application instance"):
        first.commit_terminal(
            plan["id"],
            lifecycle=ExecutionLifecycle.DISCARDED_UNSTARTED,
            message="stale instance",
        )
    recovered.commit_terminal(
        plan["id"],
        lifecycle=ExecutionLifecycle.DISCARDED_UNSTARTED,
        message="recovered instance closes staging",
    )


def test_workspace_interruption_resumes_with_retained_owner(tmp_path) -> None:
    _database, service, config, _stage = prepared_service(tmp_path)
    plan = audited_workspace(service, config)
    service.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    service.begin_run(plan["id"])
    service.transition(
        plan["id"],
        expected=(ExecutionLifecycle.RUNNING.value,),
        new_state=ExecutionLifecycle.INTERRUPTED_RESUMABLE.value,
        message="synthetic interruption",
    )

    resumed = service.resume(plan["id"], ExecutionPlanKind.WORKSPACE)

    assert resumed["lifecycle_state"] == ExecutionLifecycle.RUNNING.value
    assert service.controller()["controller"] == "workspace"


def test_restart_marks_inflight_workspace_cell_resumable(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    plan = audited_workspace(service, config)
    service.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    service.begin_run(plan["id"])
    cell = database.list_workspace_plan_cells(plan["id"])[0]
    database.update_workspace_plan_cell(cell["id"], lifecycle_state="running")

    recovered = ExecutionControlService(database)

    rows = database.list_workspace_plan_cells(plan["id"])
    assert rows[0]["lifecycle_state"] == ExecutionLifecycle.INTERRUPTED_RESUMABLE.value
    assert recovered.controller()["controller"] == "workspace"


def test_corrupt_controller_receipt_fails_closed(tmp_path) -> None:
    database, _service, _config, _stage = prepared_service(tmp_path)
    with database.connect() as con:
        con.execute(
            "UPDATE execution_controller SET state_receipt_sha256=? WHERE singleton_id=1",
            ("0" * 64,),
        )

    with pytest.raises(RuntimeError, match="integrity receipt"):
        database.get_execution_controller()


def test_new_draft_supersession_rebuilds_receipt_and_records_event(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    first = service.create_individual_draft(config)

    second = service.create_individual_draft(config)
    superseded = database.get_execution_plan(first["id"])

    assert second["id"] != first["id"]
    assert superseded["lifecycle_state"] == ExecutionLifecycle.DISCARDED_UNSTARTED.value
    with database.connect() as con:
        event = con.execute(
            """SELECT to_state,receipt_sha256 FROM execution_lifecycle_events
               WHERE plan_id=? ORDER BY id DESC LIMIT 1""",
            (first["id"],),
        ).fetchone()
    assert event["to_state"] == ExecutionLifecycle.DISCARDED_UNSTARTED.value
    assert event["receipt_sha256"] == superseded["state_receipt_sha256"]


def test_plan_checksum_drift_blocks_audit_without_state_mutation(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    plan = service.create_individual_draft(config)
    with database.connect() as con:
        con.execute(
            "UPDATE execution_plans SET design_json=? WHERE id=?",
            ('{"tampered":true}', plan["id"]),
        )

    with pytest.raises(RuntimeError, match="design checksum"):
        service.record_audit(plan["id"], {"fair": True})
    with database.connect() as con:
        state = con.execute(
            "SELECT lifecycle_state FROM execution_plans WHERE id=?", (plan["id"],)
        ).fetchone()[0]
    assert state == ExecutionLifecycle.DRAFT.value


def test_duplicate_plan_job_admission_is_rejected(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    plan = service.create_individual_draft(config)
    experiment_id = database.create_experiment(config, {})
    first_campaign = database.create_campaign(experiment_id, "", "comparison", config.to_dict(), 1)
    second_campaign = database.create_campaign(experiment_id, "", "comparison", config.to_dict(), 1)
    identity = "a" * 64
    database.add_campaign_task(
        first_campaign,
        0,
        "CALO",
        0,
        {"algorithm_seed": 1, "scenario_seed": 2, "ai_inference_seed": 3},
        "b" * 64,
        [],
        execution_plan_id=plan["id"],
        job_identity_sha256=identity,
    )

    with pytest.raises(RuntimeError, match="Duplicate scientific job admission"):
        database.add_campaign_task(
            second_campaign,
            0,
            "CALO",
            0,
            {"algorithm_seed": 1, "scenario_seed": 2, "ai_inference_seed": 3},
            "b" * 64,
            [],
            execution_plan_id=plan["id"],
            job_identity_sha256=identity,
        )


def test_schema_v1_migration_creates_backup_without_inventing_stage(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE legacy_note(value TEXT NOT NULL)")
        con.execute("INSERT INTO legacy_note(value) VALUES('retained')")
        con.execute("PRAGMA user_version=1")

    database = ResultDatabase(path)

    assert database.schema_version == DATABASE_SCHEMA_VERSION
    assert database.migration_backup_path
    assert database.get_active_algorithm_stage() is None
    with database.connect() as con:
        assert con.execute("SELECT value FROM legacy_note").fetchone()[0] == "retained"


def test_resume_rejects_campaign_with_changed_frozen_budget(tmp_path) -> None:
    database, service, config, _stage = prepared_service(tmp_path)
    plan = service.create_individual_draft(config)
    service.record_audit(plan["id"], {"fair": True})
    service.stage(plan["id"], ExecutionPlanKind.INDIVIDUAL_EXPERIMENT)
    service.begin_run(plan["id"])
    frozen = service.plan_configuration(plan["id"])
    experiment_id = database.create_experiment(frozen, {})
    campaign_id = database.create_campaign(
        experiment_id,
        "",
        "comparison",
        frozen.to_dict(),
        frozen.runs * len(frozen.algorithms),
    )
    service.transition(
        plan["id"],
        expected=(ExecutionLifecycle.RUNNING.value,),
        new_state=ExecutionLifecycle.RUNNING.value,
        message="Bind test campaign",
        campaign_id=campaign_id,
    )
    service.request_pause(plan["id"])
    service.commit_paused(plan["id"], campaign_id=campaign_id)
    with database.connect() as con:
        stored = json.loads(
            con.execute(
                "SELECT config_json FROM campaigns WHERE id=?", (campaign_id,)
            ).fetchone()[0]
        )
        stored["budget"]["max_evaluations"] += 1
        con.execute(
            "UPDATE campaigns SET config_json=? WHERE id=?",
            (json.dumps(stored), campaign_id),
        )

    with pytest.raises(RuntimeError, match="does not match the frozen plan"):
        service.resume(plan["id"], ExecutionPlanKind.INDIVIDUAL_EXPERIMENT)


class FakeExperimentManager(QObject):
    started = pyqtSignal(str)
    completed = pyqtSignal(str)
    paused = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)
    idle = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.worker = None
        self.started_configs = []

    def start_comparison(self, config) -> bool:
        self.started_configs.append(config)
        self.running = True
        return True

    def resume_campaign(self, _campaign_id: str, *, update_workspace: bool = True) -> bool:
        del update_workspace
        self.running = True
        return True

    def pause(self) -> None:
        return None

    def cancel(self) -> None:
        return None


def test_workspace_advances_to_next_cell_only_after_manager_idle(tmp_path) -> None:
    state = AppState(tmp_path / "workspace-sequence.sqlite")
    state.config.algorithms = ["CALO"]
    state.config.algorithm_parameters = {
        "CALO": {
            "use_ai": False,
            "strict_policy_binding": False,
            "allow_unqualified_policy": False,
        }
    }
    state.config.study_case_plan = ["case30", "case57"]
    state.execution_control.submit_algorithm_stage(state.config)
    plan = state.execution_control.create_workspace_draft(state.config, ("CALO",))
    state.execution_control.record_audit(plan["id"], {"fair": True})
    state.execution_control.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    manager = FakeExperimentManager()
    coordinator = WorkspaceCampaignCoordinator(state, manager)

    coordinator.run(plan["id"])
    assert len(manager.started_configs) == 1
    first_config = manager.started_configs[0]
    first_experiment = state.database.create_experiment(first_config, {})
    first_campaign = state.database.create_campaign(
        first_experiment,
        "",
        "comparison",
        first_config.to_dict(),
        first_config.runs * len(first_config.algorithms),
    )
    manager.worker = SimpleNamespace(campaign_id=first_campaign)
    manager.started.emit("experiment-cell-1")
    manager.completed.emit("experiment-cell-1")
    assert len(manager.started_configs) == 1

    manager.running = False
    manager.worker = None
    manager.idle.emit()
    assert len(manager.started_configs) == 2
