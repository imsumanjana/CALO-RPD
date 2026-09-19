from __future__ import annotations

import pytest
from PyQt6.QtCore import QObject, QSettings


@pytest.fixture
def wired_window(qtbot, tmp_path, monkeypatch):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    import calo_rpd_studio.app.main_window as module
    from calo_rpd_studio.app.session_recovery import SessionRecoveryJournal
    from calo_rpd_studio.app.settings_manager import SettingsManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingModelLibrary

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(tmp_path / "settings"))
    monkeypatch.setattr(
        module, "SessionRecoveryJournal", lambda: SessionRecoveryJournal(tmp_path / "session")
    )
    monkeypatch.setattr(
        module,
        "TrainingModelLibrary",
        lambda settings: TrainingModelLibrary(
            settings, default_directory=tmp_path / "empty-policies"
        ),
    )
    monkeypatch.setattr(module.MainWindow, "_initial_system_scan", lambda self: None)
    monkeypatch.setattr(module.MainWindow, "_check_unfinished_work", lambda self: None)
    from PyQt6.QtWidgets import QMessageBox

    dialogs = []
    for method in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox,
            method,
            lambda *a, **k: (
                dialogs.append(tuple(str(x) for x in a[1:])),
                QMessageBox.StandardButton.Ok,
            )[1],
        )
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    state = AppState(tmp_path / "wiring.sqlite")
    manager = ExperimentManager(state)
    window = module.MainWindow(state, manager, SettingsManager())
    qtbot.addWidget(
        window, before_close_func=lambda widget: widget.activity_center.detach_logging()
    )
    window._acceptance_dialogs = dialogs
    import os
    from pathlib import Path

    if os.environ.get("CALO_ACCEPTANCE_LANE") == "native":
        window.show()
        qtbot.waitExposed(window)
        directory = Path(os.environ["CALO_GUI_ARTIFACT_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        assert window.grab().save(str(directory / "native-wiring-window.png"), "PNG")
    yield window
    manager._busy = False
    state.task_status.finish("fixture teardown")


def _receiver_counts(window):
    owners = (
        window.state,
        window.experiment_manager,
        window.workflow,
        window.command_registry,
        window.sidebar,
        window.training_launch_model,
    )
    counts = {}
    for index, owner in enumerate(owners):
        for name in dir(type(owner)):
            # PyQt creates lifetime-management proxies for Python spy callables. They are not
            # application command subscriptions and may outlive disconnect until Qt cleanup.
            if name in {"destroyed", "objectNameChanged"}:
                continue
            value = getattr(owner, name, None)
            if hasattr(value, "signal") and hasattr(value, "connect"):
                counts[f"{index}:{name}"] = owner.receivers(value)
    return counts


def test_rebinding_workflow_is_idempotent_and_delivers_once(wired_window, monkeypatch):
    window = wired_window
    before = _receiver_counts(window)
    for _ in range(10):
        window._connect_workflow()
    assert _receiver_counts(window) == before
    delivered = []
    monkeypatch.setattr(window.workflow, "mark_completed", delivered.append)
    window.pages_by_key["algorithms"].stage_completed.emit()
    assert delivered == ["algorithms"]


def test_safe_close_waits_for_worker_idle_and_closes_exactly_once(wired_window, qtbot, monkeypatch):
    window = wired_window
    closes = []
    monkeypatch.setattr(window, "close", lambda: closes.append("closed"))
    window.experiment_manager._busy = True
    window._close_when_paused = True
    window._finish_deferred_close()
    qtbot.wait(10)
    assert closes == []
    window.experiment_manager._worker_finished()
    qtbot.waitUntil(lambda: closes == ["closed"], timeout=2000)
    window.experiment_manager.idle.emit()
    qtbot.wait(10)
    assert closes == ["closed"]


def test_navigation_does_not_multiply_global_receivers(wired_window, qtbot):
    window = wired_window
    before = _receiver_counts(window)
    commands = []
    window.command_registry.command_triggered.connect(commands.append)
    expected = []
    for _ in range(4):
        for spec in window.command_registry.specs:
            action = window.command_registry.action(spec.command_id)
            if spec.handler != "workspace" or not action.isEnabled():
                continue
            expected.append(spec.command_id)
            action.trigger()
            qtbot.wait(1)
    assert expected, "The normal empty-policy navigation must remain available"
    assert commands == expected
    assert window._acceptance_dialogs == [], window._acceptance_dialogs
    window.command_registry.command_triggered.disconnect(commands.append)
    assert _receiver_counts(window) == before
    assert window.state.current_experiment_id == ""
    assert not window.experiment_manager.running


def test_inline_setup_events_route_to_current_mode_once(wired_window, monkeypatch):
    window = wired_window
    events = []
    monkeypatch.setattr(
        window.workflow, "mark_completed", lambda key: events.append(("workspace", key))
    )
    monkeypatch.setattr(
        window.workflow,
        "mark_individual_completed",
        lambda key: events.append(("individual_experiment", key)),
    )
    panel = window.pages_by_key["experiment"]
    for mode in ("individual_experiment", "workspace_study", "individual_experiment"):
        panel.show_context(mode)
        events.clear()
        panel.study_formulation.stage_completed.emit()
        expected = "workspace" if mode == "workspace_study" else mode
        assert events == [(expected, "orpd")]


def test_ribbon_actions_have_unique_ids_and_no_duplicate_shortcuts(wired_window):
    registry = wired_window.command_registry
    identifiers = [spec.command_id for spec in registry.specs]
    assert len(set(identifiers)) == len(identifiers)
    objects = [registry.action(identifier) for identifier in identifiers]
    assert len({id(obj) for obj in objects}) == len(objects)
    shortcuts = [obj.shortcut().toString() for obj in objects if not obj.shortcut().isEmpty()]
    assert len(set(shortcuts)) == len(shortcuts)
    assert all(isinstance(obj.parent(), QObject) for obj in objects)


def test_failed_completion_never_promotes_workflow(wired_window, monkeypatch):
    events = []
    window = wired_window
    monkeypatch.setattr(
        window.workflow, "mark_experiment_completed", lambda: events.append("success")
    )
    monkeypatch.setattr(
        window.workflow, "mark_experiment_stopped", lambda: events.append("stopped")
    )
    window.experiment_manager.last_completion_succeeded = False
    window._manager_completion_workflow_event("failed-campaign")
    window.experiment_manager.last_completion_succeeded = None
    window._manager_completion_workflow_event("unverified-campaign")
    window.experiment_manager.last_completion_succeeded = True
    window._manager_completion_workflow_event("verified-campaign")
    assert events == ["stopped", "stopped", "success"]


def test_all_enabled_workspace_actions_share_navigation_prerequisites(wired_window):
    window = wired_window
    for spec in window.command_registry.specs:
        if spec.handler == "workspace" and spec.workspace:
            action = window.command_registry.action(spec.command_id)
            blocker = window._navigation_blocker(spec.workspace)
            if blocker:
                assert not action.isEnabled(), spec.command_id
                assert blocker[1] in action.toolTip()


def test_policy_free_comparator_stage_unlocks_individual_setup(wired_window):
    window = wired_window
    window.state.config.algorithms = ["TLBO"]
    window.state.config.algorithm_parameters = {"TLBO": {}}
    window.state.execution_control.submit_algorithm_stage(window.state.config)
    window.pages_by_key["algorithms"].stage_completed.emit()
    action = window.command_registry.action("experiment.individual")
    assert action.isEnabled(), action.toolTip()
    action.trigger()
    assert window._acceptance_dialogs == [], window._acceptance_dialogs
    assert window.stack.currentWidget() is window.pages_by_key["experiment"]
    assert not window.experiment_manager.running
    assert not window.state.policy_training_active


def test_rejected_experiment_apply_preserves_shared_configuration(wired_window):
    from copy import deepcopy

    state = wired_window.state
    panel = wired_window.pages_by_key["experiment"]
    panel.show_context("individual_experiment")
    assert state.execution_control.active_stage() is None
    before = deepcopy(state.config.to_dict())
    notifications = []
    state.config_changed.connect(lambda config: notifications.append(config))
    panel.population.setValue(17)
    panel.seed.setValue(8127)
    with pytest.raises(RuntimeError, match="Submit at least one algorithm"):
        panel.apply()
    assert state.config.to_dict() == before
    assert notifications == []


def test_rejected_formulation_apply_preserves_shared_configuration(wired_window):
    from copy import deepcopy

    state = wired_window.state
    panel = wired_window.pages_by_key["orpd"]
    before = deepcopy(state.config.to_dict())
    completed = []
    panel.stage_completed.connect(lambda: completed.append(True))
    panel.tap_min.setValue(1.1)
    panel.tap_max.setValue(0.9)
    panel.apply()
    assert wired_window._acceptance_dialogs
    assert state.config.to_dict() == before
    assert completed == []


def test_rejected_scenario_apply_preserves_shared_configuration(wired_window):
    from copy import deepcopy

    state = wired_window.state
    panel = wired_window.pages_by_key["scenarios"]
    before = deepcopy(state.config.to_dict())
    completed = []
    panel.stage_completed.connect(lambda: completed.append(True))
    panel.count.setValue(13)
    panel.branch.setText("not-an-index")
    panel.apply()
    assert wired_window._acceptance_dialogs
    assert state.config.to_dict() == before
    assert completed == []


@pytest.mark.parametrize("persist_fails", [False, True])
def test_workspace_apply_publishes_once_only_after_durable_draft(
    wired_window, monkeypatch, persist_fails
):
    from copy import deepcopy
    from calo_rpd_studio.portfolio.study_planning import PortfolioGoalPlanner

    state = wired_window.state
    config = state.config
    config.algorithms = ["TLBO"]
    config.algorithm_parameters = {"TLBO": {}}
    config.portfolio.requested_outputs = ["objective_convergence"]
    stage = state.execution_control.submit_algorithm_stage(config)
    goal = PortfolioGoalPlanner.create(config.portfolio, stage, ("TLBO",))
    state.database.replace_portfolio_goal(goal)
    panel = wired_window.pages_by_key["experiment"]
    panel.show_context("workspace_study")
    panel.refresh()
    panel.population.setValue(17)
    panel.budget.setValue(340)
    panel.seed.setValue(8831)
    before = deepcopy(state.config.to_dict())
    observed = []

    def changed(config):
        plan = state.execution_control.active_plan(panel.execution_mode)
        observed.append((deepcopy(config.to_dict()), plan))

    state.config_changed.connect(changed)
    if persist_fails:

        def reject(*args, **kwargs):
            raise OSError("injected draft storage failure")

        monkeypatch.setattr(state.execution_control, "create_workspace_draft", reject)
    panel.apply_workspace_study_setup()
    if persist_fails:
        assert "injected draft storage failure" in str(wired_window._acceptance_dialogs)
        assert state.config.to_dict() == before
        assert observed == []
    else:
        assert wired_window._acceptance_dialogs == []
        assert len(observed) == 1
        saved, plan = observed[0]
        assert plan is not None
        assert saved["population_size"] == 17
        assert saved["master_seed"] == 8831
        assert plan["design"]["config"]["population_size"] == 17
