from __future__ import annotations

from calo_rpd_studio.app.experiment_manager import ExperimentManager
from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.experiments.execution_plans import ExecutionPlanKind
from calo_rpd_studio.gui.panels.experiment_manager_panel import (
    ExperimentManagerPanel,
    ScientificAuditWorker,
)
from calo_rpd_studio.gui.panels.portfolio_manager_panel import PortfolioManagerPanel


def _submit_stage(state) -> None:
    state.config.algorithms = ["CALO", "TLBO"]
    state.config.algorithm_parameters = {
        "CALO": {
            "use_ai": False,
            "strict_policy_binding": False,
            "allow_unqualified_policy": False,
        },
        "TLBO": {},
    }
    state.execution_control.submit_algorithm_stage(state.config)


def _audited_workspace(state):
    plan = state.execution_control.create_workspace_draft(state.config, ("CALO",))
    state.execution_control.record_audit(plan["id"], {"fair": True})
    return state.execution_control.active_plan(ExecutionPlanKind.WORKSPACE)


def test_portfolio_preview_treats_missing_stage_as_prerequisite(
    qtbot, tmp_path, monkeypatch
) -> None:
    technical_errors = []
    monkeypatch.setattr(
        "calo_rpd_studio.gui.panels.portfolio_manager_panel.log_technical_error",
        lambda *args, **kwargs: technical_errors.append((args, kwargs)),
    )
    state = AppState(tmp_path / "portfolio-missing-stage.sqlite")
    panel = PortfolioManagerPanel(state)
    qtbot.addWidget(panel)

    panel.refresh_plan()

    assert "Submit at least one algorithm" in panel.plan_summary.text()
    assert "No plan or execution has started" in panel.plan_detail.text()
    assert technical_errors == []


def test_portfolio_preview_treats_empty_study_filter_as_prerequisite(
    qtbot, tmp_path, monkeypatch
) -> None:
    technical_errors = []
    monkeypatch.setattr(
        "calo_rpd_studio.gui.panels.portfolio_manager_panel.log_technical_error",
        lambda *args, **kwargs: technical_errors.append((args, kwargs)),
    )
    state = AppState(tmp_path / "portfolio-empty-filter.sqlite")
    _submit_stage(state)
    panel = PortfolioManagerPanel(state)
    qtbot.addWidget(panel)

    panel._clear_study_filter()

    assert "Select at least one submitted algorithm" in panel.plan_summary.text()
    assert "does not modify the submitted algorithm stage" in panel.plan_detail.text()
    assert technical_errors == []


def test_portfolio_preview_treats_stage_config_drift_as_resubmission_prerequisite(
    qtbot, tmp_path, monkeypatch
) -> None:
    technical_errors = []
    monkeypatch.setattr(
        "calo_rpd_studio.gui.panels.portfolio_manager_panel.log_technical_error",
        lambda *args, **kwargs: technical_errors.append((args, kwargs)),
    )
    state = AppState(tmp_path / "portfolio-stage-config-drift.sqlite")
    _submit_stage(state)
    state.config.algorithms = ["PSO"]
    panel = PortfolioManagerPanel(state)
    qtbot.addWidget(panel)

    panel.refresh_plan()

    assert "does not match the current experiment configuration" in panel.plan_summary.text()
    assert "submit them again" in panel.plan_detail.text()
    assert "retained stage was not changed" in panel.plan_detail.text()
    assert technical_errors == []


def test_portfolio_restores_the_retained_workspace_subset_instead_of_all_algorithms(
    qtbot, tmp_path
) -> None:
    state = AppState(tmp_path / "portfolio-retained-subset.sqlite")
    _submit_stage(state)
    state.execution_control.create_workspace_draft(state.config, ("CALO",))

    panel = PortfolioManagerPanel(state)
    qtbot.addWidget(panel)

    assert panel._selected_study_algorithms() == ("CALO",)
    assert panel._algorithm_items["CALO"].checkState(0).name == "Checked"
    assert panel._algorithm_items["TLBO"].checkState(0).name == "Unchecked"


def test_study_setup_embeds_shared_panels_without_a_second_algorithm_selector(
    qtbot, tmp_path
) -> None:
    state = AppState(tmp_path / "workspace-inline-study.sqlite")
    _submit_stage(state)
    state.execution_control.create_workspace_draft(state.config, ("CALO",))
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    titles = tuple(title for title, _description, _page in panel.study_setup_workflow.steps)
    assert titles == (
        "Case",
        "Formulation",
        "Budget + runs",
        "Scenarios",
        "Validate + outputs",
        "Review + launch",
    )
    assert "Algorithms" not in titles
    assert panel.study_setup_workflow.page_widgets["Case"] is panel.study_power_system
    assert panel.study_setup_workflow.page_widgets["Formulation"] is panel.study_formulation
    assert panel.study_setup_workflow.page_widgets["Scenarios"] is panel.study_scenarios

    requested = []
    panel.workspace_requested.connect(requested.append)
    for index in range(len(titles)):
        panel.study_setup_workflow.set_step(index)
    assert requested == []

    panel.show_context("workspace_study")
    assert "Portfolio subset · 1 algorithm(s): CALO" in panel.selected.text()
    assert "TLBO" not in panel.selected.text()
    assert "Portfolio-selected algorithms" in panel.plan_summary.text()

    panel.show_context("individual_experiment")
    assert "Complete submitted stage · 2 algorithm(s): CALO, TLBO" in panel.selected.text()
    assert "no second algorithm selector" in panel.plan_summary.text()


def test_programmatic_refresh_does_not_claim_that_an_unaudited_draft_changed(
    qtbot, tmp_path
) -> None:
    state = AppState(tmp_path / "workspace-draft-refresh.sqlite")
    _submit_stage(state)
    state.execution_control.create_workspace_draft(state.config, ("CALO",))
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    panel.refresh()

    assert panel.audit_state.text() == "Required before execution"
    assert "Configuration changed" not in panel.status.text()
    assert panel.stage_plan.isEnabled() is False

    panel.population.setValue(panel.population.value() + 1)

    assert panel.audit_state.text() == "Required — run fairness audit"
    assert "no fairness pass is recorded" in panel.status.text()
    assert panel.stage_plan.isEnabled() is False


def test_user_edit_invalidates_an_audited_plan_but_programmatic_refresh_does_not(
    qtbot, tmp_path
) -> None:
    state = AppState(tmp_path / "workspace-audited-refresh.sqlite")
    _submit_stage(state)
    _audited_workspace(state)
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    panel.refresh()

    assert panel.stage_plan.isEnabled() is True
    assert "Configuration changed" not in panel.audit_state.text()

    panel.population.setValue(panel.population.value() + 1)

    assert panel.audit_state.text() == "Configuration changed — audit required"
    assert panel.stage_plan.isEnabled() is False


def test_inline_study_panels_preserve_prerequisites_and_completion_signals(qtbot, tmp_path) -> None:
    state = AppState(tmp_path / "workspace-inline-prerequisites.sqlite")
    _submit_stage(state)
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    panel.set_study_prerequisite_states(
        "workspace",
        {
            "Case": ("locked", "Verified policy prerequisite."),
            "Formulation": ("locked", "Power-system prerequisite."),
            "Scenarios": ("locked", "Portfolio prerequisite."),
        }
    )
    assert panel.study_power_system.isEnabled() is False
    assert panel.study_formulation.isEnabled() is False
    assert panel.study_scenarios.isEnabled() is False
    assert panel.study_setup_workflow.prerequisite_labels["Case"].isHidden() is False
    assert (
        panel.study_setup_workflow.prerequisite_labels["Case"].text()
        == "Verified policy prerequisite."
    )

    completed = []
    panel.power_system_completed.connect(lambda: completed.append("power_system"))
    panel.formulation_completed.connect(lambda: completed.append("orpd"))
    panel.scenarios_completed.connect(lambda: completed.append("scenarios"))
    panel.study_power_system.stage_completed.emit()
    panel.study_formulation.stage_completed.emit()
    panel.study_scenarios.stage_completed.emit()
    assert completed == ["power_system", "orpd", "scenarios"]


def test_individual_setup_is_editable_without_workspace_portfolio_prerequisite(
    qtbot, tmp_path
) -> None:
    state = AppState(tmp_path / "individual-independent-setup.sqlite")
    _submit_stage(state)
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)
    panel.set_study_prerequisite_states(
        "workspace",
        {
            "Case": ("completed", ""),
            "Formulation": ("completed", ""),
            "Scenarios": ("locked", "Apply the evidence portfolio plan first."),
        },
    )
    panel.set_study_prerequisite_states(
        "individual_experiment",
        {
            "Case": ("completed", ""),
            "Formulation": ("completed", ""),
            "Scenarios": ("recommended", ""),
        },
    )

    panel.show_context("workspace_study")
    assert panel.study_scenarios.isEnabled() is False
    assert panel.runs.isReadOnly() is True
    assert panel.study_setup_workflow.heading.text() == "Workspace Study Setup"

    panel.show_context("individual_experiment")
    assert panel.study_scenarios.isEnabled() is True
    assert panel.runs.isReadOnly() is False
    assert panel.reuse_results.isHidden() is False
    assert panel.study_setup_workflow.heading.text() == "Individual Experiment Setup"
    assert "Portfolio" not in panel.runs.toolTip()


def test_individual_audit_uses_direct_result_contract_without_portfolio_planner(
    tmp_path, monkeypatch
) -> None:
    state = AppState(tmp_path / "individual-direct-audit.sqlite")
    _submit_stage(state)
    state.config.runs = 1
    state.config.require_backend_parity = False
    state.config.reuse_compatible_results = False
    plan = state.execution_control.create_individual_draft(state.config)
    config = state.execution_control.plan_configuration(plan["id"])

    def unexpected_portfolio_plan(*_args, **_kwargs):
        raise AssertionError("Individual audit called PortfolioPlanner")

    def unexpected_reuse_lookup(*_args, **_kwargs):
        raise AssertionError("Disabled Individual reuse queried stored runs")

    monkeypatch.setattr(
        "calo_rpd_studio.gui.panels.experiment_manager_panel.PortfolioPlanner.plan",
        unexpected_portfolio_plan,
    )
    monkeypatch.setattr(
        "calo_rpd_studio.gui.panels.experiment_manager_panel.ResultDatabase.find_reusable_run",
        unexpected_reuse_lookup,
    )
    worker = ScientificAuditWorker(config, str(state.database.path))
    completed = []
    failures = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == []
    assert completed[0]["portfolio_plan"] is None
    assert completed[0]["result_contract"]["owner"] == "individual_experiment"
    assert completed[0]["reusable"] == 0


def test_workspace_staging_freezes_individual_and_portfolio_editing(qtbot, tmp_path) -> None:
    state = AppState(tmp_path / "workspace-freeze.sqlite")
    _submit_stage(state)
    plan = _audited_workspace(state)
    state.execution_control.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    study = ExperimentManagerPanel(state, ExperimentManager(state))
    portfolio = PortfolioManagerPanel(state)
    qtbot.addWidget(study)
    qtbot.addWidget(portfolio)

    study.show_context("individual_experiment")
    portfolio.refresh()

    assert study.setup_card.isEnabled() is False
    assert study.study_power_system.isEnabled() is False
    assert study.study_formulation.isEnabled() is False
    assert study.study_scenarios.isEnabled() is False
    assert study.stage_plan.isEnabled() is False
    assert study.compare.isEnabled() is False
    assert "controls experiment execution" in study.ownership_banner.text()
    assert study.ownership_actions.isVisible() is False
    study.show()
    assert study.ownership_actions.isVisible() is True
    assert portfolio.section_tabs.isEnabled() is False


def test_durable_workspace_pause_enables_individual_setup_but_keeps_resume(qtbot, tmp_path) -> None:
    state = AppState(tmp_path / "workspace-paused-handoff.sqlite")
    _submit_stage(state)
    plan = _audited_workspace(state)
    state.execution_control.stage(plan["id"], ExecutionPlanKind.WORKSPACE)
    state.execution_control.begin_run(plan["id"])
    state.execution_control.request_pause(plan["id"])
    state.execution_control.commit_paused(plan["id"])
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    panel.show_context("individual_experiment")
    assert panel.setup_card.isEnabled() is True
    assert panel.study_power_system.isEnabled() is True
    assert panel.study_formulation.isEnabled() is True
    assert panel.study_scenarios.isEnabled() is True
    assert state.execution_control.controller()["controller"] == "none"

    panel.show_context("workspace_study")
    assert panel.resume_plan.isEnabled() is True


def test_individual_staging_blocks_workspace_stage_and_resume(qtbot, tmp_path) -> None:
    state = AppState(tmp_path / "individual-freeze.sqlite")
    _submit_stage(state)
    workspace = _audited_workspace(state)
    state.execution_control.stage(workspace["id"], ExecutionPlanKind.WORKSPACE)
    state.execution_control.begin_run(workspace["id"])
    state.execution_control.commit_paused(workspace["id"])
    individual = state.execution_control.create_individual_draft(state.config)
    state.execution_control.record_audit(individual["id"], {"fair": True})
    state.execution_control.stage(individual["id"], ExecutionPlanKind.INDIVIDUAL_EXPERIMENT)
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    qtbot.addWidget(panel)

    panel.show_context("workspace_study")

    assert panel.study_power_system.isEnabled() is False
    assert panel.study_formulation.isEnabled() is False
    assert panel.study_scenarios.isEnabled() is False
    assert panel.stage_plan.isEnabled() is False
    assert panel.resume_plan.isEnabled() is False
    assert "controls execution" in panel.ownership_banner.text()
