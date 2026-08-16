from __future__ import annotations

from calo_rpd_studio.app.experiment_manager import ExperimentManager
from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.experiments.execution_plans import ExecutionPlanKind
from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel
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

    assert panel.stage_plan.isEnabled() is False
    assert panel.resume_plan.isEnabled() is False
    assert "controls execution" in panel.ownership_banner.text()
